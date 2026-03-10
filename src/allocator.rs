use std::collections::{HashMap, VecDeque};

use crate::block::{Block, BLOCK_SIZE};
use crate::error::AllocError;

/// Core allocator state. Not thread-safe on its own — wrap in `Arc<Mutex<>>`.
///
/// Design:
///   - `free_list`: a Vec used as a stack for O(1) push/pop.
///   - `lru`:       a VecDeque where front = LRU and back = MRU.
///   - `blocks`:    flat Vec owning all Block structs by index == block_id.
///   - `allocated`: HashMap<block_id, ()> for O(1) membership test.
///
/// On allocation:
///   1. Pop from free_list. If empty, walk LRU front-to-back for the first
///      block with ref_count == 1 (evictable), reclaim it, and return it.
///   2. Set ref_count = 1, zero slots, push to MRU end of lru.
///
/// On deallocation:
///   Decrement ref_count. When it hits 0, push to free_list and remove from lru.
pub struct AllocatorInner {
    blocks: Vec<Block>,
    free_list: Vec<u32>,
    lru: VecDeque<u32>,
    allocated: HashMap<u32, ()>,
}

impl AllocatorInner {
    pub fn new(num_blocks: usize) -> Self {
        let blocks: Vec<Block> = (0..num_blocks as u32).map(Block::new).collect();
        // Initially all blocks are free; push in reverse so pop() yields 0, 1, 2...
        let free_list: Vec<u32> = (0..num_blocks as u32).rev().collect();
        Self {
            blocks,
            free_list,
            lru: VecDeque::new(),
            allocated: HashMap::new(),
        }
    }

    /// Allocate one block. O(1) from free list; O(n) worst-case LRU scan.
    pub fn allocate(&mut self) -> Result<u32, AllocError> {
        let block_id = if let Some(id) = self.free_list.pop() {
            id
        } else {
            self.evict_lru().ok_or(AllocError::OutOfMemory)?
        };

        let block = &mut self.blocks[block_id as usize];
        block.reset();
        block.ref_count = 1;
        self.allocated.insert(block_id, ());
        self.lru.push_back(block_id);
        Ok(block_id)
    }

    /// Decrement ref_count; return block to free list when it hits 0.
    pub fn free(&mut self, block_id: u32) -> Result<(), AllocError> {
        if !self.allocated.contains_key(&block_id) {
            return Err(AllocError::InvalidBlockId(block_id));
        }
        let block = &mut self.blocks[block_id as usize];
        if block.ref_count == 0 {
            return Err(AllocError::DoubleFree(block_id));
        }
        block.ref_count -= 1;
        if block.ref_count == 0 {
            self.allocated.remove(&block_id);
            self.free_list.push(block_id);
            self.lru_remove(block_id);
        }
        Ok(())
    }

    /// Increment ref_count (prefix-cache sharing: multiple seqs reference one block).
    pub fn inc_ref(&mut self, block_id: u32) -> Result<(), AllocError> {
        if !self.allocated.contains_key(&block_id) {
            return Err(AllocError::InvalidBlockId(block_id));
        }
        self.blocks[block_id as usize].ref_count += 1;
        Ok(())
    }

    /// Mark block as most-recently-used.
    pub fn touch(&mut self, block_id: u32) {
        if self.allocated.contains_key(&block_id) {
            self.lru_remove(block_id);
            self.lru.push_back(block_id);
        }
    }

    /// Write a value into a slot of an allocated block.
    pub fn write_slot(&mut self, block_id: u32, slot: usize, value: u64) -> Result<(), AllocError> {
        if !self.allocated.contains_key(&block_id) {
            return Err(AllocError::InvalidBlockId(block_id));
        }
        if slot >= BLOCK_SIZE {
            return Err(AllocError::InvalidBlockId(block_id));
        }
        self.blocks[block_id as usize].slots[slot] = value;
        Ok(())
    }

    /// Read slots of an allocated block.
    pub fn read_slots(&self, block_id: u32) -> Option<&[u64; BLOCK_SIZE]> {
        if self.allocated.contains_key(&block_id) {
            Some(&self.blocks[block_id as usize].slots)
        } else {
            None
        }
    }

    pub fn ref_count(&self, block_id: u32) -> Option<u32> {
        self.allocated
            .contains_key(&block_id)
            .then(|| self.blocks[block_id as usize].ref_count)
    }

    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }

    pub fn allocated_count(&self) -> usize {
        self.allocated.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.blocks.len()
    }

    // Find and evict the least-recently-used block with ref_count == 1.
    fn evict_lru(&mut self) -> Option<u32> {
        let pos = self
            .lru
            .iter()
            .position(|&id| self.blocks[id as usize].ref_count == 1)?;
        let id = self.lru.remove(pos).unwrap();
        self.allocated.remove(&id);
        Some(id)
    }

    fn lru_remove(&mut self, block_id: u32) {
        if let Some(pos) = self.lru.iter().position(|&id| id == block_id) {
            self.lru.remove(pos);
        }
    }
}
