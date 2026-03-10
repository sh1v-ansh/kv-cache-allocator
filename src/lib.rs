mod allocator;
mod block;
mod error;

pub use block::BLOCK_SIZE;
pub use error::AllocError;

use std::sync::{Arc, Mutex};
use allocator::AllocatorInner;

/// Thread-safe KV cache allocator backed by a pre-allocated block pool.
///
/// Clone is cheap — all clones share the same backing pool via Arc.
#[derive(Clone)]
pub struct KVCacheAllocator {
    inner: Arc<Mutex<AllocatorInner>>,
}

impl KVCacheAllocator {
    pub fn new(num_blocks: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(AllocatorInner::new(num_blocks))),
        }
    }

    /// Allocate one block. O(1) from free list; falls back to LRU eviction.
    pub fn allocate(&self) -> Result<u32, AllocError> {
        self.inner.lock().unwrap().allocate()
    }

    /// Decrement ref_count; returns block to free list when count reaches 0.
    pub fn free(&self, block_id: u32) -> Result<(), AllocError> {
        self.inner.lock().unwrap().free(block_id)
    }

    /// Increment ref_count for prefix-cache block sharing.
    pub fn inc_ref(&self, block_id: u32) -> Result<(), AllocError> {
        self.inner.lock().unwrap().inc_ref(block_id)
    }

    /// Mark block as most-recently-used (call on every cache hit).
    pub fn touch(&self, block_id: u32) {
        self.inner.lock().unwrap().touch(block_id)
    }

    pub fn write_slot(&self, block_id: u32, slot: usize, value: u64) -> Result<(), AllocError> {
        self.inner.lock().unwrap().write_slot(block_id, slot, value)
    }

    pub fn read_slots(&self, block_id: u32) -> Option<Vec<u64>> {
        self.inner
            .lock()
            .unwrap()
            .read_slots(block_id)
            .map(|s| s.to_vec())
    }

    pub fn ref_count(&self, block_id: u32) -> Option<u32> {
        self.inner.lock().unwrap().ref_count(block_id)
    }

    pub fn free_count(&self) -> usize {
        self.inner.lock().unwrap().free_count()
    }

    pub fn allocated_count(&self) -> usize {
        self.inner.lock().unwrap().allocated_count()
    }

    pub fn total_blocks(&self) -> usize {
        self.inner.lock().unwrap().total_blocks()
    }
}
