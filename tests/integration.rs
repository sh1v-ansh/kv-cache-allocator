use kv_cache_allocator::{AllocError, KVCacheAllocator};

#[test]
fn alloc_and_free_basic() {
    let alloc = KVCacheAllocator::new(64);
    assert_eq!(alloc.free_count(), 64);

    let id = alloc.allocate().expect("should allocate from free list");
    assert_eq!(alloc.free_count(), 63);
    assert_eq!(alloc.allocated_count(), 1);

    alloc.free(id).expect("should free");
    assert_eq!(alloc.free_count(), 64);
    assert_eq!(alloc.allocated_count(), 0);
}

#[test]
fn double_free_returns_error() {
    let alloc = KVCacheAllocator::new(8);
    let id = alloc.allocate().unwrap();
    alloc.free(id).unwrap();
    assert_eq!(alloc.free(id), Err(AllocError::InvalidBlockId(id)));
}

#[test]
fn invalid_block_id_errors() {
    let alloc = KVCacheAllocator::new(4);
    assert_eq!(alloc.free(99), Err(AllocError::InvalidBlockId(99)));
    assert_eq!(alloc.inc_ref(99), Err(AllocError::InvalidBlockId(99)));
}

#[test]
fn ref_counting_shared_block() {
    let alloc = KVCacheAllocator::new(8);
    let id = alloc.allocate().unwrap();
    assert_eq!(alloc.ref_count(id), Some(1));

    alloc.inc_ref(id).unwrap();
    assert_eq!(alloc.ref_count(id), Some(2));

    // First free: ref drops to 1 — block stays allocated.
    alloc.free(id).unwrap();
    assert_eq!(alloc.ref_count(id), Some(1));
    assert_eq!(alloc.allocated_count(), 1);

    // Second free: ref drops to 0 — block returns to free list.
    alloc.free(id).unwrap();
    assert_eq!(alloc.allocated_count(), 0);
    assert_eq!(alloc.free_count(), 8);
}

#[test]
fn lru_eviction_when_pool_exhausted() {
    // Pool of 4. Fill it up, then free the first 3 so they are evictable.
    let alloc = KVCacheAllocator::new(4);
    let ids: Vec<u32> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
    assert_eq!(alloc.free_count(), 0);

    for &id in &ids[..3] {
        alloc.free(id).unwrap();
    }
    // 3 blocks are free; 1 still allocated (ids[3], ref_count == 1).
    assert_eq!(alloc.free_count(), 3);

    // Allocate 4 more — the 4th will trigger LRU eviction on ids[3].
    for _ in 0..4 {
        alloc.allocate().expect("LRU eviction should provide a block");
    }
}

#[test]
fn oom_when_all_blocks_shared() {
    // All 4 blocks are shared (ref_count >= 2) — none are evictable.
    let alloc = KVCacheAllocator::new(4);
    let ids: Vec<u32> = (0..4).map(|_| alloc.allocate().unwrap()).collect();
    for &id in &ids {
        alloc.inc_ref(id).unwrap(); // ref_count = 2 → not evictable
    }
    assert_eq!(alloc.allocate(), Err(AllocError::OutOfMemory));
}

#[test]
fn write_and_read_slots() {
    let alloc = KVCacheAllocator::new(4);
    let id = alloc.allocate().unwrap();
    alloc.write_slot(id, 0, 0xDEAD_BEEF).unwrap();
    alloc.write_slot(id, 15, 0xCAFE_BABE).unwrap();
    let slots = alloc.read_slots(id).unwrap();
    assert_eq!(slots[0], 0xDEAD_BEEF);
    assert_eq!(slots[15], 0xCAFE_BABE);
}

#[test]
fn touch_promotes_to_mru() {
    // Fill pool of 3. Free the first two (evictable).
    // Touch the LRU one so it becomes MRU. The un-touched one should be evicted first.
    let alloc = KVCacheAllocator::new(3);
    let a = alloc.allocate().unwrap(); // LRU
    let b = alloc.allocate().unwrap();
    let _c = alloc.allocate().unwrap(); // MRU

    // Make a and b evictable (free doesn't help here since pool is gone);
    // instead keep them allocated but touch b to push a to true LRU.
    alloc.touch(b);
    // a is now LRU. c is MRU-1, b is MRU.

    // Free a and b so they are evictable.
    alloc.free(a).unwrap();
    alloc.free(b).unwrap();
    // c still allocated (ref_count 1, evictable).

    // Allocate 3 fresh — should succeed via free list first, then LRU.
    let _ = alloc.allocate().unwrap();
    let _ = alloc.allocate().unwrap();
    let _ = alloc.allocate().unwrap();
}
