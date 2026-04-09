"""
Basic smoke tests for the PyO3 KVCacheAllocator bindings.

Build first:
    maturin develop --features python
Then run:
    python python/test_bindings.py
"""

import kv_cache_allocator as kvc


def test_basic_alloc_free():
    alloc = kvc.KVCacheAllocator(64)
    assert alloc.free_count() == 64
    block_id = alloc.allocate()
    assert alloc.free_count() == 63
    alloc.free(block_id)
    assert alloc.free_count() == 64
    print("PASS test_basic_alloc_free")


def test_ref_counting():
    alloc = kvc.KVCacheAllocator(8)
    bid = alloc.allocate()
    assert alloc.ref_count(bid) == 1
    alloc.inc_ref(bid)
    assert alloc.ref_count(bid) == 2
    alloc.free(bid)
    assert alloc.ref_count(bid) == 1   # still alive
    alloc.free(bid)
    assert alloc.free_count() == 8
    print("PASS test_ref_counting")


def test_lru_eviction():
    alloc = kvc.KVCacheAllocator(4)
    ids = [alloc.allocate() for _ in range(4)]
    # Free 3; last one stays allocated (evictable via LRU).
    for bid in ids[:3]:
        alloc.free(bid)
    # Should be able to allocate 4 more via free list + LRU eviction.
    new_ids = [alloc.allocate() for _ in range(4)]
    assert len(new_ids) == 4
    print("PASS test_lru_eviction")


def test_write_read_slots():
    alloc = kvc.KVCacheAllocator(4)
    bid = alloc.allocate()
    alloc.write_slot(bid, 0, 0xDEADBEEF)
    alloc.write_slot(bid, 15, 0xCAFEBABE)
    slots = alloc.read_slots(bid)
    assert slots[0] == 0xDEADBEEF
    assert slots[15] == 0xCAFEBABE
    print("PASS test_write_read_slots")


def test_oom_error():
    alloc = kvc.KVCacheAllocator(2)
    ids = [alloc.allocate() for _ in range(2)]
    for bid in ids:
        alloc.inc_ref(bid)  # ref_count = 2, not evictable
    try:
        alloc.allocate()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
    print("PASS test_oom_error")


def test_repr():
    alloc = kvc.KVCacheAllocator(16)
    alloc.allocate()
    r = repr(alloc)
    assert "KVCacheAllocator" in r
    print(f"PASS test_repr: {r}")


if __name__ == "__main__":
    test_basic_alloc_free()
    test_ref_counting()
    test_lru_eviction()
    test_write_read_slots()
    test_oom_error()
    test_repr()
    print("\nAll binding tests passed.")
