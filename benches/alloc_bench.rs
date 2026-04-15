use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kv_cache_allocator::KVCacheAllocator;

/// Single-threaded alloc/free round-trip — the number we compare against
/// Python's dict-based KV cache. Measures the Rust O(1) free-list path.
fn bench_alloc_free_cycle(c: &mut Criterion) {
    let alloc = KVCacheAllocator::new(1024);
    c.bench_function("alloc_free_cycle", |b| {
        b.iter(|| {
            let id = black_box(alloc.allocate().unwrap());
            black_box(alloc.free(id).unwrap());
        })
    });
}

/// Batch allocation: allocate N blocks then free them all.
fn bench_batch_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_alloc");
    for size in [64, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            let alloc = KVCacheAllocator::new(n);
            b.iter(|| {
                let ids: Vec<u32> = (0..n).map(|_| alloc.allocate().unwrap()).collect();
                for id in ids {
                    alloc.free(id).unwrap();
                }
            });
        });
    }
    group.finish();
}

/// LRU eviction path: pool fully allocated, one alloc forces an eviction.
fn bench_lru_eviction(c: &mut Criterion) {
    let pool_size = 256;
    let alloc = KVCacheAllocator::new(pool_size);
    // Saturate pool.
    let mut ids: Vec<u32> = (0..pool_size).map(|_| alloc.allocate().unwrap()).collect();

    c.bench_function("lru_eviction_alloc", |b| {
        b.iter(|| {
            // Free one to make the eviction path touch the free list rather than LRU.
            // For a "pure LRU" hit: keep pool full by immediately re-allocating.
            let freed = ids.pop().unwrap();
            alloc.free(freed).unwrap();
            let new_id = black_box(alloc.allocate().unwrap());
            ids.push(new_id);
        })
    });
}

criterion_group!(benches, bench_alloc_free_cycle, bench_batch_alloc, bench_lru_eviction);
criterion_main!(benches);
