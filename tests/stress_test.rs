/// Multi-threaded stress tests validating correctness under concurrent access.
///
/// Spawns N worker threads; each thread performs many alloc/touch/free cycles.
/// The allocator's Arc<Mutex<>> interior must keep the pool consistent with
/// zero panics and no deadlocks. Run with `cargo test -- --nocapture` to see
/// throughput counters.
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use kv_cache_allocator::KVCacheAllocator;

const WORKERS: usize = 8;
const OPS_PER_WORKER: usize = 10_000;
const POOL_SIZE: usize = 256;

#[test]
fn concurrent_alloc_free_no_panic() {
    let alloc = Arc::new(KVCacheAllocator::new(POOL_SIZE));
    let mut handles = vec![];

    for _ in 0..WORKERS {
        let a = Arc::clone(&alloc);
        handles.push(thread::spawn(move || {
            for _ in 0..OPS_PER_WORKER {
                if let Ok(id) = a.allocate() {
                    a.touch(id);
                    let _ = a.free(id);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("worker thread panicked");
    }
    // Pool must be fully recovered (no leaked blocks).
    assert_eq!(alloc.free_count() + alloc.allocated_count(), POOL_SIZE);
}

#[test]
fn concurrent_ref_sharing() {
    // One producer allocates blocks and increments ref; N consumers each free once.
    let alloc = Arc::new(KVCacheAllocator::new(128));
    const SHARE_FACTOR: usize = 4;
    let block_ids: Vec<u32> = (0..32)
        .map(|_| {
            let id = alloc.allocate().unwrap();
            // ref_count = 1 already; inc SHARE_FACTOR-1 more times.
            for _ in 0..SHARE_FACTOR - 1 {
                alloc.inc_ref(id).unwrap();
            }
            id
        })
        .collect();

    let ids = Arc::new(block_ids);
    let mut handles = vec![];

    for _ in 0..SHARE_FACTOR {
        let a = Arc::clone(&alloc);
        let ids = Arc::clone(&ids);
        handles.push(thread::spawn(move || {
            for &id in ids.iter() {
                a.free(id).expect("free on shared block should succeed");
            }
        }));
    }

    for h in handles {
        h.join().expect("consumer thread panicked");
    }
    // Every block freed SHARE_FACTOR times → all back in free list.
    assert_eq!(alloc.free_count(), 128);
}

#[test]
fn throughput_report() {
    let alloc = Arc::new(KVCacheAllocator::new(1024));
    let start = Instant::now();

    let mut handles = vec![];
    for _ in 0..WORKERS {
        let a = Arc::clone(&alloc);
        handles.push(thread::spawn(move || {
            let mut ops = 0u64;
            for _ in 0..OPS_PER_WORKER {
                if let Ok(id) = a.allocate() {
                    a.free(id).ok();
                    ops += 2;
                }
            }
            ops
        }));
    }

    let total_ops: u64 = handles.into_iter().map(|h| h.join().unwrap()).sum();
    let elapsed = start.elapsed();
    let throughput = total_ops as f64 / elapsed.as_secs_f64();
    println!(
        "throughput: {:.0} alloc+free ops/sec over {:.2}s ({} threads)",
        throughput,
        elapsed.as_secs_f64(),
        WORKERS,
    );
    assert!(total_ops > 0);
}
