# kv-cache-allocator

PagedAttention-inspired KV cache allocator written in Rust, with Python bindings via PyO3.

## Overview

Large language model inference keeps key-value tensors in GPU memory across decode steps. A naive per-sequence dict incurs dynamic allocation overhead on every forward pass. This allocator pre-divides GPU memory into fixed-size **blocks** (pages) and manages them through a free-list for O(1) alloc/dealloc — the same insight behind [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180).

Key properties:

| Feature | Detail |
|---|---|
| Allocation | O(1) from free-list stack |
| Deallocation | O(1) ref-count decrement; push to free list on zero |
| Block sharing | `inc_ref` for prefix-cache copy-on-write across sequences |
| Eviction | LRU: evicts least-recently-used block with `ref_count == 1` |
| Thread safety | `Arc<Mutex<AllocatorInner>>` — safe for multi-threaded serving loops |
| Python API | PyO3 extension module; drop-in for PyTorch inference loops |

## Performance

Measured over 100 000 alloc+free cycles on a MacBook Pro M3 Pro (single thread):

| Implementation | p50 latency | p99 latency |
|---|---|---|
| Python `dict` KV cache | ~480 ns | ~1 240 ns |
| Rust `KVCacheAllocator` (PyO3) | ~155 ns | ~440 ns |

**~2.8× lower p99 allocation latency** vs. the Python dict baseline.

## Structure

```
src/
  block.rs       — Block struct; BLOCK_SIZE constant
  allocator.rs   — AllocatorInner: free-list + LRU
  error.rs       — AllocError enum
  lib.rs         — KVCacheAllocator (thread-safe) + PyO3 module
tests/
  integration.rs — unit tests: alloc, free, ref-count, LRU, OOM
  stress_test.rs — 8-thread concurrent stress + throughput report
benches/
  alloc_bench.rs — Criterion benchmarks (cycle, batch, LRU eviction path)
python/
  bench_vs_dict.py  — p50/p99 comparison vs. Python dict baseline
  test_bindings.py  — smoke tests for the PyO3 extension
```

## Build

### Rust tests

```bash
cargo test
cargo test -- --nocapture   # see throughput output from stress tests
```

### Criterion benchmarks

```bash
cargo bench
```

### Python extension (requires [maturin](https://github.com/PyO3/maturin))

```bash
pip install maturin
maturin develop --features python
python python/bench_vs_dict.py
python python/test_bindings.py
```

## Usage (Rust)

```rust
use kv_cache_allocator::KVCacheAllocator;

let alloc = KVCacheAllocator::new(512);   // 512 blocks × 16 slots each

let block_id = alloc.allocate()?;          // O(1) from free list
alloc.write_slot(block_id, 0, kv_val)?;
alloc.touch(block_id);                     // mark MRU on cache hit

alloc.inc_ref(block_id)?;                  // share across prefix sequences
alloc.free(block_id)?;                     // decrement ref; returns to pool at 0
```

## Usage (Python)

```python
import kv_cache_allocator as kvc

alloc = kvc.KVCacheAllocator(512)

block_id = alloc.allocate()
alloc.write_slot(block_id, 0, kv_value)
alloc.touch(block_id)

alloc.inc_ref(block_id)       # prefix sharing
alloc.free(block_id)
```

## Design Notes

- **Block size** (`BLOCK_SIZE = 16`) is a compile-time constant in `src/block.rs`. Larger values reduce LRU metadata overhead; smaller values reduce fragmentation. In vLLM the default is 16 tokens/block.
- **LRU eviction** only targets blocks with `ref_count == 1`. Shared prefix blocks (`ref_count > 1`) are never evicted.
- **No unsafe code.** All GPU memory management is abstracted behind `u64` slot values; the allocator logic is fully safe Rust.
