"""
Benchmark: Rust KVCacheAllocator vs. Python dict-based KV cache.

Measures p50 / p99 allocation latency for N=100_000 alloc+free cycles.
Build the extension first: maturin develop --features python

Usage:
    python python/bench_vs_dict.py
"""

import time
import statistics
import sys

# ── Python dict-based baseline ────────────────────────────────────────────────

class DictKVCache:
    """Naive Python KV cache: dict maps seq_id -> list of token KV pairs."""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.pool: dict[int, list] = {}
        self.free_ids = list(range(num_blocks))
        self.block_size = block_size

    def allocate(self) -> int:
        if not self.free_ids:
            raise MemoryError("OOM")
        bid = self.free_ids.pop()
        self.pool[bid] = [0] * self.block_size
        return bid

    def free(self, block_id: int):
        if block_id in self.pool:
            del self.pool[block_id]
            self.free_ids.append(block_id)


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def measure_latencies(alloc_fn, free_fn, n: int) -> list[float]:
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        bid = alloc_fn()
        free_fn(bid)
        latencies.append(time.perf_counter_ns() - t0)
    return latencies


def report(label: str, latencies: list[float]):
    p50 = statistics.median(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]
    mean = statistics.mean(latencies)
    print(f"{label:30s}  p50={p50:8.1f} ns  p99={p99:8.1f} ns  mean={mean:8.1f} ns")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    N = 100_000
    POOL = 1024

    print(f"Benchmarking {N:,} alloc+free cycles, pool={POOL} blocks\n")

    # Python baseline
    py_cache = DictKVCache(POOL)
    py_lat = measure_latencies(py_cache.allocate, py_cache.free, N)
    report("Python dict KV cache", py_lat)

    # Rust extension
    try:
        import kv_cache_allocator as kvc
        rs_alloc = kvc.KVCacheAllocator(POOL)
        rs_lat = measure_latencies(rs_alloc.allocate, rs_alloc.free, N)
        report("Rust KVCacheAllocator (PyO3)", rs_lat)

        py_p99 = statistics.quantiles(py_lat, n=100)[98]
        rs_p99 = statistics.quantiles(rs_lat, n=100)[98]
        speedup = py_p99 / rs_p99
        print(f"\nSpeedup (p99 latency): {speedup:.2f}x lower for Rust allocator")
    except ImportError:
        print("\nRust extension not built. Run: maturin develop --features python", file=sys.stderr)


if __name__ == "__main__":
    main()
