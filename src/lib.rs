mod allocator;
mod block;
mod error;

pub use block::BLOCK_SIZE;
pub use error::AllocError;

use std::sync::{Arc, Mutex};

use allocator::AllocatorInner;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Thread-safe KV cache allocator.
///
/// Cloning a `KVCacheAllocator` is cheap — all clones share the same backing pool.
/// Suitable for passing across thread boundaries or into Python via PyO3.
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

// ── PyO3 bindings ────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
/// Python-facing wrapper. The inner `KVCacheAllocator` is `Clone + Send + Sync`
/// so it survives the GIL release across threads.
#[pyclass(name = "KVCacheAllocator")]
pub struct PyKVCacheAllocator {
    alloc: KVCacheAllocator,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyKVCacheAllocator {
    #[new]
    fn new(num_blocks: usize) -> Self {
        Self {
            alloc: KVCacheAllocator::new(num_blocks),
        }
    }

    fn allocate(&self) -> PyResult<u32> {
        self.alloc.allocate().map_err(to_py_err)
    }

    fn free(&self, block_id: u32) -> PyResult<()> {
        self.alloc.free(block_id).map_err(to_py_err)
    }

    fn inc_ref(&self, block_id: u32) -> PyResult<()> {
        self.alloc.inc_ref(block_id).map_err(to_py_err)
    }

    fn touch(&self, block_id: u32) {
        self.alloc.touch(block_id)
    }

    fn write_slot(&self, block_id: u32, slot: usize, value: u64) -> PyResult<()> {
        self.alloc.write_slot(block_id, slot, value).map_err(to_py_err)
    }

    fn read_slots(&self, block_id: u32) -> PyResult<Vec<u64>> {
        self.alloc
            .read_slots(block_id)
            .ok_or_else(|| to_py_err(AllocError::InvalidBlockId(block_id)))
    }

    fn ref_count(&self, block_id: u32) -> PyResult<u32> {
        self.alloc
            .ref_count(block_id)
            .ok_or_else(|| to_py_err(AllocError::InvalidBlockId(block_id)))
    }

    fn free_count(&self) -> usize {
        self.alloc.free_count()
    }

    fn allocated_count(&self) -> usize {
        self.alloc.allocated_count()
    }

    fn total_blocks(&self) -> usize {
        self.alloc.total_blocks()
    }

    fn __repr__(&self) -> String {
        format!(
            "KVCacheAllocator(total={}, allocated={}, free={})",
            self.alloc.total_blocks(),
            self.alloc.allocated_count(),
            self.alloc.free_count(),
        )
    }
}

#[cfg(feature = "python")]
fn to_py_err(e: AllocError) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[cfg(feature = "python")]
#[pymodule]
fn kv_cache_allocator(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKVCacheAllocator>()?;
    m.add("BLOCK_SIZE", BLOCK_SIZE)?;
    Ok(())
}
