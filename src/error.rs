use std::fmt;

#[derive(Debug, PartialEq, Eq)]
pub enum AllocError {
    /// No free blocks and no evictable (all ref_count > 1) blocks remain.
    OutOfMemory,
    /// Caller supplied a block_id that is not currently allocated.
    InvalidBlockId(u32),
    /// Caller freed a block whose ref_count is already 0.
    DoubleFree(u32),
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::OutOfMemory => write!(f, "KV cache out of memory: no free or evictable blocks"),
            AllocError::InvalidBlockId(id) => write!(f, "invalid block id: {id}"),
            AllocError::DoubleFree(id) => write!(f, "double-free on block id: {id}"),
        }
    }
}

impl std::error::Error for AllocError {}
