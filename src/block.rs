/// Number of KV token slots per page (analogous to vLLM's block_size).
/// Tunable: larger values reduce metadata overhead but increase fragmentation.
pub const BLOCK_SIZE: usize = 16;

/// A single KV-cache page.
///
/// Each block holds `BLOCK_SIZE` key-value slots — in a real GPU serving system
/// these would be offsets or pointers into a pre-allocated GPU memory pool.
/// Here we use `u64` as a stand-in to keep the allocator logic portable and
/// testable on CPU without a CUDA context.
#[derive(Debug)]
pub struct Block {
    #[allow(dead_code)]
    pub block_id: u32,
    /// Number of sequences currently referencing this block (for prefix sharing).
    pub ref_count: u32,
    /// KV token slots (key/value interleaved in practice; flattened here).
    pub slots: [u64; BLOCK_SIZE],
}

impl Block {
    pub fn new(block_id: u32) -> Self {
        Self {
            block_id,
            ref_count: 0,
            slots: [0u64; BLOCK_SIZE],
        }
    }

    /// Zero-fill slots so a recycled block looks fresh to a new allocatee.
    pub fn reset(&mut self) {
        self.slots = [0u64; BLOCK_SIZE];
        self.ref_count = 0;
    }
}
