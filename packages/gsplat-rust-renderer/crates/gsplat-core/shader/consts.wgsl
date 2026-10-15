// Shared, binding-free kernel dimensions, matching kernels.rs.
const TILE_WIDTH = 16u;
const WG = 256u;
const ELEMENTS_PER_THREAD = 4u;
const BLOCK_SIZE = WG * ELEMENTS_PER_THREAD;
const BITS_PER_PASS = 4u;
const BIN_COUNT = 16u;
const CHECKS_PER_ITER = 8u;
const ALPHA_CUTOFF = 1.0 / 255.0;
