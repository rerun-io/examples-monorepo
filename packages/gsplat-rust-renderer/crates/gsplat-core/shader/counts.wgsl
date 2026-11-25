// Visible splats (high bit is sticky arithmetic overflow) and intersections, then from word TILE_HITS (256 bytes,
// the portable storage offset alignment) the tile count of each projected primitive by global id. Sharing one
// buffer keeps project_forward within eight storage buffers.
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;
const TILE_HITS = 64u;
fn add_intersections(amount: u32) {
    let previous = atomicAdd(&counts[1], amount);
    if previous > 0xffffffffu - amount {
        atomicOr(&counts[0], 0x80000000u);
    }
}

fn add_tile_hits(id: u32, hit: u32) {
    atomicStore(&counts[TILE_HITS + id], hit);
    add_intersections(hit);
}
