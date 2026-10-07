// Visible splats (high bit is sticky arithmetic overflow) and intersections.
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32 >>;
fn add_intersections(amount: u32) {
    let previous = atomicAdd(&counts[1], amount);
    if previous> 0xffffffffu - amount {
        atomicOr(&counts[0], 0x80000000u);
    }
}
