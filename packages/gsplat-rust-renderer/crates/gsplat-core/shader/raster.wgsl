#import <./raster_common.wgsl>

// Gaussian splat coverage, as Brush rasterize 1388f74c. 232 splats of 40 bytes and their block masks fill the
// core's 10 KiB workgroup budget.
alias Primitive = Splat;
const BATCH = 232u;
var<workgroup> blocks: array<u32, BATCH>;

// Bit b: some pixel centre of the 8x4 block of lids 32b..32b+31 (Morton order) gets alpha >= 1/255. With
// 32-wide subgroups the skip in `reaches` is uniform, and the block's pixels skip the sample together.
fn stage(t: u32, p: Splat, tile: vec2u) {
    let xy = vec2f(p.x, p.y);
    let c = vec3f(p.conic_x, p.conic_y, p.conic_z);
    let power = log(p.color_a * 255.0);
    var mask = 0u;
    for (var b = 0u; b < 8u; b++) {
        let lo = vec2f(pixel(tile, b * 32u)) + 0.5;
        mask |= select(0u, 1u << b, rect_hit(lo, lo + vec2f(7.0, 3.0), xy, c, power));
    }
    blocks[t] = mask;
}

fn reaches(t: u32, lid: u32) -> bool {
    return (blocks[t] & (1u << (lid >> 5u))) != 0u;
}

fn sample(p: Splat, pix: vec2u, surface: bool) -> Sample {
    let s = sigma(vec2f(pix) + 0.5, vec2f(p.x, p.y), vec3f(p.conic_x, p.conic_y, p.conic_z));
    let alpha = min(0.999, p.color_a * exp(-s));
    return Sample(select(0.0, alpha, s >= 0.0), max(vec3f(p.color_r, p.color_g, p.color_b), vec3f(0.0)), p.depth,
        vec3f(0.0));
}
