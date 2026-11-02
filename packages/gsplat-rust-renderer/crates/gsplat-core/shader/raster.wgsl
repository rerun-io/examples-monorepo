#import <./raster_common.wgsl>

// Gaussian splat coverage, as Brush rasterize 1388f74c; 256 splats per cooperative batch.
alias Primitive = Splat;
const BATCH = WG;

fn sample(p: Splat, pix: vec2u, surface: bool) -> Sample {
    let s = sigma(vec2f(pix) + 0.5, vec2f(p.x, p.y), vec3f(p.conic_x, p.conic_y, p.conic_z));
    let alpha = min(0.999, p.color_a * exp(-s));
    return Sample(select(0.0, alpha, s >= 0.0), max(vec3f(p.color_r, p.color_g, p.color_b), vec3f(0.0)), p.depth,
        vec3f(0.0));
}
