#import <./raster_outputs.wgsl>

// Front-to-back tile rasterizer (hand port of Brush rasterize 1388f74c) for any primitive. The importing
// module defines `alias Primitive`, `const BATCH` (BATCH primitives fit the 10 KiB workgroup budget) and
// `fn sample(p: Primitive, pix: vec2u, surface: bool) -> Sample`; alpha below the cutoff is a miss.
struct Sample { alpha: f32, color: vec3f, depth: f32, normal: vec3f, }
@group(0) @binding(1) var<storage, read> compact_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<Primitive>;
@group(0) @binding(7) var<storage, read> counts: array<u32>;
var<workgroup> batch: array<Primitive, BATCH>;
var<workgroup> skip: u32;
var<workgroup> range_lo: u32;
var<workgroup> range_hi: u32;
var<workgroup> num_done: atomic<u32>;

// Composite candidate t front to back; nothing once the pixel is done or past the batch, and alpha below the
// cutoff is a miss.
fn blend(start: u32, t: u32, remaining: u32, pix: vec2u, surface: bool, transmittance: ptr<function, f32>,
    color: ptr<function, vec3f>, done: ptr<function, bool>, out: ptr<function, Raster>) {
    if *done || t >= remaining {
        return;
    }
    let s = sample(batch[t], pix, surface);
    if !(s.alpha >= ALPHA_CUTOFF) {
        return;
    }
    let next = *transmittance * (1.0 - s.alpha);
    if next <= 1e-4 {
        *done = true;
        return;
    }
    let w = s.alpha * *transmittance;
    *color += s.color * w;
    (*out).depth += s.depth * w;
    // 2DGS median depth and its primitive: the last contribution while transmittance exceeds one half.
    let median = *transmittance > 0.5;
    (*out).median = select((*out).median, s.depth, median);
    (*out).isect = select((*out).isect, start + t, median);
    (*out).normal += s.normal * w;
    *transmittance = next;
}

fn raster(tile: vec2u, lid: u32, surface: bool) -> Raster {
    var out: Raster;
    out.isect = NO_ID;
    if lid == 0u {
        skip = select(0u, 1u, (counts[0] & 0x80000000u) != 0u || counts[1] > arrayLength(&compact_from_isect));
    }
    if workgroupUniformLoad(&skip) != 0u {
        return out;
    }
    out.pix = pixel(tile, lid);
    out.inside = all(out.pix < vec2u(frame.image.width, frame.image.height));
    if lid == 0u {
        let index = (tile.x + tile.y * frame.image.tiles_x) * 2u;
        range_lo = offsets[index];
        range_hi = offsets[index + 1u];
        atomicStore(&num_done, 0u);
    }
    let lo = workgroupUniformLoad(&range_lo);
    let hi = workgroupUniformLoad(&range_hi);
    var transmittance = 1.0;
    var color = vec3f(0.0);
    var done = !out.inside;
    if done {
        atomicAdd(&num_done, 1u);
    }
    for (var start = lo; start < hi; start += BATCH) {
        if workgroupUniformLoad(&num_done) >= WG {
            break;
        }
        let remaining = min(BATCH, hi - start);
        if lid < remaining {
            batch[lid] = projected[compact_from_isect[start + lid]];
        }
        workgroupBarrier();
        let was_done = done;
        // Four candidates per iteration: on Apple GPUs the loop overhead (naga's loop bound, the exit tests) is a
        // large share of a cheap candidate. The order and every early exit are those of one per iteration.
        for (var t = 0u; !done && t < remaining; t += 4u) {
            blend(start, t, remaining, out.pix, surface, &transmittance, &color, &done, &out);
            blend(start, t + 1u, remaining, out.pix, surface, &transmittance, &color, &done, &out);
            blend(start, t + 2u, remaining, out.pix, surface, &transmittance, &color, &done, &out);
            blend(start, t + 3u, remaining, out.pix, surface, &transmittance, &color, &done, &out);
        }
        if !was_done && done {
            atomicAdd(&num_done, 1u);
        }
    }
    let alpha = 1.0 - transmittance;
    out.rgba = vec4f(color + transmittance * frame.background.xyz, alpha);
    out.depth /= max(alpha, 1e-8);
    out.normal /= max(alpha, 1e-8);
    return out;
}
