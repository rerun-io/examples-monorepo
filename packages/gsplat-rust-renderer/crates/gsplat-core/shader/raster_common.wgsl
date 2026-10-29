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

fn raster(tile: vec2u, lid: u32, surface: bool) -> Raster {
    var out: Raster;
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
        for (var t = 0u; !done && t < remaining; t++) {
            let s = sample(batch[t], out.pix, surface);
            if !(s.alpha >= ALPHA_CUTOFF) {
                continue;
            }
            let next = transmittance * (1.0 - s.alpha);
            if next <= 1e-4 {
                done = true;
                continue;
            }
            let w = s.alpha * transmittance;
            color += s.color * w;
            out.depth += s.depth * w;
            // 2DGS median depth: the last contribution while transmittance exceeds one half.
            out.median = select(out.median, s.depth, transmittance > 0.5);
            out.normal += s.normal * w;
            transmittance = next;
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
