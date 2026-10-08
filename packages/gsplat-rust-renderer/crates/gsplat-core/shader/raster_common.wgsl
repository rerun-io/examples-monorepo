#import <./common.wgsl>

// Hand port of Brush rasterize 1388f74c. 256 splats per cooperative batch.
@group(0) @binding(1) var<storage, read> isect_ids: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<Splat>;
@group(0) @binding(7) var<storage, read> counts: array<u32>;
var<workgroup> batch: array<Splat, 256>;
var<workgroup> range_lo: u32;
var<workgroup> range_hi: u32;
var<workgroup> num_done: atomic<u32>;
var<workgroup> done_snapshot: u32;
fn compact_bits(v: u32) -> u32 {
    var x = v & 0x55u;
    x = (x | (x >> 1u)) & 0x33u;
    return (x | (x >> 2u)) & 0x0fu;
}

fn pixel(tile: u32, lid: u32) -> vec2u {
    return vec2u(tile % u.image.tiles_x, tile / u.image.tiles_x) * 16u + vec2u(compact_bits(lid), compact_bits(lid >> 1u));
}

struct Raster { rgba: vec4f, depth: f32, pix: vec2u, inside: bool, }
fn raster(tile: u32, lid: u32) -> Raster {
    if lid == 0u {
        range_lo = select(0u, 1u, (counts[0] & 0x80000000u) != 0u || counts[1] > arrayLength(&isect_ids));
    }
    if workgroupUniformLoad(&range_lo) != 0u || tile >= u.image.tiles_x * u.image.tiles_y {
        return Raster(vec4f(0.0), 0.0, vec2u(0u), false);
    }
    let pix = pixel(tile, lid);
    let inside = all(pix < vec2u(u.image.width, u.image.height));
    if lid == 0u {
        range_lo = offsets[tile * 2u];
        range_hi = offsets[tile * 2u + 1u];
        atomicStore(&num_done, 0u);
    }
    let lo = workgroupUniformLoad(&range_lo);
    let hi = workgroupUniformLoad(&range_hi);
    var transmittance = 1.0;
    var color = vec3f(0.0);
    var expected_depth = 0.0;
    var done = !inside;
    if done {
        atomicAdd(&num_done, 1u);
    }
    for (var start = lo; start < hi; start += 256u) {
        workgroupBarrier();
        if lid == 0u {
            done_snapshot = atomicLoad(&num_done);
        }
        if workgroupUniformLoad(&done_snapshot) >= 256u {
            break;
        }
        let remaining = min(256u, hi - start);
        if lid < remaining {
            batch[lid] = projected[isect_ids[start + lid]];
        }
        workgroupBarrier();
        let was_done = done;
        for (var t = 0u; !done && t < remaining; t++) {
            let p = batch[t];
            let s = sigma(vec2f(pix) + 0.5, vec2f(p.x, p.y), vec3f(p.cx, p.cy, p.cz));
            let alpha = min(0.999, p.opacity * exp(-s));
            if s >= 0.0 && alpha >= 1.0 / 255.0 {
                let next = transmittance * (1.0 - alpha);
                if next <= 1e-4 {
                    done = true;
                } else {
                    color += max(vec3f(p.r, p.g, p.b), vec3f(0.0)) * (alpha * transmittance);
                    expected_depth += p.depth * alpha * transmittance;
                    transmittance = next;
                }
            }
        }
        if !was_done && done {
            atomicAdd(&num_done, 1u);
        }
    }
    return Raster(vec4f(color + transmittance * u.background.xyz, 1.0 - transmittance),
        expected_depth / max(1.0 - transmittance, 1e-8), pix, inside);
}
