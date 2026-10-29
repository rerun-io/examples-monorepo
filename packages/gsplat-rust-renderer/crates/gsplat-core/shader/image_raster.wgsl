#import <./image_hexel.wgsl>

// Top-10 normalized blend: no transmittance, opacity or depth order.
@group(0) @binding(1) var<storage, read> compact_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<ImageHexel>;
@group(0) @binding(7) var<storage, read> counts: array<u32>;
const BATCH = 128u;
var<workgroup> batch: array<ImageHexel, BATCH>;
var<workgroup> skip: u32;
var<workgroup> range_lo: u32;
var<workgroup> range_hi: u32;
fn compact_bits(v: u32) -> u32 {
    var x = v & 0x55u;
    x = (x | (x >> 1u)) & 0x33u;
    return (x | (x >> 2u)) & 0x0fu;
}

fn weight(h: ImageHexel, d: vec2f) -> f32 {
    let u = (d.x * h.cos_theta + d.y * h.sin_theta) / (h.su + 1e-8);
    let v = (-d.x * h.sin_theta + d.y * h.cos_theta) / (h.sv + 1e-8);
    let a = 0.5 * 1.7320508;
    let sdf = max(abs(v), max(abs(a * u + 0.5 * v), abs(a * u - 0.5 * v))) - 1.0;
    return 1.0 / (1.0 + exp(sdf * h.sharpness));
}

struct Pixel { rgba: vec4f, pix: vec2u, inside: bool, }
fn raster(tile: vec2u, lid: u32) -> Pixel {
    var out: Pixel;
    if lid == 0u {
        skip = select(0u, 1u, (counts[0] & 0x80000000u) != 0u || counts[1] > arrayLength(&compact_from_isect));
        let index = (tile.x + tile.y * frame.image.tiles_x) * 2u;
        range_lo = offsets[index];
        range_hi = offsets[index + 1u];
    }
    if workgroupUniformLoad(&skip) != 0u {
        return out;
    }
    out.pix = tile * TILE_WIDTH + vec2u(compact_bits(lid), compact_bits(lid >> 1u));
    out.inside = all(out.pix < vec2u(frame.image.width, frame.image.height));
    let lo = workgroupUniformLoad(&range_lo);
    let hi = workgroupUniformLoad(&range_hi);
    let p = vec2f(out.pix);
    var ids: array<u32, TOP_K>;
    var weights: array<f32, TOP_K>;
    var filled = 0u;
    var min_slot = 0u;
    var min_weight = -1.0;
    for (var start = lo; start < hi; start += BATCH) {
        let remaining = min(BATCH, hi - start);
        workgroupBarrier();
        if lid < remaining {
            batch[lid] = projected[compact_from_isect[start + lid]];
        }
        workgroupBarrier();
        for (var t = 0u; out.inside && t < remaining; t++) {
            let h = batch[t];
            let w = weight(h, vec2f(h.x, h.y) - p);
            if w > min_weight {
                ids[min_slot] = compact_from_isect[start + t];
                weights[min_slot] = w;
                if min_weight < 0.0 && min_slot + 1u < TOP_K {
                    min_slot++;
                    filled = min_slot;
                } else {
                    filled = TOP_K;
                    min_weight = 3.4028235e38;
                    for (var k = 0u; k < TOP_K; k++) {
                        if weights[k] < min_weight {
                            min_slot = k;
                            min_weight = weights[k];
                        }
                    }
                }
            }
        }
    }
    var total = 0.0;
    for (var k = 0u; k < filled; k++) {
        total += weights[k];
    }
    let denominator = total + 1e-4;
    var color = vec3f(0.0);
    for (var k = 0u; k < filled; k++) {
        let h = projected[ids[k]];
        let d = vec2f(h.x, h.y) - p;
        let c = vec3f(h.color[0], h.color[1], h.color[2]) + 0.1 * (d.x * vec3f(h.color[3], h.color[4], h.color[5]) +
            d.y * vec3f(h.color[6], h.color[7], h.color[8]));
        color += c * (weights[k] / denominator);
    }
    out.rgba = vec4f(color, total / denominator);
    return out;
}

@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;

@compute @workgroup_size(WG) fn image_raster_float(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid);
    if r.inside {
        out_float[r.pix.x + r.pix.y * frame.image.width] = r.rgba;
    }
}

@compute @workgroup_size(WG) fn image_raster_packed(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid);
    if r.inside {
        let v = vec4u(clamp(r.rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[r.pix.x + r.pix.y * frame.image.width] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}
