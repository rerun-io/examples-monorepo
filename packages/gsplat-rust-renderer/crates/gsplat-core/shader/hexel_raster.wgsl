#import <./hexel_common.wgsl>

// Upstream render_hexels3d: per pixel, intersect the ray with each hexel plane (h = k x l),
// evaluate the soft hexagon and the linear colour field, and composite front to back.
@group(0) @binding(1) var<storage, read> compact_from_isect: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read> projected: array<Hexel>;
@group(0) @binding(7) var<storage, read> counts: array<u32>;
// 128 hexels of 80 bytes keep the cooperative batch inside the core's 10 KiB workgroup limit.
const BATCH = 128u;
var<workgroup> batch: array<Hexel, BATCH>;
var<workgroup> skip: u32;
var<workgroup> range_lo: u32;
var<workgroup> range_hi: u32;
var<workgroup> num_done: atomic<u32>;
fn compact_bits(v: u32) -> u32 {
    var x = v & 0x55u;
    x = (x | (x >> 1u)) & 0x33u;
    return (x | (x >> 2u)) & 0x0fu;
}

// World normal of a hexel facing the camera. With M = K [a b c] (rows tu, tv, tw), the
// view normal a x b is parallel to K^T (m0 x m1) and its sign test reduces to (m0 x m1) . mc.
fn hexel_normal(h: Hexel) -> vec3f {
    let m = cross(vec3f(h.t[0], h.t[3], h.t[6]), vec3f(h.t[1], h.t[4], h.t[7]));
    let pp = frame.pinhole_params.zw - 0.5;
    let n = normalize(vec3f(frame.pinhole_params.x * m.x, frame.pinhole_params.y * m.y, dot(vec3f(pp, 1.0), m)));
    let facing = select(n, -n, dot(m, vec3f(h.t[2], h.t[5], h.t[8])) > 0.0);
    let q = frame.camera_rotation;
    let t = 2.0 * cross(q.xyz, facing);
    return facing + q.w * t + cross(q.xyz, t);
}

struct Pixel { rgba: vec4f, depth: f32, median: f32, normal: vec3f, pix: vec2u, inside: bool, }
fn raster(tile: vec2u, lid: u32, surface: bool) -> Pixel {
    var out: Pixel;
    if lid == 0u {
        skip = select(0u, 1u, (counts[0] & 0x80000000u) != 0u || counts[1] > arrayLength(&compact_from_isect));
    }
    if workgroupUniformLoad(&skip) != 0u {
        return out;
    }
    out.pix = tile * TILE_WIDTH + vec2u(compact_bits(lid), compact_bits(lid >> 1u));
    out.inside = all(out.pix < vec2u(frame.image.width, frame.image.height));
    if lid == 0u {
        let index = (tile.x + tile.y * frame.image.tiles_x) * 2u;
        range_lo = offsets[index];
        range_hi = offsets[index + 1u];
        atomicStore(&num_done, 0u);
    }
    let lo = workgroupUniformLoad(&range_lo);
    let hi = workgroupUniformLoad(&range_hi);
    let px = f32(out.pix.x);
    let py = f32(out.pix.y);
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
            let h = batch[t];
            let m = plane(h);
            let p = cross(px * m.tw - m.tu, py * m.tw - m.tv);
            if p.z == 0.0 {
                continue;
            }
            let s = p.xy / p.z;
            let depth = s.x * m.tw.x + s.y * m.tw.y + m.tw.z;
            if depth < HEX_NEAR {
                continue;
            }
            let alpha = min(h.opacity / (1.0 + exp(hexagon_sdf(s) * h.sharpness)), 0.99);
            if alpha < ALPHA_CUTOFF {
                continue;
            }
            let next = transmittance * (1.0 - alpha);
            if next < 1e-4 {
                done = true;
                continue;
            }
            let w = alpha * transmittance;
            color += (vec3f(h.color[0], h.color[1], h.color[2]) + s.x * vec3f(h.color[3], h.color[4], h.color[5]) +
                s.y * vec3f(h.color[6], h.color[7], h.color[8])) * w;
            if surface {
                if transmittance > 0.5 {
                    out.median = depth;
                }
                out.normal += hexel_normal(h) * w;
            }
            out.depth += depth * w;
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

@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;
@group(0) @binding(6) var out_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var<storage, read_write> out_depths: array<vec2f>;
@group(0) @binding(10) var<storage, read_write> out_normals: array<vec4f>;

@compute @workgroup_size(WG) fn hexel_raster_float(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        out_float[r.pix.x + r.pix.y * frame.image.width] = r.rgba;
    }
}

@compute @workgroup_size(WG) fn hexel_raster_packed(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        let v = vec4u(clamp(r.rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[r.pix.x + r.pix.y * frame.image.width] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}

@compute @workgroup_size(WG) fn hexel_raster_texture(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
    }
}

@compute @workgroup_size(WG) fn hexel_raster_texture_depth(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
        textureStore(out_depth, r.pix, vec4f(r.depth));
    }
}

@compute @workgroup_size(WG) fn hexel_raster_surface(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, true);
    if r.inside {
        let i = r.pix.x + r.pix.y * frame.image.width;
        out_float[i] = r.rgba;
        out_depths[i] = vec2f(r.depth, r.median);
        out_normals[i] = vec4f(r.normal, 0.0);
    }
}
