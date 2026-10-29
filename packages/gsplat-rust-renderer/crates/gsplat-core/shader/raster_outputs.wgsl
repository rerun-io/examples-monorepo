#import <./common.wgsl>

// Output entry points shared by every primitive. The importing module defines
// `fn raster(tile: vec2u, lid: u32, surface: bool) -> Raster` and `compact_from_isect`.
// `isect` is the intersection index of the median contributor, NO_ID when nothing contributed.
struct Raster { rgba: vec4f, depth: f32, median: f32, normal: vec3f, pix: vec2u, inside: bool, isect: u32, }
const NO_ID = 0xffffffffu;

fn compact_bits(v: u32) -> u32 {
    var x = v & 0x55u;
    x = (x | (x >> 1u)) & 0x33u;
    return (x | (x >> 2u)) & 0x0fu;
}

fn pixel(tile: vec2u, lid: u32) -> vec2u {
    return tile * TILE_WIDTH + vec2u(compact_bits(lid), compact_bits(lid >> 1u));
}

@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;
@group(0) @binding(6) var out_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;
@group(0) @binding(9) var<storage, read_write> out_depths: array<vec2f>;
@group(0) @binding(10) var<storage, read_write> out_normals: array<vec4f>;
@group(0) @binding(11) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(12) var out_id: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(WG) fn raster_float(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        out_float[r.pix.x + r.pix.y * frame.image.width] = r.rgba;
    }
}

@compute @workgroup_size(WG) fn raster_packed(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        let v = vec4u(clamp(r.rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[r.pix.x + r.pix.y * frame.image.width] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}

@compute @workgroup_size(WG) fn raster_texture(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
    }
}

@compute @workgroup_size(WG) fn raster_texture_depth(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
        textureStore(out_depth, r.pix, vec4f(r.depth));
    }
}

// Viewer picking: texture_depth plus the global row of the median contributor, NO_ID when none.
@compute @workgroup_size(WG) fn raster_texture_depth_id(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, false);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
        textureStore(out_depth, r.pix, vec4f(r.depth));
        var id = NO_ID;
        if r.isect != NO_ID {
            id = global_from_compact[compact_from_isect[r.isect]];
        }
        textureStore(out_id, r.pix, vec4u(id));
    }
}

// Surface target: expected and median depth, and the alpha-weighted normal (zero for splats).
@compute @workgroup_size(WG) fn raster_surface(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid, true);
    if r.inside {
        let i = r.pix.x + r.pix.y * frame.image.width;
        out_float[i] = r.rgba;
        out_depths[i] = vec2f(r.depth, r.median);
        out_normals[i] = vec4f(r.normal, 0.0);
    }
}
