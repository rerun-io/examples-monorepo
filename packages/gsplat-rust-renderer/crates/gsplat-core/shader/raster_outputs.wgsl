#import <./raster_common.wgsl>

// Local float and packed outputs for parity, scoring, and timing.
@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;

@compute @workgroup_size(WG) fn raster_float(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid);
    if r.inside {
        out_float[r.pix.x + r.pix.y * frame.image.width] = r.rgba;
    }
}

@compute @workgroup_size(WG) fn raster_packed(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid);
    if r.inside {
        let v = vec4u(clamp(r.rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[r.pix.x + r.pix.y * frame.image.width] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}

@group(0) @binding(9) var<storage, read_write> out_depths: array<vec2f>;
@group(0) @binding(10) var<storage, read_write> out_normals: array<vec4f>;

// Surface target: splats have centre depths and no plane normal, so the normal stays zero.
@compute @workgroup_size(WG) fn raster_surface(@builtin(workgroup_id) gid: vec3u,
    @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.xy, lid);
    if r.inside {
        let i = r.pix.x + r.pix.y * frame.image.width;
        out_float[i] = r.rgba;
        out_depths[i] = vec2f(r.depth, r.median);
        out_normals[i] = vec4f(0.0);
    }
}
