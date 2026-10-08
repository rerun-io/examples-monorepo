#import <./raster_common.wgsl>

// Local float and packed outputs for parity, scoring, and timing.
@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;

@compute @workgroup_size(256) fn raster_float(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.x + gid.y * groups.x, lid);
    if r.inside {
        out_float[r.pix.x + r.pix.y * u.image.width] = r.rgba;
    }
}

@compute @workgroup_size(256) fn raster_packed(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.x + gid.y * groups.x, lid);
    if r.inside {
        let v = vec4u(clamp(r.rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[r.pix.x + r.pix.y * u.image.width] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}
