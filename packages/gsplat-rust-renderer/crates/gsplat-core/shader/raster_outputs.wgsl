#import <./raster_common.wgsl>

// Local float and packed outputs for parity, scoring, and timing.
@group(0) @binding(4) var<storage, read_write> out_float: array<vec4f>;
@group(0) @binding(5) var<storage, read_write> out_packed: array<u32>;

@compute @workgroup_size(256) fn raster_float(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let tile = gid.x + gid.y * groups.x;
    if tile >= u.image.z * u.image.w {
        return;
    }
    let rgba = raster(tile, lid).rgba;
    let pix = pixel(tile, lid);
    if all(pix < u.image.xy) {
        out_float[pix.x + pix.y * u.image.x] = rgba;
    }
}

@compute @workgroup_size(256) fn raster_packed(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let tile = gid.x + gid.y * groups.x;
    if tile >= u.image.z * u.image.w {
        return;
    }
    let rgba = raster(tile, lid).rgba;
    let pix = pixel(tile, lid);
    if all(pix < u.image.xy) {
        let v = vec4u(clamp(rgba * 255.0, vec4f(0.0), vec4f(255.0)));
        out_packed[pix.x + pix.y * u.image.x] = v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
    }
}
