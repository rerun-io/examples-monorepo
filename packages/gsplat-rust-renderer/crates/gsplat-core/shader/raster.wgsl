#import <./raster_common.wgsl>

@group(0) @binding(6) var out_texture: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(256) fn raster_texture(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.x + gid.y * groups.x, lid);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
    }
}

@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;

@compute @workgroup_size(256) fn raster_texture_depth(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let r = raster(gid.x + gid.y * groups.x, lid);
    if r.inside {
        textureStore(out_texture, r.pix, r.rgba);
        textureStore(out_depth, r.pix, vec4f(r.depth));
    }
}
