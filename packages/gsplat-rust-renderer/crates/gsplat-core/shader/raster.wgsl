#import <./raster_common.wgsl>

@group(0) @binding(6) var out_texture: texture_storage_2d < rgba8unorm, write >;

@compute @workgroup_size(256) fn raster_texture(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let tile = gid.x + gid.y * groups.x;
    if tile >= u.image.z * u.image.w {
        return;
    }
    let rgba = raster(tile, lid).rgba;
    let pix = pixel(tile, lid);
    if all(pix < u.image.xy) {
        textureStore(out_texture, pix, rgba);
    }
}

@group(0) @binding(8) var out_depth: texture_storage_2d<r32float, write>;

@compute @workgroup_size(256) fn raster_texture_depth(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let tile = gid.x + gid.y * groups.x;
    if tile >= u.image.z * u.image.w {
        return;
    }
    let result = raster(tile, lid);
    let pix = pixel(tile, lid);
    if all(pix < u.image.xy) {
        textureStore(out_texture, pix, result.rgba);
        textureStore(out_depth, pix, vec4f(result.depth));
    }
}
