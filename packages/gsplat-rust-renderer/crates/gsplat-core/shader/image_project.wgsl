#import <./image_hexel.wgsl>
#import <./counts.wgsl>

@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> global_from_compact: array<u32>;
@group(0) @binding(5) var<storage, read_write> depths: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_hits: array<u32>;
@group(0) @binding(8) var<storage, read_write> projected: array<ImageHexel>;

// Every hexel with a tile is kept. The sort key is the primitive index, so each tile
// lists candidates in upstream order (its top-K keeps the first of equal weights).
@compute @workgroup_size(WG) fn image_forward(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let id = (gid.x + gid.y * groups.x) * WG + lid;
    if id >= frame.scene.splats {
        return;
    }
    let base = id * frame.scene.coefficients;
    var finite_input = true;
    for (var i = 0u; i < frame.scene.coefficients; i++) {
        finite_input = finite_input && finite(params[base + i]);
    }
    let rect = image_tiles(params[base], params[base + 1u], params[base + 2u], params[base + 3u], params[base + 5u]);
    let hit = select(0u, (rect.z - rect.x) * (rect.w - rect.y), finite_input);
    if hit == 0u {
        return;
    }
    tile_hits[id] = hit;
    add_intersections(hit);
    let write = atomicAdd(&counts[0], 1u) & 0x7fffffffu;
    global_from_compact[write] = id;
    depths[write] = id;
}

@compute @workgroup_size(WG) fn image_visible(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let compact = (gid.x + gid.y * groups.x) * WG + lid;
    if compact >= atomicLoad(&counts[0]) {
        return;
    }
    let base = global_from_compact[compact] * frame.scene.coefficients;
    let theta = params[base + 4u];
    projected[compact] = ImageHexel(params[base], params[base + 1u], params[base + 2u], params[base + 3u],
        cos(theta), sin(theta), params[base + 5u], array<f32, 9>(params[base + 6u], params[base + 7u],
        params[base + 8u], params[base + 9u], params[base + 10u], params[base + 11u], params[base + 12u],
        params[base + 13u], params[base + 14u]));
}
