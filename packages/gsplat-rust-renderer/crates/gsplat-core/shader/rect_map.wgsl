#import <./common.wgsl>

// map_tiles for planar primitives: every tile of the rectangle, without per-tile tests.
// The importing module defines `alias Primitive` and `fn primitive_tiles(p: Primitive) -> vec4u`.
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read> projected: array<Primitive>;
@group(0) @binding(6) var<storage, read> cum_tile_hits: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_from_isect: array<u32>;
@group(0) @binding(8) var<storage, read_write> compact_from_isect: array<u32>;

@compute @workgroup_size(WG) fn map_tiles(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = (gid.x + gid.y * groups.x) * WG + lid;
    if i >= counts[0] {
        return;
    }
    let rect = primitive_tiles(projected[i]);
    var base = 0u;
    if i > 0u {
        base = cum_tile_hits[i - 1u];
    }
    let reserved = cum_tile_hits[i] - base;
    let width = rect.z - rect.x;
    // A rectangle that differs from the forward pass by rounding never writes past its reservation.
    let emitted = min((rect.w - rect.y) * width, reserved);
    for (var t = 0u; t < reserved; t++) {
        if base + t < arrayLength(&tile_from_isect) {
            var tile = frame.image.tiles_x * frame.image.tiles_y;
            if t < emitted {
                tile = t % width + rect.x + (t / width + rect.y) * frame.image.tiles_x;
            }
            tile_from_isect[base + t] = tile;
            compact_from_isect[base + t] = i;
        }
    }
}
