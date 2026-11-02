#import <./common.wgsl>

// map_tiles for planar primitives: per row of the tile rectangle, the run of tiles the primitive covers.
// The importing module defines `alias Primitive`, `fn primitive_tiles(p: Primitive) -> vec4u` and
// `fn primitive_row(p: Primitive, rect: vec4u, row: u32) -> vec2u` (tile columns [x, y) of that row).
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
    let p = projected[i];
    let rect = primitive_tiles(p);
    var base = 0u;
    if i > 0u {
        base = cum_tile_hits[i - 1u];
    }
    let reserved = cum_tile_hits[i] - base;
    // Tiles that differ from the forward pass by rounding never write past the reservation; unused
    // reserved slots get the out-of-range tile and sort to the end.
    var t = 0u;
    for (var row = rect.y; row < rect.w; row++) {
        let span = primitive_row(p, rect, row);
        for (var x = span.x; x < span.y && t < reserved; x++) {
            if base + t < arrayLength(&tile_from_isect) {
                tile_from_isect[base + t] = x + row * frame.image.tiles_x;
                compact_from_isect[base + t] = i;
            }
            t++;
        }
    }
    for (; t < reserved; t++) {
        if base + t < arrayLength(&tile_from_isect) {
            tile_from_isect[base + t] = frame.image.tiles_x * frame.image.tiles_y;
            compact_from_isect[base + t] = i;
        }
    }
}
