#import <./common.wgsl>

// Brush gather, map_gaussians, and get_tile_offsets, adapted to raw WGSL.
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(3) var<storage, read> tile_hits: array<u32>;
@group(0) @binding(4) var<storage, read_write> sorted_tile_hits: array<u32>;
@group(0) @binding(5) var<storage, read> projected: array<Splat>;
@group(0) @binding(6) var<storage, read> cum_tile_hits: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_from_isect: array<u32>;
@group(0) @binding(8) var<storage, read_write> compact_from_isect: array<u32>;
@group(0) @binding(9) var<storage, read_write> offsets: array<u32>;
@compute @workgroup_size(WG) fn gather(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = (gid.x + gid.y * groups.x) * WG + lid;
    if i < counts[0] {
        sorted_tile_hits[i] = tile_hits[global_from_compact[i]];
    }
}

@compute @workgroup_size(WG) fn map_tiles(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i = (gid.x + gid.y * groups.x) * WG + lid;
    if i >= counts[0] {
        return;
    }
    let p = projected[i];
    let xy = vec2f(p.x, p.y);
    let c = vec3f(p.conic_x, p.conic_y, p.conic_z);
    let power = log(p.color_a * 255.0);
    let bb = tile_bbox(xy, bbox_extent(c, power));
    var base = 0u;
    if i > 0u {
        base = cum_tile_hits[i - 1u];
    }
    let reserved = cum_tile_hits[i] - base;
    let width = bb.z - bb.x;
    let n = (bb.w - bb.y) * width;
    var emitted = 0u;
    for (var t = 0u; t < n; t++) {
        let tile = vec2u(t % width + bb.x, t / width + bb.y);
        if tile_hit(tile, xy, c, power) && emitted < reserved {
            if base + emitted < arrayLength(&tile_from_isect) {
                tile_from_isect[base + emitted] = tile.x + tile.y * frame.image.tiles_x;
                compact_from_isect[base + emitted] = i;
            }
            emitted++;
        }
    }
    for (var t = emitted; t < reserved; t++) {
        if base + t < arrayLength(&tile_from_isect) {
            tile_from_isect[base + t] = frame.image.tiles_x * frame.image.tiles_y;
            compact_from_isect[base + t] = i;
        }
    }
}

@compute @workgroup_size(WG) fn tile_offsets(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let base = (gid.x + gid.y * groups.x) * (WG * CHECKS_PER_ITER) + lid;
    let n = min(counts[1], arrayLength(&tile_from_isect));
    let num_tiles = frame.image.tiles_x * frame.image.tiles_y;
    for (var j = 0u; j < CHECKS_PER_ITER; j++) {
        let i = base + j * WG;
        if i < n {
            let tile = tile_from_isect[i];
            if i == 0u {
                if tile < num_tiles {
                    offsets[tile * 2u] = 0u;
                }
            } else {
                let previous = tile_from_isect[i - 1u];
                if tile != previous {
                    if tile < num_tiles {
                        offsets[tile * 2u] = i;
                    }
                    // Unlike Brush 1388f74c, close the previous real tile even when a sentinel follows.
                    if previous < num_tiles {
                        offsets[previous * 2u + 1u] = i;
                    }
                }
            }
            if i == n - 1u && tile < num_tiles {
                offsets[tile * 2u + 1u] = n;
            }
        }
    }
}
