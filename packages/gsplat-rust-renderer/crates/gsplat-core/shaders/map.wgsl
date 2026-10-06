// Brush gather, map_gaussians, and get_tile_offsets, adapted to raw WGSL.
@group(0) @binding(1) var<storage, read> counts: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read> hits: array<u32>;
@group(0) @binding(4) var<storage, read_write> gathered: array<u32>;
@group(0) @binding(5) var<storage, read> projected: array<Splat>;
@group(0) @binding(6) var<storage, read> prefix: array<u32>;
@group(0) @binding(7) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(8) var<storage, read_write> isect_ids: array<u32>;
@group(0) @binding(9) var<storage, read_write> offsets: array<u32>;
@compute @workgroup_size(256) fn gather(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i =(gid.x + gid.y * groups.x) * 256u + lid;
    if i < counts[0] {
        gathered[i] = hits[ids[i]];
    }
}

@compute @workgroup_size(256) fn map_tiles(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let i =(gid.x + gid.y * groups.x) * 256u + lid;
    if i >= counts[0] {
        return;
    }
    let p = projected[i];
    let xy = vec2f(p.x, p.y);
    let c = vec3f(p.cx, p.cy, p.cz);
    let power = log(p.opacity * 255.0);
    let bb = tile_bbox(xy, bbox_extent(c, power));
    var base = 0u;
    if i > 0u {
        base = prefix[i - 1u];
    }
    let reserved = prefix[i] - base;
    let width = bb.z - bb.x;
    let n =(bb.w - bb.y) * width;
    var emitted = 0u;
    for (var t = 0u; t < n; t ++) {
        let tile = vec2u(t % width + bb.x, t / width + bb.y);
        if tile_hit(tile, xy, c, power) && emitted < reserved {
            if base + emitted < arrayLength(& tiles) {
                tiles[base + emitted] = tile.x + tile.y * u.image.z;
                isect_ids[base + emitted] = i;
            }
            emitted ++;
        }
    }
    for (var t = emitted; t < reserved; t ++) {
        if base + t < arrayLength(& tiles) {
            tiles[base + t] = u.image.z * u.image.w;
            isect_ids[base + t] = i;
        }
    }
}

@compute @workgroup_size(256) fn tile_offsets(@builtin(workgroup_id) gid: vec3u, @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let base =(gid.x + gid.y * groups.x) * 2048u + lid;
    let n = min(counts[1], arrayLength(& tiles));
    let num_tiles = u.image.z * u.image.w;
    for (var j = 0u; j < 8u; j ++) {
        let i = base + j * 256u;
        if i < n {
            let tile = tiles[i];
            if i == 0u {
                if tile < num_tiles {
                    offsets[tile * 2u] = 0u;
                }
            } else {
                let previous = tiles[i - 1u];
                if tile != previous {
                    if tile < num_tiles {
                        offsets[tile * 2u] = i;
                    }
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
