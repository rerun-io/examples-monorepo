// Converts sorted tile ids into `[start, end)` ranges per tile.
//
// After the tile ids are sorted, each tile's intersections become one contiguous slice. This pass
// records the slice bounds so the raster workgroup for that tile can jump straight to its range.

@group(0) @binding(0) var<storage, read> sorted_tile_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> num_intersections: array<u32>;

const TILE_SIZE: u32 = 256u;
const CHECKS_PER_ITER: u32 = 8u;
fn linear_workgroup_id(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.x + num_workgroups.x * workgroup_id.y;
}

@compute
@workgroup_size(TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let total = num_intersections[0];
    if total == 0u {
        return;
    }

    let absolute_pos = linear_workgroup_id(workgroup_id, num_workgroups) * TILE_SIZE + local_index;
    let base_id = absolute_pos * CHECKS_PER_ITER;

    for (var i = 0u; i < CHECKS_PER_ITER; i++) {
        let isect_id = base_id + i;
        if isect_id >= total {
            break;
        }

        let tile_id = sorted_tile_ids[isect_id];
        if isect_id == 0u {
            tile_offsets[tile_id * 2u] = 0u;
        } else {
            let prev_tile_id = sorted_tile_ids[isect_id - 1u];
            if tile_id != prev_tile_id {
                tile_offsets[prev_tile_id * 2u + 1u] = isect_id;
                tile_offsets[tile_id * 2u] = isect_id;
            }
        }

        if isect_id + 1u == total {
            tile_offsets[tile_id * 2u + 1u] = total;
        }
    }
}
