// ═══════════════════════════════════════════════════════════════════════════
// gaussian_tile_offsets.wgsl — Stage 5: Tile Range Extraction
// ═══════════════════════════════════════════════════════════════════════════
//
// Pipeline position: Runs after the radix sort (stage 4).
//
// Purpose: After sorting, all intersections for the same tile are contiguous in
// the sorted array.  This shader finds the [start, end) range for each tile by
// detecting where the tile ID changes between adjacent entries.
//
// Think of it like finding the boundaries in a sorted list:
//   sorted_tile_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, ...]
//   tile_offsets[0] = {start: 0, end: 3}
//   tile_offsets[1] = {start: 3, end: 5}
//   tile_offsets[2] = {start: 5, end: 9}
//   ...
//
// The raster shader (stage 6) then uses these ranges to iterate only over the
// splats that actually overlap its tile — no wasted work.
//
// Each thread checks CHECKS_PER_ITER consecutive entries to amortize the
// dispatch overhead.

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
