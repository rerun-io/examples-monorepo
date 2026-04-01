// Maps each projected splat to the tiles it overlaps.
//
// Input: projected splats with a per-splat tile bounding box.
// Output: one `(tile_id, compacted_splat_id)` pair per overlapping tile.

struct MapUniformBuffer {
    total_selected: u32,
    intersection_capacity: u32,
    tile_bounds_x: u32,
    tile_bounds_y: u32,
};

struct ProjectedTileSplat {
    xy_px: vec2f,
    conic_xyy_opacity: vec4f,
    color_rgba: vec4f,
    tile_bbox_min_max: vec4u,
};

@group(0) @binding(0) var<storage, read> projected_tile_splats: array<ProjectedTileSplat>;
@group(0) @binding(1) var<storage, read> tile_hit_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> tile_hit_counts: array<u32>;
@group(0) @binding(3) var<storage, read> tile_hit_block_offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> tile_id_from_isect: array<u32>;
@group(0) @binding(5) var<storage, read_write> compact_gid_from_isect: array<u32>;
@group(0) @binding(6) var<uniform> map_uniforms: MapUniformBuffer;
@group(0) @binding(7) var<storage, read> intersection_count_words: array<u32>;
@group(0) @binding(8) var<storage, read_write> num_intersections: array<u32>;

const PROJECT_WORKGROUP_SIZE: u32 = 128u;
const TILE_WIDTH: u32 = 16u;
const COMPACTION_BLOCK_SIZE: u32 = 512u;
fn linear_workgroup_id(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.x + num_workgroups.x * workgroup_id.y;
}

fn calc_sigma(pixel_coord: vec2f, conic: vec3f, xy: vec2f) -> f32 {
    let delta = pixel_coord - xy;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
        conic.y * delta.x * delta.y;
}

fn tile_rect(tile: vec2u) -> vec4f {
    let rect_min = vec2f(tile * TILE_WIDTH);
    let rect_max = rect_min + f32(TILE_WIDTH);
    return vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

fn will_primitive_contribute(rect: vec4f, mean: vec2f, conic: vec3f, power_threshold: f32) -> bool {
    let x_left = mean.x < rect.x;
    let x_right = mean.x > rect.z;
    let in_x_range = !(x_left || x_right);

    let y_above = mean.y < rect.y;
    let y_below = mean.y > rect.w;
    let in_y_range = !(y_above || y_below);

    if in_x_range && in_y_range {
        return true;
    }

    let closest_corner = vec2f(
        select(rect.z, rect.x, x_left),
        select(rect.w, rect.y, y_above),
    );

    let width = rect.z - rect.x;
    let height = rect.w - rect.y;
    let d = vec2f(
        select(-width, width, x_left),
        select(-height, height, y_above),
    );
    let diff = mean - closest_corner;
    let t_max = vec2f(
        select(
            clamp(
                (d.x * conic.x * diff.x + d.x * conic.y * diff.y) / (d.x * conic.x * d.x),
                0.0f,
                1.0f,
            ),
            0.0f,
            in_y_range,
        ),
        select(
            clamp(
                (d.y * conic.y * diff.x + d.y * conic.z * diff.y) / (d.y * conic.z * d.y),
                0.0f,
                1.0f,
            ),
            0.0f,
            in_x_range,
        ),
    );
    let max_contribution_point = closest_corner + t_max * d;
    let max_power_in_tile = calc_sigma(mean, conic, max_contribution_point);
    return max_power_in_tile <= power_threshold;
}

@compute
@workgroup_size(PROJECT_WORKGROUP_SIZE, 1, 1)
fn map_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let compact_gid =
        linear_workgroup_id(workgroup_id, num_workgroups) * PROJECT_WORKGROUP_SIZE + local_index;
    if compact_gid >= map_uniforms.total_selected {
        return;
    }

    let num_tiles_hit = tile_hit_counts[compact_gid];
    if num_tiles_hit == 0u {
        return;
    }

    let projected = projected_tile_splats[compact_gid];
    let mean2d = projected.xy_px;
    let conic = projected.conic_xyy_opacity.xyz;
    let opacity = projected.conic_xyy_opacity.w;
    let power_threshold = log(max(opacity, 1.0 / 255.0) * 255.0);
    let tile_bbox_min = projected.tile_bbox_min_max.xy;
    let tile_bbox_max = projected.tile_bbox_min_max.zw;
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;
    let block_index = compact_gid / COMPACTION_BLOCK_SIZE;
    let base_isect_id = tile_hit_offsets[compact_gid] + tile_hit_block_offsets[block_index];

    var written = 0u;
    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        if written >= num_tiles_hit {
            break;
        }

        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;
        let rect = tile_rect(vec2u(tx, ty));
        if will_primitive_contribute(rect, mean2d, conic, power_threshold) {
            let isect_id = base_isect_id + written;
            if isect_id < map_uniforms.intersection_capacity {
                let tile_id = tx + ty * map_uniforms.tile_bounds_x;
                tile_id_from_isect[isect_id] = tile_id;
                compact_gid_from_isect[isect_id] = compact_gid;
            }
            written += 1u;
        }
    }
}

@compute
@workgroup_size(1, 1, 1)
fn clamp_count_main() {
    // `intersection_count_words[1]` is the total number of tile intersections accumulated by the
    // scan stage. Clamp it to the buffers we actually allocated so later sort/raster passes never
    // consume unwritten entries on dense scenes.
    num_intersections[0] = min(intersection_count_words[1], map_uniforms.intersection_capacity);
}
