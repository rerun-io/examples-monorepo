// Tile-local Gaussian raster pass.
//
// Each workgroup shades one tile. The inputs have already been sorted by tile and compacted, so
// the shader can iterate a contiguous intersection range and accumulate front-to-back color.

struct ProjectedTileSplat {
    xy_px: vec2f,
    conic_xyy_opacity: vec4f,
    color_rgba: vec4f,
    tile_bbox_min_max: vec4u,
};

struct RasterUniformBuffer {
    tile_bounds: vec2u,
    img_size: vec2u,
};

@group(0) @binding(0) var<storage, read> compact_gid_from_isect: array<u32>;
@group(0) @binding(1) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> projected: array<ProjectedTileSplat>;
@group(0) @binding(3) var out_img: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> uniforms: RasterUniformBuffer;

const TILE_WIDTH: u32 = 16u;
const TILE_SIZE: u32 = TILE_WIDTH * TILE_WIDTH;
const MIN_ALPHA: f32 = 1.0f / 255.0f;

var<workgroup> range_uniform: vec2u;
var<workgroup> local_batch: array<ProjectedTileSplat, TILE_SIZE>;

fn linear_workgroup_id(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.x + num_workgroups.x * workgroup_id.y;
}

fn compact_bits_16(v: u32) -> u32 {
    var x = v & 0x55555555u;
    x = (x | (x >> 1u)) & 0x33333333u;
    x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4u)) & 0x00FF00FFu;
    x = (x | (x >> 8u)) & 0x0000FFFFu;
    return x;
}

fn decode_morton_2d(morton: u32) -> vec2u {
    return vec2u(compact_bits_16(morton), compact_bits_16(morton >> 1u));
}

fn map_1d_to_2d(id: u32, tiles_per_row: u32) -> vec2u {
    let tile_id = id / TILE_SIZE;
    let within_tile_id = id % TILE_SIZE;
    let tile_x = tile_id % tiles_per_row;
    let tile_y = tile_id / tiles_per_row;
    return vec2u(tile_x * TILE_WIDTH, tile_y * TILE_WIDTH) + decode_morton_2d(within_tile_id);
}

fn calc_sigma(pixel_coord: vec2f, mean: vec2f, conic: vec3f) -> f32 {
    let delta = pixel_coord - mean;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
        conic.y * delta.x * delta.y;
}

@compute
@workgroup_size(TILE_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let tile_pixel = map_1d_to_2d(
        linear_workgroup_id(workgroup_id, num_workgroups) * TILE_SIZE + local_index,
        uniforms.tile_bounds.x,
    );
    let inside = tile_pixel.x < uniforms.img_size.x && tile_pixel.y < uniforms.img_size.y;
    let tile_loc = vec2u(tile_pixel.x / TILE_WIDTH, tile_pixel.y / TILE_WIDTH);
    let tile_id = tile_loc.x + tile_loc.y * uniforms.tile_bounds.x;

    if local_index == 0u {
        // The whole workgroup uses one shared range, so lane 0 fetches it once.
        range_uniform = vec2u(tile_offsets[tile_id * 2u], tile_offsets[tile_id * 2u + 1u]);
    }
    workgroupBarrier();
    let range = range_uniform;
    let pixel_coord = vec2f(tile_pixel) + 0.5f;
    var transmittance = 1.0f;
    var color_out = vec3f(0.0);
    var done = !inside;

    for (var batch_start = range.x; batch_start < range.y; batch_start += TILE_SIZE) {
        let remaining = min(TILE_SIZE, range.y - batch_start);
        workgroupBarrier();
        if local_index < remaining {
            // Load one batch of splats for the current tile into shared memory.
            let load_isect_id = batch_start + local_index;
            let compact_gid = compact_gid_from_isect[load_isect_id];
            local_batch[local_index] = projected[compact_gid];
        }
        workgroupBarrier();

        for (var i = 0u; !done && i < remaining; i++) {
            let splat = local_batch[i];
            let sigma = calc_sigma(pixel_coord, splat.xy_px, splat.conic_xyy_opacity.xyz);
            let alpha = min(0.999f, splat.color_rgba.a * exp(-sigma));
            if sigma >= 0.0f && alpha >= MIN_ALPHA {
                let next_transmittance = transmittance * (1.0f - alpha);
                color_out += max(splat.color_rgba.rgb, vec3f(0.0)) * (alpha * transmittance);
                transmittance = next_transmittance;
                if transmittance <= 1e-4f {
                    done = true;
                    break;
                }
            }
        }
    }

    if inside {
        textureStore(out_img, vec2i(tile_pixel), vec4f(color_out, 1.0f - transmittance));
    }
}
