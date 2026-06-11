// ═══════════════════════════════════════════════════════════════════════════
// gaussian_project.wgsl — Stages 1, 3, 4: GPU Cull, Projection, Prefix Scan
// ═══════════════════════════════════════════════════════════════════════════
//
// Pipeline position: FIRST compute stages — the GPU is handed the FULL splat
// cloud each frame.  (The stage-2 depth argsort between project_forward and
// project_visible lives in gaussian_dynamic_sort.wgsl.)  There is no CPU pre-pass; culling and depth ordering
// happen entirely on the GPU, following Brush's project_forward /
// project_visible split (brush v0.3.0 render.rs + d0eaca6f WGSL kernels).
//
//   1. project_forward_main: one thread per splat over the whole cloud.
//      Applies every visibility gate (near-plane, opacity, finite projection,
//      bbox-extent on-screen test) and, for survivors, appends
//      (global_gid, depth bits) to a compact list via atomicAdd(num_visible).
//   2. A GPU radix sort (gaussian_dynamic_sort.wgsl) then argsorts the compact
//      list ASCENDING by view-space depth (positive f32 bits sort monotonically
//      as u32), so compact order is front-to-back — exactly what the tile
//      raster's front-to-back alpha blending consumes.
//   3. project_visible_main: one thread per compact slot.  Re-reads the splat
//      via the depth-sorted global_from_compact mapping, recomputes the exact
//      3D → 2D Gaussian projection (Brush recomputes rather than caching),
//      evaluates SH color, and writes the ProjectedTileSplat + tile hit count
//      consumed by the map/sort/raster stages.  Slots past num_visible are
//      zeroed so the downstream capacity-sized prefix scan sees zeros.
//
// This shader also contains the **prefix-scan sub-passes** (scan_blocks_main,
// scan_block_sums_main) used both for the tile-hit-count scan and inside the
// radix sort.
//
// Key math concepts:
//   - Covariance: Σ = R * diag(s²) * Rᵀ  (rotation * squared-scale * rotation-transpose)
//   - Projection: Σ_2D = J * V * Σ_3D * Vᵀ * Jᵀ  (Jacobian * view-transform * covariance)
//   - SH evaluation: color = Σ(basis_i(dir) * coeff_i) for each RGB channel
//
// project_forward_main is the sole visibility gate; project_visible_main
// trusts it and never culls.  Both entry points share the same helper
// functions in this module, so their math agrees bit-for-bit.

struct ProjectUniformBuffer {
    view_from_world: mat4x4f,
    projection_from_view: mat4x4f,
    camera_world_position: vec4f,
    viewport_and_near: vec4f,
    sigma_and_counts: vec4u,
    pad: vec4u,
};

struct ScanUniformBuffer {
    total_selected: u32,
    block_count: u32,
    pad: vec2u,
};

struct ProjectedTileSplat {
    xy_px: vec2f,
    conic_xyy_opacity: vec4f,
    color_rgba: vec4f,
    tile_bbox_min_max: vec4u,
};

@group(0) @binding(0) var<storage, read> means_world: array<vec4f>;
@group(0) @binding(1) var<storage, read> quats_xyzw: array<vec4f>;
@group(0) @binding(2) var<storage, read> scales_opacity: array<vec4f>;
@group(0) @binding(3) var<storage, read> colors_dc: array<vec4f>;
@group(0) @binding(4) var<storage, read> sh_coefficients: array<vec4f>;
@group(0) @binding(5) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(8) var<uniform> project_uniforms: ProjectUniformBuffer;
@group(0) @binding(9) var<storage, read_write> projected_tile_splats: array<ProjectedTileSplat>;
@group(0) @binding(10) var<storage, read_write> projected_tile_hit_counts: array<u32>;
@group(0) @binding(11) var<storage, read> num_visible_in: array<u32>;

@group(0) @binding(12) var<storage, read_write> forward_global_from_compact: array<u32>;
@group(0) @binding(13) var<storage, read_write> forward_depth_keys: array<u32>;
@group(0) @binding(14) var<storage, read_write> forward_num_visible: atomic<u32>;

@group(0) @binding(16) var<storage, read> scan_input_values: array<u32>;
@group(0) @binding(17) var<storage, read_write> local_offsets: array<u32>;
@group(0) @binding(18) var<storage, read_write> block_offsets: array<u32>;
@group(0) @binding(19) var<uniform> scan_uniforms: ScanUniformBuffer;

@group(0) @binding(24) var<storage, read_write> block_sums_for_scan: array<u32>;
// Scan totals output.  The layout is legacy-DrawIndirect-shaped (word[0] = 6,
// an unused legacy quad vertex count; word[1] = total) and is kept that way
// for buffer compatibility with the Rust-side readback.
@group(0) @binding(25) var<storage, read_write> scan_totals_out: array<u32>;
@group(0) @binding(26) var<uniform> scan_block_sums_uniforms: ScanUniformBuffer;

const PROJECT_WORKGROUP_SIZE: u32 = 128u;
const COMPACTION_WORKGROUP_SIZE: u32 = 256u;
const COMPACTION_BLOCK_SIZE: u32 = COMPACTION_WORKGROUP_SIZE * 2u;
const SH_C0: f32 = 0.2820947917738781f;
const DEFAULT_SIGMA_COVERAGE: f32 = 3.0f;
const BRUSH_COVARIANCE_BLUR_PX: f32 = 0.3f;
const BRUSH_VISIBILITY_ALPHA_THRESHOLD: f32 = 1.0f / 255.0f;
const TILE_WIDTH: u32 = 16u;

var<workgroup> scan_scratch: array<u32, 512>;

fn zero_projected_tile_splat() -> ProjectedTileSplat {
    return ProjectedTileSplat(
        vec2f(0.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 0.0),
        vec4f(0.0, 0.0, 0.0, 0.0),
        vec4u(0u, 0u, 0u, 0u),
    );
}

fn linear_workgroup_id(workgroup_id: vec3u, num_workgroups: vec3u) -> u32 {
    return workgroup_id.x + num_workgroups.x * workgroup_id.y;
}

fn quat_to_mat_xyzw(quat: vec4f) -> mat3x3f {
    let q = normalize(quat);
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;

    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    return mat3x3f(
        vec3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3f(2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)),
        vec3f(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy)),
    );
}

fn regularize_covariance(covariance: mat2x2f) -> mat2x2f {
    var regularized = covariance;
    let off_diagonal = clamp(0.5 * (regularized[0][1] + regularized[1][0]), -1e6, 1e6);
    regularized[0][0] = max(regularized[0][0], 1e-8);
    regularized[1][1] = max(regularized[1][1], 1e-8);
    regularized[0][1] = off_diagonal;
    regularized[1][0] = off_diagonal;
    return regularized;
}

fn inverse_2x2(covariance: mat2x2f) -> mat2x2f {
    let det = determinant(covariance);
    if det <= 1e-12 {
        return mat2x2f(vec2f(0.0, 0.0), vec2f(0.0, 0.0));
    }

    let inv_det = 1.0 / det;
    return mat2x2f(
        vec2f(covariance[1][1] * inv_det, -covariance[0][1] * inv_det),
        vec2f(-covariance[1][0] * inv_det, covariance[0][0] * inv_det),
    );
}

fn num_sh_coeffs(degree: u32) -> u32 {
    return (degree + 1u) * (degree + 1u);
}

fn read_sh_coeff(base_index: u32, coeff_index: u32) -> vec3f {
    return sh_coefficients[base_index + coeff_index].xyz;
}

fn safe_normalize(value: vec3f) -> vec3f {
    let length_sq = dot(value, value);
    if length_sq > 1e-12 {
        return value * inverseSqrt(length_sq);
    }
    return vec3f(0.0, 0.0, 1.0);
}

fn is_reasonable_scalar(value: f32) -> bool {
    return abs(value) < 1e20;
}

fn is_reasonable_vec2(value: vec2f) -> bool {
    return all(abs(value) < vec2f(1e20, 1e20));
}

fn brush_camera_jacobian_rows(
    mean_camera: vec3f,
    focal_px: vec2f,
    viewport_size_px: vec2f,
    pixel_center: vec2f,
) -> array<vec3f, 2> {
    let focal_safe = max(focal_px, vec2f(1e-6));
    let lims_pos = (1.15f * viewport_size_px - pixel_center) / focal_safe;
    let lims_neg = (-0.15f * viewport_size_px - pixel_center) / focal_safe;
    let uv = mean_camera.xy / max(mean_camera.z, 1e-6);
    let uv_clipped = clamp(uv, lims_neg, lims_pos);
    let duv_dxy = focal_safe / max(mean_camera.z, 1e-6);

    return array<vec3f, 2>(
        vec3f(duv_dxy.x, 0.0, -duv_dxy.x * uv_clipped.x),
        vec3f(0.0, duv_dxy.y, -duv_dxy.y * uv_clipped.y),
    );
}

fn brush_covariance_in_pixels(
    covariance_world: mat3x3f,
    mean_camera: vec3f,
    focal_px: vec2f,
    viewport_size_px: vec2f,
    pixel_center: vec2f,
    view_linear: mat3x3f,
) -> mat2x2f {
    // This is the local linearization used by Brush-style projection: transform covariance into
    // view space, apply the camera Jacobian, then regularize in pixel space.
    let covariance_view = view_linear * covariance_world * transpose(view_linear);
    let jacobian_rows =
        brush_camera_jacobian_rows(mean_camera, focal_px, viewport_size_px, pixel_center);
    let row0 = jacobian_rows[0];
    let row1 = jacobian_rows[1];
    let cov_times_row0 = covariance_view * row0;
    let cov_times_row1 = covariance_view * row1;
    return regularize_covariance(
        mat2x2f(
            vec2f(dot(row0, cov_times_row0), dot(row0, cov_times_row1)),
            vec2f(dot(row1, cov_times_row0), dot(row1, cov_times_row1)),
        )
    );
}

fn compensate_covariance_px(covariance_px: mat2x2f) -> mat2x2f {
    var compensated = regularize_covariance(covariance_px);
    compensated[0][0] += BRUSH_COVARIANCE_BLUR_PX;
    compensated[1][1] += BRUSH_COVARIANCE_BLUR_PX;
    return regularize_covariance(compensated);
}

fn brush_bbox_extent_px(covariance_px: mat2x2f, power_threshold: f32) -> vec2f {
    return vec2f(
        sqrt(max(2.0f * power_threshold * covariance_px[0][0], 0.0)),
        sqrt(max(2.0f * power_threshold * covariance_px[1][1], 0.0)),
    );
}

fn tile_rect(tile: vec2u) -> vec4f {
    let rect_min = vec2f(tile * TILE_WIDTH);
    let rect_max = rect_min + f32(TILE_WIDTH);
    return vec4f(rect_min.x, rect_min.y, rect_max.x, rect_max.y);
}

fn get_bbox(center: vec2f, dims: vec2f, bounds: vec2u) -> vec4u {
    let min_corner = vec2u(clamp(center - dims, vec2f(0.0), vec2f(bounds)));
    let max_corner = vec2u(clamp(center + dims + vec2f(1.0), vec2f(0.0), vec2f(bounds)));
    return vec4u(min_corner, max_corner);
}

fn get_tile_bbox(pix_center: vec2f, pix_extent: vec2f, tile_bounds: vec2u) -> vec4u {
    let tile_center = pix_center / f32(TILE_WIDTH);
    let tile_extent = pix_extent / f32(TILE_WIDTH);
    return get_bbox(tile_center, tile_extent, tile_bounds);
}

fn calc_sigma(mean: vec2f, conic: vec3f, pixel_coord: vec2f) -> f32 {
    let delta = pixel_coord - mean;
    return 0.5f * (conic.x * delta.x * delta.x + conic.z * delta.y * delta.y) +
        conic.y * delta.x * delta.y;
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

fn evaluate_sh_rgb(view_direction: vec3f, degree: u32, base_index: u32) -> vec3f {
    var colors = SH_C0 * read_sh_coeff(base_index, 0u);

    if degree == 0u {
        return max(colors + vec3f(0.5), vec3f(0.0));
    }

    let x = view_direction.x;
    let y = view_direction.y;
    let z = view_direction.z;

    let fTmp0A = 0.48860251190292f;
    colors += fTmp0A *
        (-y * read_sh_coeff(base_index, 1u) +
        z * read_sh_coeff(base_index, 2u) -
        x * read_sh_coeff(base_index, 3u));

    if degree == 1u {
        return max(colors + vec3f(0.5), vec3f(0.0));
    }

    let z2 = z * z;
    let fTmp0B = -1.092548430592079f * z;
    let fTmp1A = 0.5462742152960395f;
    let fC1 = x * x - y * y;
    let fS1 = 2.0f * x * y;
    let pSH6 = 0.9461746957575601f * z2 - 0.3153915652525201f;
    let pSH7 = fTmp0B * x;
    let pSH5 = fTmp0B * y;
    let pSH8 = fTmp1A * fC1;
    let pSH4 = fTmp1A * fS1;

    colors +=
        pSH4 * read_sh_coeff(base_index, 4u) +
        pSH5 * read_sh_coeff(base_index, 5u) +
        pSH6 * read_sh_coeff(base_index, 6u) +
        pSH7 * read_sh_coeff(base_index, 7u) +
        pSH8 * read_sh_coeff(base_index, 8u);

    if degree == 2u {
        return max(colors + vec3f(0.5), vec3f(0.0));
    }

    let fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    let fTmp1B = 1.445305721320277f * z;
    let fTmp2A = -0.5900435899266435f;
    let fC2 = x * fC1 - y * fS1;
    let fS2 = x * fS1 + y * fC1;
    let pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    let pSH13 = fTmp0C * x;
    let pSH11 = fTmp0C * y;
    let pSH14 = fTmp1B * fC1;
    let pSH10 = fTmp1B * fS1;
    let pSH15 = fTmp2A * fC2;
    let pSH9 = fTmp2A * fS2;

    colors +=
        pSH9 * read_sh_coeff(base_index, 9u) +
        pSH10 * read_sh_coeff(base_index, 10u) +
        pSH11 * read_sh_coeff(base_index, 11u) +
        pSH12 * read_sh_coeff(base_index, 12u) +
        pSH13 * read_sh_coeff(base_index, 13u) +
        pSH14 * read_sh_coeff(base_index, 14u) +
        pSH15 * read_sh_coeff(base_index, 15u);

    if degree == 3u {
        return max(colors + vec3f(0.5), vec3f(0.0));
    }

    let fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    let fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    let fTmp2B = -1.770130769779931f * z;
    let fTmp3A = 0.6258357354491763f;
    let fC3 = x * fC2 - y * fS2;
    let fS3 = x * fS2 + y * fC2;
    let pSH20 = 1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6;
    let pSH21 = fTmp0D * x;
    let pSH19 = fTmp0D * y;
    let pSH22 = fTmp1C * fC1;
    let pSH18 = fTmp1C * fS1;
    let pSH23 = fTmp2B * fC2;
    let pSH17 = fTmp2B * fS2;
    let pSH24 = fTmp3A * fC3;
    let pSH16 = fTmp3A * fS3;

    colors +=
        pSH16 * read_sh_coeff(base_index, 16u) +
        pSH17 * read_sh_coeff(base_index, 17u) +
        pSH18 * read_sh_coeff(base_index, 18u) +
        pSH19 * read_sh_coeff(base_index, 19u) +
        pSH20 * read_sh_coeff(base_index, 20u) +
        pSH21 * read_sh_coeff(base_index, 21u) +
        pSH22 * read_sh_coeff(base_index, 22u) +
        pSH23 * read_sh_coeff(base_index, 23u) +
        pSH24 * read_sh_coeff(base_index, 24u);

    return max(colors + vec3f(0.5), vec3f(0.0));
}

// Shared projection state computed identically by project_forward_main and
// project_visible_main.  Both entry points call `project_splat`, so the
// culling decision (forward) and the rasterization payload (visible) agree
// bit-for-bit — the Brush model where project_forward is the sole gate.
struct ProjectedSplatInfo {
    visible: bool,
    camera_depth: f32,
    mean_px: vec2f,
    extent_px: vec2f,
    power_threshold: f32,
    compensated_covariance_px: mat2x2f,
};

fn project_splat(splat_index: u32) -> ProjectedSplatInfo {
    var info = ProjectedSplatInfo(
        false, 0.0, vec2f(0.0), vec2f(0.0), 0.0,
        mat2x2f(vec2f(0.0, 0.0), vec2f(0.0, 0.0)),
    );

    let mean_world = means_world[splat_index].xyz;
    let quat = quats_xyzw[splat_index];
    let scale_opacity = scales_opacity[splat_index];
    let scale = max(scale_opacity.xyz, vec3f(1e-6));
    let opacity_scale = max(bitcast<f32>(project_uniforms.pad.y), 0.0);
    let effective_opacity = scale_opacity.w * opacity_scale;

    let mean_view = (project_uniforms.view_from_world * vec4f(mean_world, 1.0)).xyz;
    let camera_depth = -mean_view.z;
    info.camera_depth = camera_depth;
    if !(camera_depth > project_uniforms.viewport_and_near.z) {
        return info;
    }
    if effective_opacity < BRUSH_VISIBILITY_ALPHA_THRESHOLD {
        return info;
    }

    let rotation = quat_to_mat_xyzw(quat);
    let scale_diag = mat3x3f(
        vec3f(scale.x * scale.x, 0.0, 0.0),
        vec3f(0.0, scale.y * scale.y, 0.0),
        vec3f(0.0, 0.0, scale.z * scale.z),
    );
    let covariance_world = rotation * scale_diag * transpose(rotation);
    // The world-to-view rotation is OpenGL-style (camera looks down -z), but
    // mean_camera below uses a z-forward frame (z = -view.z).  Express the
    // covariance in that same z-forward frame by negating the z row of the
    // rotation (flips sigma_xz / sigma_yz; the other entries are unchanged).
    let vc0 = project_uniforms.view_from_world[0].xyz;
    let vc1 = project_uniforms.view_from_world[1].xyz;
    let vc2 = project_uniforms.view_from_world[2].xyz;
    let view_linear = mat3x3f(
        vec3f(vc0.x, vc0.y, -vc0.z),
        vec3f(vc1.x, vc1.y, -vc1.z),
        vec3f(vc2.x, vc2.y, -vc2.z),
    );
    let viewport_size_px = project_uniforms.viewport_and_near.xy;
    let pixel_center = viewport_size_px * 0.5;
    let focal_ndc = vec2f(
        project_uniforms.projection_from_view[0].x,
        project_uniforms.projection_from_view[1].y,
    );
    let focal_px = focal_ndc * pixel_center;
    let mean_camera = vec3f(mean_view.x, mean_view.y, camera_depth);
    let mean_px = focal_px * mean_camera.xy / camera_depth + pixel_center;
    if !is_reasonable_vec2(mean_px) {
        return info;
    }

    let sigma_coverage = bitcast<f32>(project_uniforms.sigma_and_counts.x);
    let coverage_scale = max(sigma_coverage / DEFAULT_SIGMA_COVERAGE, 1e-4);
    let covariance_px = brush_covariance_in_pixels(
        covariance_world,
        mean_camera,
        focal_px,
        viewport_size_px,
        pixel_center,
        view_linear,
    );
    let compensated_covariance_px = compensate_covariance_px(covariance_px);
    let power_threshold = log(effective_opacity * 255.0);
    if !is_reasonable_scalar(power_threshold) || power_threshold <= 0.0 {
        return info;
    }

    let extent_px =
        brush_bbox_extent_px(compensated_covariance_px, power_threshold) * coverage_scale;
    if !is_reasonable_vec2(extent_px) {
        return info;
    }

    let radius_px = max(extent_px.x, extent_px.y);
    if radius_px < project_uniforms.viewport_and_near.w {
        return info;
    }

    // Extent-aware on-screen test (Brush's mean±extent vs viewport overlap):
    // replaces the old CPU |NDC| > 1.5 center cull.
    if mean_px.x + extent_px.x <= 0.0 ||
        mean_px.x - extent_px.x >= viewport_size_px.x ||
        mean_px.y + extent_px.y <= 0.0 ||
        mean_px.y - extent_px.y >= viewport_size_px.y {
        return info;
    }

    info.visible = true;
    info.mean_px = mean_px;
    info.extent_px = extent_px;
    info.power_threshold = power_threshold;
    info.compensated_covariance_px = compensated_covariance_px;
    return info;
}

// Stage 1: cull the FULL cloud and compact (global_gid, depth) pairs.
// One thread per splat, capacity-dispatched on the statically known cloud size.
@compute
@workgroup_size(PROJECT_WORKGROUP_SIZE, 1, 1)
fn project_forward_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let global_gid =
        linear_workgroup_id(workgroup_id, num_workgroups) * PROJECT_WORKGROUP_SIZE + local_index;
    let total_splats = project_uniforms.sigma_and_counts.y;
    if global_gid >= total_splats {
        return;
    }

    let info = project_splat(global_gid);
    if !info.visible {
        return;
    }

    // Positive f32 depths bitcast to u32 sort monotonically as unsigned, so
    // the radix argsort over these keys yields ascending (front-to-back) order.
    let write_id = atomicAdd(&forward_num_visible, 1u);
    forward_global_from_compact[write_id] = global_gid;
    forward_depth_keys[write_id] = bitcast<u32>(info.camera_depth);
}

// Stage 2 (after the GPU depth argsort): exact projection of visible splats.
// One thread per compact slot, capacity-dispatched on the cloud size with an
// early-exit on the GPU-side visible count.
@compute
@workgroup_size(PROJECT_WORKGROUP_SIZE, 1, 1)
fn project_visible_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let compact_id =
        linear_workgroup_id(workgroup_id, num_workgroups) * PROJECT_WORKGROUP_SIZE + local_index;
    let total_splats = project_uniforms.sigma_and_counts.y;
    if compact_id >= total_splats {
        return;
    }

    // Zero the tail so the capacity-sized tile-count scan sees zeros.
    let num_visible = num_visible_in[0];
    if compact_id >= num_visible {
        projected_tile_splats[compact_id] = zero_projected_tile_splat();
        projected_tile_hit_counts[compact_id] = 0u;
        return;
    }

    let splat_index = global_from_compact[compact_id];
    // project_forward_main already gated visibility; recompute (Brush model)
    // and trust that the result is visible.
    let info = project_splat(splat_index);

    let viewport_size_px = project_uniforms.viewport_and_near.xy;
    let base_opacity = scales_opacity[splat_index].w;
    let has_sh = project_uniforms.pad.x != 0u;
    var color = max(colors_dc[splat_index].xyz, vec3f(0.0));
    if has_sh {
        let mean_world = means_world[splat_index].xyz;
        let camera_position = project_uniforms.camera_world_position.xyz;
        let view_direction = safe_normalize(mean_world - camera_position);
        let coeffs_per_channel = project_uniforms.sigma_and_counts.z;
        let sh_degree = project_uniforms.sigma_and_counts.w;
        let coeff_base = splat_index * coeffs_per_channel;
        color = evaluate_sh_rgb(view_direction, sh_degree, coeff_base);
    }

    let mean_px_raster = vec2f(info.mean_px.x, viewport_size_px.y - info.mean_px.y);
    let tile_bounds = vec2u(
        u32(ceil(viewport_size_px.x / f32(TILE_WIDTH))),
        u32(ceil(viewport_size_px.y / f32(TILE_WIDTH))),
    );
    let tile_bbox = get_tile_bbox(mean_px_raster, info.extent_px, tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;
    let conic_px = inverse_2x2(info.compensated_covariance_px);
    let packed_conic_px = vec3f(conic_px[0][0], -conic_px[1][0], conic_px[1][1]);

    var num_tiles_hit = 0u;
    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;
        let rect = tile_rect(vec2u(tx, ty));
        if will_primitive_contribute(rect, mean_px_raster, packed_conic_px, info.power_threshold) {
            num_tiles_hit += 1u;
        }
    }

    projected_tile_splats[compact_id] = ProjectedTileSplat(
        mean_px_raster,
        vec4f(packed_conic_px, base_opacity),
        vec4f(color, base_opacity),
        vec4u(tile_bbox_min, tile_bbox_max),
    );
    projected_tile_hit_counts[compact_id] = num_tiles_hit;
}

@compute
@workgroup_size(COMPACTION_WORKGROUP_SIZE, 1, 1)
fn scan_blocks_main(
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let block_index = linear_workgroup_id(workgroup_id, num_workgroups);
    if block_index >= scan_uniforms.block_count {
        return;
    }

    let base = block_index * COMPACTION_BLOCK_SIZE;
    let local0 = local_index;
    let local1 = local_index + COMPACTION_WORKGROUP_SIZE;
    let global0 = base + local0;
    let global1 = base + local1;

    let flag0 = select(0u, scan_input_values[global0], global0 < scan_uniforms.total_selected);
    let flag1 = select(0u, scan_input_values[global1], global1 < scan_uniforms.total_selected);
    scan_scratch[local0] = flag0;
    scan_scratch[local1] = flag1;
    workgroupBarrier();

    var step = 1u;
    loop {
        if step >= COMPACTION_BLOCK_SIZE {
            break;
        }

        let add0 = select(0u, scan_scratch[local0 - step], local0 >= step);
        let add1 = select(0u, scan_scratch[local1 - step], local1 >= step);
        workgroupBarrier();
        scan_scratch[local0] += add0;
        scan_scratch[local1] += add1;
        workgroupBarrier();
        step *= 2u;
    }

    if global0 < scan_uniforms.total_selected {
        local_offsets[global0] = scan_scratch[local0] - flag0;
    }
    if global1 < scan_uniforms.total_selected {
        local_offsets[global1] = scan_scratch[local1] - flag1;
    }
    if local_index == COMPACTION_WORKGROUP_SIZE - 1u {
        block_offsets[block_index] = scan_scratch[COMPACTION_BLOCK_SIZE - 1u];
    }
}

@compute
@workgroup_size(COMPACTION_WORKGROUP_SIZE, 1, 1)
fn scan_block_sums_main(@builtin(local_invocation_index) local_index: u32) {
    let block_count = scan_block_sums_uniforms.block_count;
    let local0 = local_index;
    let local1 = local_index + COMPACTION_WORKGROUP_SIZE;

    let block_sum0 = select(0u, block_sums_for_scan[local0], local0 < block_count);
    let block_sum1 = select(0u, block_sums_for_scan[local1], local1 < block_count);
    scan_scratch[local0] = block_sum0;
    scan_scratch[local1] = block_sum1;
    workgroupBarrier();

    var step = 1u;
    loop {
        if step >= COMPACTION_BLOCK_SIZE {
            break;
        }

        let add0 = select(0u, scan_scratch[local0 - step], local0 >= step);
        let add1 = select(0u, scan_scratch[local1 - step], local1 >= step);
        workgroupBarrier();
        scan_scratch[local0] += add0;
        scan_scratch[local1] += add1;
        workgroupBarrier();
        step *= 2u;
    }

    if local0 < block_count {
        block_sums_for_scan[local0] = scan_scratch[local0] - block_sum0;
    }
    if local1 < block_count {
        block_sums_for_scan[local1] = scan_scratch[local1] - block_sum1;
    }

    workgroupBarrier();
    if local_index == 0u {
        let total_visible = select(0u, scan_scratch[block_count - 1u], block_count > 0u);
        // Legacy-DrawIndirect-shaped layout, kept for buffer compatibility:
        // word[0] is an unused legacy quad vertex count, word[1] is the total.
        scan_totals_out[0] = 6u;
        scan_totals_out[1] = total_visible;
        scan_totals_out[2] = 0u;
        scan_totals_out[3] = 0u;
    }
}

