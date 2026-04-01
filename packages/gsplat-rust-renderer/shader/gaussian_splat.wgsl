// ═══════════════════════════════════════════════════════════════════════════
// gaussian_splat.wgsl — CPU Fallback: Instanced-Quad Rendering
// ═══════════════════════════════════════════════════════════════════════════
//
// This shader is used by the CPU fallback path (when the compute tile pipeline
// is not available).  Each Gaussian splat is rendered as a screen-aligned quad
// (two triangles = 6 vertices) using GPU instancing — one instance per splat.
//
// The vertex shader expands each instance into a quad sized by `radius_ndc`,
// centered at `center_ndc`.  The fragment shader evaluates the Gaussian
// function at each pixel: it computes the Mahalanobis distance using the
// inverse 2D covariance matrix, applies an exponential falloff, and discards
// pixels beyond 3σ (where the contribution is negligible).
//
// The output uses premultiplied alpha (`color * alpha, alpha`) for correct
// alpha blending with the back-to-front sorted draw order.
//
// Key math:
//   mahalanobis = Δᵀ * Σ⁻¹ * Δ    (where Δ = pixel offset from center)
//   alpha = exp(-0.5 * mahalanobis) * opacity
//   output = (rgb * alpha, alpha)    (premultiplied alpha)

#import <types.wgsl>
#import <global_bindings.wgsl>

struct UniformBuffer {
    radius_scale: f32,
    opacity_scale: f32,
    alpha_discard_threshold: f32,
    _pad0: f32,
};

@group(1) @binding(0)
var<uniform> ubo: UniformBuffer;

struct VertexIn {
    @location(0) center_ndc: vec2f,
    @location(1) ndc_depth: f32,
    @location(2) radius_ndc: f32,
    @location(3) inv_cov_ndc_xx_xy_yy: vec3f,
    @location(4) color_opacity: vec4f,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOut {
    @location(0) delta_ndc: vec2f,
    @location(1) inv_cov_ndc_xx_xy_yy: vec3f,
    @location(2) color_opacity: vec4f,
    @builtin(position) position: vec4f,
};

fn quad_vertex(vertex_index: u32) -> vec2f {
    switch vertex_index {
        case 0u: { return vec2f(-1.0, -1.0); }
        case 1u: { return vec2f( 1.0, -1.0); }
        case 2u: { return vec2f(-1.0,  1.0); }
        case 3u: { return vec2f(-1.0,  1.0); }
        case 4u: { return vec2f( 1.0, -1.0); }
        default: { return vec2f( 1.0,  1.0); }
    }
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let local_quad = quad_vertex(in.vertex_index);
    let delta_ndc = local_quad * in.radius_ndc * max(ubo.radius_scale, 1e-4);
    out.delta_ndc = delta_ndc;
    out.inv_cov_ndc_xx_xy_yy = in.inv_cov_ndc_xx_xy_yy;
    out.color_opacity = in.color_opacity;
    out.position = vec4f(in.center_ndc + delta_ndc, in.ndc_depth, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let effective_radius_scale = max(ubo.radius_scale, 1e-4);
    let delta = in.delta_ndc / effective_radius_scale;
    let inv_cov = mat2x2f(
        in.inv_cov_ndc_xx_xy_yy.x, in.inv_cov_ndc_xx_xy_yy.y,
        in.inv_cov_ndc_xx_xy_yy.y, in.inv_cov_ndc_xx_xy_yy.z,
    );
    let mahalanobis = dot(delta, inv_cov * delta);
    if mahalanobis > 9.0 {
        discard;
    }

    let alpha = exp(-0.5 * mahalanobis)
        * in.color_opacity.w
        * max(ubo.opacity_scale, 0.0);
    if alpha < ubo.alpha_discard_threshold {
        discard;
    }

    return vec4f(in.color_opacity.xyz * alpha, alpha);
}
