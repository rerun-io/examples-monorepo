#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>
// The core stores display-space premultiplied color. Rerun blends in linear space.
@group(1) @binding(0) var raster_color: texture_2d<f32>;
@group(1) @binding(1) var raster_depth: texture_2d<f32>;
// Solid view background in display space (w = 1); w = 0 for gradient backgrounds.
struct Background { color: vec4f, }
@group(2) @binding(0) var<uniform> background: Background;
struct Composite {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fs_main(@builtin(position) position: vec4f) -> Composite {
    let pixel = min(vec2u(position.xy), textureDimensions(raster_color) - vec2u(1u));
    let rgba = textureLoad(raster_color, vec2i(pixel), 0);
    if rgba.a <= 0.0 { discard; }
    let z = textureLoad(raster_depth, vec2i(pixel), 0).r;
    let clip = frame.projection_from_view * vec4f(0.0, 0.0, -z, 1.0);
    let depth = clamp(clip.z / clip.w, 0.0, 1.0);
    if background.color.w <= 0.0 {
        return Composite(linear_from_srgba_premultiplied(rgba), depth);
    }
    // Splats and hexels are trained with display-space blending over a solid background. Reproduce
    // that exactly over the view's solid background: Rerun adds (1 - a') linear(bg), so emit
    // linear(rgb + (1 - a) bg) - (1 - a') linear(bg). The UNORM target clamps negative colour, so a'
    // keeps the largest transparency <= 1 - a for which every channel stays non-negative.
    let wanted = linear_from_srgb(saturate(rgba.rgb + (1.0 - rgba.a) * background.color.rgb));
    let behind = linear_from_srgb(background.color.rgb);
    let ratio = select(vec3f(1.0), wanted / max(behind, vec3f(1e-6)), behind > vec3f(0.0));
    let keep = clamp(min(1.0 - rgba.a, min(ratio.x, min(ratio.y, ratio.z))), 0.0, 1.0);
    return Composite(vec4f(wanted - keep * behind, 1.0 - keep), depth);
}
