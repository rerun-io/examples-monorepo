#import <./global_bindings.wgsl>
// The core stores display-space premultiplied color. Rerun blends in linear space.
@group(1) @binding(0) var raster_color: texture_2d<f32>;
@group(1) @binding(1) var raster_depth: texture_2d<f32>;
struct Composite {
    @location(0) color: vec4f,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fs_main(@builtin(position) position: vec4f) -> Composite {
    let pixel = min(vec2u(position.xy), textureDimensions(raster_color) - vec2u(1u));
    let rgba = textureLoad(raster_color, vec2i(pixel), 0);
    if rgba.a <= 0.0 { discard; }
    let straight = rgba.rgb / rgba.a;
    let linear = select(pow((straight + 0.055) / 1.055, vec3f(2.4)), straight / 12.92, straight <= vec3f(0.04045));
    let z = textureLoad(raster_depth, vec2i(pixel), 0).r;
    let clip = frame.projection_from_view * vec4f(0.0, 0.0, -z, 1.0);
    return Composite(vec4f(linear * rgba.a, rgba.a), clamp(clip.z / clip.w, 0.0, 1.0));
}
