// Fullscreen composite pass for the tile-raster path.

struct VsOut {
    @builtin(position) position: vec4f,
};

struct RasterUniformBuffer {
    tile_bounds: vec2u,
    img_size: vec2u,
};

@group(1) @binding(0) var raster_color: texture_2d<f32>;
@group(1) @binding(1) var<uniform> raster_uniforms: RasterUniformBuffer;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(-1.0, 3.0),
        vec2f(3.0, -1.0),
    );
    var out: VsOut;
    let position = positions[vertex_index];
    out.position = vec4f(position, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let dims = max(raster_uniforms.img_size, vec2u(1u, 1u));
    let pixel = min(vec2u(in.position.xy), dims - vec2u(1u, 1u));
    return textureLoad(raster_color, vec2i(pixel), 0);
}
