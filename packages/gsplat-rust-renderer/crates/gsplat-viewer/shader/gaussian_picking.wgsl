#import <./global_bindings.wgsl>
// Picking ids and outline masks from the core's median-contributor id texture (u32::MAX = none).
@group(1) @binding(1) var raster_depth: texture_2d<f32>;
@group(1) @binding(2) var raster_id: texture_2d<u32>;
// `DrawUniform` in renderer.rs: mask words hold channel A in bits 0-7, B in 8-15, bit 16 = set.
struct Draw {
    background: vec4f,
    picking_object: vec2u,
    outline: u32,
    highlight_count: u32,
    // (row, mask) pairs, two per element.
    highlights: array<vec4u, 14>,
}
@group(2) @binding(0) var<uniform> draw: Draw;
const NO_ID = 0xffffffffu;

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) ndc: vec2f,
}

// Full-view quad placed through view space: the picking pass crops `projection_from_view` to its
// rectangle but keeps `tan_half_fov`, so the quad covers the same view pixels in every pass.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var corners = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
        vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0));
    let ndc = corners[index];
    var out: VertexOut;
    out.position = frame.projection_from_view * vec4f(ndc * frame.tan_half_fov, -1.0, 1.0);
    out.ndc = ndc;
    return out;
}

struct Sample { id: u32, depth: f32, }

// The id and the composite's depth at this fragment's full-view pixel.
fn sample(ndc: vec2f) -> Sample {
    let size = textureDimensions(raster_id);
    let pixel = min(vec2u((ndc * vec2f(0.5, -0.5) + 0.5) * vec2f(size)), size - vec2u(1u));
    let z = textureLoad(raster_depth, vec2i(pixel), 0).r;
    let clip = frame.projection_from_view * vec4f(0.0, 0.0, -z, 1.0);
    return Sample(textureLoad(raster_id, vec2i(pixel), 0).r, clamp(clip.z / clip.w, 0.0, 1.0));
}

struct PickingOut {
    @location(0) id: vec4u,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_picking(in: VertexOut) -> PickingOut {
    let s = sample(in.ndc);
    if s.id == NO_ID { discard; }
    return PickingOut(vec4u(draw.picking_object, s.id, 0u), s.depth);
}

struct OutlineOut {
    @location(0) mask: vec2u,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_outline(in: VertexOut) -> OutlineOut {
    let s = sample(in.ndc);
    if s.id == NO_ID { discard; }
    var mask = draw.outline;
    for (var i = 0u; i < draw.highlight_count; i++) {
        let pair = draw.highlights[i / 2u];
        let row = select(pair.xy, pair.zw, (i & 1u) == 1u);
        if row.x == s.id {
            mask = row.y;
        }
    }
    if (mask & 0x10000u) == 0u { discard; }
    return OutlineOut(vec2u(mask & 0xffu, (mask >> 8u) & 0xffu), s.depth);
}
