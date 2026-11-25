#import <./consts.wgsl>

struct ImageInfo { width: u32, height: u32, tiles_x: u32, tiles_y: u32, }
struct SceneInfo { splats: u32, sh_degree: u32, coefficients: u32, variant: u32, }
struct LensInfo { kind: u32, padding: u32, padding2: vec2u, half_max_render_fov: f32, radial_limit: f32, padding3: vec2f, }
struct RenderFlags { log_scale_offset: f32, mip: u32, has_min_scale: u32, padding: u32, }
// Forward math hand-ported from Brush 1388f74c.
struct Uniforms {
    view_from_obj: mat4x4f,
    camera_position_in_obj: vec4f,
    pinhole_params: vec4f,
    jacobian_clamp_limits: vec4f,
    image: ImageInfo,
    scene: SceneInfo,
    background: vec4f,
    options: RenderFlags,
    coefficients: array<vec4f, 2>,
    lens: LensInfo,
    camera_rotation: vec4f,
}

struct Splat {
    x: f32,
    y: f32,
    conic_x: f32,
    conic_y: f32,
    conic_z: f32,
    color_a: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    depth: f32
}

@group(0) @binding(0) var<uniform> frame: Uniforms;
fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn finite3(v: vec3f) -> bool {
    return finite(v.x) && finite(v.y) && finite(v.z);
}

// Float-flip preserves positive-depth order and orders negative depths numerically.
fn depth_key(depth: f32) -> u32 {
    let bits = bitcast<u32>(depth);
    return bits ^ select(0x80000000u, 0xffffffffu, (bits >> 31u) != 0u);
}

fn sigma(p: vec2f, center: vec2f, conic: vec3f) -> f32 {
    let d = p - center;
    return 0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) + conic.y * d.x * d.y;
}

fn bbox_extent(c: vec3f, power: f32) -> vec2f {
    let det = c.x * c.z - c.y * c.y;
    if det <= 0.0 {
        return vec2f(-1.0);
    }
    return sqrt(2.0 * power * vec2f(c.z, c.x) * (1.0 / det));
}

// Tiles whose pixel centres (tile * TILE_WIDTH + 0.5 ..= + TILE_WIDTH - 0.5) can fall inside p +- extent.
fn tile_bbox(p: vec2f, extent: vec2f) -> vec4u {
    let bounds = vec2f(vec2u(frame.image.tiles_x, frame.image.tiles_y));
    let lo = ceil((p - extent - (f32(TILE_WIDTH) - 0.5)) / f32(TILE_WIDTH));
    let hi = floor((p + extent - 0.5) / f32(TILE_WIDTH)) + 1.0;
    return vec4u(vec2u(clamp(lo, vec2f(0.0), bounds)), vec2u(clamp(hi, vec2f(0.0), bounds)));
}

// Whether sigma <= power somewhere in the rectangle lo..hi (non-empty). sigma is convex, so outside the
// rectangle its minimum lies on an edge through the nearest corner: test the closest point of both.
fn rect_hit(lo: vec2f, hi: vec2f, p: vec2f, c: vec3f, power: f32) -> bool {
    let left = p.x < lo.x;
    let above = p.y < lo.y;
    if !(left || p.x > hi.x) && !(above || p.y > hi.y) {
        return true;
    }
    let corner = vec2f(select(hi.x, lo.x, left), select(hi.y, lo.y, above));
    let d = select(lo - hi, hi - lo, vec2<bool>(left, above));
    let diff = p - corner;
    let tx = clamp((c.x * diff.x + c.y * diff.y) / (c.x * d.x), 0.0, 1.0);
    let ty = clamp((c.y * diff.x + c.z * diff.y) / (c.z * d.y), 0.0, 1.0);
    return min(sigma(corner + vec2f(tx * d.x, 0.0), p, c), sigma(corner + vec2f(0.0, ty * d.y), p, c)) <= power;
}

// Raster samples pixel centres only, so a tile is hit when one of them can reach alpha >= 1/255.
fn tile_hit(tile: vec2u, p: vec2f, c: vec3f, power: f32) -> bool {
    let lo = vec2f(tile * TILE_WIDTH) + 0.5;
    return rect_hit(lo, lo + f32(TILE_WIDTH - 1u), p, c, power);
}
