#import <./consts.wgsl>

struct ImageInfo { width: u32, height: u32, tiles_x: u32, tiles_y: u32, }
struct SceneInfo { splats: u32, sh_degree: u32, coefficients: u32, padding: u32, }
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
    padding: vec4u,
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

fn tile_bbox(p: vec2f, extent: vec2f) -> vec4u {
    let bounds = vec2f(vec2u(frame.image.tiles_x, frame.image.tiles_y));
    return vec4u(vec2u(clamp((p - extent) / f32(TILE_WIDTH), vec2f(0.0), bounds)), vec2u(clamp((p + extent) / f32(TILE_WIDTH) + 1.0, vec2f(0.0), bounds)));
}

fn tile_hit(tile: vec2u, p: vec2f, c: vec3f, power: f32) -> bool {
    let lo = vec2f(tile * TILE_WIDTH);
    let hi = lo + f32(TILE_WIDTH);
    let left = p.x < lo.x;
    let right = p.x > hi.x;
    let above = p.y < lo.y;
    let below = p.y > hi.y;
    let in_x = !(left || right);
    let in_y = !(above || below);
    if in_x && in_y {
        return true;
    }
    let corner = vec2f(select(hi.x, lo.x, left), select(hi.y, lo.y, above));
    let d = vec2f(select(-f32(TILE_WIDTH), f32(TILE_WIDTH), left), select(-f32(TILE_WIDTH), f32(TILE_WIDTH), above));
    let diff = p - corner;
    let tx_raw = (d.x * c.x * diff.x + d.x * c.y * diff.y) / (d.x * c.x * d.x);
    let ty_raw = (d.y * c.y * diff.x + d.y * c.z * diff.y) / (d.y * c.z * d.y);
    let t = vec2f(select(clamp(tx_raw, 0.0, 1.0), 0.0, in_y), select(clamp(ty_raw, 0.0, 1.0), 0.0, in_x));
    return sigma(corner + t * d, p, c) <= power;
}
