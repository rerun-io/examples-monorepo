struct Uniforms {
    view: mat4x4f,
    camera: vec4f,
    pinhole: vec4f,
    clamp_limits: vec4f,
    image: vec4u, // width, height, tiles_x, tiles_y
    scene: vec4u, // splats, SH degree, coefficients, intersection capacity
    background: vec4f,
    options: vec4f,
    coeff0: vec4f,
    coeff1: vec4f,
    lens: vec4u,
    camera_limits: vec4f,
}
struct Splat { x:f32, y:f32, cx:f32, cy:f32, cz:f32, opacity:f32, r:f32, g:f32, b:f32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3f) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn sigma(p: vec2f, center: vec2f, conic: vec3f) -> f32 {
    let d = p - center;
    return 0.5 * (conic.x*d.x*d.x + conic.z*d.y*d.y) + conic.y*d.x*d.y;
}
fn bbox_extent(c: vec3f, power: f32) -> vec2f {
    let det = c.x*c.z - c.y*c.y;
    if det <= 0.0 { return vec2f(-1.0); }
    return sqrt(2.0 * power * vec2f(c.z,c.x) * (1.0/det));
}
fn tile_bbox(p: vec2f, extent: vec2f) -> vec4u {
    let bounds = vec2f(u.image.zw);
    return vec4u(vec2u(clamp((p-extent)/16.0,vec2f(0.0),bounds)), vec2u(clamp((p+extent)/16.0+1.0,vec2f(0.0),bounds)));
}
fn tile_hit(tile: vec2u, p: vec2f, c: vec3f, power: f32) -> bool {
    let lo = vec2f(tile*16u); let hi = lo+16.0;
    let left = p.x < lo.x; let right = p.x > hi.x;
    let above = p.y < lo.y; let below = p.y > hi.y;
    let in_x = !(left || right); let in_y = !(above || below);
    if in_x && in_y { return true; }
    let corner = vec2f(select(hi.x,lo.x,left),select(hi.y,lo.y,above));
    let d = vec2f(select(-16.0,16.0,left),select(-16.0,16.0,above));
    let diff = p - corner;
    let tx_raw = (d.x*c.x*diff.x+d.x*c.y*diff.y)/(d.x*c.x*d.x);
    let ty_raw = (d.y*c.y*diff.x+d.y*c.z*diff.y)/(d.y*c.z*d.y);
    let t = vec2f(select(clamp(tx_raw,0.0,1.0),0.0,in_y),select(clamp(ty_raw,0.0,1.0),0.0,in_x));
    return sigma(corner+t*d,p,c) <= power;
}
