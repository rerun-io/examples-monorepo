#import <./hexel_common.wgsl>
#import <./raster_common.wgsl>

// Upstream render_hexels3d: intersect the ray with the hexel plane (h = k x l), evaluate the soft hexagon
// and the linear colour field; the shared loop composites front to back.

// World normal of a hexel facing the camera. With M = K [a b c] (rows tu, tv, tw), the
// view normal a x b is parallel to K^T (m0 x m1) and its sign test reduces to (m0 x m1) . mc.
fn hexel_normal(h: Hexel) -> vec3f {
    let m = cross(vec3f(h.t[0], h.t[3], h.t[6]), vec3f(h.t[1], h.t[4], h.t[7]));
    let pp = frame.pinhole_params.zw - 0.5;
    let n = normalize(vec3f(frame.pinhole_params.x * m.x, frame.pinhole_params.y * m.y, dot(vec3f(pp, 1.0), m)));
    let facing = select(n, -n, dot(m, vec3f(h.t[2], h.t[5], h.t[8])) > 0.0);
    let q = frame.camera_rotation;
    let t = 2.0 * cross(q.xyz, facing);
    return facing + q.w * t + cross(q.xyz, t);
}

// Values, not a mutable Sample: the struct with early returns cost NVIDIA ~15 % of the hexel raster stage.
fn sample(h: Hexel, pix: vec2u, surface: bool) -> Sample {
    let m = plane(h);
    let p = cross(f32(pix.x) * m.tw - m.tu, f32(pix.y) * m.tw - m.tv);
    // hexagon_norm(p.xy) / |p.z| is the hexagon SDF + 1 at s = p.xy / p.z. Beyond `reach` alpha < 1/255,
    // so most candidates of a tile return here, before the divisions and the exp.
    if hexagon_norm(p.xy) > h.reach * abs(p.z) || p.z == 0.0 {
        return Sample(0.0, vec3f(0.0), 0.0, vec3f(0.0));
    }
    let s = p.xy / p.z;
    let depth = s.x * m.tw.x + s.y * m.tw.y + m.tw.z;
    let alpha = min(h.opacity / (1.0 + exp(hexagon_sdf(s) * h.sharpness)), 0.99);
    let color = vec3f(h.color[0], h.color[1], h.color[2]) + s.x * vec3f(h.color[3], h.color[4], h.color[5]) +
        s.y * vec3f(h.color[6], h.color[7], h.color[8]);
    var normal = vec3f(0.0);
    if surface {
        normal = hexel_normal(h);
    }
    return Sample(select(alpha, 0.0, depth < HEX_NEAR), color, depth, normal);
}
