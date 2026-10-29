#import <./hexel_common.wgsl>
#import <./counts.wgsl>

// Hexel counterparts of project_forward and project_visible. Parameters per hexel:
// centre, wxyz rotation, scales u/v, opacity, sharpness, colour, u and v colour gradients,
// then the view term (scene.coefficients is the stride, scene.variant the variant).
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> global_from_compact: array<u32>;
@group(0) @binding(5) var<storage, read_write> depths: array<u32>;
@group(0) @binding(7) var<storage, read_write> tile_hits: array<u32>;
@group(0) @binding(8) var<storage, read_write> projected: array<Hexel>;

fn param3(i: u32) -> vec3f {
    return vec3f(params[i], params[i + 1u], params[i + 2u]);
}

// Upstream compute_transmat: T = transpose(splat2world) * full_proj * ndc2pix, written
// with the pinhole intrinsics of this frame instead of an OpenGL projection matrix.
fn hexel_plane(id: u32) -> Plane {
    let base = id * frame.scene.coefficients;
    let qraw = vec4f(params[base + 3u], params[base + 4u], params[base + 5u], params[base + 6u]);
    let q = qraw * inverseSqrt(dot(qraw, qraw));
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let scale = exp(frame.options.log_scale_offset) * vec2f(params[base + 7u], params[base + 8u]);
    let view = mat3x3f(frame.view_from_obj[0].xyz, frame.view_from_obj[1].xyz, frame.view_from_obj[2].xyz);
    let a = view * (vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)) * scale.x);
    let b = view * (vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)) * scale.y);
    let c = (frame.view_from_obj * vec4f(param3(base), 1.0)).xyz;
    let f = frame.pinhole_params.xy;
    let pp = frame.pinhole_params.zw - 0.5;
    return Plane(f.x * vec3f(a.x, b.x, c.x) + pp.x * vec3f(a.z, b.z, c.z),
        f.y * vec3f(a.y, b.y, c.y) + pp.y * vec3f(a.z, b.z, c.z), vec3f(a.z, b.z, c.z));
}

fn sharpness(id: u32) -> f32 {
    return max(params[id * frame.scene.coefficients + 10u], 1e-4);
}

@compute @workgroup_size(WG) fn hexel_forward(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let id = (gid.x + gid.y * groups.x) * WG + lid;
    if id >= frame.scene.splats {
        return;
    }
    let m = hexel_plane(id);
    let opacity = params[id * frame.scene.coefficients + 9u];
    if !(m.tw.z > HEX_NEAR && finite3(m.tu) && finite3(m.tv) && finite3(m.tw) && finite(opacity)) {
        return;
    }
    let rect = hexel_tiles(m, opacity, sharpness(id));
    // Only the tiles of each row that the outline touches; most of a tilted hexel's rectangle misses.
    let o = outline(m, hexel_reach(opacity, sharpness(id)));
    var hit = 0u;
    for (var row = rect.y; row < rect.w; row++) {
        let span = row_span(o, rect, row, 1.0);
        hit += span.y - span.x;
    }
    if hit == 0u {
        return;
    }
    tile_hits[id] = hit;
    add_intersections(hit);
    let write = atomicAdd(&counts[0], 1u) & 0x7fffffffu;
    global_from_compact[write] = id;
    let bits = bitcast<u32>(m.tw.z);
    depths[write] = bits ^ 0x80000000u;
}

// Upstream renderer.py adds the view term per primitive before rasterization:
// V2 c += A d, V3 c += sum_k Y_k(d) c_k (degrees 1-3), with d from the camera to the centre.
fn view_color(base: u32) -> vec3f {
    var color = param3(base + 11u);
    if frame.scene.variant == HEXEL_V1 {
        return color;
    }
    let delta = param3(base) - frame.camera_position_in_obj.xyz;
    let d = delta * inverseSqrt(dot(delta, delta) + 1e-8);
    let k = base + 20u;
    if frame.scene.variant == HEXEL_V2 {
        return color + vec3f(dot(param3(k), d), dot(param3(k + 3u), d), dot(param3(k + 6u), d));
    }
    let x = d.x;
    let y = d.y;
    let z = d.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    var basis = array<f32, 15>(-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x,
        1.0925484305920792 * x * y, -1.0925484305920792 * y * z, 0.31539156525252005 * (2.0 * zz - xx - yy),
        -1.0925484305920792 * x * z, 0.5462742152960396 * (xx - yy),
        -0.5900435899266435 * y * (3.0 * xx - yy), 2.890611442640554 * x * y * z,
        -0.4570457994644658 * y * (4.0 * zz - xx - yy), 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
        -0.4570457994644658 * x * (4.0 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
        -0.5900435899266435 * x * (xx - 3.0 * yy));
    for (var i = 0u; i < 15u; i++) {
        color += basis[i] * param3(k + 3u * i);
    }
    return color;
}

@compute @workgroup_size(WG) fn hexel_visible(@builtin(workgroup_id) gid: vec3u,
    @builtin(num_workgroups) groups: vec3u, @builtin(local_invocation_index) lid: u32) {
    let compact = (gid.x + gid.y * groups.x) * WG + lid;
    if compact >= atomicLoad(&counts[0]) {
        return;
    }
    let id = global_from_compact[compact];
    let base = id * frame.scene.coefficients;
    let m = hexel_plane(id);
    let c = view_color(base);
    projected[compact] = Hexel(array<f32, 9>(m.tu.x, m.tu.y, m.tu.z, m.tv.x, m.tv.y, m.tv.z, m.tw.x, m.tw.y, m.tw.z),
        params[base + 9u], sharpness(id), hexel_reach(params[base + 9u], sharpness(id)),
        array<f32, 9>(c.x, c.y, c.z, params[base + 14u], params[base + 15u],
        params[base + 16u], params[base + 17u], params[base + 18u], params[base + 19u]));
}
