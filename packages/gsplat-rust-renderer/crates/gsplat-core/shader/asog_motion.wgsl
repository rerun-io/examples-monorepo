// ASOG motion de-quantisation at load: the zlib integers of the motion streams -> the packed asog rows that
// `trigs_deform.wgsl` reads (ASOG_STRIDE f32 per row). One thread per row, no lane built-ins, three storage buffers.
// CPU oracle: `gsplat_core::asog::AsogParts::motion` (f64, the reference's operation order); this pass is f32, so the
// rows agree within f32 rounding (`motion_gpu::tests`). The position x is the de-quantised canonical centre that
// `sog_dequant.wgsl` wrote before this pass.

struct AsogDequantParams {
    count: u32,
    // Threads per dispatch row (2D dispatch beyond 65535 workgroups).
    row_threads: u32,
    // Bezier samples of the reference speed (`motion.mrefSamples`, >= 1).
    samples: u32,
    pad0: u32,
    // logsigma = ls_min + code / ls_levels * ls_range.
    ls_min: f32,
    ls_range: f32,
    ls_levels: f32,
    // Anchor step (m per code): `motion.qa / 1000`.
    qa: f32,
    // `motion.qp / 1000`.
    qpv: f32,
    r_min: f32,
    mid_o: f32,
    mid_g: f32,
    // Life length in sigmas.
    sm: f32,
    // mu = code / mu_scale (s).
    mu_scale: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: AsogDequantParams;
// MOTION_CODES words per row: opacity, mu, logsigma, anchor (3), omega (9: O0, O1, O2), g (5, f32 bits), flags.
@group(0) @binding(1) var<storage, read> codes: array<i32>;
// The canonical transforms (f32 x 10: mean, quat wxyz, log scale).
@group(0) @binding(2) var<storage, read> canonical: array<f32>;
@group(0) @binding(3) var<storage, read_write> rows: array<f32>;

const MOTION_CODES: u32 = 21u;
const ASOG_STRIDE: u32 = 21u;

fn omega_code(c: u32, k: u32) -> vec3<f32> {
    let at = c + 6u + 3u * k;
    return vec3<f32>(f32(codes[at]), f32(codes[at + 1u]), f32(codes[at + 2u]));
}

@compute @workgroup_size(256)
fn asog_motion_dequant(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + gid.y * params.row_threads;
    if i >= params.count {
        return;
    }
    let c = i * MOTION_CODES;
    let sigma = exp(params.ls_min + f32(codes[c + 2u]) / params.ls_levels * params.ls_range);
    let life = params.sm * sigma;
    let a = vec3<f32>(f32(codes[c + 3u]), f32(codes[c + 4u]), f32(codes[c + 5u])) * params.qa;
    let x = vec3<f32>(canonical[i * 10u], canonical[i * 10u + 1u], canonical[i * 10u + 2u]);
    let step_o = params.qpv / max(length(x - a), params.r_min);
    let o0 = omega_code(c, 0u) * step_o;
    let o1 = omega_code(c, 1u) * (step_o * params.mid_o);
    let o2 = omega_code(c, 2u) * step_o;
    var min_norm = 3.0e38;
    for (var s = 0u; s < params.samples; s++) {
        var t = 0.0;
        if params.samples > 1u {
            t = f32(s) / f32(params.samples - 1u);
        }
        let u = 1.0 - t;
        min_norm = min(min_norm, length(u * u * o0 + 2.0 * u * t * o1 + t * t * o2));
    }
    let mref = max(min_norm, 1e-9) * params.qpv;
    let r = i * ASOG_STRIDE;
    rows[r] = f32(codes[c]) / 255.0;
    rows[r + 1u] = f32(codes[c + 1u]) / params.mu_scale;
    rows[r + 2u] = sigma;
    // flags = static | clip << 1, the same bits `Motion::pack` writes.
    rows[r + 3u] = bitcast<f32>(u32(codes[c + 20u]));
    rows[r + 4u] = a.x;
    rows[r + 5u] = a.y;
    rows[r + 6u] = a.z;
    let o = array<vec3<f32>, 3>(o0 / life, o1 / life, o2 / life);
    for (var k = 0u; k < 3u; k++) {
        rows[r + 7u + 3u * k] = o[k].x;
        rows[r + 8u + 3u * k] = o[k].y;
        rows[r + 9u + 3u * k] = o[k].z;
    }
    let g_scale = mref / (life * life);
    for (var j = 0u; j < 5u; j++) {
        var w = params.mid_g;
        if j == 0u || j == 4u {
            w = 1.0;
        }
        rows[r + 16u + j] = bitcast<f32>(codes[c + 15u + j]) * g_scale * w;
    }
}
