// Motion pre-pass, one dispatch per time change: canonical transforms + per-row motion -> this view's transforms and
// raw opacity (read by the projector at bindings 1 and 2), in the animated-sog v1 O/g form with its clips (CPU
// reference `gsplat_core::trigs::AsogMotion::row`). Scales are copied from the canonical rows; rows with zero weight
// keep the canonical pose and get raw opacity -100.
// No lane built-ins; 5 storage buffers.

struct DeformParams {
    count: u32,
    // Threads per dispatch row (2D dispatch beyond 65535 workgroups).
    row_threads: u32,
    clip_count: u32,
    // Timeline time in seconds.
    time: f32,
    // Temporal cut, clamped to [0, 0.999] on the host (0 = off).
    threshold: f32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(0) var<uniform> params: DeformParams;
// Canonical rows: mean xyz, quat wxyz, log scale xyz (10 f32 each).
@group(0) @binding(1) var<storage, read> canonical: array<f32>;
// Per-row motion, `ASOG_STRIDE` f32 each (layout in `motion_gpu.rs`).
@group(0) @binding(2) var<storage, read> motion: array<f32>;
// Asog clips, 8 f32 each: offset, active start, active end, core start, core end, fade in, fade out, unused.
@group(0) @binding(3) var<storage, read> clips: array<f32>;
@group(0) @binding(4) var<storage, read_write> transforms: array<f32>;
@group(0) @binding(5) var<storage, read_write> raw_opacity: array<f32>;

const ASOG_STRIDE: u32 = 21u;
const GAUGE_EPS: f32 = 1e-8;
const GAUGE_DENOM_CLAMP: f32 = 1e-10;
const CULLED: f32 = -100.0;

fn finite(v: f32) -> f32 {
    // WGSL has no isnan/isinf: NaN fails every comparison, and an inf is outside the f32 range.
    if v == v && abs(v) <= 3.4028234e38 {
        return v;
    }
    return 0.0;
}

fn finite3(v: vec3f) -> vec3f {
    return vec3f(finite(v.x), finite(v.y), finite(v.z));
}

// F.normalize: q / max(|q|, 1e-12).
fn normalize4(q: vec4f) -> vec4f {
    return q / max(sqrt(dot(q, q)), 1e-12);
}

// nan_to_num, then identity for (near) zero quaternions.
fn finite_quat(q: vec4f) -> vec4f {
    let f = vec4f(finite(q.x), finite(q.y), finite(q.z), finite(q.w));
    if dot(f, f) < 1e-16 {
        return vec4f(1.0, 0.0, 0.0, 0.0);
    }
    return f;
}

// Hamilton product of wxyz quaternions.
fn quat_mul(a: vec4f, b: vec4f) -> vec4f {
    return vec4f(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x,
    );
}

fn rotate(q_in: vec4f, v: vec3f) -> vec3f {
    let q = normalize4(q_in);
    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let r0 = vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y));
    let r1 = vec3f(2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x));
    let r2 = vec3f(2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    return vec3f(dot(r0, v), dot(r1, v), dot(r2, v));
}

struct Screw {
    center: vec3f,
    q_rel: vec4f,
}

// Screw motion of `center` about `a_perp` by (nu_par, omega) over dt (`trigs::screw` + `exp_se3_quat`).
fn screw(center: vec3f, a_perp: vec3f, nu_par: vec3f, omega: vec3f, dt: f32) -> Screw {
    let rho = clamp(nu_par * dt, vec3f(-1e4), vec3f(1e4));
    let w = clamp(omega * dt, vec3f(-1e4), vec3f(1e4));
    let theta = sqrt(dot(w, w));
    let theta_sq = theta * theta;
    var q: vec4f;
    var b: f32;
    var c: f32;
    if theta_sq < 1e-8 {
        let factor = 0.5 - theta_sq / 48.0;
        q = vec4f(1.0 - theta_sq / 8.0, w * factor);
        b = 0.5 - theta_sq / 24.0;
        c = 1.0 / 6.0 - theta_sq / 120.0;
    } else {
        let half = theta * 0.5;
        q = vec4f(cos(half), w * (sin(half) / theta));
        b = (1.0 - cos(theta)) / theta_sq;
        c = (theta - sin(theta)) / (theta_sq * theta);
    }
    let wxr = cross(w, rho);
    let p = rho + b * wxr + c * cross(w, wxr);
    let q_rel = normalize4(q);
    let moved = rotate(q_rel, center - a_perp) + a_perp + p;
    return Screw(finite3(moved), q_rel);
}

// max((gamma - c) / (1 - c), 0), gamma = exp(-0.5 (dt / sigma)^2).
fn temporal_gamma(dt: f32, sigma: f32) -> f32 {
    let ratio = dt / sigma;
    var gamma = exp(-0.5 * (ratio * ratio));
    let c = params.threshold;
    if c > 0.0 {
        gamma = max((gamma - c) / (1.0 - c), 0.0);
    }
    return gamma;
}

// Bezier parameter over the row's life [mu - 2 sigma, mu + 2 sigma].
fn life_tau(t: f32, mu: f32, sigma: f32) -> f32 {
    return clamp((t - (mu - 2.0 * sigma)) / (4.0 * sigma), 0.0, 1.0);
}

// Quadratic Bernstein weights.
fn bezier2(tau: f32) -> vec3f {
    let u = 1.0 - tau;
    return vec3f(u * u, 2.0 * u * tau, tau * tau);
}

fn perp_anchor(anchor: vec3f, omega: vec3f, omega_sq: f32) -> vec3f {
    if omega_sq < GAUGE_EPS {
        return anchor;
    }
    return anchor - (dot(anchor, omega) / max(omega_sq, GAUGE_DENOM_CLAMP)) * omega;
}

fn logit(alpha: f32) -> f32 {
    let a = clamp(alpha, 1e-6, 1.0 - 1e-6);
    return clamp(log(a / (1.0 - a)), -100.0, 100.0);
}

fn m3(base: u32) -> vec3f {
    return vec3f(motion[base], motion[base + 1u], motion[base + 2u]);
}

struct Row {
    center: vec3f,
    quat: vec4f,
    // Base linear opacity and the temporal x clip weight (<= 0: culled).
    opacity: f32,
    weight: f32,
}

fn asog_row(i: u32, center: vec3f, quat: vec4f) -> Row {
    let m = i * ASOG_STRIDE;
    let opacity = motion[m];
    let flags = bitcast<u32>(motion[m + 3u]);
    let clip = flags >> 1u;
    var fade = 0.0;
    var local_t = 0.0;
    if clip < params.clip_count {
        let c = clip * 8u;
        let t = params.time;
        local_t = t - clips[c];
        let active_start = clips[c + 1u];
        let active_end = clips[c + 2u];
        let core_start = clips[c + 3u];
        let core_end = clips[c + 4u];
        let fade_in = clips[c + 5u];
        let fade_out = clips[c + 6u];
        if active_start <= t && t < active_end {
            fade = 1.0;
            if fade_in > 0.0 && t < core_start {
                fade = smoothstep(0.0, 1.0, (t - (core_start - fade_in)) / fade_in);
            } else if fade_out > 0.0 && t >= core_end {
                fade = smoothstep(0.0, 1.0, (core_end + fade_out - t) / fade_out);
            }
        }
    }
    if (flags & 1u) != 0u {
        return Row(center, quat, opacity, fade);
    }
    let mu = motion[m + 1u];
    let sigma = motion[m + 2u];
    let dt = local_t - mu;
    let weight = fade * temporal_gamma(dt, sigma);
    if weight <= 0.0 {
        return Row(center, quat, opacity, weight);
    }
    let tau = life_tau(local_t, mu, sigma);
    let b = bezier2(tau);
    let omega = b.x * m3(m + 7u) + b.y * m3(m + 10u) + b.z * m3(m + 13u);
    // Quartic Bernstein of G0..G4, in the CPU's operation order.
    let u = 1.0 - tau;
    let uu = u * u;
    let tt = tau * tau;
    let g = uu * uu * motion[m + 16u] + 4.0 * uu * u * tau * motion[m + 17u] + 6.0 * uu * tt * motion[m + 18u]
        + 4.0 * u * tt * tau * motion[m + 19u] + tt * tt * motion[m + 20u];
    let omega_sq = dot(omega, omega);
    let nu_par = (g / max(omega_sq, 1e-24)) * omega;
    let a_perp = perp_anchor(m3(m + 4u), omega, omega_sq);
    let s = screw(center, a_perp, nu_par, omega, dt);
    var q = finite_quat(normalize4(quat_mul(s.q_rel, quat)));
    if q.x < 0.0 {
        q = -q;
    }
    return Row(s.center, q, opacity, weight);
}

@compute @workgroup_size(256)
fn trigs_deform(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x + gid.y * params.row_threads;
    if i >= params.count {
        return;
    }
    let t = i * 10u;
    let center = vec3f(canonical[t], canonical[t + 1u], canonical[t + 2u]);
    let quat = vec4f(canonical[t + 3u], canonical[t + 4u], canonical[t + 5u], canonical[t + 6u]);
    let row = asog_row(i, center, quat);
    transforms[t] = row.center.x;
    transforms[t + 1u] = row.center.y;
    transforms[t + 2u] = row.center.z;
    transforms[t + 3u] = row.quat.x;
    transforms[t + 4u] = row.quat.y;
    transforms[t + 5u] = row.quat.z;
    transforms[t + 6u] = row.quat.w;
    for (var k = 7u; k < 10u; k++) {
        transforms[t + k] = canonical[t + k];
    }
    if row.weight > 0.0 {
        raw_opacity[i] = logit(row.opacity * row.weight);
    } else {
        raw_opacity[i] = CULLED;
    }
}
