// SOG / ASOG de-quantisation (L2) at load: RGBA texel words -> the scene's transforms, raw opacity and SH buffers.
// The CPU decoder `gsplat_core::sog::dequantize` is the oracle. Positions and opacity read lookup tables that the host
// fills in f64 from the same functions as the CPU decoder (so they match it bit for bit); scales and SH are codebook
// lookups; quaternions are computed here in f32 (v2 smallest-three, v3 exp map).

struct SogParams {
    count: u32,
    // 2 or 3.
    version: u32,
    // SH coefficients per channel beyond DC (0 without SH).
    rest: u32,
    // Usable palette entries; labels at or beyond decode to zero SH.
    palette_count: u32,
    // Word offsets of each plane in `planes` (one u32 = one RGBA texel, R in the low byte).
    means_l: u32,
    means_u: u32,
    quats: u32,
    scales: u32,
    sh0: u32,
    labels: u32,
    centroids: u32,
    // Threads per dispatch row (2D dispatch beyond 65535 workgroups).
    row_threads: u32,
    // Scene row of this file's row 0 (Streamed SOG sets put several files in one scene).
    first_row: u32,
    // SH coefficients per channel beyond DC in the scene buffers (>= rest; the extra ones stay zero).
    out_rest: u32,
    padding: vec2u,
}

@group(0) @binding(0) var<uniform> params: SogParams;
@group(0) @binding(1) var<storage, read> planes: array<u32>;
// [0, 256) log scales, [256, 512) SH DC, [512, 768) SH AC, [768, 1024) raw logit per alpha byte,
// [1024 + 65536 a, ...) position of each 16-bit code on axis a.
@group(0) @binding(2) var<storage, read> tables: array<f32>;
@group(0) @binding(3) var<storage, read_write> transforms: array<f32>;
@group(0) @binding(4) var<storage, read_write> raw_opacity: array<f32>;
@group(0) @binding(5) var<storage, read_write> sh: array<f32>;

const SCALES: u32 = 0u;
const SH_DC: u32 = 256u;
const SH_AC: u32 = 512u;
const OPACITY: u32 = 768u;
const POSITIONS: u32 = 1024u;
const HALF_PI: f32 = 1.5707963267948966;
const INV_SQRT2: f32 = 0.7071067811865476;

fn channel(word: u32, c: u32) -> u32 {
    return (word >> (8u * c)) & 0xffu;
}

// wxyz from a v2 smallest-three word (A = 252 + index of the dropped largest component; A < 252 = identity).
fn smallest_three(word: u32) -> vec4f {
    let mode = channel(word, 3u);
    if mode < 252u {
        return vec4f(1.0, 0.0, 0.0, 0.0);
    }
    let largest = mode - 252u;
    var q = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var sum = 0.0;
    var slot = 0u;
    for (var c = 0u; c < 3u; c++) {
        if slot == largest {
            slot += 1u;
        }
        let v = (f32(channel(word, c)) / 255.0 * 2.0 - 1.0) * INV_SQRT2;
        q[slot] = v;
        sum += v * v;
        slot += 1u;
    }
    q[largest] = sqrt(max(0.0, 1.0 - sum));
    return vec4f(q[0], q[1], q[2], q[3]);
}

// wxyz from a v3 exp-map word (RGB).
fn exp_map(word: u32) -> vec3f {
    return (vec3f(f32(channel(word, 0u)), f32(channel(word, 1u)), f32(channel(word, 2u))) - 128.0) / 127.0;
}

@compute @workgroup_size(256)
fn sog_dequant(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x + gid.y * params.row_threads;
    if i >= params.count {
        return;
    }
    let lo = planes[params.means_l + i];
    let hi = planes[params.means_u + i];
    let row = params.first_row + i;
    let t = row * 10u;
    for (var a = 0u; a < 3u; a++) {
        let code = channel(lo, a) | (channel(hi, a) << 8u);
        transforms[t + a] = tables[POSITIONS + a * 65536u + code];
    }
    let quat = planes[params.quats + i];
    var q: vec4f;
    if params.version == 3u {
        let p = exp_map(quat);
        let r = length(p);
        let angle = HALF_PI * r;
        var k = HALF_PI;
        if r > 1e-12 {
            k = sin(angle) / r;
        }
        q = vec4f(cos(angle), p * k);
    } else {
        q = smallest_three(quat);
    }
    transforms[t + 3u] = q.x;
    transforms[t + 4u] = q.y;
    transforms[t + 5u] = q.z;
    transforms[t + 6u] = q.w;
    let scale = planes[params.scales + i];
    for (var a = 0u; a < 3u; a++) {
        transforms[t + 7u + a] = tables[SCALES + channel(scale, a)];
    }
    let color = planes[params.sh0 + i];
    raw_opacity[row] = tables[OPACITY + channel(color, 3u)];
    let stride = (params.out_rest + 1u) * 3u;
    let s = row * stride;
    for (var c = 0u; c < 3u; c++) {
        sh[s + c] = tables[SH_DC + channel(color, c)];
    }
    if params.rest == 0u {
        return;
    }
    let l = planes[params.labels + i];
    var label = channel(l, 0u) | (channel(l, 1u) << 8u);
    if params.version == 3u {
        label |= channel(l, 2u) << 16u;
    }
    for (var k = 0u; k < params.rest; k++) {
        let o = s + 3u * (k + 1u);
        if label < params.palette_count {
            let centroid = planes[params.centroids + label * params.rest + k];
            for (var c = 0u; c < 3u; c++) {
                sh[o + c] = tables[SH_AC + channel(centroid, c)];
            }
        } else {
            for (var c = 0u; c < 3u; c++) {
                sh[o + c] = 0.0;
            }
        }
    }
}
