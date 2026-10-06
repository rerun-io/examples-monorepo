//! Register-resident KLT: one 16-lane group per point, both passes.
use super::super::{finite::is_finite, trig};
use super::layout::*;
use super::sampling::{at, in_bounds, interp};
use cubecl::prelude::*;

/// Point-major input/output record: warp (6), validity, source (2), camera.
pub(crate) const FUSED_RUNS: usize = 10;
const RUN_WARP: usize = 0;
const RUN_VALID: usize = 6;
const RUN_SOURCE_X: usize = 7;
const RUN_SOURCE_Y: usize = 8;
const RUN_CAMERA: usize = 9;

/// Camera source/target geometry followed by the shared pattern offsets.
pub(crate) const fn meta_len(levels: usize, cameras: usize, taps: usize) -> usize {
    2 * levels * 4 * cameras + taps * 2
}

#[derive(Clone, Copy)]
pub(crate) struct FusedParams {
    pub levels: usize,
    pub taps: usize,
    pub iterations: usize,
    pub max_dist2: f32,
    pub exit_step_px: Option<f32>,
}

pub(crate) fn launch_fused<R: Runtime>(
    client: &ComputeClient<R>,
    images: [(&cubecl::server::Handle, usize); 4],
    meta: (&cubecl::server::Handle, usize),
    io: &cubecl::server::Handle,
    count: usize,
    cameras: usize,
    params: FusedParams,
) {
    // SAFETY: Callers supply validated pyramid metadata and FUSED_RUNS values per
    // point. Each 16-unit group owns one point; the kernel rejects groups beyond count.
    unsafe {
        fused_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static((count * 16).div_ceil(64) as u32, 1, 1),
            CubeDim::new_1d(64),
            BufferArg::from_raw_parts(images[0].0.clone(), images[0].1),
            BufferArg::from_raw_parts(images[1].0.clone(), images[1].1),
            BufferArg::from_raw_parts(images[2].0.clone(), images[2].1),
            BufferArg::from_raw_parts(images[3].0.clone(), images[3].1),
            BufferArg::from_raw_parts(meta.0.clone(), meta.1),
            BufferArg::from_raw_parts(io.clone(), count * FUSED_RUNS),
            count,
            cameras,
            params.max_dist2,
            params.exit_step_px.unwrap_or(0.0),
            params.levels,
            params.taps,
            params.iterations,
            params.exit_step_px.is_some(),
        );
    }
}

#[derive(Clone)]
pub(crate) struct CachedU32Upload {
    host: Vec<u32>,
    pub handle: cubecl::server::Handle,
}

impl CachedU32Upload {
    pub fn new<R: Runtime>(client: &ComputeClient<R>, capacity: usize) -> Self {
        Self {
            host: Vec::with_capacity(capacity),
            handle: client.empty(capacity * size_of::<u32>()),
        }
    }

    pub fn update<R: Runtime>(&mut self, client: &ComputeClient<R>, values: &[u32]) {
        if self.host != values {
            client.write(
                &self.handle,
                cubecl::bytes::Bytes::from_elems(values.to_vec()),
            );
            self.host.clear();
            self.host.extend_from_slice(values);
        }
    }

    pub fn len(&self) -> usize {
        self.host.len()
    }
}

pub(crate) fn encode_point(
    warp: [f32; 6],
    valid: f32,
    source: nalgebra::Vector2<f32>,
    camera: usize,
) -> [f32; FUSED_RUNS] {
    let mut values = [0.0; FUSED_RUNS];
    values[RUN_WARP..RUN_VALID].copy_from_slice(&warp);
    values[RUN_VALID] = valid;
    values[RUN_SOURCE_X] = source.x;
    values[RUN_SOURCE_Y] = source.y;
    values[RUN_CAMERA] = camera as f32;
    values
}

pub(crate) fn decode_point(
    values: &[f32],
    out: &mut crate::frontend::tracker::FlowResult,
    index: usize,
) {
    let transform = kornia_staging_imgproc::optical_flow::patch_se2::AffineCompact2f {
        linear: nalgebra::Matrix2::new(
            values[RUN_WARP],
            values[RUN_WARP + 1],
            values[RUN_WARP + 2],
            values[RUN_WARP + 3],
        )
        .into(),
        translation: nalgebra::Vector2::new(values[RUN_WARP + 4], values[RUN_WARP + 5]).into(),
    };
    out.set_track(index, values[RUN_VALID] != 0.0, &transform);
}

// Central-difference interpolation: twelve pixel loads held in registers.
#[cube]
fn gradient(image: &[u16], base: usize, stride: usize, x: f32, y: f32, output: &mut Array<f32>) {
    let ix = usize::cast_from(x);
    let iy = usize::cast_from(y);
    let dx = x - f32::cast_from(ix);
    let dy = y - f32::cast_from(iy);
    let ddx = 1.0f32 - dx;
    let ddy = 1.0f32 - dy;

    let px0y0 = at(image, base, stride, ix, iy);
    let px1y0 = at(image, base, stride, ix + 1usize, iy);
    let px0y1 = at(image, base, stride, ix, iy + 1usize);
    let px1y1 = at(image, base, stride, ix + 1usize, iy + 1usize);
    output[0usize] = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

    let pxm1y0 = at(image, base, stride, ix - 1usize, iy);
    let pxm1y1 = at(image, base, stride, ix - 1usize, iy + 1usize);
    let res_mx = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;
    let px2y0 = at(image, base, stride, ix + 2usize, iy);
    let px2y1 = at(image, base, stride, ix + 2usize, iy + 1usize);
    let res_px = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;
    output[1usize] = 0.5f32 * (res_px - res_mx);

    let px0ym1 = at(image, base, stride, ix, iy - 1usize);
    let px1ym1 = at(image, base, stride, ix + 1usize, iy - 1usize);
    let res_my = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;
    let px0y2 = at(image, base, stride, ix, iy + 2usize);
    let px1y2 = at(image, base, stride, ix + 1usize, iy + 2usize);
    let res_py = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;
    output[2usize] = 0.5f32 * (res_py - res_my);
}

#[cube]
fn reduce(values: &Array<f32>, #[comptime] taps: usize, #[comptime] lanes: usize) -> f32 {
    let mut sum = 0.0f32;
    #[unroll]
    for slot in 0..taps.div_ceil(lanes) {
        sum += values[slot];
    }
    #[unroll]
    for bit in 0..lanes.ilog2() {
        sum += plane_shuffle_xor(sum, 1u32 << bit);
    }
    sum
}

#[cube]
#[allow(clippy::too_many_arguments, unused_assignments)]
fn sample(
    a: &[u16],
    b: &[u16],
    c: &[u16],
    d: &[u16],
    base: usize,
    width: usize,
    parity: usize,
    frame: usize,
    x: f32,
    y: f32,
) -> f32 {
    let mut value = 0.0f32;
    if frame == 0usize {
        if parity == 0usize {
            value = interp(a, base, width, x, y);
        } else {
            value = interp(b, base, width, x, y);
        }
    } else {
        if parity == 0usize {
            value = interp(c, base, width, x, y);
        } else {
            value = interp(d, base, width, x, y);
        }
    }
    value
}

#[cube]
#[allow(clippy::assign_op_pattern)]
fn compose(state: &mut Array<f32>, t0: f32, t1: f32, theta: f32) {
    let cos_theta = f32::cos(theta);
    let sin_theta = trig::sin(theta);
    let length = f32::sqrt(cos_theta * cos_theta + sin_theta * sin_theta);
    let real = cos_theta / length;
    let imaginary = sin_theta / length;
    let mut sin_by_theta = imaginary / theta;
    let mut one_minus_cos_by_theta = (1.0f32 - real) / theta;
    if f32::abs(theta) < SOPHUS_EPSILON {
        let theta_sq = theta * theta;
        sin_by_theta = 1.0f32 - (1.0f32 / 6.0f32) * theta_sq;
        one_minus_cos_by_theta = 0.5f32 * theta - (1.0f32 / 24.0f32) * theta * theta_sq;
    }
    let exp_x = sin_by_theta * t0 - one_minus_cos_by_theta * t1;
    let exp_y = one_minus_cos_by_theta * t0 + sin_by_theta * t1;
    let m00 = state[0usize];
    let m01 = state[1usize];
    let m10 = state[2usize];
    let m11 = state[3usize];
    state[0usize] = m00 * real + m01 * imaginary;
    state[1usize] = m00 * -imaginary + m01 * real;
    state[2usize] = m10 * real + m11 * imaginary;
    state[3usize] = m10 * -imaginary + m11 * real;
    state[4usize] = m00 * exp_x + m01 * exp_y + state[4usize];
    state[5usize] = m10 * exp_x + m11 * exp_y + state[5usize];
}

// Fixed-index pivoted LDLT. Dynamic pivot indices
// force function-local arrays onto Mali's stack. Keep the pivot decisions, but
// express each possible swap with scalar indices so the compiler can promote
// every element. Solve H * G = J directly: forming the nine-entry inverse
// first costs more registers and is only equivalent up to float order.
#[cube]
#[allow(clippy::manual_swap)] // CubeCL expands assignments, not std::mem::swap.
fn precondition(
    h: &Array<f32>,
    j0: &mut Array<f32>,
    j1: &mut Array<f32>,
    j2: &mut Array<f32>,
    #[comptime] slots: usize,
) {
    let mut d0 = h[0usize];
    let mut l10 = h[3usize];
    let mut d1 = h[4usize];
    let mut l20 = h[6usize];
    let mut l21 = h[7usize];
    let mut d2 = h[8usize];
    let mut pivot0 = 0usize;
    let mut largest = f32::abs(d0);
    if f32::abs(d1) > largest {
        pivot0 = 1usize;
        largest = f32::abs(d1);
    }
    if f32::abs(d2) > largest {
        pivot0 = 2usize;
    }
    if pivot0 == 1usize {
        let swap = d0;
        d0 = d1;
        d1 = swap;
        let swap = l20;
        l20 = l21;
        l21 = swap;
    }
    if pivot0 == 2usize {
        let swap = d0;
        d0 = d2;
        d2 = swap;
        let swap = l10;
        l10 = l21;
        l21 = swap;
    }
    let mut pivot1 = false;
    if f32::abs(d0) > 0.0f32 {
        l10 /= d0;
        l20 /= d0;
        // The reference selects its next pivot before the delayed diagonal
        // update. Do not compare the Schur-complement diagonals here.
        pivot1 = f32::abs(d2) > f32::abs(d1);
        if pivot1 {
            let swap = d1;
            d1 = d2;
            d2 = swap;
            let swap = l10;
            l10 = l20;
            l20 = swap;
        }
        let t0 = d0 * l10;
        d1 -= l10 * t0;
        l21 -= l20 * t0;
        if f32::abs(d1) > 0.0f32 {
            l21 /= d1;
        }
        let t0 = d0 * l20;
        let t1 = d1 * l21;
        d2 -= l20 * t0 + l21 * t1;
    } else {
        pivot0 = 0usize;
    }
    #[unroll]
    for column in 0..slots {
        let mut x = j0[column];
        let mut y = j1[column];
        let mut z = j2[column];
        if pivot0 == 1usize {
            let swap = x;
            x = y;
            y = swap;
        }
        if pivot0 == 2usize {
            let swap = x;
            x = z;
            z = swap;
        }
        if pivot1 {
            let swap = y;
            y = z;
            z = swap;
        }
        y -= l10 * x;
        z -= l20 * x;
        z -= l21 * y;
        if f32::abs(d0) > LDLT_TOLERANCE {
            x /= d0;
        } else {
            x = 0.0f32;
        }
        if f32::abs(d1) > LDLT_TOLERANCE {
            y /= d1;
        } else {
            y = 0.0f32;
        }
        if f32::abs(d2) > LDLT_TOLERANCE {
            z /= d2;
        } else {
            z = 0.0f32;
        }
        y -= l21 * z;
        x -= l10 * y;
        x -= l20 * z;
        if pivot1 {
            let swap = y;
            y = z;
            z = swap;
        }
        if pivot0 == 2usize {
            let swap = x;
            x = z;
            z = swap;
        }
        if pivot0 == 1usize {
            let swap = x;
            x = y;
            y = swap;
        }
        j0[column] = x;
        j1[column] = y;
        j2[column] = z;
    }
}

/// Metadata: [camera][source/target][level][base,width,height,parity], then taps.
/// In/out: point-major [m00,m01,m10,m11,guess_x,guess_y,selected,source_x,source_y,camera].
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::collapsible_if // Keep the compile-time exit switch outside the runtime condition.
)]
pub(crate) fn fused_kernel(
    prev_a: &[u16],
    prev_b: &[u16],
    next_a: &[u16],
    next_b: &[u16],
    meta: &[u32],
    io: &mut [f32],
    count: usize,
    cameras: usize,
    max_dist2: f32,
    exit_step_px: f32,
    #[comptime] levels: usize,
    #[comptime] taps: usize,
    #[comptime] iterations: usize,
    #[comptime] exit_enabled: bool,
) {
    let lanes = comptime!(16usize);
    let unit = CUBE_POS_X * CUBE_DIM + PLANE_POS * PLANE_DIM + UNIT_POS_PLANE;
    let point = usize::cast_from(unit) / lanes;
    let lane = usize::cast_from(UNIT_POS_PLANE) % lanes;
    if point >= count {
        terminate!();
    }
    let idx = point * FUSED_RUNS;
    // Device-prepared stereo reserves one slot per cell. Entire point groups
    // share this flag, so unused capacity can leave before sampling or shuffles.
    if io[idx + RUN_VALID] == 0.0f32 {
        terminate!();
    }
    let camera = usize::cast_from(io[idx + RUN_CAMERA]);
    let pattern = cameras * comptime!(meta_len(levels, 1, 0));
    let source_x = io[idx + RUN_SOURCE_X];
    let source_y = io[idx + RUN_SOURCE_Y];
    let guess_x = io[idx + RUN_WARP + 4];
    let guess_y = io[idx + RUN_WARP + 5];
    let offset_x = source_x - guess_x;
    let offset_y = source_y - guess_y;
    let mut template_x = source_x;
    let mut template_y = source_y;
    let mut target_x = guess_x;
    let mut target_y = guess_y;
    let mut forward = Array::<f32>::new(6usize);
    let mut state = Array::<f32>::new(6usize);
    let mut alive = io[idx + RUN_VALID] != 0.0f32;
    let width0 = f32::cast_from(meta[(camera * 2usize + 1usize) * levels * 4usize + 1usize]);
    let height0 = f32::cast_from(meta[(camera * 2usize + 1usize) * levels * 4usize + 2usize]);
    let inside = !(guess_x < 0.0f32 || guess_y < 0.0f32 || guess_x >= width0 || guess_y >= height0);
    alive = alive && inside;
    for pass in 0..2usize {
        state[0usize] = 1.0f32;
        state[1usize] = 0.0f32;
        state[2usize] = 0.0f32;
        state[3usize] = 1.0f32;
        state[4usize] = target_x;
        state[5usize] = target_y;
        for step in 0..levels {
            let level = levels - 1usize - step;
            if alive {
                let scale = f32::cast_from(1usize << level);
                state[4usize] /= scale;
                state[5usize] /= scale;
                let src = ((camera * 2usize + pass) * levels + level) * 4usize;
                let dst = ((camera * 2usize + 1usize - pass) * levels + level) * 4usize;
                let sb = usize::cast_from(meta[src]);
                let sw = usize::cast_from(meta[src + 1usize]);
                let sh = usize::cast_from(meta[src + 2usize]);
                let sp = usize::cast_from(meta[src + 3usize]);
                let db = usize::cast_from(meta[dst]);
                let dw = usize::cast_from(meta[dst + 1usize]);
                let dh = usize::cast_from(meta[dst + 2usize]);
                let dp = usize::cast_from(meta[dst + 3usize]);
                let mut data = Array::<f32>::new(taps.div_ceil(lanes));
                let mut j0 = Array::<f32>::new(taps.div_ceil(lanes));
                let mut j1 = Array::<f32>::new(taps.div_ceil(lanes));
                let mut j2 = Array::<f32>::new(taps.div_ceil(lanes));
                let mut temp = Array::<f32>::new(taps.div_ceil(lanes));
                let mut valid = Array::<f32>::new(taps.div_ceil(lanes));
                let mut ox = Array::<f32>::new(taps.div_ceil(lanes));
                let mut oy = Array::<f32>::new(taps.div_ceil(lanes));
                #[unroll]
                for slot in 0..taps.div_ceil(lanes) {
                    let tap = slot * lanes + lane;
                    data[slot] = -1.0f32;
                    temp[slot] = 0.0f32;
                    valid[slot] = 0.0f32;
                    j0[slot] = 0.0f32;
                    j1[slot] = 0.0f32;
                    j2[slot] = 0.0f32;
                    ox[slot] = 0.0f32;
                    oy[slot] = 0.0f32;
                    if tap < taps {
                        let tx = f32::reinterpret(meta[pattern + 2usize * tap]);
                        let ty = f32::reinterpret(meta[pattern + 2usize * tap + 1usize]);
                        ox[slot] = tx;
                        oy[slot] = ty;
                        let x = template_x / scale + tx;
                        let y = template_y / scale + ty;
                        if in_bounds(x, y, PATCH_BORDER, sw, sh) {
                            let mut val_grad = Array::<f32>::new(3usize);
                            if pass == 0usize {
                                if sp == 0usize {
                                    gradient(prev_a, sb, sw, x, y, &mut val_grad);
                                } else {
                                    gradient(prev_b, sb, sw, x, y, &mut val_grad);
                                }
                            } else {
                                if sp == 0usize {
                                    gradient(next_a, sb, sw, x, y, &mut val_grad);
                                } else {
                                    gradient(next_b, sb, sw, x, y, &mut val_grad);
                                }
                            }
                            let v = val_grad[0usize];
                            let gx = val_grad[1usize];
                            let gy = val_grad[2usize];
                            data[slot] = v;

                            temp[slot] = v;
                            valid[slot] = 1.0f32;
                            j0[slot] = gx;
                            j1[slot] = gy;
                            j2[slot] = gx * -ty + gy * tx;
                        }
                    }
                }
                let sum = reduce(&temp, taps, lanes);
                let points = reduce(&valid, taps, lanes);
                let sx = reduce(&j0, taps, lanes);
                let sy = reduce(&j1, taps, lanes);
                let st = reduce(&j2, taps, lanes);
                let mean = sum / points;
                let inv_mean = points / sum;
                #[unroll]
                for slot in 0..taps.div_ceil(lanes) {
                    let raw = data[slot];
                    if raw >= 0.0f32 {
                        j0[slot] = (j0[slot] - sx * raw / sum) * inv_mean;
                        j1[slot] = (j1[slot] - sy * raw / sum) * inv_mean;
                        j2[slot] = (j2[slot] - st * raw / sum) * inv_mean;
                        data[slot] = raw * inv_mean;
                    }
                }
                let mut h = Array::<f32>::new(9usize);
                #[unroll]
                for row in 0..3usize {
                    #[unroll]
                    for col in 0..row + 1usize {
                        #[unroll]
                        for slot in 0..taps.div_ceil(lanes) {
                            let mut a = j0[slot];
                            let mut b = j0[slot];
                            if row == 1usize {
                                a = j1[slot];
                            }
                            if row == 2usize {
                                a = j2[slot];
                            }
                            if col == 1usize {
                                b = j1[slot];
                            }
                            if col == 2usize {
                                b = j2[slot];
                            }
                            temp[slot] = a * b;
                        }
                        h[row * 3usize + col] = reduce(&temp, taps, lanes);
                    }
                }
                precondition(&h, &mut j0, &mut j1, &mut j2, taps.div_ceil(lanes));
                #[unroll]
                for slot in 0..taps.div_ceil(lanes) {
                    let mut bad = 0.0f32;
                    if slot * lanes + lane < taps
                        && !(is_finite(j0[slot])
                            && is_finite(j1[slot])
                            && is_finite(j2[slot])
                            && is_finite(data[slot]))
                    {
                        bad = 1.0f32;
                    }
                    temp[slot] = bad;
                }
                alive = mean > f32::new(f32::EPSILON) && reduce(&temp, taps, lanes) == 0.0f32;
                for _iteration in 0..iterations {
                    if alive {
                        let mut sampled = Array::<f32>::new(taps.div_ceil(lanes));
                        #[unroll]
                        for slot in 0..taps.div_ceil(lanes) {
                            sampled[slot] = -1.0f32;
                            temp[slot] = 0.0f32;
                            valid[slot] = 0.0f32;
                            let x =
                                state[0usize] * ox[slot] + state[1usize] * oy[slot] + state[4usize];
                            let y =
                                state[2usize] * ox[slot] + state[3usize] * oy[slot] + state[5usize];
                            if slot * lanes + lane < taps && in_bounds(x, y, PATCH_BORDER, dw, dh) {
                                let v = sample(
                                    prev_a,
                                    prev_b,
                                    next_a,
                                    next_b,
                                    db,
                                    dw,
                                    dp,
                                    1usize - pass,
                                    x,
                                    y,
                                );
                                sampled[slot] = v;
                                temp[slot] = v;
                                valid[slot] = 1.0f32;
                            }
                        }
                        let sum = reduce(&temp, taps, lanes);
                        let points = reduce(&valid, taps, lanes);
                        #[unroll]
                        for slot in 0..taps.div_ceil(lanes) {
                            let mut residual = 0.0f32;
                            let mut ok = 0.0f32;
                            if slot * lanes + lane < taps
                                && sampled[slot] >= 0.0f32
                                && data[slot] >= 0.0f32
                            {
                                residual = points * sampled[slot] / sum - data[slot];
                                ok = 1.0f32;
                            }
                            sampled[slot] = residual;
                            valid[slot] = ok;
                            temp[slot] = j0[slot] * residual;
                        }
                        let inc0 = -reduce(&temp, taps, lanes);
                        #[unroll]
                        for slot in 0..taps.div_ceil(lanes) {
                            temp[slot] = j1[slot] * sampled[slot];
                        }
                        let inc1 = -reduce(&temp, taps, lanes);
                        #[unroll]
                        for slot in 0..taps.div_ceil(lanes) {
                            temp[slot] = j2[slot] * sampled[slot];
                        }
                        let inc2 = -reduce(&temp, taps, lanes);
                        let residuals = reduce(&valid, taps, lanes);
                        let mut norm = f32::abs(inc0);
                        if norm < f32::abs(inc1) {
                            norm = f32::abs(inc1);
                        }
                        if norm < f32::abs(inc2) {
                            norm = f32::abs(inc2);
                        }
                        alive = !(sum < f32::new(f32::EPSILON))
                            && residuals * 2.0f32 > taps as f32
                            && is_finite(inc0)
                            && is_finite(inc1)
                            && is_finite(inc2)
                            && norm < MAX_INCREMENT_INFINITY_NORM;
                        if alive {
                            compose(&mut state, inc0, inc1, inc2);
                            alive = in_bounds(state[4usize], state[5usize], FILTER_MARGIN, dw, dh);
                            // Uniform within the point's lane group: every lane
                            // holds the same reduced increment. No barrier is needed.
                            // Specialize this away entirely when the knob is off.
                            if exit_enabled {
                                if inc0 * inc0 + inc1 * inc1 < exit_step_px * exit_step_px
                                    && f32::abs(inc2) * 4.0f32 < exit_step_px
                                {
                                    break;
                                }
                            }
                        }
                    }
                }
                state[4usize] *= scale;
                state[5usize] *= scale;
            }
        }
        if pass == 0usize {
            #[unroll]
            for k in 0..6usize {
                forward[k] = state[k];
            }
            template_x = state[4usize];
            template_y = state[5usize];
            target_x = template_x + offset_x;
            target_y = template_y + offset_y;
        }
    }
    let dx = source_x - state[4usize];
    let dy = source_y - state[5usize];
    alive = alive && dx * dx + dy * dy < max_dist2;
    if lane == 0usize {
        let o00 = io[idx + RUN_WARP];
        let o01 = io[idx + RUN_WARP + 1];
        let o10 = io[idx + RUN_WARP + 2];
        let o11 = io[idx + RUN_WARP + 3];
        io[idx + RUN_WARP] = o00 * forward[0usize] + o01 * forward[2usize];
        io[idx + RUN_WARP + 1] = o00 * forward[1usize] + o01 * forward[3usize];
        io[idx + RUN_WARP + 2] = o10 * forward[0usize] + o11 * forward[2usize];
        io[idx + RUN_WARP + 3] = o10 * forward[1usize] + o11 * forward[3usize];
        io[idx + RUN_WARP + 4] = forward[4usize];
        io[idx + RUN_WARP + 5] = forward[5usize];
        if !inside {
            io[idx + RUN_WARP] = 1.0f32;
            io[idx + RUN_WARP + 1] = 0.0f32;
            io[idx + RUN_WARP + 2] = 0.0f32;
            io[idx + RUN_WARP + 3] = 1.0f32;
            io[idx + RUN_WARP + 4] = 0.0f32;
            io[idx + RUN_WARP + 5] = 0.0f32;
        }
        let mut flag = 0.0f32;
        if alive {
            flag = 1.0f32;
        }
        io[idx + RUN_VALID] = flag;
    }
}
