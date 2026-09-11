//! track kernels and their launchers.
use super::super::{finite::is_finite, trig};
use super::layout::*;
use super::sampling::{in_bounds, interp};
use cubecl::prelude::*;

// Per-frame launchers use launch_unchecked. Stage shape checks establish
// the binding lengths; each raw binding keeps its handle and element count
// together. The storage probe alone retains checked launch mode.
// ── the KLT tracker ──────────────────────────────────────────────────────────

/// `Sophus::SE2<float>::exp` composed onto the running warp, on unit 0.
///
/// Writes the composed `[m00, m01, m10, m11, tx, ty]` back into `state`. Three
/// details of `tracker::se2_exp` are load-bearing and reproduced literally: the
/// `SO2(cos, sin)` constructor **normalises** by `hypot`, the `V` factor divides
/// by the *normalised* components, and the small-angle branch is Sophus's
/// Taylor series below `Constants<float>::epsilon()`. `hypot` is spelled
/// `sqrt(c*c + s*s)` here. Only sine uses a bounded polynomial: division by theta
/// amplifies its measured 257-ULP native error; native cosine stays within 2 ULP.
/// MIO14 GT ATE is 9.48 cm with sin+cos and 9.72 cm with sine only (D71). Normalization
/// and fused multiply-add can still differ from the CPU by an ulp.
#[cube]
fn compose_se2_exp(state: &mut SharedMemory<f32>, t0: f32, t1: f32, theta: f32) {
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

    // `transform *= exp`: `linear * exp.linear`, `linear * exp.t + t`.
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

/// Coarse-to-fine KLT, one cube per patch.
/// Fixed-bound loops use a shared alive flag. Read it after barriers so no unit
/// skips a barrier reached by its neighbors. Forward tracking checks the initial
/// guess bounds; backward tracking does not. An invalid initial guess returns
/// identity, recorded separately from failure after the sweep starts.
#[cube(launch, launch_unchecked)]
// Negated comparisons reject NaNs. Conditional store values use locals to avoid
// the device compiler's conditional-store miscompile.
#[allow(
    clippy::too_many_arguments,
    clippy::neg_cmp_op_on_partial_ord,
    unused_assignments
)]
fn klt_kernel(
    pyramid_a: &Array<u16>,
    pyramid_b: &Array<u16>,
    meta: &Array<u32>,
    store: &Array<f32>,
    // Compose the warp in place through one read/write binding. Binding the same
    // buffer separately as read-only and read/write is invalid on some wgpu adapters.
    transforms: &mut Array<f32>,
    capacity: usize,
    taps: usize,
    num_levels: usize,
    count: usize,
    max_iterations: usize,
    check_guess_bounds: usize,
) {
    let patch = usize::cast_from(CUBE_POS_X);
    let tap = usize::cast_from(UNIT_POS_X);
    if patch >= count {
        terminate!();
    }

    let data_len = num_levels * taps * capacity;
    let row_stride = taps * capacity;
    let pattern = num_levels * 4usize;

    let mut residual = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_0 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_1 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut product_2 = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut okflag = SharedMemory::<usize>::new(TAP_SLOTS);
    // 0..6 the running warp, 6 the tap sum, 7 the in-bounds tap count,
    // 8 the level scale.
    let mut state = SharedMemory::<f32>::new(16usize);
    // 0 the running validity, 1 whether this level was entered alive,
    // 2 whether the guess passed the bounds test at all.
    let mut alive = SharedMemory::<usize>::new(4usize);

    if tap == 0usize {
        let guess_x = transforms[4usize * count + patch];
        let guess_y = transforms[5usize * count + patch];
        let level0_width = f32::cast_from(meta[1usize]);
        let level0_height = f32::cast_from(meta[2usize]);
        let mut inside = true;
        if check_guess_bounds == 1usize
            && (guess_x < 0.0f32
                || guess_y < 0.0f32
                || guess_x >= level0_width
                || guess_y >= level0_height)
        {
            inside = false;
        }
        let mut entered: usize = 0usize;
        if inside {
            entered = 1usize;
        }
        alive[0usize] = entered;
        alive[2usize] = entered;
        // `transform.linear = identity`, `transform.translation = guess`.
        state[0usize] = 1.0f32;
        state[1usize] = 0.0f32;
        state[2usize] = 0.0f32;
        state[3usize] = 1.0f32;
        state[4usize] = guess_x;
        state[5usize] = guess_y;
    }
    sync_cube();

    for step in 0..num_levels {
        let level = num_levels - 1usize - step;
        if tap == 0usize {
            alive[1usize] = alive[0usize];
            if alive[0usize] == 1usize {
                // `transform.translation /= scale`, exact for a power of two.
                let scale = f32::cast_from(1usize << level);
                state[8usize] = scale;
                state[4usize] /= scale;
                state[5usize] /= scale;
                // `patch_valid &= patches.valid(level, index)`.
                if store[4usize * data_len + level * capacity + patch] == 0.0f32 {
                    alive[0usize] = 0usize;
                }
            }
        }
        sync_cube();

        let base = usize::cast_from(meta[level * 4usize]);
        let width = usize::cast_from(meta[level * 4usize + 1usize]);
        let height = usize::cast_from(meta[level * 4usize + 2usize]);
        let parity = usize::cast_from(meta[level * 4usize + 3usize]);

        for _iteration in 0..max_iterations {
            let sampling = alive[0usize] == 1usize;
            // `residual(img, transform * pattern2, res)`,
            // with the warp applied per tap.
            if sampling && tap < taps {
                let tap_x = f32::reinterpret(meta[pattern + tap * 2usize]);
                let tap_y = f32::reinterpret(meta[pattern + tap * 2usize + 1usize]);
                let warped_x = state[0usize] * tap_x + state[1usize] * tap_y + state[4usize];
                let warped_y = state[2usize] * tap_x + state[3usize] * tap_y + state[5usize];
                if in_bounds(warped_x, warped_y, PATCH_BORDER, width, height) {
                    let mut sampled: f32 = 0.0f32;
                    if parity == 0usize {
                        sampled = interp(pyramid_a, base, width, warped_x, warped_y);
                    } else {
                        sampled = interp(pyramid_b, base, width, warped_x, warped_y);
                    }
                    residual[tap] = sampled;
                    okflag[tap] = 1usize;
                } else {
                    residual[tap] = -1.0f32;
                    okflag[tap] = 0usize;
                }
            }
            sync_cube();

            if sampling && tap == 0usize {
                let mut sum = 0.0f32;
                let mut valid_points: u32 = 0u32;
                for i in 0..taps {
                    if okflag[i] == 1usize {
                        sum += residual[i];
                        valid_points += 1u32;
                    }
                }
                // An all-black target cannot be normalised.
                if sum < f32::new(f32::EPSILON) {
                    alive[0usize] = 0usize;
                }
                state[6usize] = sum;
                state[7usize] = f32::cast_from(valid_points);
            }
            sync_cube();

            let solving = alive[0usize] == 1usize;
            if solving && tap < taps {
                // `res[i] = num_valid_points * val / sum - data[i]`.
                let sum = state[6usize];
                let points = state[7usize];
                let stored = store[(level * taps + tap) * capacity + patch];
                let sampled = residual[tap];
                if sampled >= 0.0f32 && stored >= 0.0f32 {
                    residual[tap] = points * sampled / sum - stored;
                    okflag[tap] = 1usize;
                } else {
                    residual[tap] = 0.0f32;
                    okflag[tap] = 0usize;
                }
                let slot = data_len + (level * 3usize * taps + tap) * capacity + patch;
                product_0[tap] = store[slot] * residual[tap];
                product_1[tap] = store[slot + row_stride] * residual[tap];
                product_2[tap] = store[slot + 2usize * row_stride] * residual[tap];
            }
            sync_cube();

            if solving && tap == 0usize {
                let mut residuals: u32 = 0u32;
                for i in 0..taps {
                    if okflag[i] == 1usize {
                        residuals += 1u32;
                    }
                }
                // `return num_residuals > PATTERN_SIZE / 2`.
                if residuals * 2u32 <= u32::cast_from(taps) {
                    alive[0usize] = 0usize;
                } else {
                    // `inc = -H_se2_inv_J_se2_T * res`, taps ascending.
                    let mut sum_0 = 0.0f32;
                    let mut sum_1 = 0.0f32;
                    let mut sum_2 = 0.0f32;
                    for i in 0..taps {
                        sum_0 += product_0[i];
                        sum_1 += product_1[i];
                        sum_2 += product_2[i];
                    }
                    let inc_0 = -sum_0;
                    let inc_1 = -sum_1;
                    let inc_2 = -sum_2;
                    let mut ok = is_finite(inc_0) && is_finite(inc_1) && is_finite(inc_2);
                    // Fold the infinity norm from element zero, retaining a left-hand NaN.
                    let mut infinity_norm = f32::abs(inc_0);
                    if infinity_norm < f32::abs(inc_1) {
                        infinity_norm = f32::abs(inc_1);
                    }
                    if infinity_norm < f32::abs(inc_2) {
                        infinity_norm = f32::abs(inc_2);
                    }
                    if !(infinity_norm < MAX_INCREMENT_INFINITY_NORM) {
                        ok = false;
                    }
                    if ok {
                        compose_se2_exp(&mut state, inc_0, inc_1, inc_2);
                        // The new centre must stay two pixels inside.
                        if !in_bounds(state[4usize], state[5usize], FILTER_MARGIN, width, height) {
                            ok = false;
                        }
                    }
                    if !ok {
                        alive[0usize] = 0usize;
                    }
                }
            }
            sync_cube();
        }

        if tap == 0usize && alive[1usize] == 1usize {
            // Scale translation even when this level failed.
            let scale = state[8usize];
            state[4usize] *= scale;
            state[5usize] *= scale;
        }
        sync_cube();
    }

    if tap == 0usize {
        if alive[2usize] == 1usize {
            // `transform.linear = old_linear * transform.linear`.
            // Read into locals before the first write: the four coefficients
            // are at the indices the composed warp overwrites.
            let o00 = transforms[patch];
            let o01 = transforms[count + patch];
            let o10 = transforms[2usize * count + patch];
            let o11 = transforms[3usize * count + patch];
            transforms[patch] = o00 * state[0usize] + o01 * state[2usize];
            transforms[count + patch] = o00 * state[1usize] + o01 * state[3usize];
            transforms[2usize * count + patch] = o10 * state[0usize] + o11 * state[2usize];
            transforms[3usize * count + patch] = o10 * state[1usize] + o11 * state[3usize];
            transforms[4usize * count + patch] = state[4usize];
            transforms[5usize * count + patch] = state[5usize];
            transforms[6usize * count + patch] = f32::cast_from(alive[0usize]);
        } else {
            transforms[patch] = 1.0f32;
            transforms[count + patch] = 0.0f32;
            transforms[2usize * count + patch] = 0.0f32;
            transforms[3usize * count + patch] = 1.0f32;
            transforms[4usize * count + patch] = 0.0f32;
            transforms[5usize * count + patch] = 0.0f32;
            transforms[6usize * count + patch] = 0.0f32;
        }
    }
}

/// The backward pass's inputs, from the forward result.
///
/// `off = t2 - t2_guess` with `t2 == t1` at that point, so
/// `off == source position - guess`, and `t1_recovered = forward + off`; the
/// linear part carries over from the forward result. Both terms of `off` are on
/// the host when a frame starts, so the tracker uploads it with the backward
/// patch set's positions and this kernel only adds it.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn prepare_backward_kernel(
    forward: &Array<f32>,
    offsets: &Array<f32>,
    out: &mut Array<f32>,
    count: usize,
    offset_x_base: usize,
    offset_y_base: usize,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    for k in 0..4usize {
        out[k * count + index] = forward[k * count + index];
    }
    out[4usize * count + index] = forward[4usize * count + index] + offsets[offset_x_base + index];
    out[5usize * count + index] = forward[5usize * count + index] + offsets[offset_y_base + index];
    out[6usize * count + index] = forward[6usize * count + index];
}

/// The recovered-distance test that decides a track.
///
/// The warp published is always the *forward* one; only the validity flag comes
/// from the backward pass, exactly as `trackPoints` keeps `transform_1` and
/// tests `(t1 - t1_recovered).squaredNorm()`.
#[cube(launch, launch_unchecked)]
// Use `!(dist2 < max)` so a NaN fails the distance guard.
#[allow(clippy::too_many_arguments, clippy::neg_cmp_op_on_partial_ord)]
fn finish_kernel(
    forward: &Array<f32>,
    backward: &Array<f32>,
    positions: &Array<f32>,
    out: &mut Array<f32>,
    count: usize,
    pos_x_base: usize,
    pos_y_base: usize,
    max_recovered_dist2: f32,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index >= count {
        terminate!();
    }
    for k in 0..6usize {
        out[k * count + index] = forward[k * count + index];
    }
    let mut ok =
        forward[6usize * count + index] != 0.0f32 && backward[6usize * count + index] != 0.0f32;
    if ok {
        let dx = positions[pos_x_base + index] - backward[4usize * count + index];
        let dy = positions[pos_y_base + index] - backward[5usize * count + index];
        if !(dx * dx + dy * dy < max_recovered_dist2) {
            ok = false;
        }
    }
    let mut flag: f32 = 0.0f32;
    if ok {
        flag = 1.0f32;
    }
    out[6usize * count + index] = flag;
}

/// Track every patch coarse to fine into the pyramid handed in.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_klt<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: (
        &cubecl::server::Handle,
        usize,
        &cubecl::server::Handle,
        usize,
    ),
    meta: Buffer<'_>,
    store: Buffer<'_>,
    transforms: Buffer<'_>,
    shape: PatchShape,
    max_iterations: usize,
    check_guess_bounds: bool,
) {
    super::super::submission::launch(client);
    unsafe {
        klt_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(shape.count as u32, 1, 1),
            CubeDim {
                x: TAP_UNITS,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(pyramid.0.clone(), pyramid.1),
            ArrayArg::from_raw_parts(pyramid.2.clone(), pyramid.3),
            ArrayArg::from_raw_parts(meta.0.clone(), meta.1),
            ArrayArg::from_raw_parts(store.0.clone(), store.1),
            ArrayArg::from_raw_parts(transforms.0.clone(), transforms.1),
            shape.capacity,
            shape.taps,
            shape.num_levels,
            shape.count,
            max_iterations,
            usize::from(check_guess_bounds),
        );
    }
}

/// Build the backward pass's transform inputs from the forward result.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_prepare_backward<R: Runtime>(
    client: &ComputeClient<R>,
    forward: Buffer<'_>,
    offsets: Buffer<'_>,
    out: Buffer<'_>,
    count: usize,
    bases: PositionBases,
) {
    let (cubes, units) = linear_1d(count);
    super::super::submission::launch(client);
    unsafe {
        prepare_backward_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(forward.0.clone(), forward.1),
            ArrayArg::from_raw_parts(offsets.0.clone(), offsets.1),
            ArrayArg::from_raw_parts(out.0.clone(), out.1),
            count,
            bases.x,
            bases.y,
        );
    }
}

/// Combine the two passes into the result the CPU downloads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_finish<R: Runtime>(
    client: &ComputeClient<R>,
    forward: Buffer<'_>,
    backward: Buffer<'_>,
    positions: Buffer<'_>,
    out: Buffer<'_>,
    count: usize,
    bases: PositionBases,
    max_recovered_dist2: f32,
) {
    let (cubes, units) = linear_1d(count);
    super::super::submission::launch(client);
    unsafe {
        finish_kernel::launch_unchecked::<R>(
            client,
            cubes,
            units,
            ArrayArg::from_raw_parts(forward.0.clone(), forward.1),
            ArrayArg::from_raw_parts(backward.0.clone(), backward.1),
            ArrayArg::from_raw_parts(positions.0.clone(), positions.1),
            ArrayArg::from_raw_parts(out.0.clone(), out.1),
            count,
            bases.x,
            bases.y,
            max_recovered_dist2,
        );
    }
}
