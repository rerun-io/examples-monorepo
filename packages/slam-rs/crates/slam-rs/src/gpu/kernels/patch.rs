//! patch kernels and their launchers.
use super::super::finite::is_finite;
use super::layout::*;
use super::sampling::{in_bounds, interp_grad_into};
use cubecl::prelude::*;

// Per-frame launchers use unchecked launch after stage shape validation.
// Each binding retains its handle and element count; only the storage probe uses checked launch.

/// In-place guarded pivoted LDLT on a row-major 3x3 matrix.
/// Tiny pivots get zero weight instead of division, keeping rank-deficient patch
/// factors finite. Flags replace early returns for uniform device execution.
#[cube]
fn ldlt_decompose3(mat: &mut Array<f32>, transpositions: &mut Array<usize>) {
    let mut temp = Array::<f32>::new(3usize);
    let mut done = false;

    for k in 0..3usize {
        if !done {
            // "Find largest diagonal element"; `maxCoeff`
            // reports the first index of the maximum, so ties take the earliest.
            let mut biggest = k;
            for i in (k + 1usize)..3usize {
                if f32::abs(mat[i * 3usize + i]) > f32::abs(mat[biggest * 3usize + biggest]) {
                    biggest = i;
                }
            }
            transpositions[k] = biggest;

            if k != biggest {
                // "taking care to consider only the lower triangular part"
                for j in 0..k {
                    let swap = mat[k * 3usize + j];
                    mat[k * 3usize + j] = mat[biggest * 3usize + j];
                    mat[biggest * 3usize + j] = swap;
                }
                for i in (biggest + 1usize)..3usize {
                    let swap = mat[i * 3usize + k];
                    mat[i * 3usize + k] = mat[i * 3usize + biggest];
                    mat[i * 3usize + biggest] = swap;
                }
                let swap = mat[k * 3usize + k];
                mat[k * 3usize + k] = mat[biggest * 3usize + biggest];
                mat[biggest * 3usize + biggest] = swap;
                for i in (k + 1usize)..biggest {
                    let swap = mat[i * 3usize + k];
                    mat[i * 3usize + k] = mat[biggest * 3usize + i];
                    mat[biggest * 3usize + i] = swap;
                }
            }

            // The delayed column updates through `temp`.
            if k > 0usize {
                for i in 0..k {
                    temp[i] = mat[i * 3usize + i] * mat[k * 3usize + i];
                }
                let mut correction = 0.0f32;
                for i in 0..k {
                    correction += mat[k * 3usize + i] * temp[i];
                }
                mat[k * 3usize + k] -= correction;
                for row in (k + 1usize)..3usize {
                    let mut update = 0.0f32;
                    for i in 0..k {
                        update += mat[row * 3usize + i] * temp[i];
                    }
                    mat[row * 3usize + k] -= update;
                }
            }

            // LAPACK's cutoff of exactly zero.
            let pivot = mat[k * 3usize + k];
            let pivot_is_valid = f32::abs(pivot) > 0.0f32;
            if k == 0usize && !pivot_is_valid {
                // "The entire diagonal is zero, there is nothing more to do."
                for j in 0..3usize {
                    transpositions[j] = j;
                }
                done = true;
            }
            if !done && pivot_is_valid {
                for row in (k + 1usize)..3usize {
                    mat[row * 3usize + k] /= pivot;
                }
            }
        }
    }
}

/// `LDLT::_solve_impl_transposed<true>` at size 3, in place
/// on a row-major 3x3 right-hand side.
#[cube]
fn ldlt_solve3(mat: &Array<f32>, transpositions: &Array<usize>, rhs: &mut Array<f32>) {
    // `dst = m_transpositions * rhs`: k ascending.
    for k in 0..3usize {
        let target = transpositions[k];
        if target != k {
            for j in 0..3usize {
                let swap = rhs[k * 3usize + j];
                rhs[k * 3usize + j] = rhs[target * 3usize + j];
                rhs[target * 3usize + j] = swap;
            }
        }
    }
    // `matrixL().solveInPlace(dst)`: unit-lower forward substitution.
    for row in 1..3usize {
        for column in 0..row {
            let factor = mat[row * 3usize + column];
            for j in 0..3usize {
                let value = rhs[column * 3usize + j];
                rhs[row * 3usize + j] -= factor * value;
            }
        }
    }
    // "more precisely, use pseudo-inverse of D" : the
    // tolerance is `numeric_limits<float>::min()`, the smallest positive normal.
    for i in 0..3usize {
        let d = mat[i * 3usize + i];
        if f32::abs(d) > 1.1754944e-38f32 {
            for j in 0..3usize {
                rhs[i * 3usize + j] /= d;
            }
        } else {
            for j in 0..3usize {
                rhs[i * 3usize + j] = 0.0f32;
            }
        }
    }
    // `matrixL().transpose().solveInPlace(dst)`: unit-upper back substitution.
    for step in 0..3usize {
        let row = 2usize - step;
        for column in (row + 1usize)..3usize {
            let factor = mat[column * 3usize + row];
            for j in 0..3usize {
                let value = rhs[column * 3usize + j];
                rhs[row * 3usize + j] -= factor * value;
            }
        }
    }
    // `dst = m_transpositions.transpose() * dst`: k descending.
    for step in 0..3usize {
        let k = 2usize - step;
        let target = transpositions[k];
        if target != k {
            for j in 0..3usize {
                let swap = rhs[k * 3usize + j];
                rhs[k * 3usize + j] = rhs[target * 3usize + j];
                rhs[target * 3usize + j] = swap;
            }
        }
    }
}

// ── the patch build ──────────────────────────────────────────────────────────

/// `patch::build_patch` over one patch at one level.
///
/// One cube per `(patch, level)`, one unit per pattern tap. Three reductions
/// run on unit 0 in ascending tap order — the tap sum with `grad_sum_se2`, the
/// nine entries of `H = J^T J`, and the finiteness `AND` — because in `f32` the
/// order is the value (decision D31).
///
/// The product-rule correction that comes from differentiating the `1/mean`
/// factor is applied with the **raw** tap, before the tap is
/// normalised, and the rows of out-of-bounds taps are zeroed: dropping that
/// term gives a Jacobian that looks right and converges to the wrong warp.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn patch_build_kernel(
    pyramid_a: &Array<u16>,
    pyramid_b: &Array<u16>,
    meta: &Array<u32>,
    positions: &Array<f32>,
    store: &mut Array<f32>,
    capacity: usize,
    taps: usize,
    num_levels: usize,
    count: usize,
    pos_x_base: usize,
    pos_y_base: usize,
    selected_base: usize,
) {
    let patch = usize::cast_from(CUBE_POS_X);
    let level = usize::cast_from(CUBE_POS_Y);
    let tap = usize::cast_from(UNIT_POS_X);
    if patch >= count {
        terminate!();
    }

    let data_len = num_levels * taps * capacity;
    let row_stride = taps * capacity;
    let valid_slot = 4usize * data_len + level * capacity + patch;

    // A patch the caller switched off is invalid at every level and never
    // sampled (`SourcePatches::build`'s `selected`).
    if positions[selected_base + patch] == 0.0f32 {
        if tap == 0usize {
            store[valid_slot] = 0.0f32;
        }
        terminate!();
    }

    let mut values = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_x = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_y = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut grad_t = SharedMemory::<f32>::new(TAP_SLOTS);
    let mut okflag = SharedMemory::<usize>::new(TAP_SLOTS);
    let mut red = SharedMemory::<f32>::new(16usize);

    // `const Scalar scale = 1 << level`.
    let scale = f32::cast_from(1usize << level);
    let pos_x = positions[pos_x_base + patch] / scale;
    let pos_y = positions[pos_y_base + patch] / scale;

    let base = usize::cast_from(meta[level * 4usize]);
    let width = usize::cast_from(meta[level * 4usize + 1usize]);
    let height = usize::cast_from(meta[level * 4usize + 2usize]);
    let parity = usize::cast_from(meta[level * 4usize + 3usize]);
    let pattern = num_levels * 4usize;

    if tap < taps {
        let tap_x = f32::reinterpret(meta[pattern + tap * 2usize]);
        let tap_y = f32::reinterpret(meta[pattern + tap * 2usize + 1usize]);
        let px = pos_x + tap_x;
        let py = pos_y + tap_y;
        if in_bounds(px, py, PATCH_BORDER, width, height) {
            if parity == 0usize {
                interp_grad_into(
                    pyramid_a,
                    base,
                    width,
                    px,
                    py,
                    &mut values,
                    &mut grad_x,
                    &mut grad_y,
                    tap,
                );
            } else {
                interp_grad_into(
                    pyramid_b,
                    base,
                    width,
                    px,
                    py,
                    &mut values,
                    &mut grad_x,
                    &mut grad_y,
                    tap,
                );
            }
            // `valGrad.tail<2>().transpose() * Jw_se2` with
            // `Jw_se2 = [[1, 0, -tap_y], [0, 1, tap_x]]`.
            grad_t[tap] = grad_x[tap] * -tap_y + grad_y[tap] * tap_x;
            okflag[tap] = 1usize;
        } else {
            values[tap] = -1.0f32;
            grad_x[tap] = 0.0f32;
            grad_y[tap] = 0.0f32;
            grad_t[tap] = 0.0f32;
            okflag[tap] = 0usize;
        }
    }
    sync_cube();

    if tap == 0usize {
        let mut sum = 0.0f32;
        let mut valid_points: u32 = 0u32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_t = 0.0f32;
        for i in 0..taps {
            if okflag[i] == 1usize {
                sum += values[i];
                valid_points += 1u32;
                sum_x += grad_x[i];
                sum_y += grad_y[i];
                sum_t += grad_t[i];
            }
        }
        let points = f32::cast_from(valid_points);
        red[0usize] = sum;
        // `mean = sum / n` and `mean_inv = n / sum` are
        // two separate divisions, not reciprocals of each other.
        red[1usize] = sum / points;
        red[2usize] = points / sum;
        red[3usize] = sum_x;
        red[4usize] = sum_y;
        red[5usize] = sum_t;
    }
    sync_cube();

    let sum = red[0usize];
    let mean_inv = red[2usize];
    if tap < taps {
        let raw = values[tap];
        if raw >= 0.0f32 {
            grad_x[tap] = (grad_x[tap] - red[3usize] * raw / sum) * mean_inv;
            grad_y[tap] = (grad_y[tap] - red[4usize] * raw / sum) * mean_inv;
            grad_t[tap] = (grad_t[tap] - red[5usize] * raw / sum) * mean_inv;
            values[tap] = raw * mean_inv;
        } else {
            grad_x[tap] = 0.0f32;
            grad_y[tap] = 0.0f32;
            grad_t[tap] = 0.0f32;
        }
    }
    sync_cube();

    if tap == 0usize {
        // `H_se2 = J^T J`, one rank-1 outer product per tap.
        let mut hessian = Array::<f32>::new(9usize);
        for i in 0..9usize {
            hessian[i] = 0.0f32;
        }
        for i in 0..taps {
            let r0 = grad_x[i];
            let r1 = grad_y[i];
            let r2 = grad_t[i];
            hessian[0usize] += r0 * r0;
            hessian[1usize] += r0 * r1;
            hessian[2usize] += r0 * r2;
            hessian[3usize] += r1 * r0;
            hessian[4usize] += r1 * r1;
            hessian[5usize] += r1 * r2;
            hessian[6usize] += r2 * r0;
            hessian[7usize] += r2 * r1;
            hessian[8usize] += r2 * r2;
        }
        let mut transpositions = Array::<usize>::new(3usize);
        ldlt_decompose3(&mut hessian, &mut transpositions);
        let mut inverse = Array::<f32>::new(9usize);
        for i in 0..9usize {
            inverse[i] = 0.0f32;
        }
        inverse[0usize] = 1.0f32;
        inverse[4usize] = 1.0f32;
        inverse[8usize] = 1.0f32;
        ldlt_solve3(&hessian, &transpositions, &mut inverse);
        for i in 0..9usize {
            red[7usize + i] = inverse[i];
        }
    }
    sync_cube();

    if tap < taps {
        // `H_se2_inv_J_se2_T.col(i) = H^-1 * J^T.col(i)`.
        let c0 = grad_x[tap];
        let c1 = grad_y[tap];
        let c2 = grad_t[tap];
        let p0 = red[7usize] * c0 + red[8usize] * c1 + red[9usize] * c2;
        let p1 = red[10usize] * c0 + red[11usize] * c1 + red[12usize] * c2;
        let p2 = red[13usize] * c0 + red[14usize] * c1 + red[15usize] * c2;
        let slot = data_len + (level * 3usize * taps + tap) * capacity + patch;
        store[slot] = p0;
        store[slot + row_stride] = p1;
        store[slot + 2usize * row_stride] = p2;
        let value = values[tap];
        store[(level * taps + tap) * capacity + patch] = value;
        let finite = is_finite(p0) && is_finite(p1) && is_finite(p2) && is_finite(value);
        let mut flag: usize = 0usize;
        if finite {
            flag = 1usize;
        }
        okflag[tap] = flag;
    } else {
        okflag[tap] = 1usize;
    }
    sync_cube();

    if tap == 0usize {
        // `valid = mean > eps && H^-1 J^T finite && data finite`.
        let mut finite = true;
        for i in 0..taps {
            if okflag[i] == 0usize {
                finite = false;
            }
        }
        let mut valid: f32 = 0.0f32;
        if red[1usize] > f32::new(f32::EPSILON) && finite {
            valid = 1.0f32;
        }
        store[valid_slot] = valid;
    }
}

/// Sample every patch of `positions` at every level of the pyramid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_patch_build<R: Runtime>(
    client: &ComputeClient<R>,
    pyramid: (
        &cubecl::server::Handle,
        usize,
        &cubecl::server::Handle,
        usize,
    ),
    meta: Buffer<'_>,
    positions: Buffer<'_>,
    store: Buffer<'_>,
    shape: PatchShape,
    bases: PositionBases,
) {
    super::super::submission::launch(client);
    unsafe {
        patch_build_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(shape.count as u32, shape.num_levels as u32, 1),
            CubeDim {
                x: TAP_UNITS,
                y: 1,
                z: 1,
            },
            ArrayArg::from_raw_parts(pyramid.0.clone(), pyramid.1),
            ArrayArg::from_raw_parts(pyramid.2.clone(), pyramid.3),
            ArrayArg::from_raw_parts(meta.0.clone(), meta.1),
            ArrayArg::from_raw_parts(positions.0.clone(), positions.1),
            ArrayArg::from_raw_parts(store.0.clone(), store.1),
            shape.capacity,
            shape.taps,
            shape.num_levels,
            shape.count,
            bases.x,
            bases.y,
            bases.selected,
        );
    }
}
