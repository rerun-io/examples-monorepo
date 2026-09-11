//! Whitened IMU factor assembly and exports over a pair of states.

use super::{ImuResidualJacobians, IntegratedImuMeasurement, Matrix9, Matrix15x30};
use crate::lie::{LieScalar, c};
use crate::types::{POSE_VEL_BIAS_SIZE, POSE_VEL_SIZE, PoseVelBiasStateWithLin, Vector9, Vector15};
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, Vector3};

/// Where the gyroscope bias starts inside a 15-vector state block: the `+9` of
const BIAS_GYRO_OFFSET: usize = POSE_VEL_SIZE;
/// Accelerometer bias starts at offset 12 in the full state.
const BIAS_ACCEL_OFFSET: usize = POSE_VEL_SIZE + 3;

/// Linearization inputs for an IMU factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuLinData<S: LieScalar> {
    /// Gravity in the world frame.
    pub g: Vector3<S>,
    /// `1 / gyro_bias_std`, the square-root weight of the gyro random walk
    pub gyro_bias_weight_sqrt: Vector3<S>,
    /// `1 / accel_bias_std`, the same for the accelerometer.
    pub accel_bias_weight_sqrt: Vector3<S>,
}

/// An IMU factor with 15 whitened rows over two 15-column states.
/// Rows 0–8 are preintegration, 9–11 gyro-bias random walk, and 12–14
/// accelerometer-bias random walk. Start-state columns precede end-state columns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuBlock<S: LieScalar> {
    /// `Jp`.
    pub jp: Matrix15x30<S>,
    /// `r`.
    pub r: Vector15<S>,
    /// What `linearizeImu` returns: `imu_error + bg_error + ba_error`
    pub error: S,
}

impl<S: LieScalar> ImuBlock<S> {
    /// `ImuBlock::linearizeImu`.
    ///
    /// The residual is evaluated at the *linearized* states together with its
    /// Jacobians and then, if either state has a frozen linearization
    /// point, re-evaluated at the current states for its **value only**
    /// Skipping that second evaluation is trap 7 of the architecture
    /// dossier: it drifts rather than fails.
    pub fn linearize(
        meas: &IntegratedImuMeasurement<S>,
        lin_data: &ImuLinData<S>,
        start_state: &PoseVelBiasStateWithLin<S>,
        end_state: &PoseVelBiasStateWithLin<S>,
    ) -> Self {
        let start_idx: usize = 0;
        let end_idx: usize = POSE_VEL_BIAS_SIZE;

        let start_lin = start_state.state_lin();
        let end_lin = end_state.state_lin();
        let (mut res, jacobians): (Vector9<S>, ImuResidualJacobians<S>) = meas
            .residual_with_jacobians(
                &start_lin.pose_vel_state(),
                &lin_data.g,
                &end_lin.pose_vel_state(),
                &start_lin.bias_gyro,
                &start_lin.bias_accel,
            );

        if start_state.is_linearized() || end_state.is_linearized() {
            let start = start_state.state();
            let end = end_state.state();
            res = meas.residual(
                &start.pose_vel_state(),
                &lin_data.g,
                &end.pose_vel_state(),
                &start.bias_gyro,
                &start.bias_accel,
            );
        }

        let sqrt_cov_inv: Matrix9<S> = meas.get_cov_inv_sqrt();
        let mut jp: Matrix15x30<S> = Matrix15x30::zeros();
        let mut r: Vector15<S> = Vector15::zeros();

        let imu_error: S = c::<S>(0.5) * (sqrt_cov_inv * res).norm_squared(); // `:51`

        jp.fixed_view_mut::<9, 9>(0, start_idx)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_state0)); // `:54`
        jp.fixed_view_mut::<9, 9>(0, end_idx)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_state1)); // `:55`
        jp.fixed_view_mut::<9, 6>(0, start_idx + BIAS_GYRO_OFFSET)
            .copy_from(&(sqrt_cov_inv * jacobians.d_res_d_bias)); // `:57-58`
        r.fixed_rows_mut::<9>(0).copy_from(&(sqrt_cov_inv * res)); // `:60`

        let dt: S = c::<S>(meas.get_dt_ns() as f64) * c::<S>(1e-9); // `:63`
        let sqrt_dt: S = dt.sqrt();

        let gyro_bias_weight_dt: Vector3<S> = lin_data.gyro_bias_weight_sqrt / sqrt_dt;
        let res_bg: Vector3<S> = start_state.state().bias_gyro - end_state.state().bias_gyro;
        jp.fixed_view_mut::<3, 3>(9, start_idx + BIAS_GYRO_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&gyro_bias_weight_dt));
        jp.fixed_view_mut::<3, 3>(9, end_idx + BIAS_GYRO_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&-gyro_bias_weight_dt));
        let weighted_bg: Vector3<S> = gyro_bias_weight_dt.component_mul(&res_bg);
        r.fixed_rows_mut::<3>(9).copy_from(&weighted_bg);
        let bg_error: S = c::<S>(0.5) * weighted_bg.norm_squared();

        let accel_bias_weight_dt: Vector3<S> = lin_data.accel_bias_weight_sqrt / sqrt_dt;
        let res_ba: Vector3<S> = start_state.state().bias_accel - end_state.state().bias_accel;
        jp.fixed_view_mut::<3, 3>(12, start_idx + BIAS_ACCEL_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&accel_bias_weight_dt));
        jp.fixed_view_mut::<3, 3>(12, end_idx + BIAS_ACCEL_OFFSET)
            .copy_from(&Matrix3::from_diagonal(&-accel_bias_weight_dt));
        let weighted_ba: Vector3<S> = accel_bias_weight_dt.component_mul(&res_ba);
        r.fixed_rows_mut::<3>(12).copy_from(&weighted_ba);
        let ba_error: S = c::<S>(0.5) * weighted_ba.norm_squared();

        Self {
            jp,
            r,
            error: imu_error + bg_error + ba_error,
        }
    }

    /// Scatter `JᵀJ` and `Jᵀr` at the supplied start/end state offsets.
    /// Out-of-range offsets are ignored instead of panicking (D32).
    pub fn add_dense_h_b(
        &self,
        start_idx: usize,
        end_idx: usize,
        h: &mut DMatrix<S>,
        b: &mut DVector<S>,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        // The offsets come from an `AbsOrderMap` the caller owns, so the sum has
        // to be checked before it is compared: `usize::MAX + 15` wraps to a
        // small number in release and panics in debug (decision D32).
        let Some(needed) = start_idx.max(end_idx).checked_add(size) else {
            return;
        };
        if h.nrows() < needed || h.ncols() < needed || b.nrows() < needed {
            return;
        }
        let full_h: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, { 2 * POSE_VEL_BIAS_SIZE }> =
            self.jp.transpose() * self.jp;
        let full_b: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, 1> = self.jp.transpose() * self.r;

        for (block_row, row_offset) in [(0, start_idx), (size, end_idx)] {
            for (block_column, column_offset) in [(0, start_idx), (size, end_idx)] {
                for i in 0..size {
                    for j in 0..size {
                        h[(row_offset + i, column_offset + j)] +=
                            full_h[(block_row + i, block_column + j)];
                    }
                }
            }
            for i in 0..size {
                b[row_offset + i] += full_b[block_row + i];
            }
        }
    }

    /// Scatter the 15 whitened rows into the stacked square-root system,
    /// `add_dense_Q2Jp_Q2r`.
    ///
    /// `row_start_idx` is where this interval's rows begin; the driver advances
    /// it by `POSE_VEL_BIAS_SIZE` per interval
    /// Both column blocks are **added**,
    /// not assigned, which matters only if two intervals were ever
    /// given the same rows.
    ///
    /// Out-of-range offsets are ignored rather than panicking (decision D32),
    /// exactly as [`Self::add_dense_h_b`] does.
    pub fn add_dense_q2jp_q2r(
        &self,
        start_idx: usize,
        end_idx: usize,
        row_start_idx: usize,
        q2jp: &mut DMatrix<S>,
        q2r: &mut DVector<S>,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        let Some(col_end) = start_idx.max(end_idx).checked_add(size) else {
            return;
        };
        let Some(row_end) = row_start_idx.checked_add(size) else {
            return;
        };
        if q2jp.ncols() < col_end || q2jp.nrows() < row_end || q2r.nrows() < row_end {
            return;
        }
        for (block_col, col_offset) in [(0, start_idx), (size, end_idx)] {
            for i in 0..size {
                for j in 0..size {
                    q2jp[(row_start_idx + i, col_offset + j)] += self.jp[(i, block_col + j)];
                }
            }
        }
        for i in 0..size {
            q2r[row_start_idx + i] += self.r[i];
        }
    }

    /// This factor's share of the model cost change, `backSubstitute`
    ///
    /// There is nothing to back-substitute — the IMU block has no eliminated
    /// variables — so the whole method is the `l_diff` accumulation
    /// `l_diff -= (J inc)ᵀ (0.5 (J inc) + r)` over the two states' slices of the
    /// increment.
    pub fn back_substitute(
        &self,
        start_idx: usize,
        end_idx: usize,
        pose_inc: &DVector<S>,
        l_diff: &mut S,
    ) {
        let size: usize = POSE_VEL_BIAS_SIZE;
        let fits = |offset: usize| {
            offset
                .checked_add(size)
                .is_some_and(|end| end <= pose_inc.nrows())
        };
        if !fits(start_idx) || !fits(end_idx) {
            return;
        }
        // `pose_inc_reduced` : the start state's block, then the
        // end state's.
        let mut reduced: SMatrix<S, { 2 * POSE_VEL_BIAS_SIZE }, 1> = SMatrix::zeros();
        for i in 0..size {
            reduced[i] = pose_inc[start_idx + i];
            reduced[size + i] = pose_inc[end_idx + i];
        }
        let jinc: SMatrix<S, POSE_VEL_BIAS_SIZE, 1> = self.jp * reduced;
        let mut diff: S = S::zero();
        for i in 0..size {
            diff += jinc[i] * (S::from_literal(0.5) * jinc[i] + self.r[i]);
        }
        *l_diff -= diff;
    }
}
