//! The value types the estimator's state is built from.
//!
//! Ported from `include/basalt/utils/common_types.h`,
//! `thirdparty/basalt-headers/include/basalt/imu/imu_types.h` and
//! `include/basalt/utils/imu_types.h`. Nothing here optimizes: the point is
//! that every index, every increment order and the fixed-linearization
//! bookkeeping match the C++ exactly, because a sign or an offset wrong here
//! shows up as slow drift rather than an obvious failure.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, SVector, Vector3, Vector6};

use crate::lie::{LieScalar, Se3};

/// Degrees of freedom of a pose block (`imu_types.h:46`).
pub const POSE_SIZE: usize = 6;
/// Degrees of freedom of a pose-velocity block (`imu_types.h:47`).
pub const POSE_VEL_SIZE: usize = 9;
/// Degrees of freedom of a full state block (`imu_types.h:48`).
pub const POSE_VEL_BIAS_SIZE: usize = 15;

/// The 9-vector increment a pose-velocity state takes, and the width of the
/// preintegrated IMU residual.
pub type Vector9<S> = SVector<S, POSE_VEL_SIZE>;

/// The 15-vector increment a full state takes.
pub type Vector15<S> = SVector<S, POSE_VEL_BIAS_SIZE>;

/// Identifies a frameset. basalt uses the frameset timestamp in nanoseconds as
/// the id (`common_types.h:56`, `using FrameId = int64_t`).
pub type FrameId = i64;

/// Index of a camera on the rig (`common_types.h:59`, `using CamId = std::size_t`).
pub type CamId = usize;

/// One image: the frameset it belongs to and the camera that took it
/// (`common_types.h:62-69`).
///
/// The ordering is `frame_id` first, then `cam_id` (`common_types.h:76-79`), so
/// a sorted collection groups a frameset's images together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TimeCamId {
    /// Timestamp of the frameset, in nanoseconds.
    pub frame_id: FrameId,
    /// Camera index on the rig.
    pub cam_id: CamId,
}

impl TimeCamId {
    /// An image id from its two parts.
    pub fn new(frame_id: FrameId, cam_id: CamId) -> Self {
        Self { frame_id, cam_id }
    }
}

impl std::fmt::Display for TimeCamId {
    /// `frame_id _ cam_id`, as `operator<<` prints it (`common_types.h:71-74`).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.frame_id, self.cam_id)
    }
}

/// A tracked 2D feature (`optical_flow.h:67`, `using KeypointId = size_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct KeypointId(pub u64);

/// A landmark in the estimator's database.
///
/// basalt aliases the two (`optical_flow.h:71`, `using LandmarkId = KeypointId`)
/// because a landmark inherits the id of the keypoint that spawned it; the port
/// keeps them as separate newtypes so a keypoint index cannot be passed where a
/// landmark id belongs, with explicit conversions both ways.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LandmarkId(pub u64);

impl From<KeypointId> for LandmarkId {
    fn from(value: KeypointId) -> Self {
        Self(value.0)
    }
}

impl From<LandmarkId> for KeypointId {
    fn from(value: LandmarkId) -> Self {
        Self(value.0)
    }
}

/// Something a state type refuses to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StateError {
    /// A frame was given an offset twice, which would silently overlap two
    /// blocks in the Hessian.
    #[error("frame {frame_id} already has a block in the ordering")]
    DuplicateFrame {
        /// The frame that was inserted twice.
        frame_id: FrameId,
    },
    /// Freezing the linearization point with a non-zero accumulated increment
    /// would discard that increment.
    #[error("frame {frame_id} cannot be linearized while its delta is non-zero")]
    NonZeroDeltaAtLinearization {
        /// The frame that was being frozen.
        frame_id: FrameId,
    },
    /// A block would push the stacked state vector past the address space.
    #[error("frame {frame_id}: {size} rows do not fit after the existing {total_size}")]
    OrderingOverflow {
        /// The frame that was being added.
        frame_id: FrameId,
        /// Rows already in the ordering.
        total_size: usize,
        /// Rows the caller asked for.
        size: usize,
    },
}

/// Where each frame's block starts in the stacked state vector.
///
/// Ported from `include/basalt/utils/imu_types.h:293-304`. basalt assigns
/// offsets by accumulating `total_size` as it walks the keyframe poses and then
/// the full states (`sqrt_keypoint_vio.cpp:731-762`), so the offsets follow
/// insertion order; the C++ container is a `std::map`, i.e. key-ordered, and the
/// two coincide because keyframes always carry older timestamps than the
/// states. The port stores insertion order explicitly, which is what the offset
/// arithmetic actually depends on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AbsOrderMap {
    entries: Vec<(FrameId, usize, usize)>,
    index: HashMap<FrameId, usize>,
    total_size: usize,
}

impl AbsOrderMap {
    /// An empty ordering.
    pub fn new() -> Self {
        Self::default()
    }

    /// Give `frame_id` the next `size` rows and return the offset it got.
    ///
    /// Both checks run before either collection changes, so a rejected push
    /// leaves the ordering exactly as it was. The size comes from a caller, and
    /// an unchecked sum would panic in debug and wrap the total to a smaller
    /// number in release — a state vector that then overlaps its own blocks
    /// (decision D32).
    pub fn push(&mut self, frame_id: FrameId, size: usize) -> Result<usize, StateError> {
        if self.index.contains_key(&frame_id) {
            return Err(StateError::DuplicateFrame { frame_id });
        }
        let offset: usize = self.total_size;
        let total_size: usize = offset
            .checked_add(size)
            .ok_or(StateError::OrderingOverflow {
                frame_id,
                total_size: offset,
                size,
            })?;
        self.index.insert(frame_id, self.entries.len());
        self.entries.push((frame_id, offset, size));
        self.total_size = total_size;
        Ok(offset)
    }

    /// The `(offset, size)` of a frame's block, if it has one.
    pub fn get(&self, frame_id: FrameId) -> Option<(usize, usize)> {
        self.index
            .get(&frame_id)
            .and_then(|i| self.entries.get(*i))
            .map(|(_, offset, size)| (*offset, *size))
    }

    /// Whether a frame has a block.
    pub fn contains(&self, frame_id: FrameId) -> bool {
        self.index.contains_key(&frame_id)
    }

    /// Rows in the stacked state vector: `AbsOrderMap::total_size`.
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Number of blocks: `AbsOrderMap::items`.
    pub fn items(&self) -> usize {
        self.entries.len()
    }

    /// The blocks as `(frame_id, offset, size)`, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (FrameId, usize, usize)> + '_ {
        self.entries.iter().copied()
    }
}

/// The marginalization prior, `MargLinData<Scalar>`
/// (`include/basalt/utils/imu_types.h:317-326`).
///
/// The field named `h` is **not** a Hessian: it is the Jacobian `J_m` of
/// Paper 2 Eq. (4) and `b` is the residual `r_m`, which is the only form the QR
/// linearizer accepts (`linearization_abs_qr.cpp:578`). C++ carries a squared
/// form beside it behind `is_sqrt` (`ba_base.cpp:426-438`); the port does not,
/// because `SqrtKeypointVio::new` refuses `vio_sqrt_marg == false` (D68), so
/// that flag is judged in exactly one place.
///
/// `order` gives the prior's variables their offsets, and the linearizer
/// requires them to be a prefix of the window's ordering
/// (`ba_base.cpp:383-388`).
#[derive(Debug, Clone, PartialEq)]
pub struct MargLinData<S: LieScalar> {
    /// The prior's ordering (`:323`).
    pub order: AbsOrderMap,
    /// `J_m` (`:324`).
    pub h: DMatrix<S>,
    /// `r_m` (`:325`).
    pub b: DVector<S>,
}

impl<S: LieScalar> Default for MargLinData<S> {
    /// basalt's in-class initialiser: square root, empty (`:321-325`).
    fn default() -> Self {
        Self {
            order: AbsOrderMap::new(),
            h: DMatrix::zeros(0, 0),
            b: DVector::zeros(0),
        }
    }
}

/// An SE(3) pose at a timestamp (`imu_types.h:51-105`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseState<S: LieScalar> {
    /// Timestamp of the state, in nanoseconds.
    pub t_ns: i64,
    /// Pose of the IMU (rig) frame in the world frame.
    pub t_w_i: Se3<S>,
}

impl<S: LieScalar> Default for PoseState<S> {
    fn default() -> Self {
        Self {
            t_ns: 0,
            t_w_i: Se3::identity(),
        }
    }
}

impl<S: LieScalar> PoseState<S> {
    /// A state at a timestamp.
    pub fn new(t_ns: i64, t_w_i: Se3<S>) -> Self {
        Self { t_ns, t_w_i }
    }
}

/// An SE(3) pose and a world-frame linear velocity at a timestamp
/// (`imu_types.h:109-167`).
///
/// This is the state IMU preintegration propagates: the preintegrated
/// pseudo-measurement is itself a `PoseVelState` whose `t_ns` counts elapsed
/// nanoseconds rather than absolute time (`preintegration.h:148`, `:325`).
/// C++ derives it from `PoseState`; here the pose is a field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseVelState<S: LieScalar> {
    /// Timestamp of the state, in nanoseconds.
    pub t_ns: i64,
    /// Pose of the IMU (rig) frame in the world frame.
    pub t_w_i: Se3<S>,
    /// Linear velocity in the world frame, m/s.
    pub vel_w_i: Vector3<S>,
}

impl<S: LieScalar> Default for PoseVelState<S> {
    fn default() -> Self {
        Self {
            t_ns: 0,
            t_w_i: Se3::identity(),
            vel_w_i: Vector3::zeros(),
        }
    }
}

impl<S: LieScalar> PoseVelState<S> {
    /// A pose-velocity state from its parts (`imu_types.h:124-125`).
    pub fn new(t_ns: i64, t_w_i: Se3<S>, vel_w_i: Vector3<S>) -> Self {
        Self {
            t_ns,
            t_w_i,
            vel_w_i,
        }
    }

    /// Apply a 9-vector increment, `PoseVelState::applyInc` (`imu_types.h:140-143`).
    ///
    /// The layout is `[trans(3), rot(3), vel(3)]`; the pose goes through
    /// [`Se3::apply_inc`] and the velocity is added.
    pub fn apply_inc(&mut self, inc: &Vector9<S>) {
        self.t_w_i.apply_inc(&inc.fixed_rows::<6>(0).into_owned());
        self.vel_w_i += inc.fixed_rows::<3>(6);
    }

    /// The increment that takes `self` to `other`, `PoseVelState::diff`
    /// (`imu_types.h:156-162`), the inverse of [`PoseVelState::apply_inc`].
    ///
    /// No production caller: the estimator's states are 15-dof and use
    /// [`PoseVelBiasState::diff`]. This is the 9-dof one, and it is what the
    /// preintegration's finite-difference tests measure their Jacobians with —
    /// the residual they check is `delta_state.diff(propagated)`.
    pub fn diff(&self, other: &Self) -> Vector9<S> {
        let mut res: Vector9<S> = Vector9::zeros();
        res.fixed_rows_mut::<3>(0)
            .copy_from(&(other.t_w_i.translation - self.t_w_i.translation));
        res.fixed_rows_mut::<3>(3)
            .copy_from(&(other.t_w_i.rotation * self.t_w_i.rotation.inverse()).log());
        res.fixed_rows_mut::<3>(6)
            .copy_from(&(other.vel_w_i - self.vel_w_i));
        res
    }
}

/// Pose, velocity and the two IMU biases at a timestamp
/// (`imu_types.h:184-241`), the block the estimator actually optimizes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseVelBiasState<S: LieScalar> {
    /// Timestamp of the state, in nanoseconds.
    pub t_ns: i64,
    /// Pose of the IMU (rig) frame in the world frame.
    pub t_w_i: Se3<S>,
    /// Linear velocity in the world frame, m/s.
    pub vel_w_i: Vector3<S>,
    /// Gyroscope bias, rad/s.
    pub bias_gyro: Vector3<S>,
    /// Accelerometer bias, m/s^2.
    pub bias_accel: Vector3<S>,
}

impl<S: LieScalar> Default for PoseVelBiasState<S> {
    fn default() -> Self {
        Self {
            t_ns: 0,
            t_w_i: Se3::identity(),
            vel_w_i: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
            bias_accel: Vector3::zeros(),
        }
    }
}

impl<S: LieScalar> PoseVelBiasState<S> {
    /// A full state from its parts.
    pub fn new(
        t_ns: i64,
        t_w_i: Se3<S>,
        vel_w_i: Vector3<S>,
        bias_gyro: Vector3<S>,
        bias_accel: Vector3<S>,
    ) -> Self {
        Self {
            t_ns,
            t_w_i,
            vel_w_i,
            bias_gyro,
            bias_accel,
        }
    }

    /// The pose part on its own.
    pub fn pose_state(&self) -> PoseState<S> {
        PoseState::new(self.t_ns, self.t_w_i)
    }

    /// The pose and velocity part on its own.
    ///
    /// C++ gets this for free by inheritance — `IntegratedImuMeasurement::residual`
    /// takes a `const PoseVelState&` and a `PoseVelBiasState` slices into it
    /// (`imu_block.hpp:41-43`). The port hands over an explicit copy.
    pub fn pose_vel_state(&self) -> PoseVelState<S> {
        PoseVelState::new(self.t_ns, self.t_w_i, self.vel_w_i)
    }

    /// Apply a 15-vector increment, `PoseVelBiasState::applyInc`
    /// (`imu_types.h:212-216`).
    ///
    /// The layout is `[trans(3), rot(3), vel(3), bias_gyro(3), bias_accel(3)]`:
    /// the pose goes through [`Se3::apply_inc`] and everything else is plain
    /// addition. The gyro bias comes **before** the accel bias, matching the
    /// prior weights the estimator installs at indices 9-11 and 12-14
    /// (`sqrt_keypoint_vio.cpp:92-93`).
    pub fn apply_inc(&mut self, inc: &Vector15<S>) {
        self.t_w_i.apply_inc(&inc.fixed_rows::<6>(0).into_owned());
        self.vel_w_i += inc.fixed_rows::<3>(6);
        self.bias_gyro += inc.fixed_rows::<3>(9);
        self.bias_accel += inc.fixed_rows::<3>(12);
    }

    /// The increment that takes `self` to `other`, `PoseVelBiasState::diff`
    /// (`imu_types.h:229-236`).
    ///
    /// The inverse of [`PoseVelBiasState::apply_inc`], so
    /// `self.diff(&other)` applied to `self` reproduces `other`.
    pub fn diff(&self, other: &Self) -> Vector15<S> {
        let mut res: Vector15<S> = Vector15::zeros();
        res.fixed_rows_mut::<3>(0)
            .copy_from(&(other.t_w_i.translation - self.t_w_i.translation));
        res.fixed_rows_mut::<3>(3)
            .copy_from(&(other.t_w_i.rotation * self.t_w_i.rotation.inverse()).log());
        res.fixed_rows_mut::<3>(6)
            .copy_from(&(other.vel_w_i - self.vel_w_i));
        res.fixed_rows_mut::<3>(9)
            .copy_from(&(other.bias_gyro - self.bias_gyro));
        res.fixed_rows_mut::<3>(12)
            .copy_from(&(other.bias_accel - self.bias_accel));
        res
    }
}

/// A pose block that can hold its linearization point fixed
/// (`include/basalt/utils/imu_types.h:180-291`).
///
/// Once [`PoseStateWithLin::set_linearized`] is called the Jacobians are frozen
/// at `pose_linearized`, and every later increment accumulates into `delta` and
/// is re-applied to that frozen pose rather than to the current one
/// (`imu_types.h:240-248`). Skipping the accumulation still moves the estimate,
/// which is why the failure looks like drift instead of a crash (trap 7 of the
/// architecture dossier).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseStateWithLin<S: LieScalar> {
    linearized: bool,
    delta: Vector6<S>,
    pose_linearized: PoseState<S>,
    t_w_i_current: Se3<S>,
    backup_delta: Vector6<S>,
    backup_pose_linearized: PoseState<S>,
    backup_t_w_i_current: Se3<S>,
}

impl<S: LieScalar> Default for PoseStateWithLin<S> {
    fn default() -> Self {
        Self {
            linearized: false,
            delta: Vector6::zeros(),
            pose_linearized: PoseState::default(),
            t_w_i_current: Se3::identity(),
            backup_delta: Vector6::zeros(),
            backup_pose_linearized: PoseState::default(),
            backup_t_w_i_current: Se3::identity(),
        }
    }
}

impl<S: LieScalar> PoseStateWithLin<S> {
    /// A pose block at a timestamp (`imu_types.h:192-197`).
    pub fn new(t_ns: i64, t_w_i: Se3<S>, linearized: bool) -> Self {
        Self {
            linearized,
            delta: Vector6::zeros(),
            pose_linearized: PoseState::new(t_ns, t_w_i),
            t_w_i_current: t_w_i,
            backup_delta: Vector6::zeros(),
            backup_pose_linearized: PoseState::new(t_ns, t_w_i),
            backup_t_w_i_current: t_w_i,
        }
    }

    /// The pose block a full state collapses to when it leaves the window
    /// (`imu_types.h:206-215`): the first six entries of the state's delta
    /// carry over, and the current pose is rebuilt from the frozen one.
    pub fn from_pose_vel_bias(other: &PoseVelBiasStateWithLin<S>) -> Self {
        let delta: Vector6<S> = other.delta().fixed_rows::<6>(0).into_owned();
        let pose_linearized: PoseState<S> = other.state_lin().pose_state();
        let mut t_w_i_current: Se3<S> = pose_linearized.t_w_i;
        t_w_i_current.apply_inc(&delta);
        Self {
            linearized: other.is_linearized(),
            delta,
            pose_linearized,
            t_w_i_current,
            // `backup_delta.setZero()` with the comment "unused, but avoids
            // uninitialized gcc warning" (`imu_types.h:211`); the two poses get
            // the same treatment here because Rust has no uninitialized field.
            backup_delta: Vector6::zeros(),
            backup_pose_linearized: pose_linearized,
            backup_t_w_i_current: t_w_i_current,
        }
    }

    /// Freeze the linearization point, `setLinTrue` (`imu_types.h:224-228`).
    ///
    /// basalt asserts the delta is zero here; the port returns an error instead
    /// (decision D32, the core never panics on data), because freezing on top of
    /// an accumulated increment would silently discard it.
    pub fn set_linearized(&mut self) -> Result<(), StateError> {
        if self.delta != Vector6::zeros() {
            return Err(StateError::NonZeroDeltaAtLinearization {
                frame_id: self.pose_linearized.t_ns,
            });
        }
        self.linearized = true;
        self.t_w_i_current = self.pose_linearized.t_w_i;
        Ok(())
    }

    /// Apply an increment, `applyInc` (`imu_types.h:240-248`).
    pub fn apply_inc(&mut self, inc: &Vector6<S>) {
        if self.linearized {
            self.delta += inc;
            self.t_w_i_current = self.pose_linearized.t_w_i;
            self.t_w_i_current.apply_inc(&self.delta);
        } else {
            self.pose_linearized.t_w_i.apply_inc(inc);
        }
    }

    /// Save the mutable state, `backup` (`imu_types.h:254-258`).
    ///
    /// The `linearized` flag is deliberately **not** saved: C++ does not save it
    /// either, because a rejected Levenberg-Marquardt step never changes it.
    pub fn backup(&mut self) {
        self.backup_delta = self.delta;
        self.backup_pose_linearized = self.pose_linearized;
        self.backup_t_w_i_current = self.t_w_i_current;
    }

    /// Undo the last increments, `restore` (`imu_types.h:260-264`).
    pub fn restore(&mut self) {
        self.delta = self.backup_delta;
        self.pose_linearized = self.backup_pose_linearized;
        self.t_w_i_current = self.backup_t_w_i_current;
    }

    /// The pose the residuals are evaluated at, `getPose` (`imu_types.h:250-256`).
    pub fn pose(&self) -> &Se3<S> {
        if self.linearized {
            &self.t_w_i_current
        } else {
            &self.pose_linearized.t_w_i
        }
    }

    /// The pose the Jacobians are evaluated at, `getPoseLin` (`imu_types.h:258`).
    pub fn pose_lin(&self) -> &Se3<S> {
        &self.pose_linearized.t_w_i
    }

    /// Whether the linearization point is frozen.
    pub fn is_linearized(&self) -> bool {
        self.linearized
    }

    /// The increment accumulated since freezing.
    pub fn delta(&self) -> &Vector6<S> {
        &self.delta
    }

    /// Timestamp of the block.
    pub fn t_ns(&self) -> i64 {
        self.pose_linearized.t_ns
    }
}

/// A full state block that can hold its linearization point fixed
/// (`include/basalt/utils/imu_types.h:68-178`).
///
/// The same fixed-linearization rule as [`PoseStateWithLin`], on the 15-vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseVelBiasStateWithLin<S: LieScalar> {
    linearized: bool,
    delta: Vector15<S>,
    state_linearized: PoseVelBiasState<S>,
    state_current: PoseVelBiasState<S>,
    backup_delta: Vector15<S>,
    backup_state_linearized: PoseVelBiasState<S>,
    backup_state_current: PoseVelBiasState<S>,
}

impl<S: LieScalar> Default for PoseVelBiasStateWithLin<S> {
    fn default() -> Self {
        Self {
            linearized: false,
            delta: Vector15::zeros(),
            state_linearized: PoseVelBiasState::default(),
            state_current: PoseVelBiasState::default(),
            backup_delta: Vector15::zeros(),
            backup_state_linearized: PoseVelBiasState::default(),
            backup_state_current: PoseVelBiasState::default(),
        }
    }
}

impl<S: LieScalar> PoseVelBiasStateWithLin<S> {
    /// A state block from a plain state (`imu_types.h:88-91`).
    pub fn new(state: PoseVelBiasState<S>, linearized: bool) -> Self {
        Self {
            linearized,
            delta: Vector15::zeros(),
            state_linearized: state,
            state_current: state,
            backup_delta: Vector15::zeros(),
            backup_state_linearized: state,
            backup_state_current: state,
        }
    }

    /// Freeze the linearization point, `setLinTrue` (`imu_types.h:110-114`).
    ///
    /// See [`PoseStateWithLin::set_linearized`] for why this returns an error
    /// where basalt asserts.
    pub fn set_linearized(&mut self) -> Result<(), StateError> {
        if self.delta != Vector15::zeros() {
            return Err(StateError::NonZeroDeltaAtLinearization {
                frame_id: self.state_linearized.t_ns,
            });
        }
        self.linearized = true;
        self.state_current = self.state_linearized;
        Ok(())
    }

    /// Apply a 15-vector increment, `applyInc` (`imu_types.h:116-124`).
    ///
    /// Once linearized the increment accumulates into `delta` and the current
    /// state is recomputed from the frozen one — not from the previous current
    /// state, which would compound the increments.
    pub fn apply_inc(&mut self, inc: &Vector15<S>) {
        if self.linearized {
            self.delta += inc;
            self.state_current = self.state_linearized;
            self.state_current.apply_inc(&self.delta);
        } else {
            self.state_linearized.apply_inc(inc);
        }
    }

    /// Save the mutable state, `backup` (`imu_types.h:139-143`).
    ///
    /// As for [`PoseStateWithLin::backup`], the `linearized` flag is not saved.
    pub fn backup(&mut self) {
        self.backup_delta = self.delta;
        self.backup_state_linearized = self.state_linearized;
        self.backup_state_current = self.state_current;
    }

    /// Undo the last increments, `restore` (`imu_types.h:145-149`).
    pub fn restore(&mut self) {
        self.delta = self.backup_delta;
        self.state_linearized = self.backup_state_linearized;
        self.state_current = self.backup_state_current;
    }

    /// The state the residuals are evaluated at, `getState` (`imu_types.h:126-132`).
    pub fn state(&self) -> &PoseVelBiasState<S> {
        if self.linearized {
            &self.state_current
        } else {
            &self.state_linearized
        }
    }

    /// The state the Jacobians are evaluated at, `getStateLin` (`imu_types.h:134`).
    pub fn state_lin(&self) -> &PoseVelBiasState<S> {
        &self.state_linearized
    }

    /// Whether the linearization point is frozen.
    pub fn is_linearized(&self) -> bool {
        self.linearized
    }

    /// The increment accumulated since freezing.
    pub fn delta(&self) -> &Vector15<S> {
        &self.delta
    }

    /// Timestamp of the block.
    pub fn t_ns(&self) -> i64 {
        self.state_linearized.t_ns
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use crate::lie::So3;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    const CASES: u32 = 256;

    fn config() -> ProptestConfig {
        ProptestConfig::with_cases(CASES)
    }

    fn a_state() -> PoseVelBiasState<f64> {
        PoseVelBiasState::new(
            1_000,
            Se3::new(
                So3::exp(&Vector3::new(0.1, -0.2, 0.3)),
                Vector3::new(1.0, 2.0, 3.0),
            ),
            Vector3::new(0.5, -0.5, 0.25),
            Vector3::new(1e-3, 2e-3, -1e-3),
            Vector3::new(1e-2, -2e-2, 3e-2),
        )
    }

    fn an_inc() -> Vector15<f64> {
        Vector15::from_iterator((0..15).map(|i| 0.01 * f64::from(i + 1)))
    }

    #[test]
    fn time_cam_ids_sort_by_frame_then_camera() {
        let mut ids: Vec<TimeCamId> = vec![
            TimeCamId::new(2, 0),
            TimeCamId::new(1, 3),
            TimeCamId::new(1, 0),
        ];
        ids.sort();
        assert_eq!(
            ids,
            [
                TimeCamId::new(1, 0),
                TimeCamId::new(1, 3),
                TimeCamId::new(2, 0)
            ]
        );
        assert_eq!(TimeCamId::new(17, 2).to_string(), "17_2");
    }

    #[test]
    fn keypoint_and_landmark_ids_convert_both_ways() {
        let keypoint: KeypointId = KeypointId(7);
        let landmark: LandmarkId = keypoint.into();
        assert_eq!(landmark, LandmarkId(7));
        assert_eq!(KeypointId::from(landmark), keypoint);
    }

    /// The keyframe poses go in first and the full states after, exactly as
    /// `sqrt_keypoint_vio.cpp:731-762` builds it.
    #[test]
    fn the_ordering_lays_blocks_out_end_to_end() {
        let mut order: AbsOrderMap = AbsOrderMap::new();
        assert_eq!(order.push(100, POSE_SIZE).unwrap(), 0);
        assert_eq!(order.push(200, POSE_SIZE).unwrap(), 6);
        assert_eq!(order.push(300, POSE_VEL_BIAS_SIZE).unwrap(), 12);
        assert_eq!(order.total_size(), 27);
        assert_eq!(order.items(), 3);
        assert_eq!(order.get(200), Some((6, POSE_SIZE)));
        assert_eq!(order.get(999), None);
        assert!(order.contains(300));
        assert_eq!(
            order.iter().collect::<Vec<_>>(),
            [(100, 0, 6), (200, 6, 6), (300, 12, 15)]
        );
    }

    /// A size that would overflow the total is refused, and refusing it leaves
    /// the ordering untouched rather than half-updated.
    #[test]
    fn the_ordering_rejects_a_size_that_overflows() {
        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(0, usize::MAX).unwrap();
        assert_eq!(order.total_size(), usize::MAX);

        assert_eq!(
            order.push(1, 1),
            Err(StateError::OrderingOverflow {
                frame_id: 1,
                total_size: usize::MAX,
                size: 1
            })
        );
        assert_eq!(order.total_size(), usize::MAX);
        assert_eq!(order.items(), 1);
        assert!(!order.contains(1));
        assert_eq!(order.get(1), None);
    }

    #[test]
    fn the_ordering_rejects_a_frame_twice() {
        let mut order: AbsOrderMap = AbsOrderMap::new();
        order.push(100, POSE_SIZE).unwrap();
        assert_eq!(
            order.push(100, POSE_VEL_BIAS_SIZE),
            Err(StateError::DuplicateFrame { frame_id: 100 })
        );
        assert_eq!(order.total_size(), POSE_SIZE);
    }

    /// The 15-vector layout: entries 9-11 move the gyro bias and 12-14 the
    /// accel bias, never the other way round.
    #[test]
    fn apply_inc_moves_the_right_slots() {
        let mut state: PoseVelBiasState<f64> = a_state();
        let before: PoseVelBiasState<f64> = state;
        let mut inc: Vector15<f64> = Vector15::zeros();
        inc[9] = 1.0;
        state.apply_inc(&inc);
        assert_abs_diff_eq!(
            state.bias_gyro,
            before.bias_gyro + Vector3::new(1.0, 0.0, 0.0),
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(state.bias_accel, before.bias_accel, epsilon = 1e-15);

        let mut state: PoseVelBiasState<f64> = before;
        let mut inc: Vector15<f64> = Vector15::zeros();
        inc[12] = 1.0;
        state.apply_inc(&inc);
        assert_abs_diff_eq!(state.bias_gyro, before.bias_gyro, epsilon = 1e-15);
        assert_abs_diff_eq!(
            state.bias_accel,
            before.bias_accel + Vector3::new(1.0, 0.0, 0.0),
            epsilon = 1e-15
        );
    }

    /// Before `set_linearized`, the block behaves like a plain state: there is
    /// no delta and the linearization point moves with it.
    #[test]
    fn an_unlinearized_block_moves_its_linearization_point() {
        let mut block: PoseVelBiasStateWithLin<f64> =
            PoseVelBiasStateWithLin::new(a_state(), false);
        block.apply_inc(&an_inc());
        assert_eq!(block.delta(), &Vector15::zeros());
        assert_eq!(block.state(), block.state_lin());

        let mut expected: PoseVelBiasState<f64> = a_state();
        expected.apply_inc(&an_inc());
        assert_abs_diff_eq!(
            block.state().t_w_i.translation,
            expected.t_w_i.translation,
            epsilon = 1e-15
        );
    }

    /// After `set_linearized`, the linearization point stops moving, the delta
    /// accumulates, and the current state is rebuilt from the frozen point.
    #[test]
    fn a_linearized_block_separates_current_from_linearized() {
        let start: PoseVelBiasState<f64> = a_state();
        let mut block: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(start, false);
        block.set_linearized().unwrap();

        // Two increments about different axes, so `exp(a) exp(b)` and
        // `exp(a + b)` genuinely disagree.
        let first: Vector15<f64> = an_inc();
        let mut second: Vector15<f64> = an_inc();
        second[3] = -0.3;
        second[4] = 0.25;
        second[5] = 0.4;
        block.apply_inc(&first);
        block.apply_inc(&second);

        assert_abs_diff_eq!(*block.delta(), first + second, epsilon = 1e-15);
        assert_eq!(block.state_lin(), &start);

        // The current state is `start` plus the accumulated delta in one step,
        // which is not the same as applying the two increments in a row.
        let mut once: PoseVelBiasState<f64> = start;
        once.apply_inc(&(first + second));
        assert_abs_diff_eq!(
            block.state().t_w_i.translation,
            once.t_w_i.translation,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            block.state().t_w_i.rotation.matrix(),
            once.t_w_i.rotation.matrix(),
            epsilon = 1e-14
        );

        let mut twice: PoseVelBiasState<f64> = start;
        twice.apply_inc(&first);
        twice.apply_inc(&second);
        // Translation and biases are linear, so only the rotation separates the
        // two; that is exactly the drift a missing re-application produces.
        assert!(
            (block.state().t_w_i.rotation.matrix() - twice.t_w_i.rotation.matrix()).norm() > 1e-9
        );
    }

    #[test]
    fn freezing_a_block_with_a_delta_is_refused() {
        let mut block: PoseVelBiasStateWithLin<f64> =
            PoseVelBiasStateWithLin::new(a_state(), false);
        block.set_linearized().unwrap();

        let mut block: PoseVelBiasStateWithLin<f64> = PoseVelBiasStateWithLin::new(a_state(), true);
        block.apply_inc(&an_inc());
        assert_eq!(
            block.set_linearized(),
            Err(StateError::NonZeroDeltaAtLinearization { frame_id: 1_000 })
        );
    }

    #[test]
    fn a_pose_block_keeps_the_same_separation() {
        let pose: Se3<f64> = a_state().t_w_i;
        let mut block: PoseStateWithLin<f64> = PoseStateWithLin::new(1_000, pose, false);
        block.set_linearized().unwrap();
        let inc: Vector6<f64> = Vector6::new(0.1, 0.2, 0.3, 0.01, 0.02, 0.03);
        block.apply_inc(&inc);
        block.apply_inc(&inc);

        assert_abs_diff_eq!(*block.delta(), inc * 2.0, epsilon = 1e-15);
        assert_abs_diff_eq!(
            block.pose_lin().translation,
            pose.translation,
            epsilon = 1e-15
        );
        let mut once: Se3<f64> = pose;
        once.apply_inc(&(inc * 2.0));
        assert_abs_diff_eq!(block.pose().translation, once.translation, epsilon = 1e-15);
    }

    /// A full state that leaves the window becomes a pose block carrying the
    /// first six entries of its delta (`imu_types.h:206-215`).
    #[test]
    fn a_state_block_collapses_into_a_pose_block() {
        let mut state_block: PoseVelBiasStateWithLin<f64> =
            PoseVelBiasStateWithLin::new(a_state(), false);
        state_block.set_linearized().unwrap();
        state_block.apply_inc(&an_inc());

        let pose_block: PoseStateWithLin<f64> = PoseStateWithLin::from_pose_vel_bias(&state_block);
        assert!(pose_block.is_linearized());
        assert_abs_diff_eq!(
            *pose_block.delta(),
            an_inc().fixed_rows::<6>(0).into_owned(),
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(
            pose_block.pose().translation,
            state_block.state().t_w_i.translation,
            epsilon = 1e-15
        );
    }

    proptest! {
        #![proptest_config(config())]

        /// `p0.diff(p1) == inc` whenever `p1 = p0.apply_inc(inc)`, the identity
        /// the C++ doc comment states (`imu_types.h:218-228`).
        #[test]
        fn apply_inc_and_diff_round_trip(seed in prop::array::uniform15(-0.4f64..0.4)) {
            let inc: Vector15<f64> = Vector15::from_column_slice(&seed);
            let start: PoseVelBiasState<f64> = a_state();
            let mut moved: PoseVelBiasState<f64> = start;
            moved.apply_inc(&inc);
            let recovered: Vector15<f64> = start.diff(&moved);
            prop_assert!((recovered - inc).norm() < 1e-12);
        }

        /// A linearized block's current state is always the frozen state plus
        /// the whole accumulated delta, however the increments were split.
        #[test]
        fn a_linearized_block_is_its_frozen_state_plus_delta(
            first in prop::array::uniform15(-0.2f64..0.2),
            second in prop::array::uniform15(-0.2f64..0.2),
        ) {
            let start: PoseVelBiasState<f64> = a_state();
            let mut block: PoseVelBiasStateWithLin<f64> =
                PoseVelBiasStateWithLin::new(start, false);
            block.set_linearized().unwrap();
            block.apply_inc(&Vector15::from_column_slice(&first));
            block.apply_inc(&Vector15::from_column_slice(&second));

            let mut expected: PoseVelBiasState<f64> = start;
            expected.apply_inc(block.delta());
            prop_assert_eq!(block.state_lin(), &start);
            prop_assert!(
                (block.state().t_w_i.translation - expected.t_w_i.translation).norm() < 1e-12
            );
            prop_assert!(
                (block.state().t_w_i.rotation.matrix() - expected.t_w_i.rotation.matrix()).norm()
                    < 1e-12
            );
            prop_assert!((block.state().bias_gyro - expected.bias_gyro).norm() < 1e-12);
        }

        /// Offsets tile the state vector with no gap and no overlap.
        #[test]
        fn the_ordering_tiles_the_state_vector(sizes in prop::collection::vec(1usize..16, 1..12)) {
            let mut order: AbsOrderMap = AbsOrderMap::new();
            let mut expected_offset: usize = 0;
            for (i, size) in sizes.iter().enumerate() {
                let offset: usize = order.push(i as FrameId, *size)?;
                prop_assert_eq!(offset, expected_offset);
                expected_offset += size;
            }
            prop_assert_eq!(order.total_size(), sizes.iter().sum::<usize>());
            prop_assert_eq!(order.items(), sizes.len());
        }
    }
}
