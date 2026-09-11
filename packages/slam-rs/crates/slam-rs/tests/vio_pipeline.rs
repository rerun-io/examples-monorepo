//! Self-contained whole-pipeline, retry, and prior-comparison tests.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use slam_rs::config::VioConfig;
use slam_rs::estimator::FrameUpdateOutcome;
use slam_rs::frontend::flow::{FrontendError, FrontendOptions};
use slam_rs::lie::LieScalar;
use slam_rs::{Backend, ImageView, Vio, VioError, VioResult, VioStatus};

mod common;
use common::{IMU, Pgm};
use std::sync::LazyLock;

static TIMESTAMPS: LazyLock<Vec<i64>> = LazyLock::new(|| {
    include_str!("fixtures/flow/frames/timestamps.txt")
        .lines()
        .map(|line| line.parse().unwrap())
        .collect()
});

/// Framesets `tests/fixtures/flow/frames/` carries.
const COMMITTED_FRAMESETS: usize = 3;

/// Side of the square frame the wrong-size probe offers; the rig's is 960.
const CROPPED_SIDE: usize = 64;

/// The last of them, `TIMESTAMPS[2]`.
const LAST_COMMITTED_T_NS: i64 = 37_012_000;

/// How far the IMU is pushed for the committed run: one frameset interval past
/// [`LAST_COMMITTED_T_NS`].
///
/// It has to reach *past* the last frameset. closes the last
/// preintegration by re-stamping the first sample after the frameset, so a
/// queue that stops on the frameset leaves the interval short and `track`
/// reports `NeedMoreImu` for it.
const COMMITTED_IMU_HORIZON_NS: i64 = 40_000_000;

/// The pipeline both tests drive: the fixture's config and calibration, one
/// frontend thread so the reduction shape is fixed (D31).
fn pipeline<S: LieScalar>() -> Vio<S> {
    Vio::new(
        common::config(),
        common::calibration(),
        FrontendOptions {
            threads: 1,
            ..FrontendOptions::default()
        },
    )
    .unwrap()
}

/// A config field that counts cannot be negative, and the refusal is the same
/// on both backends because it happens before either one is looked at.
///
/// The GPU arm sizes its device buffers with `optical_flow_levels as usize + 1`
/// and `optical_flow_max_iterations as usize`, so it used to cast `-1` — a
/// debug-build panic, and in release a wrap to `usize::MAX` that asked the
/// device for a pyramid it cannot hold and reported *that* instead of the
/// frontend's own `NegativeConfig`. The CPU arm validated first. The check is
/// ahead of the branch now, so both arms refuse the same field with the same
/// value, and no device is constructed for a config no backend can run.
#[test]
fn a_negative_config_count_is_refused_on_both_lanes_before_any_cast() {
    let options = || FrontendOptions {
        threads: 1,
        ..FrontendOptions::default()
    };
    for field in ["optical_flow_levels", "optical_flow_max_iterations"] {
        for backend in [Backend::Cpu, Backend::Gpu] {
            let mut config: VioConfig = common::config();
            match field {
                "optical_flow_levels" => config.optical_flow_levels = -1,
                _ => config.optical_flow_max_iterations = -1,
            }
            let refused: VioError =
                Vio::<f32>::with_backend(config, common::calibration(), options(), backend)
                    .unwrap_err();
            assert_eq!(
                refused,
                VioError::Frontend(FrontendError::NegativeConfig { field, value: -1 }),
                "{backend:?} refused {field} = -1 as {refused}"
            );
        }
    }
}

/// A read PGM as the byte view `Vio::track` takes.
fn view(pgm: &Pgm) -> ImageView<'_> {
    ImageView {
        width: pgm.width,
        height: pgm.height,
        stride: pgm.width,
        data: &pgm.pixels,
    }
}

/// The three committed framesets, and the IMU that covers them.
///
/// The whole pipeline needs real pixels: `Vio::track` runs the KLT before the
/// estimator sees anything, and a constant frame detects no corners. Three is
/// what `tests/fixtures/flow/frames/` carries, so this cannot reach
/// `opt_started` (five states). The schedule test extends the input by holding the last image.
fn drive_the_committed_framesets<S: LieScalar>(vio: &mut Vio<S>) -> Vec<VioResult> {
    let directory: PathBuf = common::fixtures().join("flow/frames");
    let cameras: usize = common::calibration().t_i_c.len();
    let framesets: usize = common::available_framesets(&directory, cameras, COMMITTED_FRAMESETS);
    assert_eq!(framesets, COMMITTED_FRAMESETS, "{}", directory.display());

    for row in IMU
        .iter()
        .take_while(|row| row.t_ns <= COMMITTED_IMU_HORIZON_NS)
    {
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
    }
    TIMESTAMPS
        .iter()
        .take(framesets)
        .enumerate()
        .map(|(frame, flow)| {
            let rasters: Vec<Pgm> = (0..cameras)
                .map(|camera| common::read_pgm(&directory, frame, camera))
                .collect();
            let views: Vec<ImageView<'_>> = rasters.iter().map(view).collect();
            vio.track(*flow, &views).unwrap()
        })
        .collect()
}

/// The frontend and the estimator are wired together, and a repeat run is
/// bit-identical (D17).
///
/// In the default lane: three framesets of 960x960 KLT twice over is 0.31 s
/// with the optimized test profile (`Cargo.toml`), where it was 14 s of the
/// suite's 15 s budget unoptimized, which is why it used to be `#[ignore]`d.
#[test]
fn the_whole_pipeline_tracks_and_repeats_bit_identically() {
    check_pipeline::<f32>();
    check_pipeline::<f64>();
}

fn check_pipeline<S: LieScalar>() {
    let mut vio: Vio<S> = pipeline();
    let results: Vec<VioResult> = drive_the_committed_framesets(&mut vio);

    assert_eq!(
        results.iter().map(|r| r.status).collect::<Vec<VioStatus>>(),
        vec![VioStatus::Tracking; COMMITTED_FRAMESETS]
    );
    assert_eq!(
        results.iter().map(|r| r.t_ns).collect::<Vec<i64>>(),
        TIMESTAMPS[..COMMITTED_FRAMESETS].to_vec()
    );
    let estimator = vio.estimator();
    assert_eq!(estimator.snapshot().states.len(), COMMITTED_FRAMESETS);
    assert_eq!(estimator.last_state_t_ns(), LAST_COMMITTED_T_NS);
    // The first frameset is always a keyframe
    // and three framesets cannot reach `opt_started`.
    let Some(stats) = vio.last_stats() else {
        panic!("three framesets measured, so the last one left stats");
    };
    assert_eq!(stats.kf_ids, vec![0]);
    assert!(!stats.opt_started);
    assert!(stats.num_landmarks > 0);

    assert_eq!(
        drive_the_committed_framesets(&mut pipeline::<S>()),
        results,
        "a repeat run is not bit-identical"
    );
}

/// Every field of the pipeline as one number.
///
/// `Vio` derives `Debug`, so this reads all of them — the frontend's pyramids,
/// clock, counter, cells and keypoints, both IMU buffers and their popped
/// samples, the published state and depth guess, the whole estimator window.
/// 85 MB of text, so it is hashed rather than kept.
///
/// The fields that cannot be compared across runs are the three wall-clock
/// blocks, which their own docs call wall-clock: each is replaced by a fixed
/// string, and each replacement is asserted to have found exactly the one block
/// that this pipeline's state says is there — so a renamed, removed or repeated
/// block fails the test instead of quietly leaving a clock in the hash.
fn fingerprint(vio: &Vio<f32>) -> u64 {
    use std::hash::{Hash, Hasher};

    let mut text: String = format!("{vio:?}");
    // Every wall-clock block comes out: they are measurements of this run's own
    // speed and differ run to run by design, where the fingerprint is what must
    // not. `FlowTimings` is the frontend's three phases and `FrontendTimings`
    // the four `Vio` publishes, one of each; `StageTimings` is the estimator's
    // six on the last measured frame, so it is in the text exactly when a frame
    // has been measured and not at all on a pipeline that has refused every one.
    // Each holds integers only, so its first `}` closes it.
    let blocks: [(&str, usize); 3] = [
        ("FlowTimings {", 1),
        ("FrontendTimings {", 1),
        ("StageTimings {", usize::from(vio.last_stats().is_some())),
    ];
    for (marker, wanted) in blocks {
        let found: usize = text.matches(marker).count();
        assert_eq!(
            found, wanted,
            "`{marker}` is in the Debug output {found} times, not {wanted}: \
             the fingerprint either hashes a wall clock or no longer covers one"
        );
        if wanted == 0 {
            continue;
        }
        let start: usize = text.find(marker).unwrap();
        let length: usize = text[start..].find('}').unwrap_or_else(|| {
            panic!("`{marker}` is never closed in the Debug output, so its wall clock cannot be cut out")
        }) + 1;
        text.replace_range(start..start + length, "<wall clock>");
    }
    let mut hasher: std::collections::hash_map::DefaultHasher = Default::default();
    text.hash(&mut hasher);
    hasher.finish()
}

/// The fixture's IMU rows up to and including `horizon`, from `next`; returns
/// where it stopped.
fn push_imu_through(vio: &mut Vio<f32>, next: usize, horizon: i64) -> usize {
    let mut index: usize = next;
    while index < IMU.len() && IMU[index].t_ns <= horizon {
        let row = &IMU[index];
        vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
        index += 1;
    }
    index
}

/// A frameset that is not the size the calibration gives its cameras is refused
/// **before** anything moves, so the corrected frameset behind it takes the
/// trajectory it would have taken on its own.
///
/// The frontend checks the size itself and undoes its own passes, but by the
/// time it looks, `track` has already spent the frontend's preintegration on the
/// interval ( eats the buffer to seed the
/// KLT), and that cannot be spent again: the retry then predicts from a shorter
/// interval and the run parts from a clean one by ~6e-8 m within a few
/// framesets. The first frameset cannot show it — there is no state to predict
/// from yet — so the probe is made against all three, initialized or not.
#[test]
fn a_wrong_size_frameset_is_refused_without_moving_the_pipeline() {
    let directory: PathBuf = common::fixtures().join("flow/frames");
    let cameras: usize = common::calibration().t_i_c.len();
    let rasters: Vec<Vec<Pgm>> = (0..COMMITTED_FRAMESETS)
        .map(|frame| {
            (0..cameras)
                .map(|camera| common::read_pgm(&directory, frame, camera))
                .collect()
        })
        .collect();

    // Nothing is ever refused: the reference run.
    let mut clean: Vio<f32> = pipeline();
    push_imu_through(&mut clean, 0, COMMITTED_IMU_HORIZON_NS);
    let wanted: Vec<VioResult> = rasters
        .iter()
        .enumerate()
        .map(|(frame, raster)| {
            let views: Vec<ImageView<'_>> = raster.iter().map(view).collect();
            clean.track(TIMESTAMPS[frame], &views).unwrap()
        })
        .collect();

    // A cropped frameset is offered before each of them, and corrected.
    let cropped: Vec<u8> = vec![0; CROPPED_SIDE * CROPPED_SIDE];
    let wrong: Vec<ImageView<'_>> = (0..cameras)
        .map(|_| ImageView {
            width: CROPPED_SIDE,
            height: CROPPED_SIDE,
            stride: CROPPED_SIDE,
            data: &cropped,
        })
        .collect();
    let mut probed: Vio<f32> = pipeline();
    push_imu_through(&mut probed, 0, COMMITTED_IMU_HORIZON_NS);
    let mut got: Vec<VioResult> = Vec::new();
    for (frame, raster) in rasters.iter().enumerate() {
        let t_ns: i64 = TIMESTAMPS[frame];

        let before: u64 = fingerprint(&probed);
        let refused: VioError = probed.track(t_ns, &wrong).unwrap_err();
        assert!(
            matches!(
                refused,
                VioError::Frontend(FrontendError::FrameSizeMismatch { .. })
            ),
            "frame {frame}: the cropped frameset was refused as {refused}, not for its size"
        );
        assert_eq!(
            fingerprint(&probed),
            before,
            "frame {frame}: the refused frameset moved the pipeline"
        );

        let views: Vec<ImageView<'_>> = raster.iter().map(view).collect();
        got.push(probed.track(t_ns, &views).unwrap());
    }

    assert_eq!(got, wanted, "the corrected run took a different trajectory");
    assert_eq!(
        fingerprint(&probed),
        fingerprint(&clean),
        "the two runs agree on the poses but not on the rest of the pipeline"
    );
}

/// A frameset the IMU does not yet cover leaves the **whole** pipeline
/// untouched, so pushing the samples and tracking the same frameset again gives
/// what a run that had them all along gives (D17: no arrival order may reach
/// the trajectory).
///
/// This is the property `VioStatus::NeedMoreImu` promises and the one the
/// frontend threatens: `track` runs the KLT before the estimator sees anything,
/// and the KLT swaps the pyramids, advances `t_ns` and the frame counter and
/// eats the frontend's own IMU buffer. A coverage test placed after all that
/// would make the retry track the frameset against itself, and the second and
/// third framesets below — refused *after* initialization — would each lose a
/// preintegration interval as well.
#[test]
fn a_refused_frameset_is_retried_bit_identically() {
    let directory: PathBuf = common::fixtures().join("flow/frames");
    let cameras: usize = common::calibration().t_i_c.len();
    let rasters: Vec<Vec<Pgm>> = (0..COMMITTED_FRAMESETS)
        .map(|frame| {
            (0..cameras)
                .map(|camera| common::read_pgm(&directory, frame, camera))
                .collect()
        })
        .collect();

    // The IMU arrives first: the reference run.
    let mut ahead: Vio<f32> = pipeline();
    push_imu_through(&mut ahead, 0, COMMITTED_IMU_HORIZON_NS);
    let wanted: Vec<VioResult> = rasters
        .iter()
        .enumerate()
        .map(|(frame, raster)| {
            let views: Vec<ImageView<'_>> = raster.iter().map(view).collect();
            ahead.track(TIMESTAMPS[frame], &views).unwrap()
        })
        .collect();

    // Every frameset arrives before the samples that cover it, is refused, and
    // is tracked again once they land.
    let mut behind: Vio<f32> = pipeline();
    let mut next: usize = 0;
    let mut got: Vec<VioResult> = Vec::new();
    for (frame, raster) in rasters.iter().enumerate() {
        let t_ns: i64 = TIMESTAMPS[frame];
        let views: Vec<ImageView<'_>> = raster.iter().map(view).collect();

        let before: u64 = fingerprint(&behind);
        let refused: VioResult = behind.track(t_ns, &views).unwrap();
        assert_eq!(
            refused.status,
            VioStatus::NeedMoreImu,
            "frame {frame} was not refused, so the retry it is asked to model never happens"
        );
        assert_eq!(
            fingerprint(&behind),
            before,
            "frame {frame}: the refused frameset moved the pipeline"
        );

        // Up to the next frameset: past this one, so can close its
        // preintegration, and not past the next, so the next is refused too.
        // The last horizon is the reference run's, so both runs end holding the
        // same samples.
        let horizon: i64 = if frame + 1 < COMMITTED_FRAMESETS {
            TIMESTAMPS[frame + 1]
        } else {
            COMMITTED_IMU_HORIZON_NS
        };
        next = push_imu_through(&mut behind, next, horizon);
        got.push(behind.track(t_ns, &views).unwrap());
    }

    assert_eq!(got, wanted, "the retried run took a different trajectory");
    assert_eq!(
        fingerprint(&behind),
        fingerprint(&ahead),
        "the two runs agree on the poses but not on the rest of the pipeline"
    );
}

const PRIOR_NULL_RATIO: f64 = 1e-5;
const PRIOR_TOLERANCE_F64: f64 = 5e-6;
const PRIOR_TOLERANCE_F32: f64 = 3e-1;

fn prior_deviation(
    prior: &DMatrix<f64>,
    rhs: &DVector<f64>,
    reference: &DMatrix<f64>,
    reference_rhs: &DVector<f64>,
) -> (f64, f64) {
    let information: DMatrix<f64> = prior.transpose() * prior;
    let reference_information: DMatrix<f64> = reference.transpose() * reference;
    let term: DVector<f64> = prior.transpose() * rhs;
    let reference_term: DVector<f64> = reference.transpose() * reference_rhs;

    let eigen: SymmetricEigen<f64, nalgebra::Dyn> =
        SymmetricEigen::new(reference_information.clone());
    let floor: f64 =
        PRIOR_NULL_RATIO * PRIOR_NULL_RATIO * eigen.eigenvalues.iter().copied().fold(0.0, f64::max);
    let whitener: DMatrix<f64> = &eigen.eigenvectors
        * DMatrix::from_diagonal(&eigen.eigenvalues.map(|value| 1.0 / value.max(floor).sqrt()));

    let error: DMatrix<f64> =
        &(whitener.transpose() * (information - &reference_information)) * &whitener;
    let deviation: f64 = SymmetricEigen::new(error).eigenvalues.amax();

    let residual: DVector<f64> = whitener.transpose() * (term - &reference_term);
    let scale: f64 = (whitener.transpose() * reference_term).norm().max(1.0);
    (deviation, residual.norm() / scale)
}

fn reference_prior() -> (DMatrix<f64>, DVector<f64>) {
    let directions: [[f64; 6]; 4] = [
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
    ];
    let singular: [f64; 4] = [1.0e4, 1.0e2, 1.0e1, 3.0];
    let mut prior: DMatrix<f64> = DMatrix::zeros(5, 6);
    for (row, (direction, sigma)) in directions.iter().zip(singular).enumerate() {
        for (column, value) in direction.iter().enumerate() {
            prior[(row, column)] = value * sigma * std::f64::consts::FRAC_1_SQRT_2;
        }
    }
    (prior, DVector::from_vec(vec![1.0, -2.0, 0.5, 0.25, 0.0]))
}

#[test]
fn a_constraint_hidden_beside_a_zero_row_fails_the_prior_comparison() {
    let (reference, reference_rhs) = reference_prior();
    let mut candidate: DMatrix<f64> = reference.clone().insert_row(reference.nrows(), 0.0);
    candidate[(reference.nrows(), 4)] = 0.5;
    let rhs: DVector<f64> = reference_rhs.clone().insert_row(reference_rhs.nrows(), 0.0);

    // What the shape said about it: one surplus row, and a smallest row norm of
    // zero, from the row the reference already had.
    assert_eq!(candidate.nrows(), reference.nrows() + 1);
    let smallest: f64 = (0..candidate.nrows())
        .map(|row| candidate.row(row).norm())
        .fold(f64::INFINITY, f64::min);
    assert_eq!(smallest, 0.0);

    // The control first: the reference against itself is exactly equal.
    let (same, _) = prior_deviation(&reference, &reference_rhs, &reference, &reference_rhs);
    assert!(
        same < PRIOR_TOLERANCE_F64,
        "the reference moved: {same:.3e}"
    );

    let (deviation, _) = prior_deviation(&candidate, &rhs, &reference, &reference_rhs);
    assert!(
        deviation > PRIOR_TOLERANCE_F32,
        "a 0.5 constraint in a direction the reference leaves free scored {deviation:.3e}, \
         inside the loosest lane tolerance {PRIOR_TOLERANCE_F32:.1e}"
    );
}

#[test]
fn a_lost_direction_fails_the_prior_comparison_at_the_same_shape_and_norm() {
    let (reference, reference_rhs) = reference_prior();
    let mut candidate: DMatrix<f64> = reference.clone();
    candidate.row_mut(3).fill(0.0);
    candidate *= (reference.norm_squared() / candidate.norm_squared()).sqrt();

    // What the shape and the digest say about it: nothing.
    assert_eq!(candidate.shape(), reference.shape());
    assert!((candidate.norm() - reference.norm()).abs() <= 1e-9 * reference.norm());

    let (deviation, _) = prior_deviation(&candidate, &reference_rhs, &reference, &reference_rhs);
    assert!(
        deviation > PRIOR_TOLERANCE_F32,
        "dropping the weakest of four constrained directions scored {deviation:.3e}, inside the \
         loosest lane tolerance {PRIOR_TOLERANCE_F32:.1e}"
    );
}

#[test]
fn the_frame_update_lane_is_deterministic_and_takes_its_branch() {
    fn drive<S: LieScalar>() -> Vec<(VioResult, FrameUpdateOutcome, bool, usize)> {
        let mut config = common::config();
        config.port_frame_update_max_iterations = 5;
        let mut vio =
            Vio::<S>::new(config, common::calibration(), FrontendOptions::default()).unwrap();
        for row in IMU.iter() {
            vio.push_imu(row.t_ns, row.gyro, row.accel).unwrap();
        }
        let directory = common::fixtures().join("flow/frames");
        let mut output = Vec::new();
        let mut optimized = false;
        for (index, &time) in TIMESTAMPS.iter().enumerate() {
            // Only three raster framesets are shipped. Hold the final image for
            // the remaining timestamps to exercise scheduling beyond warmup.
            // This is a schedule probe, not an accuracy measurement.
            let rasters: Vec<_> = (0..2)
                .map(|camera| common::read_pgm(&directory, index.min(2), camera))
                .collect();
            let views: Vec<_> = rasters.iter().map(view).collect();
            let result = vio.track(time, &views).unwrap();
            assert_eq!(result.status, VioStatus::Tracking);
            let stats = vio.last_stats().unwrap();
            if !optimized || stats.took_keyframe {
                assert_eq!(stats.frame_update, FrameUpdateOutcome::NotAttempted);
            } else {
                assert_eq!(stats.frame_update, FrameUpdateOutcome::Taken);
                assert!(stats.lm.len() <= 6);
            }
            optimized = stats.opt_started;
            output.push((
                result,
                stats.frame_update,
                stats.took_keyframe,
                stats.lm.len(),
            ));
        }
        assert!(output.iter().any(|row| row.1 == FrameUpdateOutcome::Taken));
        output
    }
    assert_eq!(drive::<f32>(), drive::<f32>());
    assert_eq!(drive::<f64>(), drive::<f64>());
}
