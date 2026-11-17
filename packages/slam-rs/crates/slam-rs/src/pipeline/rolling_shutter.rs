//! The rolling-shutter correction of the estimator's keypoints.
//!
//! Row `v` of an `H`-row image is read at `t + readout_s * (v - (H - 1) / 2) / H`
//! relative to the frame time `t` (the middle row's); image rows are sensor rows
//! and `H` is the calibrated image height, so a row's read time is its fraction
//! of the image and stays the same when a camera tracks at a reduced resolution.
//! The frontend keeps tracking the raw positions. The estimator's copy of each
//! keypoint moves to where a global shutter at the frame time would have seen
//! it, with the camera's rotation over that interval taken from the gyro.
//! Translation during the readout is ignored.
//!
//! A rig whose cameras state no readout, or a zero one, has no correction: the
//! estimator sees exactly the frontend's positions.

use crate::calib::Calibration;
use crate::camera::{CameraError, SlamCamera};
use crate::estimator::FlowObservations;
use crate::lie::So3;
use kornia_staging_slam::init::inertial::{ImuSample, sample_at};
use nalgebra::Vector3;

/// One rolling-shutter camera of the rig.
#[derive(Debug)]
struct ShutterCamera {
    model: SlamCamera<f64>,
    /// Image rows.
    height: f64,
    readout_s: f64,
    /// `R_cam_imu`.
    cam_from_imu: So3<f64>,
}

/// The correction of every camera with a non-zero readout: the rate of a frame
/// is the calibrated gyro interpolated at the frame time, less the estimator's
/// bias. The gyro buffer is trimmed by the frame times consumed, so how the IMU
/// is batched does not change a correction.
#[derive(Debug)]
pub struct RollingShutter {
    /// Per camera; `None` for a global-shutter camera, whose keypoints stay raw.
    cameras: Vec<Option<ShutterCamera>>,
    /// Calibrated gyro samples (IMU frame), oldest first, from the last one at
    /// or before the last corrected frame.
    gyro: Vec<ImuSample>,
    corrected: usize,
    no_coverage: usize,
}

impl RollingShutter {
    /// The correction of `calibration`'s rolling-shutter cameras; `None` when no
    /// camera states a non-zero readout.
    ///
    /// # Errors
    ///
    /// A camera model the staged cameras cannot build.
    pub fn for_calibration(calibration: &Calibration<f64>) -> Result<Option<Self>, CameraError> {
        let mut cameras: Vec<Option<ShutterCamera>> = Vec::new();
        for (index, model) in calibration.intrinsics.iter().enumerate() {
            let readout_s: f64 = calibration
                .readout_s
                .get(index)
                .copied()
                .flatten()
                .unwrap_or(0.0);
            cameras.push(if readout_s == 0.0 {
                None
            } else {
                Some(ShutterCamera {
                    model: SlamCamera::from_model(model)?,
                    height: f64::from(calibration.resolution[index][1]),
                    readout_s,
                    cam_from_imu: calibration.t_i_c[index].rotation.inverse(),
                })
            });
        }
        Ok(cameras.iter().any(Option::is_some).then_some(Self {
            cameras,
            gyro: Vec::new(),
            corrected: 0,
            no_coverage: 0,
        }))
    }

    /// Keep one calibrated gyro sample (IMU frame).
    pub fn push_gyro(&mut self, t_ns: i64, gyro: Vector3<f64>) {
        self.gyro.push(ImuSample {
            t_ns,
            gyro,
            accel: Vector3::zeros(),
        });
    }

    /// Frames corrected so far, and frames left raw because the gyro did not cover their time.
    pub fn counts(&self) -> (usize, usize) {
        (self.corrected, self.no_coverage)
    }

    /// Remove the rolling shutter from one frameset's keypoints. `gyro_bias` is
    /// the estimator's latest bias (IMU frame), subtracted when known. A frame
    /// the gyro does not cover on both sides stays raw and is counted
    /// ([`Self::counts`]); a keypoint whose bearing leaves its camera keeps its
    /// raw position.
    pub fn correct(
        &mut self,
        observations: &mut FlowObservations,
        gyro_bias: Option<Vector3<f64>>,
    ) {
        let t_ns: i64 = observations.t_ns;
        let before: usize = self.gyro.partition_point(|sample| sample.t_ns <= t_ns);
        if before == 0 || self.gyro.last().is_none_or(|sample| sample.t_ns < t_ns) {
            self.no_coverage += 1;
            return;
        }
        let (rate, _) = sample_at(&self.gyro, t_ns);
        // Frames arrive in time order: what precedes this frame's left neighbour is never read again.
        self.gyro.drain(..before - 1);
        self.corrected += 1;
        let rate: Vector3<f64> = rate - gyro_bias.unwrap_or_else(Vector3::zeros);
        for (camera, keypoints) in self.cameras.iter().zip(observations.cameras.iter_mut()) {
            let Some(camera) = camera else { continue };
            let omega: Vector3<f64> = camera.cam_from_imu * rate;
            for pixel in keypoints.values_mut() {
                if let Some([u, v]) =
                    camera.correct([f64::from(pixel.x), f64::from(pixel.y)], &omega)
                {
                    *pixel = nalgebra::Vector2::new(u as f32, v as f32);
                }
            }
        }
    }
}

impl ShutterCamera {
    /// One keypoint moved to the frame time.
    ///
    /// Its row gives `dt = readout_s * (row - (H - 1) / 2) / H`. The camera
    /// turned by `exp(omega * dt)` (`omega` in this camera's frame, rad/s)
    /// between the frame time and that row, so the frame-time bearing of a
    /// static point is `exp(omega * dt) * b_observed`, projected back with the
    /// camera. `None` when the pixel or the turned bearing leaves the camera.
    fn correct(&self, pixel: [f64; 2], omega: &Vector3<f64>) -> Option<[f64; 2]> {
        let dt: f64 = self.readout_s * (pixel[1] - 0.5 * (self.height - 1.0)) / self.height;
        let bearing: Vector3<f64> = Vector3::from(self.model.inner.unproject(pixel).ok()?);
        let turned: Vector3<f64> = So3::exp(&(omega * dt)) * bearing;
        self.model
            .inner
            .project([turned.x, turned.y, turned.z])
            .ok()
    }
}

#[cfg(test)]
mod tests;
