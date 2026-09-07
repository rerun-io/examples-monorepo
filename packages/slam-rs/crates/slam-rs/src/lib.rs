//! Visual-inertial odometry core.
//!
//! The crate is deliberately free of Python, Rerun and GPU code: it consumes
//! grayscale images and IMU samples and returns plain values. Python plumbing
//! (catalog feed, evaluation, logging) lives in the `slam_rs` package and the
//! bindings in `slam-rs-py`; a native runner lives in `slam-rs-cli`.
//!
//! This is the skeleton: the state machine, the error taxonomy and the value
//! types are real, the estimator is not implemented yet, so `track` never
//! reports [`VioStatus::Tracking`].

pub mod ba_base;
pub mod calib;
pub mod camera;
pub mod config;
pub mod frontend;
pub mod image;
pub mod imu;
pub mod landmark;
pub mod lie;
pub mod pyramid;
pub mod types;

use nalgebra::{Isometry3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

/// Version of the core, as declared in `crates/slam-rs/Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// How far the estimator has got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VioStatus {
    /// No pose yet: the estimator has not initialised.
    NotInitialised,
    /// The frame arrived before the IMU samples that cover it.
    NeedMoreImu,
    /// The returned pose is an estimate.
    Tracking,
}

/// Estimator configuration.
///
/// A placeholder: basalt's own JSON config replaces it once the estimator
/// lands, which is why it already deserializes and ignores nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    /// Number of cameras in the rig; every frameset carries exactly this many images.
    pub camera_count: usize,
    /// IMU samples needed before a frame can be processed.
    pub min_imu_samples: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_count: 2,
            min_imu_samples: 1,
        }
    }
}

/// A borrowed grayscale image: `height` rows of `width` bytes, `stride` bytes apart.
#[derive(Debug, Clone, Copy)]
pub struct ImageView<'a> {
    /// Row length in pixels.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
    /// Distance between the starts of two rows, in bytes.
    pub stride: usize,
    /// Pixel bytes, at least `stride * height` long.
    pub data: &'a [u8],
}

/// What one `track` call produced.
///
/// The pose is `world_from_rig` as `[tx, ty, tz, qx, qy, qz, qw]` (translation
/// in metres, quaternion xyzw, as the Python boundary expects); the rig frame
/// is the IMU frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VioResult {
    /// Estimator state for this frame.
    pub status: VioStatus,
    /// Timestamp of the frameset, in nanoseconds.
    pub t_ns: i64,
    /// Rig pose in the world frame, `[tx, ty, tz, qx, qy, qz, qw]`.
    pub world_from_rig: [f64; 7],
    /// Rig velocity in the world frame, m/s.
    pub velocity: [f64; 3],
    /// Gyroscope bias estimate, rad/s.
    pub gyro_bias: [f64; 3],
    /// Accelerometer bias estimate, m/s².
    pub accel_bias: [f64; 3],
}

/// Everything that can go wrong at the API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VioError {
    /// IMU samples must arrive strictly ordered; duplicates are rejected too.
    #[error("imu sample at {t_ns} ns does not follow the previous sample at {previous_t_ns} ns")]
    NonMonotonicImu {
        /// Timestamp of the last accepted sample.
        previous_t_ns: i64,
        /// Timestamp of the rejected sample.
        t_ns: i64,
    },
    /// The frameset does not hold one image per configured camera.
    #[error("expected {expected} images, got {actual}")]
    CameraCountMismatch {
        /// Cameras in the configured rig.
        expected: usize,
        /// Images in the frameset.
        actual: usize,
    },
    /// A row stride cannot be shorter than the row.
    #[error("camera {index}: stride {stride} is shorter than width {width}")]
    StrideTooSmall {
        /// Index of the offending camera.
        index: usize,
        /// Declared row length in pixels.
        width: usize,
        /// Declared row pitch in bytes.
        stride: usize,
    },
    /// `stride * height` does not fit in a `usize`, so no buffer can satisfy it.
    #[error("camera {index}: {height} rows of stride {stride} overflow the address space")]
    ImageSizeOverflow {
        /// Index of the offending camera.
        index: usize,
        /// Declared number of rows.
        height: usize,
        /// Declared row pitch in bytes.
        stride: usize,
    },
    /// The buffer does not hold `stride * height` bytes.
    #[error("camera {index}: {height} rows of stride {stride} do not fit in {len} bytes")]
    ShortImage {
        /// Index of the offending camera.
        index: usize,
        /// Declared number of rows.
        height: usize,
        /// Declared row pitch in bytes.
        stride: usize,
        /// Bytes actually supplied.
        len: usize,
    },
}

/// The estimator, driven one frameset at a time.
#[derive(Debug, Clone)]
pub struct Vio {
    config: Config,
    world_from_rig: Isometry3<f64>,
    velocity: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    accel_bias: Vector3<f64>,
    last_imu_t_ns: Option<i64>,
    imu_count: usize,
}

impl Vio {
    /// Build an estimator that has seen nothing yet.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            world_from_rig: Isometry3::identity(),
            velocity: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            accel_bias: Vector3::zeros(),
            last_imu_t_ns: None,
            imu_count: 0,
        }
    }

    /// The configuration this estimator runs with.
    pub fn config(&self) -> Config {
        self.config
    }

    /// Timestamp of the last accepted IMU sample, if any.
    pub fn last_imu_t_ns(&self) -> Option<i64> {
        self.last_imu_t_ns
    }

    /// Add one IMU sample: `gyro` in rad/s, `accel` in m/s², both in the rig frame.
    ///
    /// Samples must be strictly increasing in time; a duplicate or out-of-order
    /// timestamp is a [`VioError::NonMonotonicImu`], never a silent reorder.
    pub fn push_imu(&mut self, t_ns: i64, gyro: [f64; 3], accel: [f64; 3]) -> Result<(), VioError> {
        if let Some(previous_t_ns) = self.last_imu_t_ns
            && t_ns <= previous_t_ns
        {
            return Err(VioError::NonMonotonicImu {
                previous_t_ns,
                t_ns,
            });
        }
        let _ = (gyro, accel); // integrated once the preintegration lands
        self.last_imu_t_ns = Some(t_ns);
        self.imu_count += 1;
        Ok(())
    }

    /// Process one frameset: one image per camera, oldest to newest in time.
    pub fn track(&mut self, t_ns: i64, images: &[ImageView<'_>]) -> Result<VioResult, VioError> {
        if images.len() != self.config.camera_count {
            return Err(VioError::CameraCountMismatch {
                expected: self.config.camera_count,
                actual: images.len(),
            });
        }
        for (index, image) in images.iter().enumerate() {
            if image.stride < image.width {
                return Err(VioError::StrideTooSmall {
                    index,
                    width: image.width,
                    stride: image.stride,
                });
            }
            // width, height and stride are caller-controlled: an unchecked product
            // panics in debug and wraps to an accepted zero in release.
            let needed: usize =
                image
                    .stride
                    .checked_mul(image.height)
                    .ok_or(VioError::ImageSizeOverflow {
                        index,
                        height: image.height,
                        stride: image.stride,
                    })?;
            if image.data.len() < needed {
                return Err(VioError::ShortImage {
                    index,
                    height: image.height,
                    stride: image.stride,
                    len: image.data.len(),
                });
            }
        }

        let status: VioStatus = if self.imu_count < self.config.min_imu_samples {
            VioStatus::NeedMoreImu
        } else {
            // The estimator arrives with the frontend and backend PRs; until then
            // the state stays at the identity and no frame ever tracks.
            VioStatus::NotInitialised
        };
        log::debug!(
            "frame {t_ns} ns: {} images, status {status:?}",
            images.len()
        );

        Ok(VioResult {
            status,
            t_ns,
            world_from_rig: pose_to_array(&self.world_from_rig),
            velocity: self.velocity.into(),
            gyro_bias: self.gyro_bias.into(),
            accel_bias: self.accel_bias.into(),
        })
    }
}

/// Flatten an isometry into `[tx, ty, tz, qx, qy, qz, qw]`.
fn pose_to_array(pose: &Isometry3<f64>) -> [f64; 7] {
    let rotation: UnitQuaternion<f64> = pose.rotation;
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        rotation.i,
        rotation.j,
        rotation.k,
        rotation.w,
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    fn image(bytes: &[u8], width: usize, height: usize) -> ImageView<'_> {
        ImageView {
            width,
            height,
            stride: width,
            data: bytes,
        }
    }

    #[test]
    fn version_is_the_crate_version() {
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn config_round_trips_through_json() {
        let config: Config = Config {
            camera_count: 4,
            min_imu_samples: 8,
        };
        let text: String = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<Config>(&text).unwrap(), config);
    }

    #[test]
    fn a_frame_without_imu_needs_more_imu() {
        let mut vio: Vio = Vio::new(Config::default());
        let pixels: Vec<u8> = vec![0; 16];
        let images: [ImageView<'_>; 2] = [image(&pixels, 4, 4), image(&pixels, 4, 4)];
        let result: VioResult = vio.track(1_000, &images).unwrap();
        assert_eq!(result.status, VioStatus::NeedMoreImu);
        assert_eq!(result.t_ns, 1_000);
        assert_abs_diff_eq!(result.world_from_rig[6], 1.0, epsilon = 1e-12);
        assert_eq!(result.velocity, [0.0; 3]);
    }

    #[test]
    fn imu_lifts_the_frame_out_of_need_more_imu() {
        let mut vio: Vio = Vio::new(Config {
            camera_count: 1,
            min_imu_samples: 2,
        });
        let pixels: Vec<u8> = vec![0; 16];
        vio.push_imu(0, [0.0; 3], [0.0, 0.0, 9.81]).unwrap();
        assert_eq!(
            vio.track(10, &[image(&pixels, 4, 4)]).unwrap().status,
            VioStatus::NeedMoreImu
        );
        vio.push_imu(1_000, [0.0; 3], [0.0, 0.0, 9.81]).unwrap();
        assert_eq!(
            vio.track(2_000, &[image(&pixels, 4, 4)]).unwrap().status,
            VioStatus::NotInitialised
        );
    }

    #[test]
    fn repeated_imu_timestamps_are_rejected() {
        let mut vio: Vio = Vio::new(Config::default());
        vio.push_imu(5, [0.0; 3], [0.0; 3]).unwrap();
        assert_eq!(
            vio.push_imu(5, [0.0; 3], [0.0; 3]),
            Err(VioError::NonMonotonicImu {
                previous_t_ns: 5,
                t_ns: 5
            })
        );
        assert_eq!(vio.last_imu_t_ns(), Some(5));
    }

    #[test]
    fn a_frameset_of_the_wrong_width_is_rejected() {
        let mut vio: Vio = Vio::new(Config::default());
        let pixels: Vec<u8> = vec![0; 16];
        assert_eq!(
            vio.track(0, &[image(&pixels, 4, 4)]),
            Err(VioError::CameraCountMismatch {
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn a_short_buffer_is_rejected() {
        let mut vio: Vio = Vio::new(Config {
            camera_count: 1,
            min_imu_samples: 1,
        });
        let pixels: Vec<u8> = vec![0; 8];
        let short: [ImageView<'_>; 1] = [image(&pixels, 4, 4)];
        assert_eq!(
            vio.track(0, &short),
            Err(VioError::ShortImage {
                index: 0,
                height: 4,
                stride: 4,
                len: 8
            })
        );
        let narrow: [ImageView<'_>; 1] = [ImageView {
            width: 4,
            height: 2,
            stride: 2,
            data: &pixels,
        }];
        assert_eq!(
            vio.track(0, &narrow),
            Err(VioError::StrideTooSmall {
                index: 0,
                width: 4,
                stride: 2,
            })
        );
    }

    #[test]
    fn an_image_whose_size_overflows_is_rejected() {
        let mut vio: Vio = Vio::new(Config {
            camera_count: 1,
            min_imu_samples: 1,
        });
        let huge: [ImageView<'_>; 1] = [ImageView {
            width: 1,
            height: 2,
            stride: 1 << 63,
            data: &[],
        }];
        assert_eq!(
            vio.track(0, &huge),
            Err(VioError::ImageSizeOverflow {
                index: 0,
                height: 2,
                stride: 1 << 63,
            })
        );
    }

    proptest! {
        /// Whatever the first timestamp is, a second one that does not strictly
        /// follow it is rejected, and the accepted state does not move.
        #[test]
        fn non_monotonic_imu_is_always_rejected(first in -1_000_000i64..1_000_000, back in 0i64..1_000_000) {
            let mut vio: Vio = Vio::new(Config::default());
            vio.push_imu(first, [0.0; 3], [0.0; 3]).unwrap();
            let result: Result<(), VioError> = vio.push_imu(first - back, [0.0; 3], [0.0; 3]);
            prop_assert_eq!(
                result,
                Err(VioError::NonMonotonicImu { previous_t_ns: first, t_ns: first - back })
            );
            prop_assert_eq!(vio.last_imu_t_ns(), Some(first));
        }
    }
}
