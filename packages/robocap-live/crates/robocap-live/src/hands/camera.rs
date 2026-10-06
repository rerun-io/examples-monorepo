//! One rig camera for hand geometry ([`RigCameraModel`]), in the two precisions handtrack uses for it:
//! - the lens (kornia-3d `KannalaBrandt4`, KB4, or a plain pinhole) plus `cam_from_rig` in float64, for the KeyNet crops and the
//!   overlays (handtrack's `geometry/camera.py`);
//! - [`FitParams`]: the same calibration rounded to float32, as handtrack's `CameraRig` holds it and handfit's Python binding
//!   receives it, for the tracker's planning projections and the fit ([`fit_view`]). The model keeps the two apart, which keeps
//!   both float contracts with Python exact.
//!
//! The port of handtrack's `geometry/camera.py` for the cameras RoboCap has: its Fisheye62 formula with `k5 = k6 = p1 = p2 = 0`
//! is exactly kornia-3d's Kannala-Brandt `KannalaBrandt4` (the `r^2 <= pi^2` clip never binds, since theta <= pi). Projection is
//! unclipped (points behind the camera and outside the image still project), as in handtrack; use [`in_front`] for z > 0.

use kornia_staging_3d::camera::{CameraModel, KannalaBrandt4, Pinhole, UnprojectError};
use nalgebra::{Isometry3, Matrix3, Vector2, Vector3};

use crate::frame::{NUM_CAMERAS, Rig, RigCamera};

use handfit::View;
use handfit::nalgebra::{Matrix4, SMatrix, SVector};

use super::letterbox::BarLetterbox;
use crate::nets::NUM_LANDMARKS;

/// Errors building camera models from a rig.
#[derive(Debug, thiserror::Error)]
pub enum CameraError {
    /// The rig does not have one camera per RoboCap camera.
    #[error("rig has {0} cameras, expected {NUM_CAMERAS}")]
    CameraCount(usize),
    /// A lens the port does not support (Fisheye62 with tangential or k5/k6 terms).
    #[error("camera {name}: unsupported lens {detail}")]
    UnsupportedLens { name: String, detail: String },
    /// A non-finite or degenerate calibration value.
    #[error("camera {name}: invalid calibration ({detail})")]
    Invalid { name: String, detail: String },
}

/// The lens of one camera.
#[derive(Clone, Debug)]
pub enum Lens {
    /// Kannala-Brandt KB4 fisheye (staged `KannalaBrandt4`, unprojected on its physical branch), RoboCap's model.
    Fisheye(KannalaBrandt4<f64>),
    /// A distortion-free pinhole (handtrack's SHOW3D lens).
    Pinhole(Pinhole<f64>),
}

/// One rig camera: lens, image size and the rigid transform from the rig frame, its net-frame letterbox, and the same
/// calibration rounded to float32 for the tracker and the fit.
#[derive(Clone, Debug)]
pub struct RigCameraModel {
    /// Camera name (one of `frame::CAMERA_NAMES`).
    pub name: String,
    lens: Lens,
    /// The calibration's rotation exactly as stored (handtrack applies the matrix as it is).
    rotation: Matrix3<f64>,
    translation: Vector3<f64>,
    size: (u32, u32),
    /// The camera's net-frame letterbox.
    pub net: BarLetterbox,
    /// The calibration rounded to float32: the tracker's planning projections and the fit ([`fit_view`]).
    pub fit: FitParams,
}

/// One camera's calibration rounded to float32 (and held as float64), as handtrack's `CameraRig` holds it and handfit's Python
/// binding receives it.
#[derive(Clone, Debug)]
pub struct FitParams {
    /// Camera from rig, metres.
    pub cam_from_rig: Matrix4<f64>,
    /// (fx, fy).
    pub focal: Vector2<f64>,
    /// (cx, cy).
    pub principal: Vector2<f64>,
    /// Fisheye62 `[k1..k6, p1, p2]`; `None` = pinhole.
    pub distortion: Option<SVector<f64, 8>>,
}

impl FitParams {
    /// Unclipped projection into the camera's own pixels (handtrack `geometry.camera.project`: Fisheye62 or pinhole, through
    /// handfit's lens).
    pub fn project(&self, point_cam: &Vector3<f64>) -> Vector2<f64> {
        handfit::residual::project_lens(
            point_cam,
            &self.focal,
            &self.principal,
            self.distortion.as_ref(),
            None,
        )
    }

    /// A world point into the camera (handtrack `world_to_cameras`: p_cam = cam_from_rig · rig_from_world · p_world), with
    /// `world` the float32 headset pose ([`world_matrix`]).
    pub fn cam_from_world_point(&self, world: &Matrix4<f64>, point: &[f64; 3]) -> Vector3<f64> {
        let rotation = world.fixed_view::<3, 3>(0, 0);
        let rig = rotation.transpose()
            * (Vector3::new(point[0], point[1], point[2]) - world.fixed_view::<3, 1>(0, 3));
        self.cam_from_rig.fixed_view::<3, 3>(0, 0) * rig
            + self.cam_from_rig.fixed_view::<3, 1>(0, 3)
    }
}

impl RigCameraModel {
    /// Build the model of one catalog camera.
    ///
    /// # Errors
    ///
    /// [`CameraError::UnsupportedLens`] for Fisheye62 with nonzero `k5, k6, p1, p2`; [`CameraError::Invalid`] for non-finite values
    /// (also after rounding to float32), a non-positive focal length, or an image size without a net-frame letterbox
    /// ([`BarLetterbox::for_size`]).
    pub fn from_rig_camera(camera: &RigCamera) -> Result<Self, CameraError> {
        let invalid = |detail: &str| CameraError::Invalid {
            name: camera.name.clone(),
            detail: detail.to_string(),
        };
        let [fx, fy] = camera.focal;
        let [cx, cy] = camera.principal;
        if !(fx.is_finite() && fy.is_finite() && cx.is_finite() && cy.is_finite())
            || fx <= 0.0
            || fy <= 0.0
        {
            return Err(invalid("focal/principal"));
        }
        let lens = match camera.fisheye62 {
            None => Lens::Pinhole(
                Pinhole::new([fx, fy, cx, cy])
                    .map_err(|_| invalid("camera calibration or branch overflow"))?,
            ),
            Some(k) => {
                if k.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("fisheye62"));
                }
                if k[4] != 0.0 || k[5] != 0.0 || k[6] != 0.0 || k[7] != 0.0 {
                    return Err(CameraError::UnsupportedLens {
                        name: camera.name.clone(),
                        detail: format!(
                            "Fisheye62 with k5={} k6={} p1={} p2={} (only KB4 is ported)",
                            k[4], k[5], k[6], k[7]
                        ),
                    });
                }
                Lens::Fisheye(
                    KannalaBrandt4::new([fx, fy, cx, cy, k[0], k[1], k[2], k[3]])
                        .map_err(|_| invalid("camera calibration or branch overflow"))?,
                )
            }
        };
        let m = camera.cam_from_rig;
        if m.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("cam_from_rig"));
        }
        let rotation = Matrix3::new(
            m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2],
        );
        let translation = Vector3::new(m[0][3], m[1][3], m[2][3]);
        let fit = FitParams {
            cam_from_rig: Matrix4::from_fn(|i, k| f32_round(m[i][k])),
            focal: Vector2::new(f32_round(fx), f32_round(fy)),
            principal: Vector2::new(f32_round(cx), f32_round(cy)),
            distortion: camera
                .fisheye62
                .map(|k| SVector::<f64, 8>::from_fn(|i, _| f32_round(k[i]))),
        };
        let rounded = fit
            .cam_from_rig
            .iter()
            .chain(fit.focal.iter())
            .chain(fit.principal.iter())
            .chain(fit.distortion.iter().flat_map(|d| d.iter()));
        if rounded.into_iter().any(|x| !x.is_finite()) {
            return Err(invalid("a value beyond float32"));
        }
        let net = BarLetterbox::for_size(camera.width, camera.height).ok_or_else(|| {
            invalid(&format!(
                "no net-frame letterbox for a {}x{} camera",
                camera.width, camera.height
            ))
        })?;
        Ok(Self {
            name: camera.name.clone(),
            lens,
            rotation,
            translation,
            size: (camera.width, camera.height),
            net,
            fit,
        })
    }

    /// The lens.
    pub fn lens(&self) -> &Lens {
        &self.lens
    }

    /// Pixel of a camera-frame point (metres), unclipped; `None` only where the lens has no image (the optical centre,
    /// the negative optical axis).
    pub fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>> {
        match &self.lens {
            Lens::Fisheye(fisheye) => {
                let radius = p_cam.x.hypot(p_cam.y);
                if radius < 1e-12 && p_cam.z < 1e-12 {
                    return None;
                }
                let pixel = fisheye.project_unchecked([p_cam.x, p_cam.y, p_cam.z]);
                pixel
                    .iter()
                    .all(|v| v.is_finite())
                    .then(|| Vector2::from(pixel))
            }
            Lens::Pinhole(pinhole) => {
                // handtrack clamps |z| < 1e-9; in_front owns its visibility policy.
                let z = if p_cam.z.abs() < 1e-9 { 1e-9 } else { p_cam.z };
                Some(pinhole.project_unchecked([p_cam.x, p_cam.y, z]).into())
            }
        }
    }

    /// Unit ray through a pixel. RoboCap saturates pixels outside the KB4 physical branch.
    pub fn unproject(&self, px: &Vector2<f64>) -> Vector3<f64> {
        let pixel = [px.x, px.y];
        let ray = match &self.lens {
            Lens::Fisheye(fisheye) => match fisheye.unproject(pixel) {
                Err(UnprojectError::OutsideDomain) => fisheye.boundary_bearing(pixel),
                result => result,
            },
            Lens::Pinhole(pinhole) => pinhole.unproject(pixel),
        };
        ray.unwrap_or([f64::NAN; 3]).into()
    }

    /// A rig-frame point into the camera frame, with the calibration matrix as stored (handtrack's `transform_points`).
    pub fn cam_from_rig_point(&self, p_rig: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * p_rig + self.translation
    }

    /// A world point into the camera frame: `cam_from_rig · rig_from_world · p` (handtrack's `world_to_cameras`).
    pub fn cam_from_world_point(
        &self,
        world_from_rig: &Isometry3<f64>,
        p_world: &Vector3<f64>,
    ) -> Vector3<f64> {
        let p_rig = world_from_rig
            .rotation
            .inverse_transform_vector(&(p_world - world_from_rig.translation.vector));
        self.cam_from_rig_point(&p_rig)
    }

    /// (width, height) in pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Whether a pixel lies inside `[0, W) x [0, H)` (handtrack's `inside_image`; non-finite pixels fail).
    pub fn inside_image(&self, px: &Vector2<f64>) -> bool {
        px.x >= 0.0 && px.y >= 0.0 && px.x < self.size.0 as f64 && px.y < self.size.1 as f64
    }
}

/// Finite points with z > 0 (handtrack's `in_front`).
pub fn in_front(p_cam: &Vector3<f64>) -> bool {
    p_cam.iter().all(|v| v.is_finite()) && p_cam.z > 0.0
}

/// The models of all six rig cameras, in index order.
///
/// # Errors
///
/// [`CameraError`] when the rig does not have six cameras or a camera does not build ([`RigCameraModel::from_rig_camera`]).
pub fn rig_models(rig: &Rig) -> Result<Vec<RigCameraModel>, CameraError> {
    if rig.cameras.len() != NUM_CAMERAS {
        return Err(CameraError::CameraCount(rig.cameras.len()));
    }
    rig.cameras
        .iter()
        .map(RigCameraModel::from_rig_camera)
        .collect()
}

/// A value rounded to float32 and back.
pub fn f32_round(x: f64) -> f64 {
    x as f32 as f64
}

/// One view of a hand as handfit's fit takes it, built as handfit's Python binding builds it: camera-from-world from the float32
/// `cam_from_rig` and `world_from_rig` ([`handfit::residual::cam_from_world`]), keypoints, weights and relative distances rounded
/// to float32 (distances 0 where unobserved).
pub fn fit_view(
    camera: &RigCameraModel,
    world: &Matrix4<f64>,
    keypoints_px: &[[f64; 2]; NUM_LANDMARKS],
    weights: &[f64; NUM_LANDMARKS],
    d_rel_mm: &[f64; NUM_LANDMARKS],
) -> View {
    let fit = &camera.fit;
    let (rotation, translation) = handfit::residual::cam_from_world(&fit.cam_from_rig, world);
    View {
        rotation,
        translation,
        focal: fit.focal,
        principal: fit.principal,
        distortion: fit.distortion,
        pixels: SMatrix::from_fn(|i, k| {
            if keypoints_px[i][k].is_finite() {
                f32_round(keypoints_px[i][k])
            } else {
                0.0
            }
        }),
        weights: SVector::from_fn(|i, _| f32_round(weights[i])),
        distances: SVector::from_fn(|i, _| {
            if weights[i] > 0.0 {
                f32_round(d_rel_mm[i])
            } else {
                0.0
            }
        }),
    }
}

/// The headset pose as handtrack's float32 4x4 `world_from_rig`.
pub fn world_matrix(world_from_rig: &Isometry3<f64>) -> Matrix4<f64> {
    world_from_rig.to_homogeneous().map(f32_round)
}
