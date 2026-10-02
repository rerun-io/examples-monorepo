//! Perspective KeyNet crops (handtrack `labels/perspective.py`): UmeTrack's crop cameras.
//!
//! A crop camera shares its source camera's centre and looks at the hand: the minimal rotation taking the optical axis to the
//! target direction, then a roll about the new axis by the camera's mounting angle, then an optional x mirror (right hands, so
//! KeyNet sees left hands only). Its pinhole focal fits the hand in the 96 x 96 crop with a margin. Every crop pixel's ray goes
//! back through the source lens and samples the native 1920x1080 frame (kornia-style maps + bilinear remap in `kornia_ext`).
//! Crop pixel centres are 0..95 with the centre at 47.5.

use kornia_image::{Image, ImageError, ImageSize};
use nalgebra::{Matrix3, Vector2, Vector3};

use super::camera::{Lens, RigCameraModel};
use super::letterbox::BarLetterbox;
use crate::kornia_ext::remap::remap_f32_from_u8;
use crate::kornia_ext::virtual_camera::{RayProjection, VirtualPinhole, maps_from_virtual_pinhole_f32, maps_from_virtual_pinhole_kb4_f32};
use crate::nets::{KEYNET_CROP, NUM_LANDMARKS};

/// Crop side in pixels.
pub const CROP_SIZE: usize = KEYNET_CROP;
/// The crop's centre pixel coordinate, (96 - 1) / 2.
pub const CROP_CENTRE: f64 = (CROP_SIZE as f64 - 1.0) / 2.0;
/// The farthest crop point sits at 1/1.2 of the half-side from the centre.
pub const CROP_MARGIN: f64 = 1.2;
/// A crop camera must look less than ~87 degrees off its source camera's axis.
pub const MIN_AXIS_COSINE: f64 = 0.05;
/// Crop rays with source-frame z at or below this sample nothing (zero).
pub const MIN_RAY_Z: f64 = 1e-6;
/// The crop's size.
pub const CROP_IMAGE_SIZE: ImageSize = ImageSize { width: CROP_SIZE, height: CROP_SIZE };

/// One pinhole crop camera, sharing its source camera's centre.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CropCamera {
    /// crop_from_camera, roll included, mirror not.
    pub rotation: Matrix3<f64>,
    /// Crop pixels per unit of normalised image coordinate; NaN when unusable.
    pub focal: f64,
    /// Mirror the crop's x axis (right hands).
    pub mirror: bool,
}

impl CropCamera {
    /// handtrack's `_usable`: finite focal and rotation.
    pub fn usable(&self) -> bool {
        self.focal.is_finite() && self.rotation.iter().all(|v| v.is_finite())
    }

    /// `to_crop`: a camera-frame point into crop pixels (mirror applied), and its depth along the crop axis.
    pub fn to_crop(&self, p_cam: &Vector3<f64>) -> (Vector2<f64>, f64) {
        let local = self.rotation * p_cam;
        let depth = local.z;
        let safe = if depth.abs() < 1e-9 { 1e-9 } else { depth };
        let u = local.x / safe * self.focal + CROP_CENTRE;
        let v = local.y / safe * self.focal + CROP_CENTRE;
        (Vector2::new(if self.mirror { (CROP_SIZE as f64 - 1.0) - u } else { u }, v), depth)
    }

    /// `from_crop`: the camera-frame ray (not normalised) through crop pixel `uv`, mirror undone: the ray the sampling maps use.
    pub fn from_crop(&self, uv: &Vector2<f64>) -> Vector3<f64> {
        Vector3::from(self.virtual_pinhole().ray(uv.x, uv.y, CROP_SIZE))
    }

    /// The crop as a kornia-style virtual pinhole (for the remap maps).
    pub fn virtual_pinhole(&self) -> VirtualPinhole {
        let r = &self.rotation;
        VirtualPinhole {
            rotation: [[r[(0, 0)], r[(0, 1)], r[(0, 2)]], [r[(1, 0)], r[(1, 1)], r[(1, 2)]], [r[(2, 0)], r[(2, 1)], r[(2, 2)]]],
            focal: self.focal,
            principal: [CROP_CENTRE, CROP_CENTRE],
            mirror_x: self.mirror,
        }
    }
}

/// The smallest rotation taking +z to the unit `direction` (Rodrigues; UmeTrack's `from_two_vectors`). `1 + cos` is floored at
/// 1e-6 so directions at or behind the image plane stay finite (callers drop them).
pub fn minimal_rotation(direction: &Vector3<f64>) -> Matrix3<f64> {
    let axis = Vector3::z().cross(direction);
    let cosine = direction.z;
    let skew = axis.cross_matrix();
    Matrix3::identity() + skew + skew * skew / (1.0 + cosine).max(1e-6)
}

/// The rotation by `angle` about z.
pub fn roll(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
}

/// crop_from_camera for a crop camera aimed along `direction` (camera frame) and rolled by `roll_rad` about its axis.
pub fn look_at(direction: &Vector3<f64>, roll_rad: f64) -> Matrix3<f64> {
    let unit = direction / direction.norm().max(1e-12);
    (minimal_rotation(&unit) * roll(roll_rad)).transpose()
}

/// `crop_cameras` (UmeTrack's `gen_crop_parameters_from_points`): aim at the valid points' bounding-box centre, then fit them
/// with `margin`. Without valid points, or with a valid point behind the crop camera, the focal is NaN.
pub fn crop_camera_from_points(points_cam: &[Vector3<f64>; NUM_LANDMARKS], valid: &[bool; NUM_LANDMARKS], roll_rad: f64, mirror: bool,
                               margin: f64) -> CropCamera {
    let big = 1e9;
    let mut low = Vector3::repeat(big);
    let mut high = Vector3::repeat(-big);
    for (point, _) in points_cam.iter().zip(valid).filter(|(_, ok)| **ok) {
        low = low.inf(point);
        high = high.sup(point);
    }
    let centre = (low + high) * 0.5;
    let rotation = look_at(&centre, roll_rad);
    let mut extent = 0.0f64;
    let mut all_ahead = true;
    let mut finite = true;
    for (point, ok) in points_cam.iter().zip(valid) {
        if !ok {
            continue;
        }
        let local = rotation * point;
        all_ahead &= local.z > 1e-4;
        let z = local.z.max(1e-4);
        extent = extent.max((local.x / z).abs().max((local.y / z).abs()));
        finite &= point.iter().all(|v| v.is_finite());
    }
    let forward = centre.z > MIN_AXIS_COSINE * centre.norm();
    let usable = valid.iter().any(|ok| *ok) && forward && all_ahead && extent > 1e-6 && finite;
    let focal = if usable { CROP_CENTRE / (extent.max(1e-6) * margin) } else { f64::NAN };
    CropCamera { rotation, focal, mirror }
}

/// The acquisition crop camera: aimed through the DetNet circle's centre, with a focal that gives the circle's angular radius
/// the crop margin (`PerspectiveKeyNetEstimator._cameras`' circle branch). A non-finite circle gives a NaN focal.
pub fn crop_camera_from_circle(camera: &RigCameraModel, letterbox: &BarLetterbox, circle_net: [f32; 3], roll_rad: f64, mirror: bool) -> CropCamera {
    let [cx, cy, r] = circle_net.map(f64::from);
    let centre = letterbox.from_net(&Vector2::new(cx, cy));
    let radius = r / letterbox.scale;
    let ray0 = camera.unproject(&centre);
    let ray1 = camera.unproject(&(centre + Vector2::new(radius, 0.0)));
    let angle = ray0.dot(&ray1).clamp(-1.0, 1.0).acos();
    // Python's max(nan, 1e-3) is nan: a NaN angle keeps the NaN focal.
    let angle = if angle.is_nan() { angle } else { angle.max(1e-3) };
    let focal = CROP_CENTRE / (angle.tan() * CROP_MARGIN);
    CropCamera { rotation: look_at(&ray0, roll_rad), focal, mirror }
}

/// The crop camera an unusable view samples with (identity rotation, focal 1), as handtrack does before zeroing the view.
pub fn placeholder_crop(mirror: bool) -> CropCamera {
    CropCamera { rotation: Matrix3::identity(), focal: 1.0, mirror }
}

impl RayProjection for RigCameraModel {
    fn project_ray(&self, ray: [f64; 3]) -> Option<[f64; 2]> {
        self.project(&Vector3::new(ray[0], ray[1], ray[2])).map(|p| [p.x, p.y])
    }
}

/// Reusable remap maps for crop sampling.
pub struct CropMaps {
    /// Source x per crop pixel.
    pub map_x: Image<f32, 1>,
    /// Source y per crop pixel.
    pub map_y: Image<f32, 1>,
}

impl CropMaps {
    /// Maps for a 96 x 96 crop.
    ///
    /// # Errors
    ///
    /// Never in practice (kornia's allocation check).
    pub fn new() -> Result<Self, ImageError> {
        Ok(Self { map_x: Image::from_size_val(CROP_IMAGE_SIZE, 0.0)?, map_y: Image::from_size_val(CROP_IMAGE_SIZE, 0.0)? })
    }
}

/// `sample_crops` for one crop: the 96 x 96 crop in [0, 1] (bilinear, zero outside the image and behind the camera) of `full`
/// through `camera`'s lens, mirror applied.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` when `dst` is not 96 x 96.
///
/// `turned_180`: `full` is the camera's image turned 180 degrees (the sensor's readout of a camera the vendor turns): the maps
/// read it at `(w - 1 - x, h - 1 - y)`, so the crop is the upright one without turning the frame.
pub fn sample_crop(full: &Image<u8, 1>, camera: &RigCameraModel, crop: &CropCamera, maps: &mut CropMaps, dst: &mut Image<f32, 1>, turned_180: bool) -> Result<(), ImageError> {
    match camera.lens() {
        // The vectorised float32 KB4 maps (Python computes its crops in float32 too).
        Lens::Fisheye(fisheye) => maps_from_virtual_pinhole_kb4_f32(&fisheye.camera, &crop.virtual_pinhole(), MIN_RAY_Z, &mut maps.map_x, &mut maps.map_y)?,
        Lens::Pinhole { .. } => maps_from_virtual_pinhole_f32(camera, &crop.virtual_pinhole(), MIN_RAY_Z, &mut maps.map_x, &mut maps.map_y)?,
    }
    if turned_180 {
        let (w, h) = ((full.width() - 1) as f32, (full.height() - 1) as f32);
        maps.map_x.as_slice_mut().iter_mut().for_each(|x| *x = w - *x);
        maps.map_y.as_slice_mut().iter_mut().for_each(|y| *y = h - *y);
    }
    remap_f32_from_u8(full, dst, &maps.map_x, &maps.map_y, 1.0 / 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crop_from_an_upside_down_frame_matches_the_crop_from_the_upright_frame() -> Result<(), Box<dyn std::error::Error>> {
        let rig_camera = crate::frame::RigCamera {
            name: "left_eye".into(),
            width: 1920,
            height: 1080,
            cam_from_rig: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            focal: [700.0, 700.0],
            principal: [955.0, 545.0],
            fisheye62: Some([0.05, -0.01, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0]),
        };
        let camera = RigCameraModel::from_rig_camera(&rig_camera)?;
        let size = ImageSize { width: 1920, height: 1080 };
        // A smooth image (bilinear sampling of noise would amplify float rounding into large differences).
        let pixels: Vec<u8> = (0..1920 * 1080).map(|i| ((i % 1920) / 8 + (i / 1920) / 5) as u8).collect();
        let upright = Image::<u8, 1>::new(size, pixels.clone())?;
        let turned = Image::<u8, 1>::new(size, pixels.into_iter().rev().collect())?;
        let mut maps = CropMaps::new()?;
        for (direction, mirror) in [(Vector3::new(0.3, -0.2, 1.0), false), (Vector3::new(-0.4, 0.3, 1.0), true)] {
            let crop = CropCamera { rotation: look_at(&direction, 0.0), focal: 140.0, mirror };
            let mut expected = Image::<f32, 1>::from_size_val(CROP_IMAGE_SIZE, 0.0)?;
            let mut got = Image::<f32, 1>::from_size_val(CROP_IMAGE_SIZE, 0.0)?;
            sample_crop(&upright, &camera, &crop, &mut maps, &mut expected, false)?;
            sample_crop(&turned, &camera, &crop, &mut maps, &mut got, true)?;
            let worst = expected.as_slice().iter().zip(got.as_slice()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "mirror {mirror}: worst difference {worst}");
            assert!(expected.as_slice().iter().any(|&v| v > 0.1), "the crop saw the image");
        }
        Ok(())
    }

    #[test]
    fn look_at_points_the_crop_axis_along_the_direction() {
        let direction = Vector3::new(0.3, -0.2, 1.0);
        let crop_from_camera = look_at(&direction, 0.4);
        let axis = crop_from_camera * direction.normalize();
        assert!((axis - Vector3::z()).norm() < 1e-12);
        assert!((crop_from_camera * crop_from_camera.transpose() - Matrix3::identity()).norm() < 1e-12);
    }

    #[test]
    fn to_crop_and_from_crop_are_inverse_up_to_depth() {
        let crop = CropCamera { rotation: look_at(&Vector3::new(0.1, 0.2, 1.0), 0.0), focal: 120.0, mirror: true };
        let point = Vector3::new(0.12, 0.25, 1.1);
        let (uv, depth) = crop.to_crop(&point);
        let ray = crop.from_crop(&uv);
        assert!((ray * depth - point).norm() < 1e-12, "{ray} {depth}");
    }

    #[test]
    fn points_fit_inside_the_margin_and_invalid_rows_get_nan() {
        let mut points = [Vector3::new(0.0, 0.0, 0.5); NUM_LANDMARKS];
        for (i, p) in points.iter_mut().enumerate() {
            p.x = -0.05 + 0.005 * i as f64;
            p.y = 0.02 * ((i % 3) as f64);
        }
        let valid = [true; NUM_LANDMARKS];
        let crop = crop_camera_from_points(&points, &valid, 0.0, false, CROP_MARGIN);
        let farthest = points.iter().map(|p| (crop.to_crop(p).0 - Vector2::repeat(CROP_CENTRE)).amax()).fold(0.0, f64::max);
        assert!((farthest - CROP_CENTRE / CROP_MARGIN).abs() < 1e-9, "{farthest}");
        assert!(crop_camera_from_points(&points, &[false; NUM_LANDMARKS], 0.0, false, CROP_MARGIN).focal.is_nan());
    }
}
