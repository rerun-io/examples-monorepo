//! Remap maps for a "virtual" pinhole camera that shares a source camera's centre: every virtual pixel's ray goes through the
//! source camera's own lens (e.g. kornia-3d's KB4 `FisheyeCamera`) to a source pixel, so `remap` cuts a distortion-free
//! perspective crop looking anywhere in a fisheye image (UmeTrack's perspective hand crops). Target upstream: kornia-3d
//! `camera` (the map builder) beside kornia-imgproc `calibration::distortion`'s undistort maps.

use super::{CameraModel, KannalaBrandt4};
use kornia_image::{Image, ImageError};

mod kernels;

/// A pinhole camera with square pixels at the source camera's centre, rotated by `rotation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualPinhole {
    /// virtual_from_source, row-major (a source-frame direction `d` is `rotation · d` in the virtual frame).
    pub rotation: [[f64; 3]; 3],
    /// Pixels per unit of normalised image coordinate.
    pub focal: f64,
    /// Principal point (pixel centres at integers), e.g. `(n - 1) / 2` for a centred `n x n` crop.
    pub principal: [f64; 2],
    /// Mirror the virtual image's x axis (column `u` shows the ray of column `width - 1 - u`).
    pub mirror_x: bool,
}

impl VirtualPinhole {
    /// The source-frame ray (not normalised, virtual z = 1) through virtual pixel `(u, v)`, mirror applied.
    ///
    /// # Arguments
    ///
    /// * `u`, `v` - Virtual pixel coordinates.
    /// * `width` - The virtual image width (for the mirror).
    ///
    /// # Returns
    ///
    /// `rotationᵀ · ((u' - cx) / f, (v - cy) / f, 1)` with `u' = width - 1 - u` when mirrored.
    pub fn ray(&self, u: f64, v: f64, width: usize) -> [f64; 3] {
        let u = if self.mirror_x {
            (width as f64 - 1.0) - u
        } else {
            u
        };
        let local = [
            (u - self.principal[0]) / self.focal,
            (v - self.principal[1]) / self.focal,
            1.0,
        ];
        let r = &self.rotation;
        [
            r[0][0] * local[0] + r[1][0] * local[1] + r[2][0] * local[2],
            r[0][1] * local[0] + r[1][1] * local[1] + r[2][1] * local[2],
            r[0][2] * local[0] + r[1][2] * local[1] + r[2][2] * local[2],
        ]
    }
}

/// Fill `remap` maps so that the destination shows `view` through the source lens `lens`.
///
/// A virtual pixel whose ray has source-frame `z <= min_z` (behind or grazing the source image plane) or does not project
/// gets NaN, which zero-border remap samples as zero. Forward projection is unchecked:
/// the caller owns the depth/crop policy; the inverse branch does not clip maps.
///
/// # Arguments
///
/// * `lens` - The source camera's lens.
/// * `view` - The virtual pinhole.
/// * `min_z` - Smallest source-frame z of a usable ray (1e-6 in UmeTrack's crops).
/// * `map_x`, `map_y` - Output source coordinates, the virtual image's size.
///
/// # Returns
///
/// `Ok(())` when the maps were written.
///
/// # Errors
///
/// `ImageError::InvalidImageSize` when the maps differ in size, or
/// `ImageError::UnsupportedDevice` when either output is not host-accessible.
///
/// # Example
///
/// ```
/// use kornia_staging_3d::camera::KannalaBrandt4;
/// use kornia_image::{Image, ImageSize};
/// use kornia_staging_3d::camera::virtual_camera::{VirtualPinhole, maps_from_virtual_pinhole_f32};
/// let lens = KannalaBrandt4::new([500.0,500.0,960.0,540.0,0.0,0.0,0.0,0.0]).expect("valid camera calibration");
/// let view = VirtualPinhole { rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], focal: 100.0, principal: [47.5, 47.5], mirror_x: false };
/// let size = ImageSize { width: 96, height: 96 };
/// let (mut map_x, mut map_y) = (Image::<f32, 1>::from_size_val(size, 0.0).unwrap(), Image::<f32, 1>::from_size_val(size, 0.0).unwrap());
/// maps_from_virtual_pinhole_f32(&lens, &view, 1e-6, &mut map_x, &mut map_y).unwrap();
/// // The crop centre looks down the optical axis.
/// assert!((map_x.as_slice()[47 * 96 + 47] - (960.0 - 0.5 * 500.0 / 100.0)).abs() < 1e-2);
/// ```
pub fn maps_from_virtual_pinhole_f32<L: CameraModel<f64>>(
    lens: &L,
    view: &VirtualPinhole,
    min_z: f64,
    map_x: &mut Image<f32, 1>,
    map_y: &mut Image<f32, 1>,
) -> Result<(), ImageError> {
    if !map_x.storage.domain().is_host_accessible() || !map_y.storage.domain().is_host_accessible()
    {
        return Err(ImageError::UnsupportedDevice);
    }
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(
            map_x.cols(),
            map_x.rows(),
            map_y.cols(),
            map_y.rows(),
        ));
    }
    let width = map_x.cols();
    for (index, (x, y)) in map_x
        .as_slice_mut()
        .iter_mut()
        .zip(map_y.as_slice_mut().iter_mut())
        .enumerate()
    {
        let (u, v) = ((index % width) as f64, (index / width) as f64);
        let ray = view.ray(u, v, width);
        match (ray[2] > min_z).then(|| lens.project_unchecked(ray)) {
            Some([px, py]) if px.is_finite() && py.is_finite() => {
                *x = px as f32;
                *y = py as f32;
            }
            _ => {
                *x = f32::NAN;
                *y = f32::NAN;
            }
        }
    }
    Ok(())
}

/// Opt-in approximate [`maps_from_virtual_pinhole_f32`] for a KB4 source lens, in f32 and vectorisable: the same maps to ~1e-4 px at 1080p (float32
/// precision, what PyTorch-side crops are computed in), several times faster than the generic f64 path.
///
/// # Arguments
///
/// * `lens` - The KB4 source camera.
/// * `view` - The virtual pinhole.
/// * `min_z` - Smallest source-frame z of a usable ray.
/// * `map_x`, `map_y` - Output source coordinates, the virtual image's size.
///
/// # Returns
///
/// `Ok(())` when the maps were written (NaN for rays with `z <= min_z`).
///
/// # Errors
///
/// `ImageError::InvalidImageSize` when the maps differ in size, or
/// `ImageError::UnsupportedDevice` when either output is not host-accessible.
pub fn maps_from_virtual_pinhole_kb4_f32(
    lens: &KannalaBrandt4<f64>,
    view: &VirtualPinhole,
    min_z: f64,
    map_x: &mut Image<f32, 1>,
    map_y: &mut Image<f32, 1>,
) -> Result<(), ImageError> {
    if !map_x.storage.domain().is_host_accessible() || !map_y.storage.domain().is_host_accessible()
    {
        return Err(ImageError::UnsupportedDevice);
    }
    if map_x.size() != map_y.size() {
        return Err(ImageError::InvalidImageSize(
            map_x.cols(),
            map_x.rows(),
            map_y.cols(),
            map_y.rows(),
        ));
    }
    let width = map_x.cols();
    if width == 0 {
        return Ok(());
    }
    // ray = rotationᵀ · (a, b, 1) = a · row0 + b · row1 + row2, with a = (u' - cx) / f and b = (v - cy) / f.
    let r = &view.rotation;
    let row = |i: usize| [r[i][0] as f32, r[i][1] as f32, r[i][2] as f32];
    let (row0, row1, row2) = (row(0), row(1), row(2));
    let inv_focal = (1.0 / view.focal) as f32;
    let (px, py) = (view.principal[0] as f32, view.principal[1] as f32);
    let [fx, fy, cx, cy, k1, k2, k3, k4] = lens.params().map(|v| v as f32);
    let min_z = min_z as f32;
    // Column u looks along source column u' = u, or width - 1 - u when mirrored: a = a0 + u * step, tabulated per block of
    // columns (a stack buffer) for the kernel.
    let (a0, step) = if view.mirror_x {
        ((width as f32 - 1.0 - px) * inv_focal, -inv_focal)
    } else {
        (-px * inv_focal, inv_focal)
    };
    let lens = kernels::Kb4 {
        row0,
        fx,
        fy,
        cx,
        cy,
        k: [k1, k2, k3, k4],
        min_z,
    };
    const BLOCK: usize = 64;
    let mut columns = [0f32; BLOCK];
    for (v, (xs, ys)) in map_x
        .as_slice_mut()
        .chunks_exact_mut(width)
        .zip(map_y.as_slice_mut().chunks_exact_mut(width))
        .enumerate()
    {
        let b = (v as f32 - py) * inv_focal;
        let base = [
            b * row1[0] + row2[0],
            b * row1[1] + row2[1],
            b * row1[2] + row2[2],
        ];
        for (block, (xs, ys)) in xs.chunks_mut(BLOCK).zip(ys.chunks_mut(BLOCK)).enumerate() {
            let columns = &mut columns[..xs.len()];
            for (k, a) in columns.iter_mut().enumerate() {
                *a = a0 + (block * BLOCK + k) as f32 * step;
            }
            kernels::kb4_row(&lens, base, columns, xs, ys);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    use crate::camera::Pinhole;

    fn maps(width: usize, height: usize) -> Result<(Image<f32, 1>, Image<f32, 1>), ImageError> {
        let size = ImageSize { width, height };
        Ok((
            Image::from_size_val(size, 0.0)?,
            Image::from_size_val(size, 0.0)?,
        ))
    }

    #[test]
    #[allow(unsafe_code)] // Exercise the foreign-buffer API with an owned test allocation.
    fn every_map_output_must_be_host_accessible_even_when_empty() {
        use kornia_tensor::MemoryDomain;
        use std::sync::Arc;

        let lens = KannalaBrandt4::new([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let view = VirtualPinhole {
            rotation: IDENTITY,
            focal: 100.0,
            principal: [0.0, 0.0],
            mirror_x: false,
        };
        for width in [0, 2] {
            for (x_device, y_device) in [(true, false), (false, true), (true, true)] {
                for approximate in [false, true] {
                    let make_map = |device: bool| {
                        let size = ImageSize { width, height: 1 };
                        if !device {
                            return Image::from_size_val(size, 0.0).unwrap();
                        }
                        // Restrict a live host allocation to simulate device-only residency.
                        // The Arc keeps its aligned, initialized bytes alive; no device pointer
                        // is dereferenced, and the residency guard must prevent slice access.
                        let backing = Arc::new(vec![0.0f32; width]);
                        unsafe {
                            Image::from_borrowed(
                                size,
                                backing.as_ptr(),
                                MemoryDomain::Device { id: 0 },
                                backing,
                            )
                            .unwrap()
                        }
                    };
                    let (mut x, mut y) = (make_map(x_device), make_map(y_device));
                    let result = if approximate {
                        maps_from_virtual_pinhole_kb4_f32(&lens, &view, 1e-6, &mut x, &mut y)
                    } else {
                        maps_from_virtual_pinhole_f32(&lens, &view, 1e-6, &mut x, &mut y)
                    };
                    assert!(matches!(result, Err(ImageError::UnsupportedDevice)));
                }
            }
        }
    }

    #[test]
    fn a_virtual_camera_equal_to_the_source_maps_pixels_to_themselves() -> Result<(), ImageError> {
        let view = VirtualPinhole {
            rotation: IDENTITY,
            focal: 100.0,
            principal: [50.0, 40.0],
            mirror_x: false,
        };
        let (mut map_x, mut map_y) = maps(8, 4)?;
        maps_from_virtual_pinhole_f32(
            &Pinhole::new([100.0, 100.0, 50.0, 40.0]).unwrap(),
            &view,
            1e-6,
            &mut map_x,
            &mut map_y,
        )?;
        assert!(
            (map_x.as_slice()[3] - 3.0).abs() < 1e-4
                && (map_y.as_slice()[3 * 8 + 1] - 3.0).abs() < 1e-4
        );
        Ok(())
    }

    #[test]
    fn the_kb4_fast_path_matches_the_generic_maps() -> Result<(), ImageError> {
        let lens = KannalaBrandt4::new([630.7, 628.8, 946.7, 539.5, 0.077, -0.063, 0.080, -0.029])
            .expect("valid camera calibration");
        for (direction, mirror) in [
            ([0.0, 0.0, 1.0], false),
            ([0.6, -0.4, 0.7], true),
            ([-0.9, 0.3, 0.3], false),
        ] {
            let d: [f64; 3] = direction;
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            // A crop axis along d: rows of virtual_from_source are an orthonormal frame with z = d.
            let z = [d[0] / n, d[1] / n, d[2] / n];
            let x0 = [z[2], 0.0, -z[0]];
            let xn = (x0[0] * x0[0] + x0[2] * x0[2]).sqrt();
            let x = [x0[0] / xn, 0.0, x0[2] / xn];
            let y = [
                z[1] * x[2] - z[2] * x[1],
                z[2] * x[0] - z[0] * x[2],
                z[0] * x[1] - z[1] * x[0],
            ];
            let view = VirtualPinhole {
                rotation: [x, y, z],
                focal: 140.0,
                principal: [47.5, 47.5],
                mirror_x: mirror,
            };
            let (mut gx, mut gy) = maps(96, 96)?;
            let (mut fx, mut fy) = maps(96, 96)?;
            maps_from_virtual_pinhole_f32(&lens, &view, 1e-6, &mut gx, &mut gy)?;
            maps_from_virtual_pinhole_kb4_f32(&lens, &view, 1e-6, &mut fx, &mut fy)?;
            for i in 0..96 * 96 {
                let (a, b) = (
                    (gx.as_slice()[i], gy.as_slice()[i]),
                    (fx.as_slice()[i], fy.as_slice()[i]),
                );
                assert_eq!(a.0.is_nan(), b.0.is_nan(), "{i}");
                if !a.0.is_nan() {
                    assert!(
                        (a.0 - b.0).abs() < 2e-3 && (a.1 - b.1).abs() < 2e-3,
                        "{direction:?} {i}: {a:?} vs {b:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn the_mirror_reverses_columns_and_rays_behind_the_camera_are_nan() -> Result<(), ImageError> {
        let view = VirtualPinhole {
            rotation: IDENTITY,
            focal: 100.0,
            principal: [50.0, 40.0],
            mirror_x: true,
        };
        let (mut map_x, mut map_y) = maps(8, 1)?;
        maps_from_virtual_pinhole_f32(
            &Pinhole::new([100.0, 100.0, 50.0, 40.0]).unwrap(),
            &view,
            1e-6,
            &mut map_x,
            &mut map_y,
        )?;
        assert!((map_x.as_slice()[0] - 7.0).abs() < 1e-4);
        // Looking backwards: z = -1 for every pixel.
        let back = VirtualPinhole {
            rotation: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            ..view
        };
        maps_from_virtual_pinhole_f32(
            &Pinhole::new([100.0, 100.0, 50.0, 40.0]).unwrap(),
            &back,
            1e-6,
            &mut map_x,
            &mut map_y,
        )?;
        assert!(map_x.as_slice().iter().all(|v| v.is_nan()));
        Ok(())
    }
}
