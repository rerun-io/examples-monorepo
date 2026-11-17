//! The Insta360 factory lens model: the unified (Mei) sphere with the vendor's V3 or V6 distortion, and the vendor's
//! rotation convention. `kornia-staging-io` parses the `.insv` lens strings into [`LensCalibration`].

/// One lens of the factory calibration, stated for the video track's pixels.
///
/// The factory model is the unified (Mei) sphere: a ray `P` maps to `m = (X, Y) / (Z + xi |P|)`, then a distortion
/// of `m`, then `u = fx x_d + cx`, `v = fy y_d + cy`. V3 distorts with `1 + k1 r^2 + k2 r^4 + k3 r^6` and Brown
/// tangential `p1 p2`; V6 with `1 + d0 r^2 + .. + d4 r^10`, tangential `d5..d8` and thin prism `d9..d12`
/// (Gyroflow `insta360.rs`; video-types `geometry.rs`).
#[derive(Clone, Debug, PartialEq)]
pub struct LensCalibration {
    /// Lens-string version the values come from, 3 or 6.
    pub version: u8,
    /// Mei sphere offset.
    pub xi: f64,
    /// Focal along x, track pixels (Mei form: px per unit of `m`).
    pub fx: f64,
    /// Focal along y, track pixels.
    pub fy: f64,
    /// Principal point x, track pixels (pixel-centre convention, first pixel centre at 0).
    pub cx: f64,
    /// Principal point y, track pixels.
    pub cy: f64,
    /// Radial coefficients: `k1 k2 k3` (V3) or `d0..d4` (V6).
    pub radial: Vec<f64>,
    /// Tangential coefficients: `p1 p2` (V3) or `d5..d8` (V6).
    pub tangential: Vec<f64>,
    /// Thin-prism coefficients `d9..d12` (V6); empty for V3.
    pub prism: Vec<f64>,
    /// Factory yaw, pitch, roll, degrees.
    pub ypr_deg: [f64; 3],
    /// Factory translation, metres, in the lens-0 frame with lens 0 at the origin.
    pub translation_m: [f64; 3],
    /// Rotation taking rig-frame vectors into this lens's frame (row-major). The rig frame has lens 0's axes
    /// (x right, y down, z along lens 0's optical axis).
    pub lens_from_rig_rotation: [[f64; 3]; 3],
    /// Translation of the `lens_from_rig` transform, metres. The rig origin is the midpoint of the two lens centres.
    pub lens_from_rig_translation_m: [f64; 3],
    /// `R_P_lens`: rotation taking this lens's vectors into the panorama frame `P` (row-major). `P` is the vendor rig
    /// frame (telemetry-parser `rig.rs::rotation`): `+Z` the nominal lens-0 axis, `+Y` image-down of lens 0 (the body
    /// axis toward the handle), `+X` right. `R_P_lens = Rv(0, 0, 90; lens 0) Rv(ypr_i; i)^T` with the vendor rotation
    /// `Rv` of [`vendor_rotation`]. Unlike `lens_from_rig_rotation`, lens 0 is not the identity here: the factory
    /// relative rotation, not the nominal 180 deg, puts the two seams within 0.25 px on the X4 Air clip.
    pub pano_from_lens_rotation: [[f64; 3]; 3],
    /// This lens's centre in the panorama frame `P`, metres, with the origin at the mean of the lens centres:
    /// `R_P_lens0 (translation_m - mean)`. The factory translations are in lens 0's frame, which `R_P_lens0` turns
    /// into `P`; on the X4 Air the two centres sit about 16.56 mm either side of the origin along `Z`.
    pub pano_centre_m: [f64; 3],
}

/// Largest angle off the optical axis the factory model covers: lens class 131 has a 200 deg field
/// (telemetry-parser `rig.rs`). Beyond it the image circle ends and [`LensCalibration::project`] returns `None`.
pub const LENS_HALF_FIELD_RAD: f64 = 100.0 * std::f64::consts::PI / 180.0;

impl LensCalibration {
    /// Pixels per radian at the optical axis, `f / (1 + xi)` with `f` the mean focal.
    pub fn px_per_rad(&self) -> f64 {
        0.5 * (self.fx + self.fy) / (1.0 + self.xi)
    }

    /// Image radius, track pixels, of a ray `theta` radians off the optical axis, through the radial part of the
    /// model only (mean focal).
    pub fn radius_px(&self, theta: f64) -> f64 {
        let m = theta.sin() / (theta.cos() + self.xi);
        let r2 = m * m;
        let gain = 1.0
            + self
                .radial
                .iter()
                .enumerate()
                .map(|(i, k)| k * r2.powi(i as i32 + 1))
                .sum::<f64>();
        0.5 * (self.fx + self.fy) * m * gain
    }

    /// The full factory projection: a ray in this lens's frame to a track pixel (pixel centres at integers).
    ///
    /// Mei sphere `m = (x, y) / (z + xi |d|)`, `r2 = |m|^2`, then
    /// V3: `x_d = m_x R + 2 p1 m_x m_y + p2 (r2 + 2 m_x^2)`, `y_d = m_y R + p1 (r2 + 2 m_y^2) + 2 p2 m_x m_y`;
    /// V6: `x_d = m_x R + (r2 + 2 m_x^2) a + 2 m_x m_y b + d9 r2 + d11 r2^2`,
    /// `y_d = m_y R + (r2 + 2 m_y^2) b + 2 m_x m_y a + d10 r2 + d12 r2^2` with `a = d5 + d7 r2`, `b = d6 + d8 r2`;
    /// `R = 1 + sum_i radial_i r2^(i+1)`; `u = fx x_d + cx`, `v = fy y_d + cy` (video-types `MeiDistortion`).
    /// The ray need not be unit length.
    ///
    /// # Returns
    /// `None` for a zero ray or one more than [`LENS_HALF_FIELD_RAD`] off the optical axis.
    pub fn project(&self, ray: [f64; 3]) -> Option<[f64; 2]> {
        let [x, y, z] = ray;
        let norm = (x * x + y * y + z * z).sqrt();
        if norm == 0.0 || z < norm * LENS_HALF_FIELD_RAD.cos() {
            return None;
        }
        let den = z + self.xi * norm;
        let (mx, my) = (x / den, y / den);
        let r2 = mx * mx + my * my;
        let mut power = 1.0;
        let mut gain = 1.0;
        for k in &self.radial {
            power *= r2;
            gain += k * power;
        }
        let t = &self.tangential;
        let (xd, yd) = if self.version == 3 {
            (
                mx * gain + 2.0 * t[0] * mx * my + t[1] * (r2 + 2.0 * mx * mx),
                my * gain + t[0] * (r2 + 2.0 * my * my) + 2.0 * t[1] * mx * my,
            )
        } else {
            let (a, b, p) = (t[0] + t[2] * r2, t[1] + t[3] * r2, &self.prism);
            (
                mx * gain
                    + (r2 + 2.0 * mx * mx) * a
                    + 2.0 * mx * my * b
                    + p[0] * r2
                    + p[2] * r2 * r2,
                my * gain
                    + (r2 + 2.0 * my * my) * b
                    + 2.0 * mx * my * a
                    + p[1] * r2
                    + p[3] * r2 * r2,
            )
        };
        Some([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// [`LensCalibration::project`] of a ray given in the panorama frame `P` (rotated by `pano_from_lens_rotation^T`).
    pub fn project_from_panorama(&self, ray_p: [f64; 3]) -> Option<[f64; 2]> {
        let r = &self.pano_from_lens_rotation;
        self.project(std::array::from_fn(|j| {
            (0..3).map(|i| r[i][j] * ray_p[i]).sum()
        }))
    }
}

/// A row-major 3x3 matrix.
pub type Mat3 = [[f64; 3]; 3];

/// The product `a b`.
pub fn mul(a: &Mat3, b: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

/// The transpose of `a`.
pub fn transpose(a: &Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

/// Active rotation by `deg` degrees about axis 0 (x), 1 (y) or 2 (z).
fn axis_rotation(axis: usize, deg: f64) -> Mat3 {
    let (s, c) = deg.to_radians().sin_cos();
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut r = [[0.0; 3]; 3];
    r[axis][axis] = 1.0;
    r[a][a] = c;
    r[b][b] = c;
    r[a][b] = -s;
    r[b][a] = s;
    r
}

/// The vendor's world-to-lens rotation `Ry(pitch + 90) Rz(yaw) Rx(roll)`, times `Rz(180)` for lens 0 of two.
pub fn vendor_rotation(ypr_deg: [f64; 3], first_of_two: bool) -> Mat3 {
    let r = mul(
        &mul(
            &axis_rotation(1, ypr_deg[1] + 90.0),
            &axis_rotation(2, ypr_deg[0]),
        ),
        &axis_rotation(0, ypr_deg[2]),
    );
    if first_of_two {
        mul(&r, &axis_rotation(2, 180.0))
    } else {
        r
    }
}
