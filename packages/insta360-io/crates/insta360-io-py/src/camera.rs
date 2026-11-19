//! A Mei camera as a calibration states it: a model name, `fx fy cx cy` and the coefficients in that model's order.
use kornia_staging_3d::camera::{CameraModel, Mei, MeiDistortion};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A unified (Mei) sphere camera with the Insta360 factory distortion.
///
/// Built from what a calibration states (`model` `mei_v3` or `mei_v6`, `fx fy cx cy`, `coefficients`
/// `xi, half_field_rad, k1, k2, k3, p1, p2` or `xi, half_field_rad, d0 .. d12`), or read from a clip's lens.
#[pyclass(module = "insta360_io", frozen, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct MeiCamera {
    pub camera: Mei<f64>,
}

/// The camera of `model` with `[fx, fy, cx, cy]` and `coefficients` in that model's order.
pub fn mei_camera(
    model: &str,
    focal_principal: [f64; 4],
    coefficients: &[f64],
) -> Result<Mei<f64>, String> {
    let [fx, fy, cx, cy] = focal_principal;
    let (&[xi, half_field_rad], rest) = coefficients.split_first_chunk::<2>().ok_or_else(|| {
        format!(
            "{model} needs xi and half_field_rad first; got {} coefficients",
            coefficients.len()
        )
    })?;
    let distortion = match (model, rest) {
        ("mei_v3", &[k1, k2, k3, p1, p2]) => MeiDistortion::V3 {
            k: [k1, k2, k3],
            p: [p1, p2],
        },
        ("mei_v6", rest) if rest.len() == 13 => MeiDistortion::V6 {
            d: rest.try_into().expect("13 coefficients"),
        },
        ("mei_v3" | "mei_v6", _) => {
            return Err(format!(
                "{model} takes {} coefficients; got {}",
                if model == "mei_v3" { 7 } else { 15 },
                coefficients.len()
            ))
        }
        _ => return Err(format!("{model:?} is not a Mei model (mei_v3, mei_v6)")),
    };
    Mei::new([fx, fy, cx, cy, xi], half_field_rad, distortion).map_err(|_| {
        format!("{model} [{fx}, {fy}, {cx}, {cy}] {coefficients:?} is not a valid Mei camera")
    })
}

#[pymethods]
impl MeiCamera {
    /// Raises `ValueError` on an unknown model, a wrong coefficient count or invalid values.
    #[new]
    fn new(
        model: &str,
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        coefficients: Vec<f64>,
    ) -> PyResult<Self> {
        mei_camera(model, [fx, fy, cx, cy], &coefficients)
            .map(|camera| Self { camera })
            .map_err(PyValueError::new_err)
    }

    /// `mei_v3` or `mei_v6`.
    #[getter]
    fn model(&self) -> &'static str {
        match self.camera.distortion() {
            MeiDistortion::V3 { .. } => "mei_v3",
            MeiDistortion::V6 { .. } => "mei_v6",
        }
    }

    /// Focal along x, pixels (Mei form: pixels per unit of the sphere-projected point).
    #[getter]
    fn fx(&self) -> f64 {
        self.camera.params()[0]
    }

    /// Focal along y, pixels.
    #[getter]
    fn fy(&self) -> f64 {
        self.camera.params()[1]
    }

    /// Principal point x, pixels (pixel centres at integers).
    #[getter]
    fn cx(&self) -> f64 {
        self.camera.params()[2]
    }

    /// Principal point y, pixels.
    #[getter]
    fn cy(&self) -> f64 {
        self.camera.params()[3]
    }

    /// The coefficients in the order of `model`.
    #[getter]
    fn coefficients(&self) -> Vec<f64> {
        let head = [self.camera.params()[4], self.camera.half_field_rad()];
        match self.camera.distortion() {
            MeiDistortion::V3 { k, p } => head.into_iter().chain(k).chain(p).collect(),
            MeiDistortion::V6 { d } => head.into_iter().chain(d).collect(),
        }
    }

    /// Project `float64[n, 3]` points in the camera frame (any norm) to `float64[n, 2]` pixels and a `bool[n]`
    /// validity: false where the forward model rejects the point (outside the half field, zero or non-finite),
    /// and the pixel is NaN there. Pixel centres at integers.
    fn project<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<bool>>)> {
        let shape = points.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyValueError::new_err(format!(
                "points must be float64[n, 3]; got shape {shape:?}"
            )));
        }
        let view = points.as_array();
        let n = shape[0];
        let camera = self.camera;
        let (pixels, valid) = py.detach(|| {
            let mut pixels = Array2::<f64>::from_elem((n, 2), f64::NAN);
            let mut valid = Array1::<bool>::from_elem(n, false);
            for (i, row) in view.outer_iter().enumerate() {
                if let Ok([u, v]) = camera.project([row[0], row[1], row[2]]) {
                    pixels[[i, 0]] = u;
                    pixels[[i, 1]] = v;
                    valid[i] = true;
                }
            }
            (pixels, valid)
        });
        Ok((pixels.into_pyarray(py), valid.into_pyarray(py)))
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<MeiCamera>()
}
