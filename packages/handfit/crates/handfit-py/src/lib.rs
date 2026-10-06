//! Checked numpy boundary; all borrowed inputs are copied before releasing the GIL.
use handfit::nalgebra::{Matrix3, Matrix4, SMatrix, SVector, Vector2, Vector3};
use handfit::residual::cam_from_world;
use handfit::{fit, Config, JacobianMode, Model, Pose, View};
use numpy::{PyArray1, PyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};

fn floats(array: &PyReadonlyArrayDyn<'_, f32>, shape: &[usize], name: &str) -> PyResult<Vec<f64>> {
    if array.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "{name}: expected {shape:?}, got {:?}",
            array.shape()
        )));
    }
    Ok(array.as_array().iter().map(|x| *x as f64).collect())
}
fn ints(array: &PyReadonlyArrayDyn<'_, i64>, shape: &[usize], name: &str) -> PyResult<Vec<i64>> {
    if array.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "{name}: expected {shape:?}, got {:?}",
            array.shape()
        )));
    }
    Ok(array.as_array().iter().copied().collect())
}

#[derive(Clone)]
struct Camera {
    transform: Matrix4<f64>,
    model: handfit::residual::CameraModelKind<f64>,
}

#[pyclass(module = "handfit._core")]
struct HandFitter {
    model: Model,
    config: Config,
    cameras: Vec<Camera>,
}

#[pyclass(module = "handfit._core", get_all)]
struct FitOutput {
    rotation: Py<PyArray2<f32>>,
    translation: Py<PyArray1<f32>>,
    joint_angles: Py<PyArray1<f32>>,
    e_2d: f64,
    e_dist: f64,
    e_temporal: f64,
    energy: f64,
    iterations: usize,
    converged: bool,
    termination: String,
    rigid_iterations: Vec<usize>,
    chosen_hypotheses: Vec<usize>,
    full_iterations: Vec<usize>,
    winner: Option<usize>,
}

#[pymethods]
impl HandFitter {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        axes: PyReadonlyArrayDyn<'_, f32>,
        pivots: PyReadonlyArrayDyn<'_, f32>,
        rest: PyReadonlyArrayDyn<'_, f32>,
        weights: PyReadonlyArrayDyn<'_, f32>,
        limits: PyReadonlyArrayDyn<'_, f32>,
        indices: PyReadonlyArrayDyn<'_, i64>,
        topology: PyReadonlyArrayDyn<'_, i64>,
        config: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Self> {
        let indices = ints(&indices, &[21, 3], "bone indices")?;
        let topology = ints(&topology, &[4, 22], "topology")?;
        let bone_weights = floats(&weights, &[21, 3], "bone weights")?;
        // A config that is not a vector is refused as one of the wrong length.
        let numbers: Vec<f64> = if config.ndim() == 1 {
            config.as_array().iter().copied().collect()
        } else {
            Vec::new()
        };
        let config =
            Config::from_numbers(&numbers).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let model = Model::new(
            SMatrix::from_row_slice(&floats(&axes, &[20, 3], "axes")?),
            SMatrix::from_row_slice(&floats(&pivots, &[20, 3], "pivots")?),
            SMatrix::from_row_slice(&floats(&rest, &[21, 3], "rest")?),
            SMatrix::from_row_slice(&bone_weights),
            SMatrix::from_row_slice(&floats(&limits, &[20, 2], "limits")?),
            &indices,
            &topology,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            model,
            config,
            cameras: Vec::new(),
        })
    }

    #[pyo3(signature=(cam_from_rig, focal, principal, distortion=None))]
    fn add_camera(
        &mut self,
        cam_from_rig: PyReadonlyArrayDyn<'_, f32>,
        focal: PyReadonlyArrayDyn<'_, f32>,
        principal: PyReadonlyArrayDyn<'_, f32>,
        distortion: Option<PyReadonlyArrayDyn<'_, f32>>,
    ) -> PyResult<usize> {
        let transform = Matrix4::from_row_slice(&floats(&cam_from_rig, &[4, 4], "cam_from_rig")?);
        if transform.iter().any(|v| !v.is_finite()) { return Err(PyValueError::new_err("non-finite camera transform")); }
        let focal = Vector2::from_row_slice(&floats(&focal, &[2], "focal")?);
        let principal = Vector2::from_row_slice(&floats(&principal, &[2], "principal")?);
        let distortion = distortion.map(|d| floats(&d, &[8], "distortion").map(|a| SVector::from_row_slice(&a))).transpose()?;
        let model = handfit::residual::camera_model(&focal, &principal, distortion.as_ref()).map_err(|source| {
            PyValueError::new_err(handfit::residual::ViewValidationError::InvalidCalibration { view: self.cameras.len(), source }.to_string())
        })?;
        let camera = Camera { transform, model };
        let index = self.cameras.len();
        self.cameras.push(camera);
        Ok(index)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(sides, rotations, translations, angles, camera_indices, world_from_rig, pixels, weights, distances, *, has_previous=None, central_difference=false))]
    fn fit(
        &self,
        py: Python<'_>,
        sides: PyReadonlyArrayDyn<'_, i64>,
        rotations: PyReadonlyArrayDyn<'_, f32>,
        translations: PyReadonlyArrayDyn<'_, f32>,
        angles: PyReadonlyArrayDyn<'_, f32>,
        camera_indices: PyReadonlyArrayDyn<'_, i64>,
        world_from_rig: PyReadonlyArrayDyn<'_, f32>,
        pixels: PyReadonlyArrayDyn<'_, f32>,
        weights: PyReadonlyArrayDyn<'_, f32>,
        distances: PyReadonlyArrayDyn<'_, f32>,
        has_previous: Option<PyReadonlyArrayDyn<'_, bool>>,
        central_difference: bool,
    ) -> PyResult<Vec<FitOutput>> {
        if sides.ndim() != 1 {
            return Err(PyValueError::new_err("sides must have rank 1"));
        }
        let n = sides.shape()[0];
        let warm = if let Some(present) = has_previous {
            if present.shape() != [n] {
                return Err(PyValueError::new_err(
                    "has_previous: expected one flag per hand",
                ));
            }
            present.as_array().iter().copied().collect::<Vec<_>>()
        } else {
            vec![true; n]
        };
        let sides = ints(&sides, &[n], "sides")?;
        let rotations = floats(&rotations, &[n, 3, 3], "rotations")?;
        let translations = floats(&translations, &[n, 3], "translations")?;
        let angles = floats(&angles, &[n, 22], "angles")?;
        let ids = ints(&camera_indices, &[n, 2], "camera_indices")?;
        // Each view may have its own world_from_rig. This also accepts moving-rig observations.
        let world = floats(&world_from_rig, &[n, 2, 4, 4], "world_from_rig")?;
        let pixels = floats(&pixels, &[n, 2, 21, 2], "pixels")?;
        let weights = floats(&weights, &[n, 2, 21], "weights")?;
        let distances = floats(&distances, &[n, 2, 21], "distances")?;
        if sides.iter().any(|s| *s != 0 && *s != 1) {
            return Err(PyValueError::new_err("side must be 0 or 1"));
        }
        let mut inputs = Vec::with_capacity(n);
        for h in 0..n {
            let pose = Pose {
                rotation: Matrix3::from_row_slice(&rotations[h * 9..h * 9 + 9]),
                translation: Vector3::from_row_slice(&translations[h * 3..h * 3 + 3]),
                angles: SVector::from_row_slice(&angles[h * 22..h * 22 + 22]),
            };
            let mut views = Vec::new();
            for v in 0..2 {
                let idx = 2 * h + v;
                if ids[idx] == -1 {
                    continue;
                }
                let camera = self
                    .cameras
                    .get(ids[idx] as usize)
                    .ok_or_else(|| PyValueError::new_err("invalid camera index"))?;
                let transform = Matrix4::from_row_slice(&world[idx * 16..idx * 16 + 16]);
                if transform.iter().any(|x| !x.is_finite()) {
                    return Err(PyValueError::new_err("non-finite world_from_rig"));
                }
                let (rotation, translation) = cam_from_world(&camera.transform, &transform);
                let view = View {
                    rotation,
                    translation,
                    camera: camera.model,
                    pixels: SMatrix::from_row_slice(&pixels[idx * 42..idx * 42 + 42]),
                    weights: SVector::from_row_slice(&weights[idx * 21..idx * 21 + 21]),
                    distances: SVector::from_row_slice(&distances[idx * 21..idx * 21 + 21]),
                };
                for i in 0..21 {
                    if !view.weights[i].is_finite()
                        || view.weights[i] < 0.0
                        || (view.weights[i] > 0.0
                            && (!view.distances[i].is_finite()
                                || !view.pixels[(i, 0)].is_finite()
                                || !view.pixels[(i, 1)].is_finite()))
                    {
                        return Err(PyValueError::new_err("invalid observed keypoint or weight"));
                    }
                }
                views.push(view);
            }
            inputs.push((pose, if sides[h] == 0 { 1.0 } else { -1.0 }, views, warm[h]));
        }
        let mode = if central_difference {
            JacobianMode::CentralDifference
        } else {
            JacobianMode::Analytic
        };
        let results = py.detach(|| {
            inputs
                .iter()
                .map(|(pose, mirror, views, warm)| {
                    if *warm {
                        fit(&self.model, &self.config, pose, *mirror, views, mode)
                    } else {
                        handfit::cold::initial_pose(&self.model, &self.config, *mirror, views, mode)
                    }
                })
                .collect::<Result<Vec<_>, _>>()
        });
        results
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_iter()
            .map(|r| {
                let rotation: Vec<Vec<f32>> = (0..3)
                    .map(|i| (0..3).map(|k| r.pose.rotation[(i, k)] as f32).collect())
                    .collect();
                Ok(FitOutput {
                    rotation: PyArray2::from_vec2(py, &rotation)?.unbind(),
                    translation: PyArray1::from_vec(
                        py,
                        r.pose.translation.iter().map(|x| *x as f32).collect(),
                    )
                    .unbind(),
                    joint_angles: PyArray1::from_vec(
                        py,
                        r.pose.angles.iter().map(|x| *x as f32).collect(),
                    )
                    .unbind(),
                    e_2d: r.energies[0] as f32 as f64,
                    e_dist: r.energies[1] as f32 as f64,
                    e_temporal: r.energies[2] as f32 as f64,
                    energy: r.energies[3] as f32 as f64,
                    iterations: r.iterations,
                    converged: r.converged,
                    termination: r.termination.as_str().to_string(),
                    rigid_iterations: r
                        .cold
                        .as_ref()
                        .map_or_else(Vec::new, |d| d.rigid_iterations.clone()),
                    chosen_hypotheses: r.cold.as_ref().map_or_else(Vec::new, |d| d.chosen.clone()),
                    full_iterations: r
                        .cold
                        .as_ref()
                        .map_or_else(Vec::new, |d| d.full_iterations.clone()),
                    winner: r.cold.as_ref().map(|d| d.winner),
                })
            })
            .collect()
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<HandFitter>()?;
    m.add_class::<FitOutput>()?;
    Ok(())
}
