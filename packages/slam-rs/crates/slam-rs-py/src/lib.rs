//! Python bindings for the slam-rs core, exposed as `slam_rs._core`.
//!
//! Every array crossing the boundary is validated here and copied into owned
//! Rust storage, so the GIL is released for the whole core call and a producer
//! thread can keep decoding. A bad dtype, rank, shape or memory layout raises
//! `ValueError`, never a panic.

use numpy::{
    PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods,
    ToPyArray,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use slam_rs::FrontendLane;
use slam_rs::calib::Calibration as CoreCalibration;
use slam_rs::config::VioConfig as CoreVioConfig;
use slam_rs::frontend::detect::CellGrid;
use slam_rs::frontend::flow::{
    FlowFrame as CoreFlowFrame, FrameToFrameOpticalFlow, FrontendError, FrontendOptions,
    PosePrediction,
};
use slam_rs::image::ImageU16;

/// Map any core error onto `ValueError`, which is what every refusal here is.
fn value_error<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// basalt's `VioConfig`, as `data/**/*_config.json` carries it.
///
/// A fresh instance is `VioConfig::VioConfig()`, the same defaults the C++
/// constructor sets; [`VioConfig::from_json`] then overwrites whatever keys a
/// file names, leaving the rest alone, as cereal does.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct VioConfig {
    inner: CoreVioConfig,
}

#[pymethods]
impl VioConfig {
    /// basalt's own defaults (`src/utils/vio_config.cpp:47-128`).
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreVioConfig::default(),
        }
    }

    /// Read one of basalt's config files; keys it omits keep their default.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreVioConfig::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Write the config back in basalt's shape, `value0` wrapper and all.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(value_error)
    }

    /// `config.optical_flow_image_safe_radius`: the circular mask that hides a
    /// fisheye's black corners, in pixels. Zero disables it.
    ///
    /// This is the one frontend field that differs per device rather than per
    /// dataset — 472 on Index, 340 on G2 — and the reference manifest carries it
    /// per segment, so it is settable here rather than only through a file.
    #[getter]
    fn optical_flow_image_safe_radius(&self) -> f32 {
        self.inner.optical_flow_image_safe_radius
    }

    #[setter]
    fn set_optical_flow_image_safe_radius(&mut self, radius: f32) {
        self.inner.optical_flow_image_safe_radius = radius;
    }

    fn __repr__(&self) -> String {
        format!(
            "VioConfig(optical_flow_type={:?}, optical_flow_pattern={}, optical_flow_levels={}, optical_flow_image_safe_radius={})",
            self.inner.optical_flow_type,
            self.inner.optical_flow_pattern,
            self.inner.optical_flow_levels,
            self.inner.optical_flow_image_safe_radius
        )
    }
}

/// basalt's camera-IMU calibration: extrinsics, intrinsics and the noise model.
#[pyclass(module = "slam_rs._core", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Calibration {
    inner: CoreCalibration<f64>,
}

#[pymethods]
impl Calibration {
    /// Read one of basalt's calibration files.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: CoreCalibration::<f64>::from_json_str(text).map_err(value_error)?,
        })
    }

    /// Write the calibration back in basalt's shape, `value0` wrapper and all.
    fn to_json(&self) -> PyResult<String> {
        self.inner.to_json_string().map_err(value_error)
    }

    /// Cameras on the rig.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// Each camera's `(width, height)` in pixels, in rig order.
    #[getter]
    fn resolution(&self) -> Vec<(u32, u32)> {
        self.inner
            .resolution
            .iter()
            .map(|size| (size[0], size[1]))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("Calibration(camera_count={})", self.inner.camera_count())
    }
}

/// The frontend options both entry points build, from the two knobs they expose.
///
/// Everything else in `FrontendOptions` is a property of the port rather than of
/// a run, so `None` means the default rather than "unset". `threads` is read by
/// `CpuPatchTracker::new` alone, so it does nothing on the GPU lane.
fn frontend_options(threads: usize, max_keypoints: Option<usize>) -> FrontendOptions {
    let defaults: FrontendOptions = FrontendOptions::default();
    FrontendOptions {
        threads,
        max_keypoints: max_keypoints.unwrap_or(defaults.max_keypoints),
        ..defaults
    }
}

/// The C-contiguous elements of an array borrowed out of Python.
///
/// Both halves of one check, so the refusal has one wording. `as_slice` alone
/// accepts Fortran order, whose bytes run down the columns and would transpose
/// every row-major read here, so the C flag is asked explicitly; `as_slice`'s
/// own refusal — which the flag has already ruled out — lands on the same
/// sentence rather than a second copy of it. `what` names the array in the
/// message and `hint` is what the caller should wrap.
fn contiguous<'a, T, D>(
    readonly: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
    hint: &str,
) -> PyResult<&'a [T]>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    let refused = || {
        PyValueError::new_err(format!(
            "{what} must be C-contiguous; pass numpy.ascontiguousarray({hint})"
        ))
    };
    if !readonly.is_c_contiguous() {
        return Err(refused());
    }
    readonly.as_slice().map_err(|_| refused())
}

/// Borrow one `uint8[h, w]` array out of Python: rank and dtype checked.
///
/// The two consumers differ only in what they do with the bytes — [`Vio::track`]
/// copies them into a `Vec`, [`OpticalFlow::process`] widens them straight into a
/// reused [`ImageU16`] — so the checks live here and neither repeats them. The
/// layout is [`gray_pixels`]' half of the same pair.
fn gray_array<'py>(
    object: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<PyReadonlyArray2<'py, u8>> {
    let array: &Bound<'py, PyArray2<u8>> = object.cast::<PyArray2<u8>>().map_err(|_| {
        PyValueError::new_err(format!("image {index} must be a 2-D uint8 numpy array"))
    })?;
    Ok(array.readonly())
}

/// The pixels, width and height of an array [`gray_array`] has accepted.
fn gray_pixels<'a>(
    readonly: &'a PyReadonlyArray2<'_, u8>,
    index: usize,
) -> PyResult<(&'a [u8], usize, usize)> {
    let shape: &[usize] = readonly.shape();
    let pixels: &[u8] = contiguous(readonly, &format!("image {index}"), "image")?;
    Ok((pixels, shape[1], shape[0]))
}

/// One camera's keypoints for one frameset, copied out of the frontend.
///
/// Owned rather than borrowed: the frontend's buffers are reused frame to frame,
/// so a view would change under the caller on the next `process`.
#[derive(Debug, Default)]
struct CameraKeypoints {
    /// Keypoint ids, ascending.
    ids: Vec<i64>,
    /// `[m00, m01, tx, m10, m11, ty]` per keypoint, keypoint index slow-varying.
    ///
    /// The keypoint's pixel is `[tx, ty]`, so this is where a position comes
    /// from as well ([`FlowFrame::positions`]).
    transforms: Vec<f32>,
    /// Occupancy counts, row-major over the frame's grid.
    occupancy: Vec<i32>,
    /// Ids handed out on this frameset: detections plus stereo matches.
    num_new: usize,
}

/// What one [`OpticalFlow::process`] call produced.
#[pyclass(module = "slam_rs._core", frozen)]
#[derive(Debug)]
pub struct FlowFrame {
    t_ns: i64,
    grid: CellGrid,
    cameras: Vec<CameraKeypoints>,
}

impl FlowFrame {
    /// One camera's keypoints, or an `IndexError`.
    fn camera(&self, camera: usize) -> PyResult<&CameraKeypoints> {
        self.cameras.get(camera).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "camera {camera} is past the end of a {}-camera rig",
                self.cameras.len()
            ))
        })
    }
}

#[pymethods]
impl FlowFrame {
    /// Frameset timestamp, nanoseconds.
    #[getter]
    fn t_ns(&self) -> i64 {
        self.t_ns
    }

    /// Cameras on the rig.
    #[getter]
    fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// `optical_flow_detection_grid_size`: the occupancy cell's side in pixels.
    #[getter]
    fn cell_size(&self) -> usize {
        self.grid.cell
    }

    /// `(x_start, y_start)`: the top-left corner of cell `(0, 0)` in pixels.
    ///
    /// basalt centres the grid on the frame, so the leftover `width % cell` is
    /// split between the two edges (`keypoints.cpp:140-144`).
    #[getter]
    fn cell_origin(&self) -> (usize, usize) {
        (self.grid.x_start, self.grid.y_start)
    }

    /// One camera's keypoint ids, ascending: `int64[n]`.
    fn ids<'py>(&self, py: Python<'py>, camera: usize) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(self.camera(camera)?.ids.to_pyarray(py))
    }

    /// One camera's keypoint positions in pixels: `float32[n, 2]`.
    ///
    /// Read out of the warps' translation column rather than stored beside them:
    /// they are the same two floats.
    fn positions<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        let positions: Vec<f32> = keypoints
            .transforms
            .chunks_exact(6)
            .flat_map(|warp| [warp[2], warp[5]])
            .collect();
        positions.to_pyarray(py).reshape((keypoints.ids.len(), 2))
    }

    /// One camera's 2x3 warps, `[[m00, m01, tx], [m10, m11, ty]]`: `float32[n, 2, 3]`.
    fn transforms<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        keypoints
            .transforms
            .to_pyarray(py)
            .reshape((keypoints.ids.len(), 2, 3))
    }

    /// One camera's occupancy counts: `int32[rows, columns]`.
    ///
    /// The grid is camera 0's, as basalt's is (`frame_to_frame_optical_flow.h:119`),
    /// whatever the camera's own resolution is.
    fn occupancy<'py>(
        &self,
        py: Python<'py>,
        camera: usize,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let keypoints: &CameraKeypoints = self.camera(camera)?;
        keypoints
            .occupancy
            .to_pyarray(py)
            .reshape((self.grid.rows, self.grid.columns))
    }

    /// Ids one camera gained on this frameset: detections plus stereo matches.
    fn num_new(&self, camera: usize) -> PyResult<usize> {
        Ok(self.camera(camera)?.num_new)
    }

    /// Keypoints one camera carries.
    fn num_tracks(&self, camera: usize) -> PyResult<usize> {
        Ok(self.camera(camera)?.ids.len())
    }

    fn __repr__(&self) -> String {
        let counts: Vec<usize> = self.cameras.iter().map(|camera| camera.ids.len()).collect();
        format!("FlowFrame(t_ns={}, keypoints={counts:?})", self.t_ns)
    }
}

/// basalt's `FrameToFrameOpticalFlow`, driven one frameset at a time.
///
/// Pattern 51 only: every shipped config sets `optical_flow_pattern = 51`, and a
/// config asking for another one is refused rather than silently tracked with
/// the wrong pattern.
#[pyclass(module = "slam_rs._core")]
pub struct OpticalFlow {
    inner: FrontendLane,
    /// The widened frameset, reused so a steady stream never reallocates.
    images: Vec<ImageU16>,
}

#[pymethods]
impl OpticalFlow {
    /// Build a frontend for one rig.
    ///
    /// One of basalt's own files arrives through [`Calibration::from_json`] and
    /// [`VioConfig::from_json`], so this takes the two classes only.
    #[new]
    #[pyo3(signature = (calibration, config, *, threads = 1, max_keypoints = None))]
    fn new(
        calibration: PyRef<'_, Calibration>,
        config: PyRef<'_, VioConfig>,
        threads: usize,
        max_keypoints: Option<usize>,
    ) -> PyResult<Self> {
        // The standalone frontend is the CPU lane: the GPU choice belongs on
        // `Vio`, which is what a caller runs a whole pipeline through.
        let inner: FrontendLane = FrontendLane::Cpu(
            FrameToFrameOpticalFlow::new(
                config.inner.clone(),
                &calibration.inner,
                frontend_options(threads, max_keypoints),
            )
            .map_err(value_error)?,
        );
        let cameras: usize = inner.camera_count();
        Ok(Self {
            inner,
            images: vec![ImageU16::default(); cameras],
        })
    }

    /// Cameras this frontend expects in every frameset.
    #[getter]
    fn camera_count(&self) -> usize {
        self.inner.camera_count()
    }

    /// The next keypoint id that will be handed out.
    #[getter]
    fn last_keypoint_id(&self) -> u64 {
        self.inner.last_keypoint_id()
    }

    /// Timestamp of the last accepted frameset, or None before the first.
    ///
    /// The core's own clock, not a copy of it: the two cannot drift apart when a
    /// frameset is refused.
    #[getter]
    fn t_ns(&self) -> Option<i64> {
        self.inner.t_ns()
    }

    /// Track and detect on one frameset of `camera_count` `uint8[h, w]` images.
    ///
    /// The images are widened into the frontend's own buffers while the GIL is
    /// held and the whole tracking pass then runs without it, so a decoder thread
    /// keeps running. A frameset the frontend refuses leaves it exactly as the
    /// last accepted one did.
    fn process(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        images: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<FlowFrame> {
        // Refuse the frameset's width before widening anything: the core would
        // refuse it too, but only after every image had been copied in. The
        // buffer is sized once, at construction, so nothing here resizes it.
        let cameras: usize = self.inner.camera_count();
        if images.len() != cameras {
            return Err(value_error(FrontendError::CameraCountMismatch {
                expected: cameras,
                actual: images.len(),
            }));
        }
        // Widen here, where the GIL is still held: the numpy borrow cannot
        // outlive it. `fill_from_u8_strided` reuses the buffer whenever the
        // geometry is unchanged, which is every frame after the first.
        for (index, image) in images.iter().enumerate() {
            let readonly: PyReadonlyArray2<'_, u8> = gray_array(image, index)?;
            let (pixels, width, height) = gray_pixels(&readonly, index)?;
            self.images[index]
                .fill_from_u8_strided(pixels, width, height, width)
                .map_err(|error| PyValueError::new_err(format!("image {index}: {error}")))?;
        }

        let Self { inner, images } = self;
        py.detach(|| -> Result<FlowFrame, ProcessError> {
            inner
                .process_frame(t_ns, images, &PosePrediction::default(), &[])
                .map_err(ProcessError::Frontend)?;
            flow_frame(inner, t_ns)
        })
        .map_err(PyErr::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpticalFlow(camera_count={}, frame_counter={})",
            self.inner.camera_count(),
            self.inner.frame_counter()
        )
    }
}

/// What a `process` call can refuse once the GIL is already released.
enum ProcessError {
    /// The frontend itself refused the frameset.
    Frontend(FrontendError),
    /// A keypoint id outgrew the `int64` the boundary hands to numpy.
    IdOverflow(u64),
}

impl From<ProcessError> for PyErr {
    fn from(error: ProcessError) -> Self {
        match error {
            ProcessError::Frontend(inner) => value_error(inner),
            ProcessError::IdOverflow(id) => {
                PyValueError::new_err(format!("keypoint id {id} does not fit in an int64"))
            }
        }
    }
}

/// Copy the frontend's committed frame into an owned [`FlowFrame`].
///
/// A pure read: `num_new` comes from the frontend's own watermark
/// ([`FrameToFrameOpticalFlow::last_keypoint_id_before_frame`]), so this answers
/// the same thing inside the `process` that produced the frame and afterwards
/// through [`Vio::flow_frame`].
///
/// `t_ns` is the timestamp `process_frame` has just accepted. The core carries
/// its own as an `Option` — `None` until the first frameset commits — and this
/// runs only after a commit, so taking the value the caller passed keeps the
/// Python-facing `FlowFrame.t_ns` a plain `int` with no impossible branch.
fn flow_frame(flow: &FrontendLane, t_ns: i64) -> Result<FlowFrame, ProcessError> {
    let previous_last_id: u64 = flow.last_keypoint_id_before_frame();
    let grid: CellGrid = flow.occupancy_grid();
    let frame: &CoreFlowFrame = flow.frame();
    let mut cameras: Vec<CameraKeypoints> = Vec::with_capacity(frame.cameras.len());
    for (index, keypoints) in frame.cameras.iter().enumerate() {
        let count: usize = keypoints.len();
        let mut ids: Vec<i64> = Vec::with_capacity(count);
        let mut transforms: Vec<f32> = Vec::with_capacity(6 * count);
        let mut num_new: usize = 0;
        for (slot, id) in keypoints.ids.iter().enumerate() {
            ids.push(i64::try_from(id.0).map_err(|_| ProcessError::IdOverflow(id.0))?);
            // Ids are handed out in one ascending run, so everything at or above
            // the counter's value before the call is new on this frameset.
            if id.0 >= previous_last_id {
                num_new += 1;
            }
            // `[m00, m01, m10, m11, tx, ty]`, into row-major 2x3.
            let warp: [f32; 6] = keypoints.transforms.coefficients(slot);
            transforms.extend_from_slice(&[warp[0], warp[1], warp[4], warp[2], warp[3], warp[5]]);
        }
        cameras.push(CameraKeypoints {
            ids,
            transforms,
            occupancy: flow.cell_counts(index).to_vec(),
            num_new,
        });
    }
    Ok(FlowFrame {
        t_ns,
        grid,
        cameras,
    })
}

/// The compiled core of the `slam_rs` package.
#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", slam_rs::VERSION)?;
    module.add_class::<Calibration>()?;
    module.add_class::<VioConfig>()?;
    module.add_class::<OpticalFlow>()?;
    module.add_class::<FlowFrame>()?;
    Ok(())
}
