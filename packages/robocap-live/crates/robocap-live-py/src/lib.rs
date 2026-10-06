//! Python bindings for robocap-live's hands-layer writer, exposed as `robocap_live._core`.
//!
//! Each catalog frame is copied from its `uint8[1080, 1920]` array into a Rust-owned image before it is queued, so Python can
//! reuse the array after `push` returns. [`HandsLayer`] runs robocap-live's own pipeline (`robocap_live::layer`, the
//! scheduler of `robocap-live --source replay --slam reference`) on its threads; a push hands the frameset over and releases
//! the GIL while it waits for room, so a Python decoder thread keeps running. A bad dtype, shape or layout raises
//! `ValueError`, never a panic.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use kornia_image::Image;
use kornia_staging_sensors::{CameraFrame, CaptureMeta};
use numpy::{
    PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use robocap_live::frame::{CAMERA_NAMES, FULL_SIZE, NUM_CAMERAS, RigCamera};
use robocap_live::hands::HandsConfig;
use robocap_live::layer::{HandsLayerConfig, HandsLayerWriter, LayerError};
use robocap_live::log::scene::HandOverlays;
use robocap_live::nets::ort::{OrtConfig, OrtDevice, OrtNets};
use robocap_live::nets::{HandNets, NoNets};
use robocap_live::sched::PipelineConfig;
use robocap_live::slam::ReferencePoses;

fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Bad input is a `ValueError`; a network, tracker or file failure is a `RuntimeError`.
fn layer_error(error: LayerError) -> PyErr {
    match error {
        LayerError::Input(_) => value_error(error),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Copy a C-contiguous float64 array of `rows x N` values out of Python.
fn rows<const N: usize>(
    array: &PyReadonlyArray2<'_, f64>,
    name: &str,
    count: usize,
) -> PyResult<Vec<[f64; N]>> {
    if array.shape() != [count, N] {
        return Err(value_error(format!(
            "{name} must have shape ({count}, {N}), got {:?}",
            array.shape()
        )));
    }
    // Fortran order passes as_slice() too, with its values running down the columns.
    let values: &[f64] = array.as_slice().ok().filter(|_| array.is_c_contiguous()).ok_or_else(|| value_error(format!("{name} must be C-contiguous")))?;
    Ok(values.as_chunks::<N>().0.iter().map(|row| std::array::from_fn(|i| row[i])).collect())
}

/// Copy row-major 4x4 matrices after validating their dtype, shape and layout.
fn matrices(array: &Bound<'_, PyAny>, name: &str, count: usize) -> PyResult<Vec<[f64; 16]>> {
    let poses = array
        .cast::<PyArray3<f64>>()
        .map_err(|_| value_error(format!("{name} must be a float64 array of shape (n, 4, 4)")))?
        .readonly();
    if poses.shape() != [count, 4, 4] {
        return Err(value_error(format!(
            "{name} must have shape ({count}, 4, 4), got {:?}",
            poses.shape()
        )));
    }
    let values = poses
        .as_slice()
        .ok()
        .filter(|_| poses.is_c_contiguous())
        .ok_or_else(|| value_error(format!("{name} must be C-contiguous")))?;
    Ok(values.as_chunks::<16>().0.iter().map(|row| std::array::from_fn(|i| row[i])).collect())
}

/// The six calibrated cameras of a RoboCap rig, in `CAMERA_NAMES` order.
#[pyclass(module = "robocap_live._core", frozen, skip_from_py_object)]
pub struct Rig {
    inner: robocap_live::frame::Rig,
}

#[pymethods]
impl Rig {
    /// Build the rig from the catalog's per-camera calibration (one row per camera, camera order).
    ///
    /// `resolution_wh` is (width, height) pixels, `cam_from_rig` row-major camera-from-rig 4x4 (metres), `focal` (fx, fy),
    /// `principal` (cx, cy) pixels and `fisheye62` the `[k1..k6, p1, p2]` Fisheye62 coefficients.
    #[new]
    #[pyo3(signature = (*, names, resolution_wh, cam_from_rig, focal, principal, fisheye62, source, device))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        names: Vec<String>,
        resolution_wh: PyReadonlyArray2<'_, f64>,
        cam_from_rig: &Bound<'_, PyAny>,
        focal: PyReadonlyArray2<'_, f64>,
        principal: PyReadonlyArray2<'_, f64>,
        fisheye62: PyReadonlyArray2<'_, f64>,
        source: String,
        device: String,
    ) -> PyResult<Self> {
        let count: usize = names.len();
        let poses = matrices(cam_from_rig, "cam_from_rig", count)?;
        let sizes: Vec<[f64; 2]> = rows(&resolution_wh, "resolution_wh", count)?;
        let focals: Vec<[f64; 2]> = rows(&focal, "focal", count)?;
        let principals: Vec<[f64; 2]> = rows(&principal, "principal", count)?;
        let coefficients: Vec<[f64; 8]> = rows(&fisheye62, "fisheye62", count)?;
        let cameras: Vec<RigCamera> = (0..count)
            .map(|c| {
                let [width, height] = sizes[c];
                if width.fract() != 0.0 || height.fract() != 0.0 || width < 1.0 || height < 1.0 {
                    return Err(value_error(format!(
                        "camera {c}: resolution {width}x{height} is not whole pixels"
                    )));
                }
                let pose = &poses[c];
                Ok(RigCamera {
                    name: names[c].clone(),
                    width: width as u32,
                    height: height as u32,
                    cam_from_rig: std::array::from_fn(|row| {
                        std::array::from_fn(|col| pose[4 * row + col])
                    }),
                    focal: focals[c],
                    principal: principals[c],
                    fisheye62: Some(coefficients[c]),
                })
            })
            .collect::<PyResult<_>>()?;
        let inner = robocap_live::frame::Rig {
            cameras,
            source,
            device,
        };
        inner.validate().map_err(value_error)?;
        Ok(Self { inner })
    }

    /// The camera names in index order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner
            .cameras
            .iter()
            .map(|camera| camera.name.clone())
            .collect()
    }

    /// Where the calibration came from.
    #[getter]
    fn source(&self) -> &str {
        &self.inner.source
    }

    /// `cap_a` or `cap_b`.
    #[getter]
    fn device(&self) -> &str {
        &self.inner.device
    }

    /// The rig as robocap-live's `rig.json`.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(value_error)
    }
}

/// The tracker's split of one step (or of a whole layer), milliseconds.
#[pyclass(module = "robocap_live._core", frozen, get_all, skip_from_py_object)]
#[derive(Clone, Copy)]
pub struct HandStageTimes {
    detnet_ms: f64,
    crops_ms: f64,
    keynet_ms: f64,
    fit_ms: f64,
    tracker_ms: f64,
}

/// A finished layer's totals.
#[pyclass(module = "robocap_live._core", frozen, skip_from_py_object)]
pub struct LayerSummary {
    #[pyo3(get)]
    framesets: u64,
    #[pyo3(get)]
    with_pose: u64,
    #[pyo3(get)]
    held_pose: u64,
    #[pyo3(get)]
    tracked: (u64, u64),
    #[pyo3(get)]
    reported: (u64, u64),
    #[pyo3(get)]
    scale: f64,
    #[pyo3(get)]
    scale_final: bool,
    #[pyo3(get)]
    hand_stages: HandStageTimes,
    #[pyo3(get)]
    log_worker_ms_mean: f64,
    stages: BTreeMap<String, f64>,
}

#[pymethods]
impl LayerSummary {
    /// One pipeline stage's summed wall time in milliseconds (the stages overlap).
    fn stage_total_ms(&self, stage: &str) -> PyResult<f64> {
        self.stages.get(stage).copied().ok_or_else(|| {
            value_error(format!(
                "no stage {stage:?}: {:?}",
                self.stages.keys().collect::<Vec<_>>()
            ))
        })
    }
}

/// One camera's frame for the core: a Rust-owned snapshot in row-major order, for every NumPy layout.
fn camera_frame(object: &Bound<'_, PyAny>, camera: usize, timestamp_ns: i64) -> PyResult<CameraFrame> {
    let array = object
        .cast::<PyArray2<u8>>()
        .map_err(|_| value_error(format!("frame {camera} must be a 2-D uint8 numpy array")))?;
    let readonly = array.readonly();
    if readonly.shape() != [FULL_SIZE.height, FULL_SIZE.width] {
        return Err(value_error(format!(
            "frame {camera} has shape {:?}, expected ({}, {})",
            readonly.shape(),
            FULL_SIZE.height,
            FULL_SIZE.width
        )));
    }
    let pixels = readonly
        .as_array()
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset()
        .0;
    let image = Image::new(FULL_SIZE, pixels).map_err(value_error)?;
    Ok(CameraFrame {
        meta: CaptureMeta {
            sequence: 0,
            timestamp_ns,
            camera_slot: camera,
        },
        full: Arc::new(image),
    })
}

fn ort_device(device: &str) -> PyResult<OrtDevice> {
    match device {
        "auto" => Ok(OrtDevice::Auto),
        "cpu" => Ok(OrtDevice::Cpu),
        "cuda" => Ok(OrtDevice::Cuda(0)),
        other => other
            .strip_prefix("cuda:")
            .and_then(|index| index.parse().ok())
            .map(OrtDevice::Cuda)
            .ok_or_else(|| {
                value_error(format!(
                    "device {other:?}: expected auto, cpu, cuda or cuda:<n>"
                ))
            }),
    }
}

/// One segment's hands layer: push every frameset in time order, then `finish`.
#[pyclass(module = "robocap_live._core")]
pub struct HandsLayer {
    writer: Option<HandsLayerWriter>,
    nets: String,
}

impl HandsLayer {
    fn writer(&mut self) -> PyResult<&mut HandsLayerWriter> {
        self.writer
            .as_mut()
            .ok_or_else(|| value_error("the layer is finished"))
    }
}

#[pymethods]
impl HandsLayer {
    /// Load the networks, build the tracker for `rig`, open `output` (truncated if it exists: never a registered layer) and
    /// start robocap-live's pipeline on the framesets to come.
    ///
    /// `reference_t_ns` and `reference_world_from_rig` are the segment's SLAM poses (row-major 4x4, NaN rows for none) keyed
    /// by frameset time: each frameset takes the pose within 2 ms of its time, and one without a pose the newest earlier one, as
    /// the runtime's `--slam reference` replay does. `nets` is `ort` (DetNet and KeyNet from `models_dir` on ONNX Runtime, the
    /// library from `ort_dylib`, else `ORT_DYLIB_PATH`; `device` auto, cpu, cuda or cuda:<n>) or `none` (no hands: tests).
    #[new]
    #[pyo3(signature = (
        rig, output, recording_id, reference_t_ns, reference_world_from_rig, *, nets = "ort", models_dir = None, device = "auto",
        ort_dylib = None, ort_threads = 0, overlays = "debug",
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        rig: PyRef<'_, Rig>,
        output: PathBuf,
        recording_id: String,
        reference_t_ns: PyReadonlyArray1<'_, i64>,
        reference_world_from_rig: &Bound<'_, PyAny>,
        nets: &str,
        models_dir: Option<PathBuf>,
        device: &str,
        ort_dylib: Option<PathBuf>,
        ort_threads: usize,
        overlays: &str,
    ) -> PyResult<Self> {
        let hand_overlays: HandOverlays = overlays.parse().map_err(value_error)?;
        let reference = reference_poses(&reference_t_ns, reference_world_from_rig)?;
        let rig = rig.inner.clone();
        let nets: Box<dyn HandNets> = match nets {
            "ort" => {
                let dir = models_dir.ok_or_else(|| {
                    value_error("nets='ort' needs models_dir (detnet_full.onnx, keynet.onnx)")
                })?;
                let options = OrtConfig {
                    device: ort_device(device)?,
                    dylib: ort_dylib,
                    intra_threads: ort_threads,
                };
                Box::new(
                    py.detach(|| OrtNets::new(&dir, &options))
                        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
                )
            }
            "none" => Box::new(NoNets),
            other => return Err(value_error(format!("nets {other:?}: expected ort or none"))),
        };
        let hands = HandsConfig {
            // Offline, as a lossless replay: switch to the calibrated scale on the frameset that finishes the solve, the same
            // one every run.
            scale_wait: true,
            ..HandsConfig::default()
        };
        let config = HandsLayerConfig {
            output,
            recording_id,
            hand_overlays,
            hands,
            downsample_threads: PipelineConfig::default().downsample_threads,
        };
        let description = nets.describe();
        let writer = HandsLayerWriter::new(&rig, nets, reference, config).map_err(layer_error)?;
        Ok(Self {
            nets: description,
            writer: Some(writer),
        })
    }

    /// The networks in use (ONNX Runtime names the device it actually got).
    #[getter]
    fn nets(&self) -> &str {
        &self.nets
    }

    /// Hand one frameset to the pipeline and return its index; blocks (without the GIL) while the pipeline is behind.
    ///
    /// `frames` holds six `uint8[1080, 1920]` luma arrays in camera order, None for a missing camera. Every array is copied
    /// before it is queued; the caller may reuse or mutate it after this returns. `t_ns` is the frameset's catalog
    /// `video_time` and `camera_t_ns` each present camera's own (default `t_ns`).
    #[pyo3(signature = (t_ns, frames, camera_t_ns = None))]
    fn push(
        &mut self,
        py: Python<'_>,
        t_ns: i64,
        frames: Vec<Option<Bound<'_, PyAny>>>,
        camera_t_ns: Option<Vec<i64>>,
    ) -> PyResult<u64> {
        if frames.len() != NUM_CAMERAS {
            return Err(value_error(format!(
                "{} frames, expected {NUM_CAMERAS} ({})",
                frames.len(),
                CAMERA_NAMES.join(", ")
            )));
        }
        if camera_t_ns
            .as_ref()
            .is_some_and(|times| times.len() != NUM_CAMERAS)
        {
            return Err(value_error(format!(
                "camera_t_ns must hold {NUM_CAMERAS} times"
            )));
        }
        let writer = self.writer()?;
        let mut cameras: [Option<CameraFrame>; NUM_CAMERAS] = Default::default();
        for (camera, (slot, frame)) in cameras.iter_mut().zip(&frames).enumerate() {
            if let Some(frame) = frame {
                let timestamp_ns: i64 = camera_t_ns.as_ref().map_or(t_ns, |times| times[camera]);
                *slot = Some(camera_frame(frame, camera, timestamp_ns)?);
            }
        }
        #[cfg(test)]
        tests::retain_submitted(&cameras);
        py.detach(|| writer.push(t_ns, cameras))
            .map_err(layer_error)
    }

    /// End the input, drain the pipeline, close the file and return the layer's totals; the layer takes no frameset after this.
    fn finish(&mut self, py: Python<'_>) -> PyResult<LayerSummary> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| value_error("the layer is finished"))?;
        let inner = py.detach(|| writer.finish()).map_err(layer_error)?;
        let counts = inner.counts;
        let t = counts.hand_timings;
        Ok(LayerSummary {
            framesets: counts.framesets,
            with_pose: counts.with_pose,
            held_pose: counts.held_pose,
            tracked: (counts.tracked[0], counts.tracked[1]),
            reported: (counts.reported[0], counts.reported[1]),
            scale: counts.scale,
            scale_final: counts.scale_final,
            hand_stages: HandStageTimes {
                detnet_ms: t.detnet_ms,
                crops_ms: t.crops_ms,
                keynet_ms: t.keynet_ms,
                fit_ms: t.fit_ms,
                tracker_ms: t.tracker_ms,
            },
            log_worker_ms_mean: inner.log.worker_ms_mean,
            stages: inner
                .run
                .stages
                .into_iter()
                .map(|(name, stage)| (name, stage.mean * stage.count as f64))
                .collect(),
        })
    }

    /// Stop the pipeline without finishing the layer (its file is left incomplete); a no-op once finished or aborted. Waits for
    /// the pipeline's threads without the GIL, which a plain drop of the object would hold.
    fn abort(&mut self, py: Python<'_>) {
        if let Some(writer) = self.writer.take() {
            py.detach(|| drop(writer));
        }
    }
}

/// The reference pose table: frameset times and row-major 4x4 poses, NaN rows for framesets without one.
fn reference_poses(
    t_ns: &PyReadonlyArray1<'_, i64>,
    world_from_rig: &Bound<'_, PyAny>,
) -> PyResult<ReferencePoses> {
    let times: &[i64] = t_ns
        .as_slice()
        .map_err(|_| value_error("reference_t_ns must be a contiguous int64 array"))?;
    let poses = matrices(world_from_rig, "reference_world_from_rig", times.len())?;
    Ok(ReferencePoses::new(
        times.iter().copied().zip(poses).collect(),
        0,
        None,
    ))
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("CAMERA_NAMES", CAMERA_NAMES.to_vec())?;
    module.add("FULL_WIDTH", FULL_SIZE.width)?;
    module.add("FULL_HEIGHT", FULL_SIZE.height)?;
    module.add_class::<Rig>()?;
    module.add_class::<HandsLayer>()?;
    module.add_class::<LayerSummary>()?;
    module.add_class::<HandStageTimes>()?;
    Ok(())
}

#[cfg(test)]
mod tests;
