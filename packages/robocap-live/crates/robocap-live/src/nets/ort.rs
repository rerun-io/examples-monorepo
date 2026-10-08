//! DetNet and KeyNet through ONNX Runtime (`ort` 2.0.0-rc.11, FP32): CUDA on hosts with an NVIDIA GPU, CPU otherwise.
//!
//! Built only with the `ort` cargo feature, so the cap build carries no ONNX Runtime. The library is loaded at run time
//! (`load-dynamic`, as kornia-rs's `examples/onnx` does): from [`OrtConfig::dylib`], else from `ORT_DYLIB_PATH`. On
//! a CUDA host both the library and the CUDA libraries come from the monorepo's `handtrack` pixi env (no pip):
//!
//! ```text
//! ENV=<checkout>/.pixi/envs/handtrack
//! ORT_DYLIB_PATH=$ENV/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.29.0
//! LD_LIBRARY_PATH=$ENV/lib            # cuDNN 9, cuBLAS, cudart 13 for libonnxruntime_providers_cuda.so
//! ```
//!
//! The graphs are `detnet_full.onnx` (the 640x480 frame / 255 with the 4x4 pool inside, exactly handtrack's `DetNetDetector`)
//! and `keynet.onnx`, both with a dynamic batch axis, so a call runs its whole batch at once. TF32 is off on CUDA so the
//! outputs stay at FP32 parity with PyTorch.

use std::path::{Path, PathBuf};

use ::ort::ep::{CPU, CUDA, ExecutionProviderDispatch};
use ::ort::session::Session;
use ::ort::session::builder::GraphOptimizationLevel;
use ::ort::value::TensorRef;

use super::{
    CROP_LEN, DETNET_HEIGHT, DETNET_WIDTH, DISTANCE_LEN, DetNetRaw, HEATMAP_LEN, HandNets, KEYNET_CROP, KeyNetRaw, NUM_LANDMARKS, NetFrame, NetsError,
};

/// DetNet graph inside a models directory.
pub const DETNET_ONNX: &str = "detnet_full.onnx";
/// KeyNet graph inside a models directory.
pub const KEYNET_ONNX: &str = "keynet.onnx";

const FRAME_LEN: usize = DETNET_WIDTH * DETNET_HEIGHT;

/// Where the graphs run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrtDevice {
    /// CUDA on this GPU; an error if the CUDA execution provider cannot be registered.
    Cuda(i32),
    /// CUDA on GPU 0 when it registers, else the CPU.
    Auto,
    /// The CPU.
    Cpu,
}

/// How [`OrtNets`] loads ONNX Runtime and the graphs.
#[derive(Clone, Debug)]
pub struct OrtConfig {
    /// Where the graphs run.
    pub device: OrtDevice,
    /// `libonnxruntime.so`; `None` uses `ORT_DYLIB_PATH`.
    pub dylib: Option<PathBuf>,
    /// CPU threads per session (0 = ONNX Runtime's default).
    pub intra_threads: usize,
}

impl Default for OrtConfig {
    fn default() -> Self {
        Self { device: OrtDevice::Auto, dylib: None, intra_threads: 0 }
    }
}

fn load_error(what: &str, error: impl std::fmt::Display) -> NetsError {
    NetsError::Load { what: what.to_string(), message: error.to_string() }
}

fn run_error(net: &'static str, error: impl std::fmt::Display) -> NetsError {
    NetsError::Run { net, message: error.to_string() }
}

/// [`HandNets`] on ONNX Runtime.
pub struct OrtNets {
    detnet: Session,
    keynet: Session,
    description: String,
    frames: Vec<f32>,
}

fn session(path: &Path, providers: &[ExecutionProviderDispatch], threads: usize) -> Result<Session, NetsError> {
    let what: String = path.display().to_string();
    let mut builder = Session::builder().map_err(|e| load_error(&what, e))?;
    builder = builder.with_optimization_level(GraphOptimizationLevel::Level3).map_err(|e| load_error(&what, e))?;
    if threads > 0 {
        builder = builder.with_intra_threads(threads).map_err(|e| load_error(&what, e))?;
    }
    builder = builder.with_execution_providers(providers).map_err(|e| load_error(&what, e))?;
    builder.commit_from_file(path).map_err(|e| load_error(&what, e))
}

impl OrtNets {
    /// Loads ONNX Runtime and the two graphs ([`DETNET_ONNX`], [`KEYNET_ONNX`]) from `models_dir`.
    ///
    /// # Arguments
    ///
    /// * `models_dir` - the directory holding the `.onnx` files.
    /// * `options` - device, library path and threads.
    ///
    /// # Returns
    ///
    /// The backend; [`HandNets::describe`] names the device that was actually used.
    ///
    /// # Errors
    ///
    /// [`NetsError::Load`] when no library path is given or it cannot be loaded, when a graph cannot be read, or when
    /// [`OrtDevice::Cuda`] was asked for and the CUDA execution provider does not register.
    pub fn new(models_dir: impl AsRef<Path>, options: &OrtConfig) -> Result<Self, NetsError> {
        let dylib: PathBuf = match &options.dylib {
            Some(path) => path.clone(),
            None => std::env::var_os("ORT_DYLIB_PATH")
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .ok_or_else(|| load_error("onnxruntime", "set ORT_DYLIB_PATH (or OrtConfig::dylib) to libonnxruntime.so"))?,
        };
        let builder = ::ort::init_from(&dylib).map_err(|e| load_error(&dylib.display().to_string(), e))?;
        builder.with_name("robocap-live").commit();
        let dir: &Path = models_dir.as_ref();
        let cuda = |device: i32| CUDA::default().with_device_id(device).with_tf32(false).build().error_on_failure();
        let pair = |providers: &[ExecutionProviderDispatch]| -> Result<(Session, Session), NetsError> {
            Ok((session(&dir.join(DETNET_ONNX), providers, options.intra_threads)?, session(&dir.join(KEYNET_ONNX), providers, options.intra_threads)?))
        };
        let cpu = || [CPU::default().build()];
        let ((detnet, keynet), device): ((Session, Session), String) = match options.device {
            OrtDevice::Cpu => (pair(&cpu())?, "cpu".into()),
            OrtDevice::Cuda(device) => (pair(&[cuda(device)])?, format!("cuda:{device}")),
            OrtDevice::Auto => match pair(&[cuda(0)]) {
                Ok(sessions) => (sessions, "cuda:0".into()),
                Err(error) => (pair(&cpu())?, format!("cpu (cuda unavailable: {error})")),
            },
        };
        let description: String = format!("ort {device} fp32 ({DETNET_ONNX}, {KEYNET_ONNX}; {})", dylib.display());
        Ok(Self { detnet, keynet, description, frames: Vec::new() })
    }
}

fn output<'a>(outputs: &'a ::ort::session::SessionOutputs<'_>, name: &str, net: &'static str, len: usize) -> Result<&'a [f32], NetsError> {
    let value = outputs.get(name).ok_or_else(|| run_error(net, format!("no output named {name}")))?;
    let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| run_error(net, e))?;
    if data.len() != len {
        return Err(run_error(net, format!("output {name} has {} values, expected {len}", data.len())));
    }
    Ok(data)
}

impl HandNets for OrtNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let count: usize = frames.len();
        self.frames.resize(count * FRAME_LEN, 0.0);
        for (frame, dst) in frames.iter().zip(self.frames.as_chunks_mut::<{ FRAME_LEN }>().0.iter_mut()) {
            frame.write_unit_f32(dst)?;
        }
        let input = TensorRef::from_array_view(([count, 1, DETNET_HEIGHT, DETNET_WIDTH], self.frames.as_slice())).map_err(|e| run_error("detnet", e))?;
        let outputs = self.detnet.run(::ort::inputs!["image" => input]).map_err(|e| run_error("detnet", e))?;
        let center: &[f32] = output(&outputs, "center", "detnet", count * 4)?;
        let radius: &[f32] = output(&outputs, "radius", "detnet", count * 2)?;
        let presence: &[f32] = output(&outputs, "presence_logit", "detnet", count * 2)?;
        Ok((0..count)
            .map(|i| DetNetRaw {
                center: [[center[4 * i], center[4 * i + 1]], [center[4 * i + 2], center[4 * i + 3]]],
                radius: [radius[2 * i], radius[2 * i + 1]],
                presence_logit: [presence[2 * i], presence[2 * i + 1]],
            })
            .collect())
    }

    fn keynet(&mut self, crops: &[&[f32]], keypoints: &[[f32; 3 * NUM_LANDMARKS]]) -> Result<Vec<KeyNetRaw>, NetsError> {
        if crops.len() != keypoints.len() {
            return Err(NetsError::Input { net: "keynet", message: format!("{} crops but {} keypoint priors", crops.len(), keypoints.len()) });
        }
        if crops.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(crop) = crops.iter().find(|crop| crop.len() != CROP_LEN) {
            return Err(NetsError::Input { net: "keynet", message: format!("crop has {} values, expected 96x96", crop.len()) });
        }
        let count: usize = crops.len();
        let crop_data: Vec<f32> = crops.concat();
        let prior_data: Vec<f32> = keypoints.iter().flatten().copied().collect();
        let crop = TensorRef::from_array_view(([count, 1, KEYNET_CROP, KEYNET_CROP], crop_data.as_slice())).map_err(|e| run_error("keynet", e))?;
        let prior = TensorRef::from_array_view(([count, 63], prior_data.as_slice())).map_err(|e| run_error("keynet", e))?;
        let outputs = self.keynet.run(::ort::inputs!["crop" => crop, "keypoints" => prior]).map_err(|e| run_error("keynet", e))?;
        let heatmaps: &[f32] = output(&outputs, "heatmaps", "keynet", count * HEATMAP_LEN)?;
        let distance: &[f32] = output(&outputs, "distance", "keynet", count * DISTANCE_LEN)?;
        let presence: &[f32] = output(&outputs, "presence_logit", "keynet", count)?;
        let pinch: Option<&[f32]> = match outputs.get("pinch_logit") {
            Some(_) => Some(output(&outputs, "pinch_logit", "keynet", count)?),
            None => None,
        };
        Ok((0..count)
            .map(|i| KeyNetRaw {
                heatmaps: heatmaps[i * HEATMAP_LEN..(i + 1) * HEATMAP_LEN].to_vec(),
                distance: distance[i * DISTANCE_LEN..(i + 1) * DISTANCE_LEN].to_vec(),
                presence_logit: presence[i],
                pinch_logit: pinch.map(|values| values[i]),
            })
            .collect())
    }

    fn describe(&self) -> String {
        self.description.clone()
    }
}
