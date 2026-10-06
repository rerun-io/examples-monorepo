//! DetNet and KeyNet on the RK3588 NPU through Rockchip's RKNN runtime (`librknnrt.so`, C API 2.3.2), loaded at run time.
//!
//! The library is `dlopen`ed with `libloading` and never linked, so the binary builds and starts without it. Two layers:
//!
//! - [`RknnRuntime`] + [`RknnModel`]: a self-contained, model-agnostic binding (one context per model and NPU core, u8/f32
//!   inputs, float outputs written into caller buffers, typed errors, `rknn_destroy` on drop). It has no robocap-specific code
//!   and is the piece to upstream (kornia-rs has no inference crate; see UPSTREAM.md).
//! - [`RknnNets`]: the [`HandNets`] backend. DetNet runs on core 0; KeyNet has one context on core 1 and one on core 2, and a
//!   batch of more than one crop is split between them on scoped threads.
//!
//! The models have the `/255` baked in (`rknn_convert.py`: mean 0, std 255), so the image input is on the u8 scale. DetNet's
//! 640x480 frame is average-pooled 4x4 on the CPU to the 120x160 input the model expects, in one pass from the small image's rows
//! straight into the model's input type ([`PooledInput`]; the black bars are zeros). How the image is fed follows the
//! model's input type ([`ImageFeed`]): an INT8 model gets u8 (the rounded pool [`pool4_u8`](kornia_staging_imgproc::resize::pool4_u8), the crop rounded to u8; the NPU
//! quantises its input at about that step anyway); an FP16 model gets the unrounded image (the exact pool [`pool4_mean_f32`](kornia_staging_imgproc::resize::pool4_mean_f32),
//! the float crop), normalised and converted to fp16 on our side and passed through in its native NHWC layout, because
//! rounding the pooled input to u8 alone moves DetNet's centres by 2.9 px mean on s66 frames. KeyNet's crop goes into one buffer
//! of the same type ([`CropInput`]); its prior keypoints are always float32.

use std::path::Path;

mod api;
pub use api::{InputData, NpuCore, RknnError, RknnModel, RknnRuntime, RunTiming, TensorInfo};

use kornia_staging_imgproc::resize::{pool4_from_sums, pool4_mean_f32, pool4_u8};

use super::{
    CROP_LEN, DETNET_HEIGHT, DETNET_WIDTH, DISTANCE_LEN, DetNetRaw, HEATMAP_LEN, HandNets, KEYNET_CROP, KeyNetRaw, NUM_LANDMARKS, NetFrame, NetsError,
};

/// Where Rockchip's images install the runtime.
pub const DEFAULT_LIBRARY: &str = "/usr/lib/librknnrt.so";
/// DetNet model file inside a models directory (see `models/MODELS.md` for why this precision).
pub const DEFAULT_DETNET: &str = "detnet_b1_fp16.rknn";
/// KeyNet model file inside a models directory (single crop; see `models/MODELS.md`).
pub const DEFAULT_KEYNET: &str = "keynet_b1_int8_mmse.rknn";
/// DetNet's pooled input: the 640x480 frame average-pooled 4x4.
pub const POOLED_WIDTH: usize = DETNET_WIDTH / 4;
/// DetNet's pooled input height.
pub const POOLED_HEIGHT: usize = DETNET_HEIGHT / 4;
/// Values in DetNet's pooled input, 160 x 120.
pub const POOLED_LEN: usize = POOLED_WIDTH * POOLED_HEIGHT;

// ---------------------------------------------------------------------------------------------------------------------------
impl From<RknnError> for NetsError {
    fn from(error: RknnError) -> Self {
        match error {
            RknnError::Library { .. } | RknnError::Symbol { .. } | RknnError::Read { .. } => {
                NetsError::Load { what: "rknn".into(), message: error.to_string() }
            }
            RknnError::Shape { .. } => NetsError::Input { net: "rknn", message: error.to_string() },
            RknnError::Call { .. } => NetsError::Run { net: "rknn", message: error.to_string() },
        }
    }
}

/// How an image input is handed to the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFeed {
    /// u8 values (INT8 models): the runtime normalises and quantises them.
    U8,
    /// f32 values on the u8 scale, unrounded (FP16 models whose native layout is not plain NHWC fp16): the runtime normalises
    /// and converts them.
    F32,
    /// fp16 values of the normalised image (`u8 / 255`), passed through in the model's native NHWC fp16 layout. Measured on
    /// Cap B: `rknn_inputs_set` drops from 0.3-1.1 ms (f32) to 0.02 ms, with bit-identical outputs.
    F16Native,
}

impl ImageFeed {
    /// The feed for a model's image input: u8 when it is quantised (int8/uint8); fp16 pass-through when its native layout is a
    /// dense one-channel NHWC fp16 tensor of `height x width`; f32 otherwise.
    pub fn for_input(info: &TensorInfo, native: &TensorInfo, height: usize, width: usize) -> Self {
        if info.dtype == 2 || info.dtype == 3 {
            return ImageFeed::U8;
        }
        let dense: bool = native.fmt == 1
            && native.dtype == 1
            && native.dims == [1, height as u32, width as u32, 1]
            && (native.w_stride == 0 || native.w_stride as usize == width)
            && native.size_with_stride == height * width * 2;
        if dense { ImageFeed::F16Native } else { ImageFeed::F32 }
    }
}

// The CPU pre-processing (kornia-style free functions on slices; see UPSTREAM.md). The 4x4 pooling is `kornia_staging_imgproc::resize`.

fn pool_error(error: kornia_image::ImageError) -> NetsError {
    NetsError::Input { net: "detnet", message: format!("pool4: {error}") }
}

/// DetNet's pooled 160x120 input, in the type of the model's [`ImageFeed`], written in one pass from a [`NetFrame`]: the 4x4
/// block sums of its pixel rows go straight into the feed's type, and the black bars are zeros. The arithmetic is the one of
/// pooling the padded 640x480 frame ([`pool4_u8`], [`pool4_mean_f32`], then [`f16_bytes_from_f32`] at 1/255 for fp16), so the
/// bytes are the same, without the padded frame or an f32 image in between.
#[derive(Clone, Debug)]
pub enum PooledInput {
    /// [`ImageFeed::U8`]: the rounded means.
    U8(Vec<u8>),
    /// [`ImageFeed::F32`]: the exact means on the u8 scale.
    F32(Vec<f32>),
    /// [`ImageFeed::F16Native`]: the exact means / 255 as little-endian fp16.
    F16(Vec<[u8; 2]>),
}

impl PooledInput {
    /// An input buffer for a model fed with `feed`.
    pub fn new(feed: ImageFeed) -> Self {
        let len: usize = POOLED_LEN;
        match feed {
            ImageFeed::U8 => PooledInput::U8(vec![0; len]),
            ImageFeed::F32 => PooledInput::F32(vec![0.0; len]),
            ImageFeed::F16Native => PooledInput::F16(vec![[0; 2]; len]),
        }
    }

    /// Pools `frame` into the buffer and returns it as the model's input.
    ///
    /// # Arguments
    ///
    /// * `frame` - The net frame; its pixel rows must start and end on 4-row block boundaries.
    ///
    /// # Returns
    ///
    /// The input to hand to [`RknnModel::run`], borrowing the buffer.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] when the frame does not fit the 640x480 net frame or its rows are not on block boundaries.
    pub fn fill(&mut self, frame: &NetFrame<'_>) -> Result<InputData<'_>, NetsError> {
        let rows: usize = frame.rows()?;
        if !frame.top.is_multiple_of(4) || !rows.is_multiple_of(4) {
            let message: String = format!("rows {}..{} are not on 4-row pooling blocks", frame.top, frame.top + rows);
            return Err(NetsError::Input { net: "detnet", message });
        }
        let (first, last): (usize, usize) = (frame.top / 4 * POOLED_WIDTH, (frame.top + rows) / 4 * POOLED_WIDTH);
        match self {
            PooledInput::U8(dst) => {
                dst[..first].fill(0);
                dst[last..].fill(0);
                pool4_u8(frame.pixels, DETNET_WIDTH, rows, &mut dst[first..last]).map_err(pool_error)?;
                Ok(InputData::U8(dst))
            }
            PooledInput::F32(dst) => {
                dst[..first].fill(0.0);
                dst[last..].fill(0.0);
                pool4_mean_f32(frame.pixels, DETNET_WIDTH, rows, &mut dst[first..last]).map_err(pool_error)?;
                Ok(InputData::F32(dst))
            }
            PooledInput::F16(dst) => {
                dst[..first].fill([0; 2]);
                dst[last..].fill([0; 2]);
                let scale: f32 = 1.0 / 255.0;
                pool4_from_sums(frame.pixels, DETNET_WIDTH, rows, &mut dst[first..last], |sum| {
                    half::f16::from_f32(f32::from(sum) / 16.0 * scale).to_bits().to_le_bytes()
                })
                .map_err(pool_error)?;
                Ok(InputData::Native(dst.as_flattened()))
            }
        }
    }
}

/// KeyNet's 96x96 crop in the type of the model's [`ImageFeed`], written from the [0, 1] float crop into one buffer (as
/// [`PooledInput`] is DetNet's).
#[derive(Clone, Debug)]
pub enum CropInput {
    /// [`ImageFeed::U8`]: `round(x * 255)`, clamped ([`u8_from_unit_f32`]).
    U8(Vec<u8>),
    /// [`ImageFeed::F32`]: `x * 255`, unrounded.
    F32(Vec<f32>),
    /// [`ImageFeed::F16Native`]: `x` as little-endian fp16 ([`f16_bytes_from_f32`] at scale 1).
    F16(Vec<u8>),
}

impl CropInput {
    /// An input buffer for a model fed with `feed`.
    pub fn new(feed: ImageFeed) -> Self {
        match feed {
            ImageFeed::U8 => CropInput::U8(vec![0; CROP_LEN]),
            ImageFeed::F32 => CropInput::F32(vec![0.0; CROP_LEN]),
            ImageFeed::F16Native => CropInput::F16(vec![0; 2 * CROP_LEN]),
        }
    }

    /// Writes `crop` into the buffer and returns it as the model's input.
    ///
    /// # Arguments
    ///
    /// * `crop` - 96 x 96 values in [0, 1], left-hand orientation.
    ///
    /// # Returns
    ///
    /// The input to hand to [`RknnModel::run`], borrowing the buffer.
    ///
    /// # Errors
    ///
    /// [`NetsError::Input`] when `crop` does not have 96 x 96 values.
    pub fn fill(&mut self, crop: &[f32]) -> Result<InputData<'_>, NetsError> {
        if crop.len() != CROP_LEN {
            return Err(NetsError::Input { net: "keynet", message: format!("crop has {} values, expected 96x96", crop.len()) });
        }
        match self {
            CropInput::U8(dst) => {
                u8_from_unit_f32(crop, dst)?;
                Ok(InputData::U8(dst))
            }
            CropInput::F32(dst) => {
                for (out, &value) in dst.iter_mut().zip(crop) {
                    *out = value * 255.0;
                }
                Ok(InputData::F32(dst))
            }
            CropInput::F16(dst) => {
                f16_bytes_from_f32(crop, 1.0, dst)?;
                Ok(InputData::Native(dst))
            }
        }
    }
}

/// Writes `values * scale` as little-endian IEEE half-precision bytes (round to nearest even): the buffer of an
/// [`ImageFeed::F16Native`] input, passed as [`InputData::Native`].
///
/// # Arguments
///
/// * `values` - the values, row-major.
/// * `scale` - a factor applied first (`1 / 255` turns the u8 scale into the models' normalised input).
/// * `dst` - `2 * values.len()` bytes.
///
/// # Errors
///
/// [`NetsError::Input`] when `dst` has another size.
pub fn f16_bytes_from_f32(values: &[f32], scale: f32, dst: &mut [u8]) -> Result<(), NetsError> {
    if dst.len() != values.len() * 2 {
        return Err(NetsError::Input { net: "rknn", message: format!("f16_bytes_from_f32: {} values into {} bytes", values.len(), dst.len()) });
    }
    for (out, &value) in dst.as_chunks_mut::<2>().0.iter_mut().zip(values) {
        out.copy_from_slice(&half::f16::from_f32(value * scale).to_bits().to_le_bytes());
    }
    Ok(())
}

/// Rounds [0, 1] floats to u8 (`round(x * 255)`, clamped): KeyNet's crop as the NPU model takes it.
///
/// # Arguments
///
/// * `src` - values in [0, 1] (out-of-range values clamp).
/// * `dst` - as many bytes as `src` has values.
///
/// # Errors
///
/// [`NetsError::Input`] when the lengths differ.
pub fn u8_from_unit_f32(src: &[f32], dst: &mut [u8]) -> Result<(), NetsError> {
    if src.len() != dst.len() {
        return Err(NetsError::Input { net: "keynet", message: format!("u8_from_unit_f32: {} values into {} bytes", src.len(), dst.len()) });
    }
    for (out, &value) in dst.iter_mut().zip(src) {
        *out = (value * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------------
// The HandNets backend.

/// Output slots of a KeyNet model, found by name.
#[derive(Clone, Copy, Debug)]
struct KeyNetOutputs {
    heatmaps: usize,
    distance: usize,
    presence: usize,
    pinch: Option<usize>,
}

impl KeyNetOutputs {
    fn new(model: &str, outputs: &[TensorInfo]) -> Result<Self, RknnError> {
        let shape = |message: String| RknnError::Shape { model: model.to_string(), message };
        let find = |name: &str| outputs.iter().position(|output| output.name == name);
        let slots: KeyNetOutputs = KeyNetOutputs {
            heatmaps: find("heatmaps").ok_or_else(|| shape("no output named heatmaps".into()))?,
            distance: find("distance").ok_or_else(|| shape("no output named distance".into()))?,
            presence: find("presence_logit").ok_or_else(|| shape("no output named presence_logit".into()))?,
            pinch: find("pinch_logit"),
        };
        let expected = [(Some(slots.heatmaps), HEATMAP_LEN), (Some(slots.distance), DISTANCE_LEN), (Some(slots.presence), 1), (slots.pinch, 1)];
        for (slot, len) in expected {
            if let Some(slot) = slot.filter(|&slot| outputs[slot].n_elems != len) {
                return Err(shape(format!("output {} has {} values, expected {len}", outputs[slot].name, outputs[slot].n_elems)));
            }
        }
        Ok(slots)
    }
}

/// One KeyNet context with its scratch buffers.
struct KeyNetWorker {
    model: RknnModel,
    slots: KeyNetOutputs,
    feed: ImageFeed,
    input: CropInput,
    buffers: Vec<Vec<f32>>,
}

impl KeyNetWorker {
    fn new(model: RknnModel) -> Result<Self, RknnError> {
        let shape = |message: String| RknnError::Shape { model: model.name().to_string(), message };
        let slots = KeyNetOutputs::new(model.name(), model.outputs())?;
        let inputs: &[TensorInfo] = model.inputs();
        if inputs.len() != 2 || inputs[0].n_elems != CROP_LEN || inputs[1].n_elems != 63 {
            return Err(shape(format!("expected inputs crop [1,1,96,96] and keypoints [1,63] (single-crop model), got {inputs:?}")));
        }
        let buffers: Vec<Vec<f32>> = model.outputs().iter().map(|output| vec![0.0; output.n_elems]).collect();
        let feed: ImageFeed = ImageFeed::for_input(&inputs[0], &model.native_inputs()[0], KEYNET_CROP, KEYNET_CROP);
        Ok(Self { model, slots, feed, input: CropInput::new(feed), buffers })
    }

    fn run(&mut self, crop: &[f32], keypoints: &[f32; 3 * NUM_LANDMARKS]) -> Result<KeyNetRaw, NetsError> {
        let image: InputData<'_> = self.input.fill(crop)?;
        let mut outputs: Vec<&mut [f32]> = self.buffers.iter_mut().map(Vec::as_mut_slice).collect();
        self.model.run(&[image, InputData::F32(keypoints)], &mut outputs)?;
        let slots: KeyNetOutputs = self.slots;
        Ok(KeyNetRaw {
            heatmaps: self.buffers[slots.heatmaps].clone(),
            distance: self.buffers[slots.distance].clone(),
            presence_logit: self.buffers[slots.presence][0],
            pinch_logit: slots.pinch.map(|slot| self.buffers[slot][0]),
        })
    }
}

/// One DetNet context with its scratch buffers.
struct DetNetWorker {
    model: RknnModel,
    slots: [usize; 3],
    feed: ImageFeed,
    input: PooledInput,
    buffers: Vec<Vec<f32>>,
}

impl DetNetWorker {
    fn new(model: RknnModel) -> Result<Self, RknnError> {
        let shape = |message: String| RknnError::Shape { model: model.name().to_string(), message };
        let find = |name: &str| model.output_index(name).ok_or_else(|| shape(format!("no output named {name}")));
        let slots: [usize; 3] = [find("center")?, find("radius")?, find("presence_logit")?];
        if model.inputs().len() != 1 || model.inputs()[0].n_elems != POOLED_LEN {
            return Err(shape(format!("expected one pooled [1,1,120,160] input, got {:?}", model.inputs())));
        }
        if slots.iter().zip([4, 2, 2]).any(|(&slot, len)| model.outputs()[slot].n_elems != len) {
            return Err(shape(format!("unexpected output sizes {:?}", model.outputs())));
        }
        let feed: ImageFeed = ImageFeed::for_input(&model.inputs()[0], &model.native_inputs()[0], POOLED_HEIGHT, POOLED_WIDTH);
        let buffers: Vec<Vec<f32>> = model.outputs().iter().map(|output| vec![0.0; output.n_elems]).collect();
        Ok(Self { model, slots, feed, input: PooledInput::new(feed), buffers })
    }

    fn run(&mut self, frame: &NetFrame<'_>) -> Result<DetNetRaw, NetsError> {
        let image: InputData<'_> = self.input.fill(frame)?;
        let mut outputs: Vec<&mut [f32]> = self.buffers.iter_mut().map(Vec::as_mut_slice).collect();
        self.model.run(&[image], &mut outputs)?;
        let [center, radius, presence] = self.slots.map(|slot| &self.buffers[slot]);
        Ok(DetNetRaw {
            center: [[center[0], center[1]], [center[2], center[3]]],
            radius: [radius[0], radius[1]],
            presence_logit: [presence[0], presence[1]],
        })
    }
}

/// Runs `items` through `workers`: inline on the first worker for one item (or one worker), otherwise in contiguous chunks,
/// one scoped thread per worker, the k-th made by `thread(k)` (`Builder::new()`; the tests pass one the OS refuses); results
/// come back in input order. A thread the OS refuses is a [`NetsError::Run`]: the threads started before it run their chunks
/// to the end and are joined, its chunk and the later ones do not run, so no item runs twice.
fn run_spread<W: Send, I: Sync, O: Send>(
    workers: &mut [W],
    items: &[I],
    net: &'static str,
    thread: impl Fn(usize) -> std::thread::Builder,
    run: impl Fn(&mut W, &I) -> Result<O, NetsError> + Sync,
) -> Result<Vec<O>, NetsError> {
    let count: usize = workers.len();
    let Some(first) = workers.first_mut() else {
        return Err(NetsError::Run { net, message: "no contexts".into() });
    };
    if items.len() <= 1 || count == 1 {
        return items.iter().map(|item| run(first, item)).collect();
    }
    let chunk: usize = items.len().div_ceil(count);
    let run = &run;
    let parts: Vec<Result<Vec<O>, NetsError>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(count);
        let mut refused: Option<NetsError> = None;
        for (k, (worker, part)) in workers.iter_mut().zip(items.chunks(chunk)).enumerate() {
            let work = move || part.iter().map(|item| run(worker, item)).collect::<Result<Vec<O>, NetsError>>();
            match thread(k).spawn_scoped(scope, work) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    refused = Some(NetsError::Run { net, message: format!("starting worker thread {k}: {error}") });
                    break;
                }
            }
        }
        let mut parts: Vec<Result<Vec<O>, NetsError>> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or_else(|_| Err(NetsError::Run { net, message: "a worker thread panicked".into() })))
            .collect();
        parts.extend(refused.map(Err));
        parts
    });
    let mut outputs: Vec<O> = Vec::with_capacity(items.len());
    for part in parts {
        outputs.extend(part?);
    }
    Ok(outputs)
}

/// [`HandNets`] on the RK3588 NPU. DetNet's first context is on core 0 and KeyNet's on cores 1 and 2; a call with several
/// frames (or crops) spreads them over all of its contexts on scoped threads, so six DetNet frames take about two frames' time.
pub struct RknnNets {
    detnet: Vec<DetNetWorker>,
    keynet: Vec<KeyNetWorker>,
    description: String,
}

impl RknnNets {
    /// Opens the runtime and loads [`DEFAULT_DETNET`] (3 contexts: cores 0, 1, 2) and [`DEFAULT_KEYNET`] (2 contexts: cores 1,
    /// 2) from `models_dir`.
    ///
    /// # Errors
    ///
    /// As [`RknnNets::with_files`].
    pub fn open(models_dir: impl AsRef<Path>) -> Result<Self, NetsError> {
        let dir: &Path = models_dir.as_ref();
        Self::with_files(DEFAULT_LIBRARY, dir.join(DEFAULT_DETNET), dir.join(DEFAULT_KEYNET), 3, 2)
    }

    /// Loads DetNet contexts on cores 0, 1, 2 and KeyNet contexts on cores 1, 2, 0 (the first `detnet_contexts` and
    /// `keynet_contexts` of each order) from explicit files.
    ///
    /// # Arguments
    ///
    /// * `library` - `librknnrt.so`.
    /// * `detnet` - a pooled-input DetNet model (input [1,1,120,160], outputs center/radius/presence_logit).
    /// * `keynet` - a single-crop KeyNet model (inputs crop [1,1,96,96] and keypoints [1,63]).
    /// * `detnet_contexts`, `keynet_contexts` - 1 to 3 contexts each.
    ///
    /// # Errors
    ///
    /// [`NetsError::Load`] when the library or a model cannot be loaded, [`NetsError::Input`] when a model has other tensors
    /// than expected.
    pub fn with_files(
        library: impl AsRef<Path>,
        detnet: impl AsRef<Path>,
        keynet: impl AsRef<Path>,
        detnet_contexts: usize,
        keynet_contexts: usize,
    ) -> Result<Self, NetsError> {
        let runtime: RknnRuntime = RknnRuntime::open(library)?;
        let detnet_cores: [NpuCore; 3] = [NpuCore::Core0, NpuCore::Core1, NpuCore::Core2];
        let detnet: Vec<DetNetWorker> = detnet_cores
            .iter()
            .take(detnet_contexts.clamp(1, 3))
            .map(|&core| RknnModel::load(&runtime, detnet.as_ref(), core).and_then(DetNetWorker::new))
            .collect::<Result<_, _>>()?;
        let keynet_cores: [NpuCore; 3] = [NpuCore::Core1, NpuCore::Core2, NpuCore::Core0];
        let keynet: Vec<KeyNetWorker> = keynet_cores
            .iter()
            .take(keynet_contexts.clamp(1, 3))
            .map(|&core| RknnModel::load(&runtime, keynet.as_ref(), core).and_then(KeyNetWorker::new))
            .collect::<Result<_, _>>()?;
        let (api, driver) = detnet[0].model.versions().clone();
        let description: String = format!(
            "rknn {} ({:?} in) x{} + {} ({:?} in) x{} (api {api}, driver {driver})",
            detnet[0].model.name(),
            detnet[0].feed,
            detnet.len(),
            keynet[0].model.name(),
            keynet[0].feed,
            keynet.len()
        );
        Ok(Self { detnet, keynet, description })
    }
}

impl HandNets for RknnNets {
    fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        run_spread(&mut self.detnet, frames, "detnet", |_| std::thread::Builder::new(), |worker, frame| worker.run(frame))
    }

    fn keynet(&mut self, crops: &[&[f32]], keypoints: &[[f32; 3 * NUM_LANDMARKS]]) -> Result<Vec<KeyNetRaw>, NetsError> {
        if crops.len() != keypoints.len() {
            return Err(NetsError::Input { net: "keynet", message: format!("{} crops but {} keypoint priors", crops.len(), keypoints.len()) });
        }
        if let Some(crop) = crops.iter().find(|crop| crop.len() != CROP_LEN) {
            return Err(NetsError::Input { net: "keynet", message: format!("crop has {} values, expected 96x96", crop.len()) });
        }
        let pairs: Vec<(&[f32], &[f32; 3 * NUM_LANDMARKS])> = crops.iter().copied().zip(keypoints).collect();
        run_spread(&mut self.keynet, &pairs, "keynet", |_| std::thread::Builder::new(), |worker, (crop, prior)| worker.run(crop, prior))
    }

    fn describe(&self) -> String {
        self.description.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keynet_rejects_a_malformed_pinch_head() {
        let outputs: Vec<TensorInfo> = [("heatmaps", HEATMAP_LEN), ("distance", DISTANCE_LEN), ("presence_logit", 1)]
            .into_iter().map(|(name, len)| TensorInfo { name: name.into(), ..tensor(&[len as u32], 0, 0, 0, len * 4) }).collect();
        assert!(KeyNetOutputs::new("keynet", &outputs).is_ok());
        for length in [0, 1, 2] {
            let mut heads = outputs.clone();
            heads.push(TensorInfo { name: "pinch_logit".into(), ..tensor(&[length as u32], 0, 0, 0, length * 4) });
            assert_eq!(KeyNetOutputs::new("keynet", &heads).is_ok(), length == 1, "pinch_logit: {length} values");
        }
    }

    #[test]
    fn unit_floats_round_to_u8() {
        let mut dst: [u8; 4] = [0; 4];
        assert!(u8_from_unit_f32(&[0.0, 1.0, 0.5, -0.2], &mut dst).is_ok());
        assert_eq!(dst, [0, 255, 128, 0]);
    }

    fn tensor(dims: &[u32], fmt: i32, dtype: i32, w_stride: u32, size_with_stride: usize) -> TensorInfo {
        TensorInfo { name: "image".into(), dims: dims.to_vec(), n_elems: dims.iter().product::<u32>() as usize, fmt, dtype, size_with_stride, w_stride }
    }

    #[test]
    fn image_feed_follows_the_input_and_native_layout() {
        // The FP16 models' image on Cap B: NHWC fp16, dense rows (w_stride = width).
        let fp16: TensorInfo = tensor(&[1, 120, 160, 1], 1, 1, 160, 120 * 160 * 2);
        assert_eq!(ImageFeed::for_input(&fp16, &fp16, 120, 160), ImageFeed::F16Native);
        let unset_stride: TensorInfo = tensor(&[1, 120, 160, 1], 1, 1, 0, 120 * 160 * 2);
        assert_eq!(ImageFeed::for_input(&fp16, &unset_stride, 120, 160), ImageFeed::F16Native);
        // Padded rows, NC1HWC2 or another size need the runtime's conversion.
        let padded: TensorInfo = tensor(&[1, 120, 160, 1], 1, 1, 176, 120 * 176 * 2);
        assert_eq!(ImageFeed::for_input(&fp16, &padded, 120, 160), ImageFeed::F32);
        let nc1hwc2: TensorInfo = tensor(&[1, 1, 120, 160, 8], 2, 1, 160, 120 * 160 * 16);
        assert_eq!(ImageFeed::for_input(&fp16, &nc1hwc2, 120, 160), ImageFeed::F32);
        assert_eq!(ImageFeed::for_input(&fp16, &fp16, 96, 96), ImageFeed::F32);
        // Quantised inputs take u8 whatever the native layout.
        let int8: TensorInfo = tensor(&[1, 120, 160, 1], 1, 2, 160, 120 * 160);
        assert_eq!(ImageFeed::for_input(&int8, &int8, 120, 160), ImageFeed::U8);
    }

    #[test]
    fn f16_bytes_are_scaled_little_endian_halves() {
        let mut dst: [u8; 6] = [0; 6];
        assert!(f16_bytes_from_f32(&[255.0, 0.0, 127.5], 1.0 / 255.0, &mut dst).is_ok());
        // 1.0 = 0x3c00, 0.0 = 0x0000, 0.5 = 0x3800.
        assert_eq!(dst, [0x00, 0x3c, 0x00, 0x00, 0x00, 0x38]);
        assert!(f16_bytes_from_f32(&[1.0], 1.0, &mut dst).is_err());
    }

    #[test]
    fn a_refused_worker_thread_is_an_error_and_no_item_runs_twice() {
        // An impossible stack size makes the OS refuse the thread without exhausting host resources (as in handfit's cold start).
        assert!(std::thread::Builder::new().stack_size(usize::MAX).spawn(|| ()).is_err());
        let items: Vec<usize> = (0..7).collect();
        let record = |seen: &mut Vec<usize>, &item: &usize| -> Result<usize, NetsError> {
            seen.push(item);
            Ok(10 * item)
        };
        // Three workers take the chunks [0, 1, 2], [3, 4, 5], [6].
        for refused in 0..3 {
            let mut workers: Vec<Vec<usize>> = vec![Vec::new(); 3];
            let thread = |k: usize| if k == refused { std::thread::Builder::new().stack_size(usize::MAX) } else { std::thread::Builder::new() };
            let result: Result<Vec<usize>, NetsError> = run_spread(&mut workers, &items, "keynet", thread, record);
            assert!(matches!(result, Err(NetsError::Run { net: "keynet", .. })), "thread {refused} refused: {result:?}");
            // The threads before the refused one ran their whole chunks; nothing else ran.
            let ran: Vec<usize> = workers.iter().flatten().copied().collect();
            assert_eq!(ran, items[..3 * refused], "thread {refused} refused");
        }
        let mut workers: Vec<Vec<usize>> = vec![Vec::new(); 3];
        let result: Result<Vec<usize>, NetsError> = run_spread(&mut workers, &items, "keynet", |_| std::thread::Builder::new(), record);
        assert_eq!(result.ok(), Some(items.iter().map(|item| 10 * item).collect()));
        assert_eq!(workers, [vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
    }

    #[test]
    fn missing_library_is_a_typed_error() {
        let error: Result<RknnRuntime, RknnError> = RknnRuntime::open("/nonexistent/librknnrt.so");
        assert!(matches!(error, Err(RknnError::Library { .. })));
    }
}
