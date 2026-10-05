//! Model-independent binding to the RKNN runtime C API (2.3.2).
#![deny(missing_docs)]

use std::ffi::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use libloading::Library;

// The C API (rknn_api.h, RKNN runtime 2.3.2, 64-bit layout).

type RknnContext = u64;

const RKNN_SUCC: c_int = 0;
const RKNN_MAX_DIMS: usize = 16;
const RKNN_MAX_NAME_LEN: usize = 256;
const RKNN_QUERY_IN_OUT_NUM: c_int = 0;
const RKNN_QUERY_INPUT_ATTR: c_int = 1;
const RKNN_QUERY_OUTPUT_ATTR: c_int = 2;
const RKNN_QUERY_PERF_RUN: c_int = 4;
const RKNN_QUERY_SDK_VERSION: c_int = 5;
const RKNN_QUERY_NATIVE_INPUT_ATTR: c_int = 8;
const RKNN_TENSOR_FLOAT32: c_int = 0;
const RKNN_TENSOR_UINT8: c_int = 3;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RknnInputOutputNum {
    n_input: u32,
    n_output: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RknnTensorAttr {
    index: u32,
    n_dims: u32,
    dims: [u32; RKNN_MAX_DIMS],
    name: [c_char; RKNN_MAX_NAME_LEN],
    n_elems: u32,
    size: u32,
    fmt: c_int,
    type_: c_int,
    qnt_type: c_int,
    fl: i8,
    zp: i32,
    scale: f32,
    w_stride: u32,
    size_with_stride: u32,
    pass_through: u8,
    h_stride: u32,
}

impl RknnTensorAttr {
    fn zeroed(index: u32) -> Self {
        Self {
            index,
            n_dims: 0,
            dims: [0; RKNN_MAX_DIMS],
            name: [0; RKNN_MAX_NAME_LEN],
            n_elems: 0,
            size: 0,
            fmt: 0,
            type_: 0,
            qnt_type: 0,
            fl: 0,
            zp: 0,
            scale: 0.0,
            w_stride: 0,
            size_with_stride: 0,
            pass_through: 0,
            h_stride: 0,
        }
    }
}

#[repr(C)]
struct RknnInput {
    index: u32,
    buf: *mut c_void,
    size: u32,
    pass_through: u8,
    type_: c_int,
    fmt: c_int,
}

#[repr(C)]
struct RknnOutput {
    want_float: u8,
    is_prealloc: u8,
    index: u32,
    buf: *mut c_void,
    size: u32,
}

#[repr(C)]
struct RknnSdkVersion {
    api_version: [c_char; 256],
    drv_version: [c_char; 256],
}

#[repr(C)]
#[derive(Default)]
struct RknnPerfRun {
    run_duration: i64,
}

type InitFn = unsafe extern "C" fn(*mut RknnContext, *mut c_void, u32, u32, *mut c_void) -> c_int;
type DestroyFn = unsafe extern "C" fn(RknnContext) -> c_int;
type QueryFn = unsafe extern "C" fn(RknnContext, c_int, *mut c_void, u32) -> c_int;
type InputsSetFn = unsafe extern "C" fn(RknnContext, u32, *mut RknnInput) -> c_int;
type RunFn = unsafe extern "C" fn(RknnContext, *mut c_void) -> c_int;
type OutputsGetFn = unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput, *mut c_void) -> c_int;
type OutputsReleaseFn = unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput) -> c_int;
type SetCoreMaskFn = unsafe extern "C" fn(RknnContext, c_int) -> c_int;

/// The resolved entry points. The function pointers stay valid while `_library` is alive, and every holder keeps it alive.
struct Api {
    init: InitFn,
    destroy: DestroyFn,
    query: QueryFn,
    inputs_set: InputsSetFn,
    run: RunFn,
    outputs_get: OutputsGetFn,
    outputs_release: OutputsReleaseFn,
    set_core_mask: SetCoreMaskFn,
    _library: Library,
}

// ---------------------------------------------------------------------------------------------------------------------------
// The binding.

/// Errors of the RKNN binding.
#[derive(Debug, thiserror::Error)]
pub enum RknnError {
    /// `librknnrt.so` could not be opened.
    #[error("opening {path}: {source}")]
    Library {
        /// The library path tried.
        path: PathBuf,
        /// The loader's error.
        source: libloading::Error,
    },
    /// A C API symbol is missing from the library.
    #[error("symbol {name} in {path}: {source}")]
    Symbol {
        /// The symbol name.
        name: &'static str,
        /// The library path.
        path: PathBuf,
        /// The loader's error.
        source: libloading::Error,
    },
    /// A model file could not be read.
    #[error("reading {path}: {source}")]
    Read {
        /// The model path.
        path: PathBuf,
        /// The I/O error.
        source: std::io::Error,
    },
    /// A C API call returned an error code (`RKNN_ERR_*`, e.g. -5 = invalid parameter, -6 = invalid model).
    #[error("{call} failed for {model}: RKNN error {code}")]
    Call {
        /// The C function.
        call: &'static str,
        /// The model's name.
        model: String,
        /// The negative return code.
        code: i32,
    },
    /// The model does not have the inputs or outputs the caller expects, or the caller's buffers do not fit them.
    #[error("{model}: {message}")]
    Shape {
        /// The model's name.
        model: String,
        /// What did not match.
        message: String,
    },
}

/// An NPU core (or a set of them) for [`RknnModel::load`]; the values are `rknn_core_mask`'s.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpuCore {
    /// Let the driver choose.
    Auto,
    /// Core 0.
    Core0,
    /// Core 1.
    Core1,
    /// Core 2.
    Core2,
}

impl NpuCore {
    fn mask(self) -> c_int {
        match self {
            NpuCore::Auto => 0,
            NpuCore::Core0 => 1,
            NpuCore::Core1 => 2,
            NpuCore::Core2 => 4,
        }
    }
}

/// One input or output tensor of a loaded model, as the runtime describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    /// The tensor's name (the ONNX name).
    pub name: String,
    /// Its dimensions.
    pub dims: Vec<u32>,
    /// Elements per call.
    pub n_elems: usize,
    /// `rknn_tensor_format` of the model's tensor (0 NCHW, 1 NHWC, 3 undefined).
    pub fmt: i32,
    /// `rknn_tensor_type` of the model's tensor (0 f32, 1 f16, 2 i8, 3 u8).
    pub dtype: i32,
    /// Bytes of the tensor including the runtime's row stride (the size a pass-through buffer must have).
    pub size_with_stride: usize,
    /// Row stride in elements (0 = the width).
    pub w_stride: u32,
}

/// The data for one model input in [`RknnModel::run`]. The runtime converts it to the model's own type and layout.
#[derive(Clone, Copy, Debug)]
pub enum InputData<'a> {
    /// u8 values (the image inputs, whose normalisation is baked into the model).
    U8(&'a [u8]),
    /// f32 values.
    F32(&'a [f32]),
    /// Raw bytes already in the model's native layout and type ([`RknnModel::native_inputs`]), passed through unconverted
    /// (no normalisation either: the bytes are what the first layer reads).
    Native(&'a [u8]),
}

impl InputData<'_> {
    fn len(&self) -> usize {
        match self {
            InputData::U8(values) => values.len(),
            InputData::F32(values) => values.len(),
            InputData::Native(bytes) => bytes.len(),
        }
    }
}

/// Microseconds spent in one [`RknnModel::run_timed`] call, by phase.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunTiming {
    /// `rknn_inputs_set` (CPU conversion of the inputs).
    pub inputs_set_us: f64,
    /// `rknn_run` (submission and wait).
    pub run_us: f64,
    /// `rknn_outputs_get` + `rknn_outputs_release` (dequantisation to float).
    pub outputs_get_us: f64,
    /// The driver's own NPU time (`RKNN_QUERY_PERF_RUN`).
    pub npu_us: f64,
}

/// A loaded `librknnrt.so`.
#[derive(Clone)]
pub struct RknnRuntime {
    api: Arc<Api>,
    path: PathBuf,
}

fn symbol<T: Copy>(library: &Library, name: &'static str, path: &Path) -> Result<T, RknnError> {
    // SAFETY: `T` is the function-pointer type that rknn_api.h 2.3.2 declares for `name`; the pointer is used only while
    // the library stays loaded (it is kept in `Api` beside the pointers).
    let found: libloading::Symbol<'_, T> = unsafe { library.get(name.as_bytes()) }
        .map_err(|source| RknnError::Symbol { name, path: path.to_path_buf(), source })?;
    Ok(*found)
}

impl RknnRuntime {
    /// Opens the RKNN runtime library.
    ///
    /// # Arguments
    ///
    /// * `path` - the library, usually `/usr/lib/librknnrt.so`.
    ///
    /// # Returns
    ///
    /// The runtime, from which models are loaded.
    ///
    /// # Errors
    ///
    /// [`RknnError::Library`] when the file cannot be opened, [`RknnError::Symbol`] when an entry point is missing.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RknnError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        // SAFETY: loading librknnrt runs its initialisers; Rockchip's runtime has no unsound global constructors, and the
        // library stays loaded for as long as any model refers to it.
        let library: Library = unsafe { Library::new(&path) }.map_err(|source| RknnError::Library { path: path.clone(), source })?;
        let api: Api = Api {
            init: symbol(&library, "rknn_init", &path)?,
            destroy: symbol(&library, "rknn_destroy", &path)?,
            query: symbol(&library, "rknn_query", &path)?,
            inputs_set: symbol(&library, "rknn_inputs_set", &path)?,
            run: symbol(&library, "rknn_run", &path)?,
            outputs_get: symbol(&library, "rknn_outputs_get", &path)?,
            outputs_release: symbol(&library, "rknn_outputs_release", &path)?,
            set_core_mask: symbol(&library, "rknn_set_core_mask", &path)?,
            _library: library,
        };
        Ok(Self { api: Arc::new(api), path })
    }

    /// The library path this runtime was opened from.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// One RKNN context: a model loaded on one NPU core. Not shareable between threads at once, but movable to another thread.
pub struct RknnModel {
    api: Arc<Api>,
    ctx: RknnContext,
    name: String,
    inputs: Vec<TensorInfo>,
    native_inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
    versions: (String, String),
    _model: Vec<u8>,
}

fn c_string(chars: &[c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl RknnModel {
    /// Loads a `.rknn` file into a new context bound to `core`.
    ///
    /// # Arguments
    ///
    /// * `runtime` - the opened runtime.
    /// * `path` - the `.rknn` file.
    /// * `core` - the NPU core the context runs on.
    ///
    /// # Returns
    ///
    /// The model, with its tensor descriptions queried.
    ///
    /// # Errors
    ///
    /// [`RknnError::Read`] when the file cannot be read; [`RknnError::Call`] when `rknn_init`, `rknn_set_core_mask` or a query
    /// fails.
    pub fn load(runtime: &RknnRuntime, path: impl AsRef<Path>, core: NpuCore) -> Result<Self, RknnError> {
        let path: &Path = path.as_ref();
        let mut model: Vec<u8> = std::fs::read(path).map_err(|source| RknnError::Read { path: path.to_path_buf(), source })?;
        let name: String = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
        let size: u32 = u32::try_from(model.len()).map_err(|_| RknnError::Shape { model: name.clone(), message: "model > 4 GiB".into() })?;
        let api: Arc<Api> = runtime.api.clone();
        let mut ctx: RknnContext = 0;
        // SAFETY: `model` is a valid buffer of `size` bytes that outlives the context (it is kept in the struct); `ctx` is a
        // valid out-pointer; a null extend pointer is allowed by the API.
        let code: c_int = unsafe { (api.init)(&mut ctx, model.as_mut_ptr().cast(), size, 0, std::ptr::null_mut()) };
        if code != RKNN_SUCC {
            return Err(RknnError::Call { call: "rknn_init", model: name, code });
        }
        let mut loaded: Self =
            Self { api, ctx, name, inputs: Vec::new(), native_inputs: Vec::new(), outputs: Vec::new(), versions: Default::default(), _model: model };
        if core != NpuCore::Auto {
            // SAFETY: `ctx` is the live context created above.
            let code: c_int = unsafe { (loaded.api.set_core_mask)(loaded.ctx, core.mask()) };
            loaded.check("rknn_set_core_mask", code)?;
        }
        let mut io: RknnInputOutputNum = RknnInputOutputNum::default();
        loaded.query(RKNN_QUERY_IN_OUT_NUM, &mut io)?;
        loaded.inputs = (0..io.n_input).map(|index| loaded.tensor(RKNN_QUERY_INPUT_ATTR, index)).collect::<Result<_, _>>()?;
        loaded.native_inputs = (0..io.n_input).map(|index| loaded.tensor(RKNN_QUERY_NATIVE_INPUT_ATTR, index)).collect::<Result<_, _>>()?;
        loaded.outputs = (0..io.n_output).map(|index| loaded.tensor(RKNN_QUERY_OUTPUT_ATTR, index)).collect::<Result<_, _>>()?;
        let mut version: RknnSdkVersion = RknnSdkVersion { api_version: [0; 256], drv_version: [0; 256] };
        loaded.query(RKNN_QUERY_SDK_VERSION, &mut version)?;
        loaded.versions = (c_string(&version.api_version), c_string(&version.drv_version));
        Ok(loaded)
    }

    fn check(&self, call: &'static str, code: c_int) -> Result<(), RknnError> {
        if code == RKNN_SUCC { Ok(()) } else { Err(RknnError::Call { call, model: self.name.clone(), code }) }
    }

    fn query<T>(&self, command: c_int, info: &mut T) -> Result<(), RknnError> {
        let size: u32 = size_of::<T>() as u32;
        // SAFETY: `info` is a valid, writable `T`, and `T` is the struct rknn_api.h pairs with `command`.
        let code: c_int = unsafe { (self.api.query)(self.ctx, command, (info as *mut T).cast(), size) };
        self.check("rknn_query", code)
    }

    fn tensor(&self, command: c_int, index: u32) -> Result<TensorInfo, RknnError> {
        let mut attr: RknnTensorAttr = RknnTensorAttr::zeroed(index);
        self.query(command, &mut attr)?;
        let dims: Vec<u32> = attr.dims[..(attr.n_dims as usize).min(RKNN_MAX_DIMS)].to_vec();
        Ok(TensorInfo {
            name: c_string(&attr.name),
            dims,
            n_elems: attr.n_elems as usize,
            fmt: attr.fmt,
            dtype: attr.type_,
            size_with_stride: attr.size_with_stride as usize,
            w_stride: attr.w_stride,
        })
    }

    /// The model's file name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The model's inputs, in index order.
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// The model's outputs, in index order.
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// The runtime's (API, driver) version strings.
    pub fn versions(&self) -> &(String, String) {
        &self.versions
    }

    /// The model's inputs in the runtime's native (NPU) layout and type, in index order: what an [`InputData::Native`] buffer
    /// must hold (`size_with_stride` bytes, rows `w_stride` elements apart).
    pub fn native_inputs(&self) -> &[TensorInfo] {
        &self.native_inputs
    }

    /// The index of the output called `name`, if the model has one.
    pub fn output_index(&self, name: &str) -> Option<usize> {
        self.outputs.iter().position(|output| output.name == name)
    }

    /// Runs the model once: sets every input, runs, and writes every output as float32 into `outputs`.
    ///
    /// # Arguments
    ///
    /// * `inputs` - one entry per model input, in index order, each with exactly that input's element count.
    /// * `outputs` - one buffer per model output, in index order, each with exactly that output's element count. Values come
    ///   back in the ONNX layout (row-major NCHW for 4-d tensors).
    ///
    /// # Errors
    ///
    /// [`RknnError::Shape`] when the counts do not match; [`RknnError::Call`] when a runtime call fails.
    pub fn run(&mut self, inputs: &[InputData<'_>], outputs: &mut [&mut [f32]]) -> Result<(), RknnError> {
        self.run_inner(inputs, outputs, false).map(|_| ())
    }

    /// [`RknnModel::run`], also timing each phase and querying the driver's NPU time.
    ///
    /// # Errors
    ///
    /// As [`RknnModel::run`].
    pub fn run_timed(&mut self, inputs: &[InputData<'_>], outputs: &mut [&mut [f32]]) -> Result<RunTiming, RknnError> {
        self.run_inner(inputs, outputs, true)
    }

    fn run_inner(&mut self, inputs: &[InputData<'_>], outputs: &mut [&mut [f32]], timed: bool) -> Result<RunTiming, RknnError> {
        if inputs.len() != self.inputs.len() || outputs.len() != self.outputs.len() {
            return Err(RknnError::Shape {
                model: self.name.clone(),
                message: format!("{} inputs and {} outputs given, model has {} and {}", inputs.len(), outputs.len(), self.inputs.len(), self.outputs.len()),
            });
        }
        let mut c_inputs: Vec<RknnInput> = Vec::with_capacity(inputs.len());
        for ((info, native), data) in self.inputs.iter().zip(&self.native_inputs).zip(inputs) {
            let (expected, unit): (usize, &str) = if matches!(data, InputData::Native(_)) { (native.size_with_stride, "bytes") } else { (info.n_elems, "values") };
            if data.len() != expected {
                return Err(RknnError::Shape { model: self.name.clone(), message: format!("input {} needs {expected} {unit}, got {}", info.name, data.len()) });
            }
            let (buf, size, type_, fmt, pass_through): (*mut c_void, usize, c_int, c_int, u8) = match data {
                InputData::U8(values) => (values.as_ptr().cast_mut().cast(), values.len(), RKNN_TENSOR_UINT8, info.fmt, 0),
                InputData::F32(values) => (values.as_ptr().cast_mut().cast(), values.len() * 4, RKNN_TENSOR_FLOAT32, info.fmt, 0),
                InputData::Native(bytes) => (bytes.as_ptr().cast_mut().cast(), bytes.len(), native.dtype, native.fmt, 1),
            };
            c_inputs.push(RknnInput { index: c_inputs.len() as u32, buf, size: size as u32, pass_through, type_, fmt });
        }
        let mut c_outputs: Vec<RknnOutput> = Vec::with_capacity(outputs.len());
        for (index, (info, buffer)) in self.outputs.iter().zip(outputs.iter_mut()).enumerate() {
            if buffer.len() != info.n_elems {
                return Err(RknnError::Shape { model: self.name.clone(), message: format!("output {} has {} values, buffer holds {}", info.name, info.n_elems, buffer.len()) });
            }
            c_outputs.push(RknnOutput { want_float: 1, is_prealloc: 1, index: index as u32, buf: buffer.as_mut_ptr().cast(), size: (buffer.len() * 4) as u32 });
        }
        let started: Instant = Instant::now();
        // SAFETY: every `buf` points to a live caller slice of `size` bytes; the runtime only reads input buffers during this
        // call (it converts or copies them into its own memory, `pass_through` 1 included).
        let code: c_int = unsafe { (self.api.inputs_set)(self.ctx, c_inputs.len() as u32, c_inputs.as_mut_ptr()) };
        self.check("rknn_inputs_set", code)?;
        let set: Instant = Instant::now();
        // SAFETY: live context; a null extend runs blocking.
        let code: c_int = unsafe { (self.api.run)(self.ctx, std::ptr::null_mut()) };
        self.check("rknn_run", code)?;
        let ran: Instant = Instant::now();
        // SAFETY: every output is pre-allocated (`is_prealloc` 1) with a live caller buffer of exactly the output's float size.
        let code: c_int = unsafe { (self.api.outputs_get)(self.ctx, c_outputs.len() as u32, c_outputs.as_mut_ptr(), std::ptr::null_mut()) };
        self.check("rknn_outputs_get", code)?;
        // SAFETY: releases the outputs just fetched; for pre-allocated buffers the runtime frees nothing of ours.
        let code: c_int = unsafe { (self.api.outputs_release)(self.ctx, c_outputs.len() as u32, c_outputs.as_mut_ptr()) };
        self.check("rknn_outputs_release", code)?;
        let done: Instant = Instant::now();
        let mut timing: RunTiming = RunTiming::default();
        if timed {
            let mut perf: RknnPerfRun = RknnPerfRun::default();
            self.query(RKNN_QUERY_PERF_RUN, &mut perf)?;
            timing = RunTiming {
                inputs_set_us: (set - started).as_secs_f64() * 1e6,
                run_us: (ran - set).as_secs_f64() * 1e6,
                outputs_get_us: (done - ran).as_secs_f64() * 1e6,
                npu_us: perf.run_duration as f64,
            };
        }
        Ok(timing)
    }
}

impl Drop for RknnModel {
    fn drop(&mut self) {
        // SAFETY: `ctx` was created by `rknn_init` and is destroyed exactly once, here; nothing uses it afterwards.
        let _code: c_int = unsafe { (self.api.destroy)(self.ctx) };
    }
}

impl std::fmt::Debug for RknnModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RknnModel").field("name", &self.name).field("inputs", &self.inputs).field("outputs", &self.outputs).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_attr_layout_matches_the_c_header() {
        // rknn_tensor_attr on aarch64/x86_64: 4 + 4 + 64 + 256 + 4 + 4 + 3*4 + 1 (+3 pad) + 4 + 4 + 4 + 4 + 1 (+3 pad) + 4 = 376.
        assert_eq!(size_of::<RknnTensorAttr>(), 376);
        assert_eq!(size_of::<RknnInput>(), 32);
        assert_eq!(size_of::<RknnOutput>(), 24);
    }

}
