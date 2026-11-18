"""ALIKED feature extraction — the Python replacement for `feature_extractor_main`'s per-image half.

COLMAP 4.2 ships ALIKED itself (`FeatureExtractorType.ALIKED_N16ROT`,
`ALIKED_N32`), so this stage is a thin, typed driver over
`pycolmap.extract_features` rather than a TensorRT runner. Two facts make that
work:

* **The pipeline owns the identifiers.** `colsfm.database.create_database`
  writes the cameras, rig, frames and images first; COLMAP's `ImageReader` then
  looks each name up, finds the row and keeps its `image_id`, `camera_id` and
  `frame_id`. Nothing is renumbered and no camera is invented.
* **The variant matches the blob.** `pycusfm/models/aliked_lightglue/aliked.onnx`
  carries a 32-channel SDDH offset convolution (`Conv[32, 128, 3, 3]` feeding a
  `Conv[32, 32, 1, 1]`), i.e. `M = 16` sampling positions, the ALIKED-n16
  architecture — corroborated by the RaCo export's own checkpoint,
  `~/.cache/torch/hub/checkpoints/aliked-n16.pth`. `ALIKED_N32` would be `M = 32`
  (`Conv[64, 128, 3, 3]`). So `ALIKED_N16ROT` is the closest COLMAP variant and
  the default here. Its weights are trained with rotation augmentation, so the
  descriptors are not the blob's bit for bit — the plan does not ask for that.

COLMAP downloads its ONNX models on first use from its own GitHub release
assets and caches them in `~/.cache/colmap/<sha256>-<filename>`; nothing ships
inside the conda package.

**Environment note.** ONNX Runtime's CUDA execution provider needs `libcudnn.so`
for the convolutions, and the `colsfm` pixi environment declares no cuDNN. The
failure is not an exception: ONNX Runtime throws inside a COLMAP worker thread
and the process aborts with SIGABRT. `resolve_device` therefore probes for cuDNN
*before* the call and falls back to the CPU provider, which is ~50x slower
(1.8 s per 1920x1200 image against 0.03 s on the GPU).

**The other backend.** `FeatureOptions.backend = "raco"` runs RaCo-ALIKED on a batch-dynamic TensorRT engine
(`colsfm.features_raco`), which is what a run does by default. Everything above this paragraph describes the
`pycolmap` path, the ablation for a machine with no engine: `raco` needs `tensorrt`, `cuda-python`, an engine built
for this GPU and the exported graph (`colsfm.model_assets`, `pixi run -e colsfm colsfm-raco-export`). Both write the
same two database columns for the same `image_id`s, so the choice is invisible to every later stage. (The fork's third
backend, the cuSFM blob's own `aliked.onnx` engine, is not part of this package.)
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal, TypeAlias

import pycolmap

from colsfm.database import image_ids_by_name, keypoint_counts
from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSource, is_catalog_input, read_manifest
from colsfm.keypoint_masks import MaskSource, open_mask_source

AlikedVariant: TypeAlias = Literal["ALIKED_N16ROT", "ALIKED_N32"]
"""The two ALIKED graphs COLMAP 4.2 can download and run."""

FeatureBackend: TypeAlias = Literal["pycolmap", "raco"]
"""Which ALIKED runs: COLMAP's own ONNX one, or RaCo-ALIKED on a TensorRT engine (the default).

`raco` is `colsfm.features_raco` — fabio-sim's RaCo-ALIKED through a batch-dynamic engine that runs 8 images at once.
A checkout without its graph is told to export one or to pass `--features-backend pycolmap` (`colsfm.model_assets`)."""

RacoEngineChoice: TypeAlias = Literal["auto", "fixed", "dynamic"]
"""Which RaCo engine runs an image once `native_resolution` is on.

`auto` picks per image size: the fixed-shape engine when the size group is one
that engine already accepts (Galileo's 1920x1200), the shape-dynamic one
otherwise (KITTI's 1226x370). `fixed` and `dynamic` force one of them, which is
what a measurement of the two wants; see `colsfm.features_raco`."""

DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]
"""Requested compute device; `auto` picks CUDA when it is actually usable."""

ResolvedDevice: TypeAlias = Literal["cuda", "cpu"]
"""The device a stage really ran on."""

TensorRTExtraction: TypeAlias = tuple[dict[int, int], float]
"""What `colsfm.features_raco.extract_raco` returns: keypoints per image id, and seconds."""

BLOB_DETECTOR_THRESHOLD: Final[float] = 0.005
"""`aliked_detector.detector_threshold` in `pycusfm/configs/isaac/keypoint_creation_config.pb.txt`."""

RACO_MIN_SCORE: Final[float] = 0.0
"""Score gate on the `raco` backend, which is none.

The RaCo graph's `scores` output is upstream's selection *priority*, a strictly
decreasing `arange(2048, 0, -1) / 2048`, not a detector response — see
`colsfm.features_raco`. Its smallest value, 0.00049, is below the blob's 0.005
`detector_threshold`, so reusing `FeatureOptions.min_score` here would drop ten
points per image on the strength of a number that means nothing."""

BLOB_MAX_KEYPOINTS: Final[int] = 2048
"""Keypoints the blob's ONNX detector head emits per image, which is what actually binds.

`max_key_points_for_tensorrt: 4000` never fires: the shipped graph's top-k is
fixed at 2048, so every Galileo keyframe holds 2047-2048 points
(feature_extractor_main.md §6.3).
"""

CUDNN_LIBRARY_NAME: Final[str] = "libcudnn.so"
"""What ONNX Runtime's CUDA provider dlopens before it will run a convolution."""


@dataclass(frozen=True, slots=True)
class FeatureOptions:
    """How ALIKED runs over one image set."""

    backend: FeatureBackend = "pycolmap"
    """Which ALIKED implementation runs; see `FeatureBackend`.

    `pycolmap` here even though a *run* defaults to `raco`: this is the stage's own
    API, taken by tests and tools that want features and no TensorRT engine, and it
    should not stop working on a machine that has no graph. `colsfm.run_config`
    always passes the run's choice explicitly, so the two never disagree in a run.

    On `raco` the `variant`, `max_image_size`, `num_threads`, `device` and
    `gpu_index` fields are not read: the engine runs on CUDA device 0."""
    variant: AlikedVariant = "ALIKED_N16ROT"
    """ALIKED graph; `ALIKED_N16ROT` is the architecture the blob's ONNX uses."""
    max_num_features: int = BLOB_MAX_KEYPOINTS
    """Keypoint budget per image, matching the blob's fixed top-2048 head."""
    min_score: float = BLOB_DETECTOR_THRESHOLD
    """Detector score gate; the blob's `detector_threshold`, applied with `>=`."""
    max_image_size: int = -1
    """Longest side COLMAP resizes to before detection; -1 keeps the original size."""
    num_threads: int = -1
    """COLMAP extraction threads; -1 lets COLMAP choose."""
    device: DeviceChoice = "auto"
    """Compute device; `auto` falls back to the CPU provider when cuDNN is missing."""
    gpu_index: str = "-1"
    """CUDA device index as COLMAP's comma-separated string; `"-1"` lets COLMAP choose."""
    native_resolution: bool = True
    """On `raco` only: run each image at its own size instead of stretching it.

    True runs a 1226x370 KITTI frame over 1248x384 rather than 1920x1200, on
    whichever engine `raco_engine` picks for that size. False is the legacy
    stretch on the fixed-shape graph, which is what `docs/kitti-06-results.md`
    §9 measured. `pycolmap` ignores it: it is already native."""
    raco_batch_pixels: int | None = None
    """On `raco` only: the shape-dynamic engines' pixel budget per execution (`PipelineOptions.raco_batch_pixels`);
    None keeps `colsfm.features_raco.DYNAMIC_BATCH_PIXEL_BUDGET`."""
    raco_engine: RacoEngineChoice = "auto"
    """On `raco` with `native_resolution` only: which engine runs each size.

    `auto` sends a size group the fixed-shape engine already accepts through
    that engine and everything else through the shape-dynamic one, so Galileo
    keeps its 2.7 s stage and KITTI keeps its 4.87 ms/image. See
    `RacoEngineChoice`."""


DEFAULT_FEATURE_OPTIONS: Final[FeatureOptions] = FeatureOptions()
"""Shared immutable default, so the signatures below hold no constructor call."""


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """What one extraction run produced."""

    keypoint_counts: dict[int, int] = field(default_factory=dict)
    """Keypoints stored per `image_id`."""
    device: ResolvedDevice = "cpu"
    """The device the run actually used."""
    elapsed_seconds: float = 0.0
    """Wall time of the `pycolmap.extract_features` call."""

    def summary(self) -> str:
        """One line naming the image count, the keypoint range and the wall time.

        Returns:
            A printable summary; `"no images"` when nothing was extracted.
        """
        if not self.keypoint_counts:
            return f"extracted no images on {self.device} in {self.elapsed_seconds:.2f}s"
        counts: list[int] = sorted(self.keypoint_counts.values())
        return (
            f"extracted {len(counts)} images on {self.device} in {self.elapsed_seconds:.2f}s, "
            f"keypoints {counts[0]}-{counts[-1]} (median {counts[len(counts) // 2]})"
        )


def cudnn_is_available() -> bool:
    """Report whether ONNX Runtime will find cuDNN.

    Returns:
        True when `libcudnn.so` loads. ONNX Runtime dlopens it by that exact
        soname-less name, so a versioned `libcudnn.so.9` on its own is not enough.
    """
    try:
        ctypes.CDLL(CUDNN_LIBRARY_NAME)
    except OSError:
        return False
    return True


def resolve_device(device: DeviceChoice = "auto") -> ResolvedDevice:
    """Turn a device request into the device COLMAP's ONNX models can really use.

    Args:
        device: `auto`, `cuda` or `cpu`.

    Returns:
        `cuda` when a CUDA build, a CUDA device and cuDNN are all present, else `cpu`.

    Raises:
        RuntimeError: When `cuda` is requested and any of the three is missing.
            Letting the call through instead aborts the process: ONNX Runtime
            raises inside a COLMAP worker thread, which terminates.
    """
    if device == "cpu":
        return "cpu"
    reasons: list[str] = []
    if not pycolmap.has_cuda:
        reasons.append("pycolmap was built without CUDA")
    elif pycolmap.get_num_cuda_devices() < 1:
        reasons.append("no CUDA device is visible")
    if not cudnn_is_available():
        reasons.append(f"{CUDNN_LIBRARY_NAME} is not loadable, so ONNX Runtime's CUDA provider cannot run convolutions")
    if not reasons:
        return "cuda"
    if device == "cuda":
        raise RuntimeError("CUDA was requested but is unusable: " + "; ".join(reasons))
    print(f"colsfm: falling back to the CPU provider because {'; '.join(reasons)}")
    return "cpu"


def _extraction_options(options: FeatureOptions, device: ResolvedDevice) -> pycolmap.FeatureExtractionOptions:
    """Build the pycolmap options for one extraction run.

    Args:
        options: The typed settings.
        device: The already-resolved device.

    Returns:
        `FeatureExtractionOptions` with the ALIKED sub-options filled in. In
        COLMAP 4.2 `use_gpu`, `gpu_index`, `num_threads` and `max_image_size`
        live on this object, not on the per-extractor one.
    """
    extraction_options: pycolmap.FeatureExtractionOptions = pycolmap.FeatureExtractionOptions()
    extraction_options.type = getattr(pycolmap.FeatureExtractorType, options.variant)
    extraction_options.use_gpu = device == "cuda"
    extraction_options.gpu_index = options.gpu_index
    extraction_options.num_threads = options.num_threads
    extraction_options.max_image_size = options.max_image_size
    extraction_options.aliked.max_num_features = options.max_num_features
    extraction_options.aliked.min_score = options.min_score
    return extraction_options


def extract_features(
    database_path: Path,
    image_root: Path,
    image_names: Sequence[str],
    options: FeatureOptions = DEFAULT_FEATURE_OPTIONS,
    frames: FrameSource | None = None,
) -> ExtractionReport:
    """Run ALIKED over the named images and store keypoints and descriptors.

    The database must already hold an image row per name (see
    `colsfm.database.create_database`); COLMAP reuses those rows and their
    cameras rather than creating its own.

    Args:
        database_path: An existing COLMAP database.
        image_root: The run's input directory: an image folder, or a catalog input whose keyframes the frame
            source decodes from the segment's video streams (`colsfm.frames`).
        image_names: Images to extract, as stored in the database.
        options: Extraction settings.
        frames: The run's frame source, for the `raco` backend (it reads pixels); one is opened on `image_root`
            for this call when None.

    Returns:
        Per-image keypoint counts, the device used and the wall time.

    Raises:
        FileNotFoundError: When the database or the image root is missing.
        RuntimeError: When `options.device` is `cuda` and CUDA is unusable.
        ImportError: When `options.backend` is `raco` and TensorRT or
            `cuda-python` is not importable.
        ValueError: When the catalog input names keypoint masks and the backend is not `raco`.
    """
    if not database_path.is_file():
        raise FileNotFoundError(f"No COLMAP database at {database_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"No image directory at {image_root}")

    # A catalog input may name keypoint masks per camera stream; only the raco backend applies them.
    masks: MaskSource | None = open_mask_source(read_manifest(image_root / FRAME_SOURCE_FILE) if is_catalog_input(image_root) else None)
    if masks is not None and options.backend != "raco":
        raise ValueError(f"keypoint masks are applied by the raco backend only, not {options.backend}")

    if options.backend == "raco":
        # Imported here, not at module scope: TensorRT and `cuda-python` are only needed by this branch, and a machine
        # without a usable engine must still be able to run the `pycolmap` backend.
        from colsfm.features_raco import extract_raco
        from colsfm.frames import open_frame_source

        # The graph's `scores` output is a selection rank, not a detector response, so the blob's
        # `detector_threshold` is meaningless against it; `colsfm.features_raco` documents why the gate is dropped.
        source: FrameSource = open_frame_source(image_root) if frames is None else frames
        try:
            extracted: TensorRTExtraction = extract_raco(
                database_path,
                image_root,
                image_names,
                min_score=RACO_MIN_SCORE,
                max_num_features=options.max_num_features,
                native_resolution=options.native_resolution,
                raco_engine=options.raco_engine,
                batch_pixels=options.raco_batch_pixels,
                frames=source,
                masks=masks,
            )
        finally:
            if frames is None:
                source.close()
        tensorrt_report: ExtractionReport = ExtractionReport(keypoint_counts=extracted[0], device="cuda", elapsed_seconds=extracted[1])
        print(f"colsfm.features[{options.backend}]: {tensorrt_report.summary()}")
        return tensorrt_report

    device: ResolvedDevice = resolve_device(options.device)
    extraction_options: pycolmap.FeatureExtractionOptions = _extraction_options(options, device)
    started: float = time.perf_counter()
    pycolmap.extract_features(
        database_path,
        image_root,
        image_names=list(image_names),
        extraction_options=extraction_options,
        device=pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu,
    )
    elapsed_seconds: float = time.perf_counter() - started

    name_to_id: dict[str, int] = image_ids_by_name(database_path)
    wanted_ids: list[int] = [name_to_id[name] for name in image_names if name in name_to_id]
    report: ExtractionReport = ExtractionReport(
        keypoint_counts=keypoint_counts(database_path, wanted_ids),
        device=device,
        elapsed_seconds=elapsed_seconds,
    )
    print(f"colsfm.features: {report.summary()}")
    return report
