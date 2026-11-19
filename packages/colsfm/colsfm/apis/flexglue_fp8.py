"""Calibrate FlexGlue's FP8 matcher on image sequences and write its graph beside the prepared matcher.

Run once per machine, model and `--min-covisibility` (each threshold has its own graph), before matching (a run refuses
to start without the FP8 graph): `pixi run -e colsfm
colsfm-flexglue-fp8` calibrates on the eight cameras of the Galileo sample. The transformer's activations depend on the
images only (the quantized layers come before the keypoint head), so any sequences of real frames calibrate it. A
valid FP8 graph is kept; a FlexGlue graph change (rewrite version, options) makes it stale and this recalibrates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import torch
from torch import Tensor

from colsfm.flexglue_fp8 import (
    CALIBRATION_PAIRS,
    SOURCE_PROPERTY,
    calibration_graph,
    calibration_pairs,
    fp8_source,
    fp8_targets,
    quantize,
)
from colsfm.flexglue_trt import (
    DESCRIPTOR_BATCH,
    FlexGlueTensorRT,
    descriptor_profile,
    matcher_profile,
    prepare_descriptor_graph,
    prepare_matcher_graph,
)
from colsfm.frames import ImageFolderFrames
from colsfm.matching import FLEXGLUE_MODEL_DIR, FLEXGLUE_MODEL_FILES, FlexGlueOptions
from colsfm.tensorrt_runtime import resolve_engine

IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg")
"""Frame files a calibration directory may hold."""

LATTICE_SIDE: int = 32
"""Keypoints per side of the lattice each calibration image carries: the head needs keypoints, the ranges do not."""


@dataclass
class Config:
    """The calibration sequences and the model directory."""

    image_dirs: list[Path]
    """Directories of frames, one sequence each, in name order (at least 12 frames each)."""
    pairs: int = CALIBRATION_PAIRS
    """Image pairs to calibrate on."""
    model_dir: Path = field(default_factory=lambda: FLEXGLUE_MODEL_DIR)
    """FlexGlue's graphs (`colsfm-download-flexglue`)."""
    min_covisibility: float = FlexGlueOptions().min_covisibility
    """The matching run's `MatchingOptions.flexglue_min_covisibility`: each threshold is folded into its own FP8 graph."""


def main(config: Config) -> Path:
    """Calibrate the activation maxima and write the FP8 matcher graph, then build its engine.

    Args:
        config: Sequences and model directory.

    Returns:
        The FP8 graph.
    """
    started: float = time.perf_counter()
    models: Path = config.model_dir
    options: FlexGlueOptions = FlexGlueOptions(min_covisibility=config.min_covisibility)
    matcher_graph: Path = prepare_matcher_graph(models / FLEXGLUE_MODEL_FILES["matcher_model_path_fp16"], models / "trt", options)
    destination: Path = models / options.fp8_file()
    if fp8_source(destination) == matcher_graph.name:
        print(f"[colsfm] flexglue fp8: {destination.name} present, calibrated from {matcher_graph.name}")
        return destination
    sequences: list[list[Path]] = [sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) for directory in config.image_dirs]
    pairs: list[tuple[Path, Path]] = calibration_pairs(sequences, config.pairs)
    descriptor_engine: Path = resolve_engine(
        prepare_descriptor_graph(models / FLEXGLUE_MODEL_FILES["descriptor_model_path"], models / "trt"), descriptor_profile(), precision="fp16-trunk"
    )
    model: onnx.ModelProto = onnx.load(matcher_graph)
    targets = fp8_targets(model.graph)
    names: list[str] = targets.calibrated()
    images: list[Path] = list(dict.fromkeys(path for pair in pairs for path in pair))
    ids: dict[Path, int] = {path: index for index, path in enumerate(images)}
    grid: np.ndarray = (np.arange(LATTICE_SIDE, dtype=np.float32) + 0.5) / LATTICE_SIDE
    maxima: dict[str, Tensor] = {}
    with TemporaryDirectory(prefix=".flexglue-fp8-", dir=models / "trt") as work:
        calibration_path: Path = Path(work) / f"{matcher_graph.stem}_calibration.onnx"
        onnx.save(calibration_graph(model, targets), calibration_path)
        # The calibration graph serves as the matcher: encoding needs none, and calibration runs only this graph.
        flexglue: FlexGlueTensorRT = FlexGlueTensorRT(
            descriptor_engine, resolve_engine(calibration_path, matcher_profile()), cache_images=len(images)
        )
        for start in range(0, len(images), DESCRIPTOR_BATCH):
            batch: list[Path] = images[start : start + DESCRIPTOR_BATCH]
            decoded: list[tuple[int, Tensor, np.ndarray]] = []
            for path in batch:
                ((_, rgb),) = ImageFolderFrames(path.parent).frames([path.name])
                height, width = int(rgb.shape[1]), int(rgb.shape[2])
                lattice: np.ndarray = np.stack(np.meshgrid(grid * width, grid * height, indexing="xy"), -1).reshape(-1, 2).astype(np.float32)
                decoded.append((ids[path], rgb, lattice))
            flexglue.encode(decoded)
        for first, second in pairs:
            a, b = flexglue.images[ids[first]], flexglue.images[ids[second]]
            outputs: dict[str, Tensor] = flexglue.matchers[0](
                {
                    "features0": a.features,
                    "features1": b.features,
                    "kpts0": a.keypoints,
                    "kpts1": b.keypoints,
                    "image_size0": flexglue.size,
                    "image_size1": flexglue.size,
                }
            )
            for name in names:  # running maxima on the GPU: no host round trip per pair
                largest: Tensor = outputs[name].abs().amax().float()
                maxima[name] = largest if name not in maxima else torch.maximum(maxima[name], largest)
        del flexglue  # the runtimes and the cached feature maps: free them for the FP8 engine build
    torch.cuda.empty_cache()
    amax: dict[str, float] = {name: float(value) for name, value in maxima.items()}
    quantized: onnx.ModelProto = quantize(model, targets, amax)
    onnx.helper.set_model_props(
        quantized,
        {
            "colsfm_fp8_calibration": f"{len(pairs)} pairs over {', '.join(str(d) for d in config.image_dirs)}",
            SOURCE_PROPERTY: matcher_graph.name,
        },
    )
    with TemporaryDirectory(prefix=".flexglue-fp8-", dir=models) as stage:
        staged: Path = Path(stage) / destination.name
        onnx.save(quantized, staged)
        staged.replace(destination)
    resolve_engine(destination, matcher_profile())
    print(
        f"[colsfm] flexglue fp8: {len(targets.linears)} linears and {len(targets.cores)} attention cores to FP8, maxima from {len(pairs)} pairs "
        f"({min(amax.values()):.3g} .. {max(amax.values()):.3g}) -> {destination.name}, engine built, {time.perf_counter() - started:.0f} s"
    )
    return destination
