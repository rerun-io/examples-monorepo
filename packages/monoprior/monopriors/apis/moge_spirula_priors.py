"""MoGe-2 (TensorRT) depth and normal priors for every cube face, written in spirula's dataset folder layout.

Every face of the 6-face cubemap is a full centred 90 deg pinhole, so MoGe runs on each face image directly and the
prior matches the training image pixel for pixel.

Output (third-party input format for spirula, an export at the boundary, never a source of truth):
``<dataset>/depths/<face>/<t_ns>.png`` 16-bit z-depth in mm (0 = invalid) and
``<dataset>/normals/<face>/<t_ns>.png`` RGB ``127.5 + 127.5 n`` in the OpenCV camera frame, black = invalid.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from jaxtyping import Bool, Float32, UInt8
from numpy import ndarray

from monopriors.cube_faces import CubeFace, depth_png_mm, normal_png_rgb
from monopriors.cube_layer import iter_framesets, open_rrd, read_cube_faces
from monopriors.models.moge_v2.export import resolve_num_tokens, token_grid_hw
from monopriors.models.moge_v2.predictor import MoGeV2GeometryOutput, MoGeV2TrtPredictor


@dataclass
class MoGeSpirulaPriorsConfig:
    """Write MoGe-2 depth + normal priors for a cube-faces layer into a spirula dataset folder."""

    faces_rrd: Path
    """The cube-faces layer ``.rrd`` (``/world/rig_NN/cam_NN`` named ``cube6_<direction>``)."""
    dataset_dir: Path
    """The spirula dataset folder; ``depths/`` and ``normals/`` are written inside."""
    timeline: str = "video_time"
    """Timeline of the face videos."""
    only_exported_images: bool = True
    """Only frames that have ``images/<face>/<t_ns>.jpg`` in the dataset (the registered ones); all frames when off
    or when the dataset has no ``images/`` yet."""
    encoder: Literal["vits", "vitb", "vitl"] = "vitl"
    """MoGe-2 encoder."""
    resolution_level: int = 9
    """MoGe detail level 0-9 (9 = 3600 tokens, an 840 x 840 square engine)."""
    batch_size: int = 8
    """TensorRT engine batch size."""
    mask_threshold: float = 0.5
    """MoGe validity probability below which depth and normal are written as invalid."""
    max_framesets: int | None = None
    """Stop after this many framesets (for a quick look)."""

    def __post_init__(self) -> None:
        """Validate ranges."""
        if self.batch_size < 1 or not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("batch_size must be positive and mask_threshold in (0, 1)")


@dataclass(frozen=True, slots=True)
class _Job:
    """One network input and where its prediction goes."""

    face: CubeFace
    t_ns: int
    rgb: UInt8[ndarray, "n n 3"]


def _exported_stamps(dataset_dir: Path, faces: tuple[CubeFace, ...]) -> dict[str, set[int]] | None:
    """Stamps per face that have an exported training image, or None when there is no ``images/`` folder."""
    images_dir: Path = dataset_dir / "images"
    if not images_dir.is_dir():
        return None
    return {face.name: {int(path.stem) for path in (images_dir / face.name).glob("*.jpg")} for face in faces}


def _write_prior(
    dataset_dir: Path, job: _Job, depth_m: Float32[ndarray, "n n"], normals: Float32[ndarray, "n n 3"], valid: Bool[ndarray, "n n"]
) -> None:
    for kind, image in (
        ("depths", depth_png_mm(depth_m, valid)),
        ("normals", cv2.cvtColor(normal_png_rgb(normals, valid), cv2.COLOR_RGB2BGR)),
    ):
        path: Path = dataset_dir / kind / job.face.name / f"{job.t_ns}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"could not write {path}")
        path.chmod(0o644)


def main(config: MoGeSpirulaPriorsConfig) -> None:
    """Run MoGe-2 on every (selected) face of every frameset and write the priors."""
    side_tokens_hw: tuple[int, int] = token_grid_hw(1.0, resolve_num_tokens(config.resolution_level))
    with open_rrd(config.faces_rrd, "faces") as layer_segment:
        faces: tuple[CubeFace, ...] = read_cube_faces(layer_segment)
        sides: set[int] = {face.width for face in faces}
        if len(sides) != 1:
            raise ValueError(f"faces of different full sizes {sorted(sides)}; one square engine expected")
        exported: dict[str, set[int]] | None = _exported_stamps(config.dataset_dir, faces) if config.only_exported_images else None
        wanted: set[int] | None = None if exported is None else set().union(*exported.values())
        for face in faces:
            for kind in ("depths", "normals"):
                (config.dataset_dir / kind / face.name).mkdir(parents=True, exist_ok=True)
        print(
            f"{len(faces)} faces, full side {sides.pop()} px, MoGe-2 {config.encoder} engine {side_tokens_hw[0]}x{side_tokens_hw[1]}, batch {config.batch_size}"
        )
        predictor: MoGeV2TrtPredictor = MoGeV2TrtPredictor(
            encoder=config.encoder,
            heads="geometry",
            network_hw_options=(side_tokens_hw,),
            resolution_level=config.resolution_level,
            batch_size=config.batch_size,
        )
        pending: list[_Job] = []
        writes: list[Future[None]] = []
        n_images: int = 0
        n_framesets: int = 0
        start: float = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:

            def flush() -> None:
                nonlocal n_images
                if not pending:
                    return
                batch: UInt8[torch.Tensor, "b n n 3"] = torch.from_numpy(np.stack([job.rgb for job in pending])).to(device="cuda")
                output: MoGeV2GeometryOutput = predictor.predict_geometry(batch)
                depth_m: Float32[ndarray, "b n n"] = output.depth_bhw.cpu().numpy()
                normals: Float32[ndarray, "b n n 3"] = output.normals_bhw3.cpu().numpy()
                valid: Bool[ndarray, "b n n"] = (output.mask_bhw > config.mask_threshold).cpu().numpy()
                for index, job in enumerate(pending):
                    writes.append(pool.submit(_write_prior, config.dataset_dir, job, depth_m[index], normals[index], valid[index]))
                n_images += len(pending)
                pending.clear()

            for frameset in iter_framesets(layer_segment, faces, config.timeline, wanted):
                for face in faces:
                    if exported is not None and frameset.t_ns not in exported[face.name]:
                        continue
                    rgb: UInt8[ndarray, "n n 3"] = frameset.rgb_by_face[face.name]
                    pending.append(_Job(face=face, t_ns=frameset.t_ns, rgb=rgb))
                    if len(pending) == config.batch_size:
                        flush()
                n_framesets += 1
                if config.max_framesets is not None and n_framesets >= config.max_framesets:
                    break
            flush()
            for write in writes:
                write.result()
        seconds: float = time.perf_counter() - start
        print(
            f"wrote priors for {n_images} images ({n_framesets} framesets) in {seconds:.1f} s = {n_images / max(seconds, 1e-9):.1f} images/s -> {config.dataset_dir}"
        )
