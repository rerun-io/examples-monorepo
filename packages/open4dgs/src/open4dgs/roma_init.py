"""The upstream Open4DGS init (RoMa v2 spatial matches + MEMFOF temporal flow) as an ``InitCloud``.

Upstream's ``prep/init/prep-roma-trigs-memory.py`` (``recipes/open4dgs-core-src``, installed under
``$CONDA_PREFIX/share/open4dgs-core-src``) is a single ``main()`` over ``sys.argv`` that reads a reference-camera JSON and
per-camera image folders from disk and writes the keyframes NPZ through its module-level ``_atomic_savez``.
``run_roma_init`` keeps upstream unpatched: it writes the clip's undistorted frames, the masks and the PINHOLE
reference cameras into a scratch directory that it deletes afterwards, loads the script as a module, swaps
``_atomic_savez`` for a function that captures the final arrays (upstream's per-keyframe checkpoint NPZs inside the
scratch directory still go through the original), runs ``main()`` and turns the arrays into an ``InitCloud``.

Only the cameras of ``clip`` reach upstream: pass a clip of training cameras, never held-out ones.
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from jaxtyping import Float32, Int32
from numpy import ndarray
from torch import Tensor

from open4dgs.catalog_clip import MultiViewClip
from open4dgs.init_cloud import InitCloud

INIT_SCRIPT_RELATIVE: str = "share/open4dgs-core-src/prep/init/prep-roma-trigs-memory.py"
"""Where ``recipes/open4dgs-core-src`` installs the upstream init script, relative to the environment prefix."""
METHOD: str = "roma_v2_memfof"
"""``InitCloud.method`` of this init."""
_DEFAULT_MODEL_DIR: str = "/mnt/nas/ssd-8tb/open4dgs/models"
_JPEG_QUALITY: int = 95
_REQUIRED: tuple[str, ...] = ("positions", "colors", "velocities", "times", "source_frames", "frame_start", "frame_end", "keyframe_step", "fps")
_FLAGS: tuple[str, ...] = ("has_velocity", "is_static", "dynamic_protect", "dynamic_allowed", "stage_keep")


def _model_path(variable: str, relative: str) -> Path:
    return Path(os.environ.get(variable, f"{os.environ.get('OPEN4DGS_MODEL_DIR', _DEFAULT_MODEL_DIR)}/{relative}"))


@dataclass(frozen=True, slots=True)
class RomaInitConfig:
    """Settings of one upstream init run (each maps to one upstream flag)."""

    frame_start: int
    """First keyframe, a source ``frame_index`` inside the clip (``--frame-start``)."""
    frame_end: int
    """Last frame of the training window, a source ``frame_index`` (``--frame-end``). The clip must reach
    ``frame_end + keyframe_step`` (and start ``keyframe_step`` before ``frame_start``) for the edge keyframes' tracks."""
    keyframe_step: int = 3
    """Frames between keyframes (``--frame-step``; also the MEMFOF triplet spacing)."""
    camera_neighbors: int = 4
    """RoMa pairs per camera with its nearest cameras (``--camera-neighbors``)."""
    wide_baseline_per_camera: int = 1
    """Extra wide-baseline pairs per camera (``--wide-baseline-per-camera``)."""
    roma_v2_setting: str = "fast"
    """RoMa v2 inference setting (``turbo``, ``fast``, ``base``, ``precise``, ...)."""
    memfof_max_side: int = 1920
    """MEMFOF input longer side (``--memfof-max-side``)."""
    subkeyframe_min_speed: float | None = None
    """Speed-aware sub-keyframes (``--subkeyframe-min-speed``): None keeps upstream's default (on), 0 turns them off."""
    mask_min_fg_alpha: float = 0.5
    """Mask value at or above which a pixel is foreground (``--mask-min-fg-alpha``)."""
    seed: int = 42
    device: str = "cuda"
    romav2_weights: Path = field(default_factory=lambda: _model_path("ROMAV2_WEIGHTS", "romav2/romav2.0.1.pt"))
    """RoMa v2 checkpoint (romav2-src patch 0002 reads ``ROMAV2_WEIGHTS``)."""
    memfof_checkpoint: Path = field(default_factory=lambda: _model_path("OPEN4DGS_MEMFOF_CHECKPOINT", "memfof/MEMFOF-Tartan-T-TSKH"))
    """MEMFOF ``from_pretrained`` directory."""
    extra_args: tuple[str, ...] = ()
    """More upstream flags, appended verbatim."""


def init_script_path() -> Path:
    """The installed upstream init script (``OPEN4DGS_INIT_SCRIPT`` if set, else under ``sys.prefix``)."""
    return Path(os.environ.get("OPEN4DGS_INIT_SCRIPT", str(Path(sys.prefix) / INIT_SCRIPT_RELATIVE)))


def camera_folder(name: str) -> str:
    """Upstream image folder of a camera name (``rig_01/cam_00`` -> ``rig_01__cam_00``)."""
    return name.replace("/", "__")


def reference_cameras(clip: MultiViewClip) -> list[dict[str, object]]:
    """Upstream ``reference_export.json`` cameras: PINHOLE ``[fx, fy, cx, cy]``, world->camera ``R``, ``t`` (metres)."""
    cameras: list[dict[str, object]] = []
    for index, name in enumerate(clip.camera_names):
        world_T_cam: ndarray = clip.world_T_cam[index].double().numpy()
        cam_R_world: ndarray = world_T_cam[:3, :3].T
        cam_t_world: ndarray = -cam_R_world @ world_T_cam[:3, 3]
        K: ndarray = clip.K[index].double().numpy()
        width, height = clip.sizes[index]
        cameras.append(
            {
                "name": camera_folder(name),
                "folder": camera_folder(name),
                "R": cam_R_world.tolist(),
                "t": cam_t_world.tolist(),
                "camera_model_name": "PINHOLE",
                "width": int(width),
                "height": int(height),
                "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])],
            }
        )
    return cameras


def materialise(clip: MultiViewClip, masks: Sequence[Tensor] | None, root: Path, mask_frames: Sequence[int] | None = None) -> None:
    """Write ``reference_export.json``, ``images/<folder>/<frame:06d>.jpg`` and ``masks/<folder>/<frame:06d>.png`` (0-255).

    ``mask_frames`` are the source frames of the masks (default: the clip's frames); frames without a mask get no mask
    file, which upstream reads as "no mask" (it only needs them at keyframes, not at the tracking neighbours)."""
    (root / "reference_export.json").write_text(json.dumps({"cameras": reference_cameras(clip)}))
    frame_indices: list[int] = clip.frame_indices.tolist()
    for index, name in enumerate(clip.camera_names):
        image_dir: Path = root / "images" / camera_folder(name)
        image_dir.mkdir(parents=True)
        frames: ndarray = clip.frames[index].cpu().numpy()
        for slot, frame in enumerate(frame_indices):
            cv2.imwrite(str(image_dir / f"{frame:06d}.jpg"), cv2.cvtColor(frames[slot], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if masks is None:
            continue
        mask_dir: Path = root / "masks" / camera_folder(name)
        mask_dir.mkdir(parents=True)
        camera_masks: Tensor = masks[index]
        masked_frames: list[int] = list(mask_frames) if mask_frames is not None else frame_indices
        if camera_masks.shape != (len(masked_frames), *clip.frames[index].shape[1:3]):
            raise ValueError(
                f"{name}: masks {tuple(camera_masks.shape)} do not match {len(masked_frames)} frames of {tuple(clip.frames[index].shape[1:3])}"
            )
        scale: float = 255.0 if camera_masks.dtype in (torch.bool, torch.float16, torch.float32, torch.float64) else 1.0
        mask_u8: ndarray = (camera_masks.float() * scale).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        for slot, frame in enumerate(masked_frames):
            cv2.imwrite(str(mask_dir / f"{frame:06d}.png"), mask_u8[slot])


def load_init_module(script: Path) -> ModuleType:
    """Import upstream's init script as a module (its sibling modules resolve from the script's directory)."""
    if str(script.parent) not in sys.path:
        sys.path.insert(0, str(script.parent))
    spec = importlib.util.spec_from_file_location("open4dgs_prep_roma_trigs_memory", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"cannot import the upstream init script {script}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def upstream_argv(config: RomaInitConfig, root: Path, output_path: Path, with_masks: bool) -> list[str]:
    """Upstream's command line for a materialised ``root``."""
    argv: list[str] = [
        "prep-roma-trigs-memory.py",
        "--reference-json", str(root / "reference_export.json"),
        "--images-root", str(root / "images"),
        "--output-path", str(output_path),
        "--frame-start", str(config.frame_start),
        "--frame-end", str(config.frame_end),
        "--frame-step", str(config.keyframe_step),
        "--camera-neighbors", str(config.camera_neighbors),
        "--wide-baseline-per-camera", str(config.wide_baseline_per_camera),
        "--roma-v2-setting", config.roma_v2_setting,
        "--memfof-max-side", str(config.memfof_max_side),
        "--memfof-checkpoint", str(config.memfof_checkpoint),
        "--seed", str(config.seed),
        "--device", config.device,
    ]  # fmt: skip
    if config.subkeyframe_min_speed is not None:
        argv += ["--subkeyframe-min-speed", str(config.subkeyframe_min_speed)]
    if with_masks:
        argv += ["--masks-root", str(root / "masks"), "--mask-min-fg-alpha", str(config.mask_min_fg_alpha)]
    return argv + list(config.extra_args)


def run_upstream(module: ModuleType, argv: list[str], output_path: Path) -> dict[str, ndarray]:
    """Run upstream ``main()`` with ``argv`` and return the arrays it would have written to ``output_path``."""
    captured: dict[str, ndarray] = {}
    original: Callable[..., None] = vars(module)["_atomic_savez"]

    def capture(path: Path, *, compressed: bool = False, **arrays: ndarray) -> None:
        if Path(path) == output_path:
            captured.update(arrays)
        else:
            original(path, compressed=compressed, **arrays)

    saved_argv: list[str] = sys.argv
    vars(module)["_atomic_savez"] = capture
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = saved_argv
        vars(module)["_atomic_savez"] = original
    if not captured:
        raise RuntimeError("the upstream init returned without writing its keyframes arrays")
    return captured


def _flag(arrays: Mapping[str, ndarray], name: str, order: ndarray) -> ndarray | None:
    return np.asarray(arrays[name]).reshape(-1).astype(np.bool_)[order] if name in arrays else None


def init_cloud_from_upstream(arrays: Mapping[str, ndarray], *, dataset: str, segment_id: str) -> InitCloud:
    """Convert upstream's keyframes arrays to an ``InitCloud``.

    Upstream frames are source frames; ``frame_start``/``frame_end`` are logical (0-based) and ``times`` are already
    normalised over the window. Rows are re-grouped by keyframe (upstream appends sub-keyframe rows at the end).
    Upstream-only columns (``accelerations``, ``track_flags``, ``local_stride``) are dropped.
    """
    missing: list[str] = [name for name in _REQUIRED if name not in arrays]
    if missing:
        raise ValueError(f"upstream init arrays lack {missing}")
    velocity_units: str = str(arrays.get("velocity_units", "world_per_video_frame"))
    if velocity_units != "world_per_video_frame":
        raise ValueError(f"velocity_units={velocity_units!r}: InitCloud holds world units per video frame")
    source_frame_start: int = int(arrays.get("source_frame_start", 0))
    frames: Int32[ndarray, " n"] = (np.asarray(arrays["source_frames"]).reshape(-1) - source_frame_start).astype(np.int32)
    order: ndarray = np.argsort(frames, kind="stable")
    durations: Float32[ndarray, "n 1"] | None = (
        np.asarray(arrays["durations"], dtype=np.float32).reshape(-1, 1)[order] if "durations" in arrays else None
    )
    return InitCloud(
        positions=np.ascontiguousarray(np.asarray(arrays["positions"], dtype=np.float32)[order]),
        velocities=np.ascontiguousarray(np.asarray(arrays["velocities"], dtype=np.float32)[order]),
        colors=np.ascontiguousarray(np.asarray(arrays["colors"], dtype=np.float32)[order]),
        times=np.ascontiguousarray(np.asarray(arrays["times"], dtype=np.float32).reshape(-1, 1)[order]),
        frames=np.ascontiguousarray(frames[order]),
        frame_start=int(arrays["frame_start"]),
        frame_end=int(arrays["frame_end"]),
        keyframe_step=int(arrays["keyframe_step"]),
        fps=float(arrays["fps"]),
        source_frame_start=source_frame_start,
        dataset=dataset,
        segment_id=segment_id,
        method=METHOD,
        durations=durations,
        **{name: _flag(arrays, name, order) for name in _FLAGS},
    )


def run_roma_init(
    clip: MultiViewClip,
    masks: Sequence[Tensor] | None,
    config: RomaInitConfig,
    *,
    dataset: str,
    segment_id: str,
    mask_frames: Sequence[int] | None = None,
) -> InitCloud:
    """Run the upstream RoMa v2 + MEMFOF init on every camera of ``clip`` and return its cloud.

    Args:
        clip: Undistorted training cameras only (held-out cameras must not be in it). Its frames must be contiguous
            (``frame_step=1``) and cover ``config.frame_start - keyframe_step .. config.frame_end + keyframe_step``.
        masks: Per camera, foreground masks ``[frames, h, w]`` aligned with ``clip.frames`` (bool / float in [0, 1] /
            uint8 0-255); None runs without masks (upstream then keeps the background too).
        mask_frames: Source frames of ``masks`` when they cover only part of the clip (e.g. the training window
            without the tracking margin); default the clip's frames.
        config: Upstream settings.
        dataset: Catalog dataset of the segment (recorded in the cloud).
        segment_id: Segment id (recorded in the cloud).
    """
    if masks is not None and len(masks) != len(clip.camera_names):
        raise ValueError(f"{len(masks)} mask stacks for {len(clip.camera_names)} cameras")
    frame_indices: list[int] = clip.frame_indices.tolist()
    if frame_indices != list(range(frame_indices[0], frame_indices[0] + len(frame_indices))):
        raise ValueError("the clip's frames must be contiguous (read it with frame_step=1)")
    if not frame_indices[0] <= config.frame_start < config.frame_end <= frame_indices[-1]:
        raise ValueError(f"window {config.frame_start}..{config.frame_end} is outside the clip {frame_indices[0]}..{frame_indices[-1]}")
    module: ModuleType = load_init_module(init_script_path())
    root: Path = Path(tempfile.mkdtemp(prefix="open4dgs-roma-init-"))
    saved_weights: str | None = os.environ.get("ROMAV2_WEIGHTS")
    os.environ["ROMAV2_WEIGHTS"] = str(config.romav2_weights)
    try:
        materialise(clip, masks, root, mask_frames)
        output_path: Path = root / "keyframes.npz"
        arrays: dict[str, ndarray] = run_upstream(module, upstream_argv(config, root, output_path, masks is not None), output_path)
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if saved_weights is None:
            os.environ.pop("ROMAV2_WEIGHTS", None)
        else:
            os.environ["ROMAV2_WEIGHTS"] = saved_weights
    return init_cloud_from_upstream(arrays, dataset=dataset, segment_id=segment_id)
