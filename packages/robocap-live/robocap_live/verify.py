"""Graph-tier check of robocap-live's hands layer, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

The robocap dataset has no ground truth (its ``slam_rs`` layer is our own output), so the reference is the PRE-PORT golden:
``score`` compares a golden with the same case's ``pre`` golden (the ``pre`` golden itself scores 0, ``pre2`` gives the
run-to-run noise). It reads only the two golden rrds, never the pipeline.

Golden contract (the ported graph writes the same entities on the segment's ``video_time``): the hands layer exactly as the
Rust layer writer logs it (``/world/hands/<side>/...`` in the world, ``/world/rig_00/cam_NN/pinhole/hands/<side>/...`` per
camera), plus ``/verify/hands/run``: static ``AnyValues`` with the layer's counts (``LayerSummary``) and the run's wall time.
"""

import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase, golden_path
from datafusion import col, lit
from jaxtyping import Float32, Float64, Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry, DatasetView
from simplecv.catalog_video import CatalogVideo, read_catalog_videos

from robocap_live import _core
from robocap_live.apis.hands_layer import DATASET, Config, LayerRun, default_ort_dylib, main
from robocap_live.catalog_segment import CAMERA_ENTITIES, TIMELINE, group_framesets, nearest_poses, read_poses, read_rig, y_planes

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
"""``packages/robocap-live``: case paths (``models_dir``) are relative to it."""
KEYFRAME_LEAD_NS: int = 5_000_000_000
"""How far before a window the packet read starts, so the decoder begins at a keyframe."""
SIDES: tuple[str, ...] = ("left", "right")
KEYNET: tuple[str, ...] = tuple(f"/world/rig_00/cam_{camera:02d}/pinhole/hands/{side}/keynet" for camera in range(6) for side in SIDES)
"""Per camera and hand: KeyNet's 21 image keypoints (stored-video pixels) on every frameset the hand was cropped."""
LANDMARKS: tuple[str, ...] = tuple(f"/world/hands/{side}/keypoints" for side in SIDES)
"""Per hand: the fitted 21 world landmarks (metres) on every frameset the hand was reported."""


class _WindowedEntry:
    """A dataset view whose readers keep only ``lo <= video_time < hi`` (a short clip's packets, not the whole segment)."""

    def __init__(self, view: DatasetEntry | DatasetView, lo: int, hi: int) -> None:
        self.view, self.lo, self.hi = view, lo, hi

    def filter_segments(self, segment: str) -> "_WindowedEntry":
        return _WindowedEntry(self.view.filter_segments(segment), self.lo, self.hi)

    def filter_contents(self, contents: list[str]) -> "_WindowedEntry":
        return _WindowedEntry(self.view.filter_contents(contents), self.lo, self.hi)

    def reader(self, index: str) -> object:
        t = col(index).cast(pa.int64())
        return self.view.reader(index=index).filter((t >= lit(self.lo)) & (t < lit(self.hi)))


def _pre_port_window(case: Case, partial: Path) -> dict[str, float]:
    """The pre-port hands layer of a window: ``robocap_live.catalog_segment``'s readers, decoder and grouping and the
    ``_core.HandsLayer`` loop of ``apis/hands_layer.main``, fed only the window's framesets (``main`` reads whole segments)."""
    assert case.dataset is not None and case.segment is not None and case.window_s is not None
    lo, hi = round(case.window_s[0] * 1e9), round(case.window_s[1] * 1e9)
    started: float = time.perf_counter()
    dataset: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()  # an entry must outlive its queries
    rig: _core.Rig = read_rig(dataset, case.segment, f"catalog {DATASET}")
    windowed = _WindowedEntry(dataset, lo - KEYFRAME_LEAD_NS, hi)
    videos = read_catalog_videos(windowed, case.segment, [f"{camera}/pinhole/video" for camera in CAMERA_ENTITIES], TIMELINE)  # type: ignore[arg-type]
    trimmed: list[CatalogVideo] = []
    skip: list[int] = []
    inside: list[Int64[ndarray, " n"]] = []
    for camera, video in enumerate(videos):
        keep: Int64[ndarray, " k"] = np.flatnonzero(video.t_ns >= lo)
        first_keyframe: int = max(i for i, keyframe in enumerate(video.keyframes[: int(keep[0]) + 1]) if keyframe)
        trimmed.append(CatalogVideo(video.times[first_keyframe:], video.samples[first_keyframe:], video.keyframes[first_keyframe:], video.codec))
        skip.append(int(keep[0]) - first_keyframe)
        inside.append(video.t_ns[keep])
        if not np.array_equal(keep, np.arange(keep[0], keep[-1] + 1)):
            raise ValueError(f"camera {camera}: the window's frames are not contiguous")
    index = group_framesets(inside, int(case.params["group_tolerance_ns"]))
    poses = nearest_poses(index.t_ns, read_poses(dataset, case.segment), int(case.params["pose_tolerance_ns"]))
    read_s: float = time.perf_counter() - started
    core = _core.HandsLayer(
        rig, partial, case.segment, poses.t_ns, poses.world_from_rig, models_dir=PACKAGE_DIR / case.params["models_dir"],
        device=case.params["device"], ort_dylib=default_ort_dylib(), ort_threads=int(case.params["ort_threads"]), overlays=case.params["overlays"],
    )  # fmt: skip
    decoders: list[Iterator[UInt8[ndarray, "h w"]]] = [y_planes(video, 30, int(case.params["decode_threads"])) for video in trimmed]
    for decoder, count in zip(decoders, skip, strict=True):
        for _ in range(count):
            next(decoder)
    loop_started: float = time.perf_counter()
    try:
        for row in range(len(index.t_ns)):
            t_ns: int = int(index.t_ns[row])
            frames: list[UInt8[ndarray, "h w"] | None] = [next(decoders[c]) if index.frame[row, c] >= 0 else None for c in range(len(decoders))]
            camera_t_ns: list[int] = [int(t) if f >= 0 else t_ns for t, f in zip(index.cam_t_ns[row], index.frame[row], strict=True)]
            core.push(t_ns, frames, camera_t_ns)
        summary = core.finish()
    except BaseException:
        core.abort()
        raise
    return _counts(summary) | {"read_s": read_s, "loop_s": time.perf_counter() - loop_started}


def _pre_port_segment(case: Case, output_dir: Path) -> tuple[Path, dict[str, float]]:
    """The pre-port hands layer of a whole segment: ``apis/hands_layer.main`` itself, with registration off."""
    assert case.segment is not None
    layer = main(Config(
        segment=case.segment, output_dir=output_dir, catalog=SegmentRef(case.dataset or "robocap", case.segment).catalog_url,
        models_dir=PACKAGE_DIR / case.params["models_dir"], device=case.params["device"],  # type: ignore[arg-type]
        ort_threads=int(case.params["ort_threads"]), overlays=case.params["overlays"],  # type: ignore[arg-type]
        decode_threads=int(case.params["decode_threads"]), register=False,
    ))  # fmt: skip
    return Path(layer.output), _counts(layer) | {"read_s": layer.catalog_read_s, "loop_s": layer.loop_s}


def _counts(layer: _core.LayerSummary | LayerRun) -> dict[str, float]:
    return {"framesets": layer.framesets, "with_pose": layer.with_pose, "held_pose": layer.held_pose, "tracked_left": layer.tracked[0],
            "tracked_right": layer.tracked[1], "reported_left": layer.reported[0], "reported_right": layer.reported[1], "scale": layer.scale,
            "scale_final": float(layer.scale_final)}  # fmt: skip


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """Pre-port: the current hands-layer pipeline (CPU ONNX Runtime) on the case's segment or window; its layer file's chunks
    are copied into ``recording`` unchanged, and the counts go to ``/verify/hands/run``."""
    started: float = time.perf_counter()
    with TemporaryDirectory() as tmp:
        if case.window_s is not None:
            layer: Path = Path(tmp) / f".{case.segment}.window.partial"
            counts: dict[str, float] = _pre_port_window(case, layer)
        else:
            layer, counts = _pre_port_segment(case, Path(tmp))
        recording.send_chunks(rr.chunk.RrdReader(layer).stream())
    counts["wall_s"] = time.perf_counter() - started
    recording.log("/verify/hands/run", rr.AnyValues(**cast(dict[str, Any], counts)), static=True)
    print(f"{case.name} {phase}: {counts}")


def _rows(table: pa.Table, entity: str, archetype: str, width: int) -> tuple[Int64[ndarray, " n"], Float32[ndarray, "n k w"]]:
    """The non-empty rows of ``<entity>:<archetype>:positions``: their ``video_time`` and the points (every row holds 21)."""
    column: str = f"{entity}:{archetype}:positions"
    if column not in table.column_names:
        return np.empty(0, dtype=np.int64), np.empty((0, 21, width), dtype=np.float32)
    cells: pa.Array = table[column].combine_chunks()
    lengths: Int64[ndarray, " r"] = np.asarray(pc.call_function("list_value_length", [cells]).fill_null(0), dtype=np.int64)
    keep: np.ndarray = lengths > 0
    if not np.all(lengths[keep] == 21):
        raise ValueError(f"{column}: rows with {sorted(set(lengths[keep].tolist()))} points; expected 21")
    times: Int64[ndarray, " n"] = np.asarray(table[TIMELINE].combine_chunks().cast(pa.int64()))[keep]
    points: Float32[ndarray, "n k w"] = np.asarray(cells.filter(pa.array(keep)).flatten().flatten(), dtype=np.float32).reshape(-1, 21, width)
    return times, points


def _read(golden: Path) -> tuple[pa.Table, dict[str, float]]:
    """The KeyNet and landmark rows of a golden, and its static run counts."""
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        table: pa.Table = dataset.filter_contents([*KEYNET, *LANDMARKS]).reader(index=TIMELINE).to_arrow_table()
        statics: pa.Table = dataset.filter_contents(["/verify/hands/run"]).reader(index=None).to_arrow_table()
    counts: dict[str, float] = {
        c.removeprefix("/verify/hands/run:"): float(np.asarray(statics[c].to_pylist()[0]).reshape(-1)[0])
        for c in statics.column_names
        if c.startswith("/verify/hands/run:")
    }
    return table, counts


def score(case: Case, golden: Path) -> dict[str, float]:
    """Compare a golden with the case's ``pre`` golden (pre-port reference; ``pre`` itself scores 0).

    Banded: ``kp2d_px_mean`` / ``kp2d_px_p95`` (KeyNet image keypoints of the frameset, camera and hand both goldens hold, in
    stored-video pixels), ``landmarks_mm_mean`` (world landmarks of the hands both report, mm), ``reported_delta_pct``
    (frameset-hands reported by exactly one of the two, % of the reference's). Reported, not banded: the golden's counts.
    """
    reference: Path = golden if golden.with_suffix("").name == f"{case.name}.pre" else golden_path("robocap-live", "hands", case.name, "pre")
    if not reference.exists():
        raise FileNotFoundError(f"{reference}: the pre-port reference golden of case {case.name!r} is missing (run --phase pre first)")
    ours, counts = _read(golden)
    theirs, _ = _read(reference)
    kp2d: list[Float32[ndarray, " m"]] = []
    for entity in KEYNET:
        (t_ours, p_ours), (t_ref, p_ref) = _rows(ours, entity, "Points2D", 2), _rows(theirs, entity, "Points2D", 2)
        _, i_ours, i_ref = np.intersect1d(t_ours, t_ref, assume_unique=True, return_indices=True)
        kp2d.append(np.linalg.norm(p_ours[i_ours] - p_ref[i_ref], axis=-1).reshape(-1))
    landmarks_mm: list[Float32[ndarray, " m"]] = []
    reported_ours: set[tuple[str, int]] = set()
    reported_ref: set[tuple[str, int]] = set()
    for entity in LANDMARKS:
        (t_ours, p_ours), (t_ref, p_ref) = _rows(ours, entity, "Points3D", 3), _rows(theirs, entity, "Points3D", 3)
        _, i_ours, i_ref = np.intersect1d(t_ours, t_ref, assume_unique=True, return_indices=True)
        landmarks_mm.append(1e3 * np.linalg.norm(p_ours[i_ours] - p_ref[i_ref], axis=-1).reshape(-1))
        reported_ours |= {(entity, int(t)) for t in t_ours}
        reported_ref |= {(entity, int(t)) for t in t_ref}
    pixels: Float64[ndarray, " m"] = np.concatenate(kp2d).astype(np.float64)
    millimetres: Float64[ndarray, " m"] = np.concatenate(landmarks_mm).astype(np.float64)
    return {
        "kp2d_px_mean": float(pixels.mean()) if pixels.size else float("nan"),
        "kp2d_px_p95": float(np.percentile(pixels, 95)) if pixels.size else float("nan"),
        "landmarks_mm_mean": float(millimetres.mean()) if millimetres.size else float("nan"),
        "reported_delta_pct": 100.0 * len(reported_ours ^ reported_ref) / len(reported_ref) if reported_ref else float("nan"),
        "kp2d_matched": float(pixels.size // 21),
        "reported": float(len(reported_ours)),
        **{f"run_{name}": value for name, value in counts.items()},
    }
