"""Replay the language-neutral golden set (``golden_fit.npz`` + ``golden_fit.json``) through the native fit, numpy only.

The golden set holds real tracker fit calls (recorded from handtrack's torch fit, see handtrack's record/export scripts). Every
call's warm and cold hands are fitted in one native call, as the tracker does; the time per call and the difference to the recorded torch
result are reported. Runs on any host with numpy and a built ``handfit._core`` (the RoboCap cap, the Mac mini): no torch needed.
"""

import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray

from handfit import FitOutput, HandFitter

MODEL_FIELDS: tuple[str, ...] = (
    "joint_rotation_axes",
    "joint_rest_positions",
    "landmark_rest_positions",
    "landmark_rest_bone_weights",
    "joint_limits",
    "landmark_rest_bone_indices",
    "joint_parent",
    "joint_frame_index",
    "joint_first_child",
    "joint_next_sibling",
)
CONFIG_FIELDS: tuple[str, ...] = (
    "dist_weight",
    "temporal_weight",
    "temporal_translation_unit_m",
    "joint_limit_margin_rad",
    "max_iterations",
    "relative_tolerance",
    "absolute_tolerance",
    "initial_damping",
    "init_iterations",
    "init_relative_tolerance",
    "rotation_hypotheses",
    "full_fit_hypotheses",
    "finger_starts",
)
"""The recorded FitConfig's fields in ``HandFitter`` order, after phi (``export_cold_golden`` writes the same order)."""


@dataclass(frozen=True, slots=True)
class Config:
    """Replay the golden set's hands through the native fit and time it."""

    golden: Path
    """The npz written by export_golden.py; its .json index sits beside it."""
    repeats: int = 5
    """Timed passes over every call; the minimum per call is kept (the other runs absorb cache and frequency noise)."""


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """The hands of one recorded tracker call, ready for one native ``fit``."""

    model_id: int
    rows: Int64[ndarray, "h"]
    camera_indices: Int64[ndarray, "h 2"]


def build_fitter(arrays: dict[str, ndarray], model_id: int, phi: float, fit_config: dict[str, float]) -> HandFitter:
    """The native fitter for one hand model of the golden set, with the recorded FitConfig."""
    model: dict[str, ndarray] = {name: arrays[f"model{model_id}_{name}"] for name in MODEL_FIELDS}
    topology: Int64[ndarray, "4 22"] = np.stack(
        [model["joint_parent"], model["joint_frame_index"], model["joint_first_child"], model["joint_next_sibling"]]
    ).astype(np.int64)
    config: Float64[ndarray, "14"] = np.array([phi, *(fit_config[name] for name in CONFIG_FIELDS)], dtype=np.float64)
    return HandFitter(
        model["joint_rotation_axes"][:20].astype(np.float32),
        model["joint_rest_positions"][:20].astype(np.float32),
        model["landmark_rest_positions"].astype(np.float32),
        model["landmark_rest_bone_weights"].astype(np.float32),
        model["joint_limits"][:20].astype(np.float32),
        model["landmark_rest_bone_indices"].astype(np.int64),
        topology,
        config,
    )


def register_cameras(fitter: HandFitter, arrays: dict[str, ndarray], rows: Int64[ndarray, "r"]) -> dict[int, Int64[ndarray, "2"]]:
    """Add each distinct camera of these rows to the fitter once; per row, the camera id of each view slot (-1 = absent)."""
    known: dict[bytes, int] = {}
    per_row: dict[int, Int64[ndarray, "2"]] = {}
    for row in rows.tolist():
        ids: Int64[ndarray, "2"] = np.full(2, -1, dtype=np.int64)
        for view in range(int(arrays["num_views"][row])):
            fisheye: bool = bool(arrays["is_fisheye"][row, view])
            parts: tuple[Float32[ndarray, "..."], ...] = (
                arrays["cam_from_rig"][row, view],
                arrays["focal"][row, view],
                arrays["principal"][row, view],
                arrays["fisheye62"][row, view] if fisheye else np.zeros(0, np.float32),
            )
            key: bytes = b"".join(np.ascontiguousarray(p, dtype=np.float32).tobytes() for p in parts)
            if key not in known:
                known[key] = fitter.add_camera(
                    parts[0].astype(np.float32), parts[1].astype(np.float32), parts[2].astype(np.float32),
                    parts[3].astype(np.float32) if fisheye else None,
                )
            ids[view] = known[key]
        per_row[row] = ids
    return per_row


def main(config: Config) -> None:
    arrays: dict[str, ndarray] = dict(np.load(config.golden))
    index: dict = json.loads(config.golden.with_suffix(".json").read_text())
    warm: Bool[ndarray, "r"] = arrays["has_previous"].astype(bool)
    fitters: dict[int, HandFitter] = {}
    cameras: dict[int, Int64[ndarray, "2"]] = {}
    for model_id in sorted(set(arrays["model_id"].tolist())):
        rows: Int64[ndarray, "r"] = np.flatnonzero(arrays["model_id"] == model_id)
        phi: float = float(arrays["phi"][rows[0]])
        fitters[model_id] = build_fitter(arrays, model_id, phi, index["fit_config"])
        cameras.update(register_cameras(fitters[model_id], arrays, rows))
    calls: list[RecordedCall] = []
    for call in sorted(set(arrays["call"].tolist())):
        rows = np.flatnonzero(arrays["call"] == call)
        if len(rows):
            calls.append(RecordedCall(int(arrays["model_id"][rows[0]]), rows, np.stack([cameras[r] for r in rows.tolist()])))
    best: list[float] = [float("inf")] * len(calls)
    outputs: list[list[FitOutput]] = []
    for repeat in range(config.repeats):
        for position, call in enumerate(calls):
            r: Int64[ndarray, "h"] = call.rows
            start: float = time.perf_counter()
            result: list[FitOutput] = fitters[call.model_id].fit(
                arrays["side"][r].astype(np.int64),
                arrays["prev_rotation"][r].astype(np.float32),
                arrays["prev_translation"][r].astype(np.float32),
                arrays["prev_joint_angles"][r].astype(np.float32),
                call.camera_indices,
                arrays["world_from_rig"][r].astype(np.float32),
                arrays["keypoints_px"][r].astype(np.float32),
                arrays["weights"][r].astype(np.float32),
                arrays["d_rel_mm"][r].astype(np.float32),
                has_previous=warm[r],
            )
            best[position] = min(best[position], time.perf_counter() - start)
            if repeat == 0:
                outputs.append(result)
    translation_mm: list[float] = []
    energy_excess: list[float] = []
    flips: int = 0
    for call, result in zip(calls, outputs, strict=True):
        for row, out in zip(call.rows.tolist(), result, strict=True):
            translation_mm.append(1000.0 * float(np.linalg.norm(out.translation - arrays["out_translation"][row])))
            want: float = float(arrays["out_energy"][row])
            energy_excess.append(max(0.0, (out.energy - want) / max(abs(want), 1e-12)))
            flips += int(out.converged != bool(arrays["out_converged"][row]))
    by_hands: dict[str, list[float]] = {}
    for call, seconds in zip(calls, best, strict=True):
        kind: str = "warm" if bool(warm[call.rows].all()) else "with cold"
        by_hands.setdefault(f"{kind}, {len(call.rows)} hand(s)", []).append(seconds)
    print(f"handfit golden replay on {platform.node()} ({platform.machine()}, {platform.system()}): {len(calls)} calls, "
          f"{len(translation_mm)} hands, best of {config.repeats}")
    for hands, times in sorted(by_hands.items()):
        print(f"  {hands}: n={len(times)}  median {1000 * statistics.median(times):.3f} ms  mean {1000 * statistics.mean(times):.3f} ms")
    print(f"  total over the calls: {1000 * sum(best):.1f} ms; per frame (400): {1000 * sum(best) / 400:.3f} ms")
    q = np.percentile(translation_mm, [50, 90, 100])
    print(f"  vs recorded torch: wrist translation median {q[0]:.4f} mm, p90 {q[1]:.4f} mm, max {q[2]:.3f} mm; "
          f"energy excess median {np.median(energy_excess):.2g}, p90 {np.percentile(energy_excess, 90):.2g}; converged flips {flips}")
