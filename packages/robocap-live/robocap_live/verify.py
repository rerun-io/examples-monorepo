"""Graph-tier check of robocap-live's hands layer, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

The robocap dataset has no ground truth (its ``slam_rs`` layer is our own output), so the reference is the PRE-PORT golden:
``score`` compares a golden with the same case's ``pre`` golden (the ``pre`` golden itself scores 0, ``pre2`` gives the
run-to-run noise). It reads only the two golden rrds, never the pipeline.

Golden contract (the ported graph writes the same entities on the segment's ``video_time``): the hands layer exactly as the
Rust layer writer logs it (``/world/hands/<side>/...`` in the world, ``/world/rig_00/cam_NN/pinhole/hands/<side>/...`` per
camera), plus ``/verify/hands/run``: static ``AnyValues`` with the layer's counts (``LayerRun``) and the run's wall time.
"""

import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import rerun as rr
from dagkit.catalog import served
from dagkit.runners import RunReport, run_threaded
from dagkit.sinks import LayerFileSink
from dagkit.verify import Case, Phase, pre_golden, run_counts
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from serde import to_dict

from robocap_live.apis.hands_layer import PACKAGE_DIR as PACKAGE_DIR
from robocap_live.apis.hands_layer import build, case_config
from robocap_live.nodes.hands import HandsNode

TIMELINE: str = "video_time"
SIDES: tuple[str, ...] = ("left", "right")
KEYNET: tuple[str, ...] = tuple(f"/world/rig_00/cam_{camera:02d}/pinhole/hands/{side}/keynet" for camera in range(6) for side in SIDES)
"""Per camera and hand: KeyNet's 21 image keypoints (stored-video pixels) on every frameset the hand was cropped."""
LANDMARKS: tuple[str, ...] = tuple(f"/world/hands/{side}/keypoints" for side in SIDES)
"""Per hand: the fitted 21 world landmarks (metres) on every frameset the hand was reported."""


def run(case: Case, phase: Phase, recording: rr.RecordingStream, slam: Literal["vio", "catalog"] = "catalog") -> None:
    """Run the App's graphs (threaded, lossless): the published hands layer (unregistered, in a temporary directory) is copied into
    ``recording`` unchanged, its ``LayerRun`` and the wall time go to ``/verify/hands/run``."""
    started: float = time.perf_counter()
    viewer = rr.RecordingStream("robocap-live-verify")
    viewer.set_sinks()  # no sink: the App's viewer output is dropped, not buffered (a disabled stream refuses its blueprint)
    with TemporaryDirectory() as tmp:
        for graph in build(case_config(case, Path(tmp), slam), viewer):
            report: RunReport = run_threaded(graph, mode="lossless")
            print(f"{case.name} {phase}: {graph.name} {report.status} in {report.wall_s:.1f} s")
        hands, publish = cast(HandsNode, graph.nodes["hands"].obj), cast(LayerFileSink, graph.nodes["publish"].obj)
        assert publish.published is not None and hands.run is not None, f"{case.name}: the run published no layer ({report.status})"
        recording.send_chunks(rr.chunk.RrdReader(publish.published).stream())
    recording.log("/verify/hands/run", rr.AnyValues(**to_dict(hands.run), wall_s=time.perf_counter() - started), static=True)


run_robocap: partial[None] = partial(run, slam="vio")


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
    with served(golden) as dataset:
        return dataset.filter_contents([*KEYNET, *LANDMARKS]).reader(index=TIMELINE).to_arrow_table(), run_counts(dataset, "hands")


def score(case: Case, golden: Path) -> dict[str, float]:
    """Compare a golden with the case's ``pre`` golden (pre-port reference; ``pre`` itself scores 0).

    Banded: ``kp2d_px_mean`` / ``kp2d_px_p95`` (KeyNet image keypoints of the frameset, camera and hand both goldens hold, in
    stored-video pixels), ``landmarks_mm_mean`` (world landmarks of the hands both report, mm), ``reported_delta_pct``
    (frameset-hands reported by exactly one of the two, % of the reference's) and ``kp2d_delta_pct`` (the same for KeyNet's
    frameset-camera-hands, so a lost camera pane fails). Reported, not banded: the golden's counts.
    """
    reference: Path = pre_golden(case, golden)
    ours, counts = _read(golden)
    theirs, _ = (ours, counts) if reference == golden else _read(reference)
    errors: dict[str, list[Float32[ndarray, " m"]]] = {"Points2D": [], "Points3D": []}
    held: dict[str, tuple[int, float]] = {}  # archetype -> (rows ours holds, % of the reference's rows held by one golden only)
    for entities, archetype, width, unit in ((KEYNET, "Points2D", 2, 1.0), (LANDMARKS, "Points3D", 3, 1e3)):
        rows_ours: set[tuple[str, int]] = set()
        rows_ref: set[tuple[str, int]] = set()
        for entity in entities:
            (t_ours, p_ours), (t_ref, p_ref) = _rows(ours, entity, archetype, width), _rows(theirs, entity, archetype, width)
            _, i_ours, i_ref = np.intersect1d(t_ours, t_ref, assume_unique=True, return_indices=True)
            errors[archetype].append(unit * np.linalg.norm(p_ours[i_ours] - p_ref[i_ref], axis=-1).reshape(-1))
            rows_ours |= {(entity, int(t)) for t in t_ours}
            rows_ref |= {(entity, int(t)) for t in t_ref}
        held[archetype] = len(rows_ours), 100.0 * len(rows_ours ^ rows_ref) / len(rows_ref) if rows_ref else float("nan")
    pixels: Float64[ndarray, " m"] = np.concatenate(errors["Points2D"]).astype(np.float64)
    millimetres: Float64[ndarray, " m"] = np.concatenate(errors["Points3D"]).astype(np.float64)
    return {
        "kp2d_px_mean": float(pixels.mean()) if pixels.size else float("nan"),
        "kp2d_px_p95": float(np.percentile(pixels, 95)) if pixels.size else float("nan"),
        "landmarks_mm_mean": float(millimetres.mean()) if millimetres.size else float("nan"),
        "reported_delta_pct": held["Points3D"][1],
        "kp2d_delta_pct": held["Points2D"][1],
        "kp2d_matched": float(pixels.size // 21),
        "reported": float(held["Points3D"][0]),
        **{f"run_{name}": value for name, value in counts.items()},
    }
