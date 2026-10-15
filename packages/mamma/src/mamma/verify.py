"""Graph-tier check of mamma, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

``run`` writes a golden rrd; ``score`` reads only that golden plus the segment's GT (``mosh_gt`` meshes, person masks)
from the catalog. Golden contract (on the segment's ``video_time``; the pre-port goldens wrote the same scored paths):

- ``/world/mamma/people/person_NN/body/mesh``: ``Mesh3D`` vertex positions (SMPL-X, world metres) per fitted frame;
  ``triangle_indices`` static. ``body/smplx``: ``AnyValues`` pose (165 axis-angle), betas, trans per fitted frame.
- ``/world/mamma/people/person_NN/landmarks3d``: triangulated landmarks (``Points3D``) per frame.
- ``/world/<camera>/pinhole/mamma/masks/person_NN``: palette PNG (index 1 = the person) at engine size per frame, under a
  static ``Transform3D`` scale to the stored video size; ``.../mamma/landmarks/person_NN`` (``Points2D`` + visibility) and
  ``.../mamma/persons`` (``Boxes2D``, class id = person + 1). Since the port the scale sits on ``.../pinhole/mamma``
  (the pre-port goldens put it on each mask and wrote boxes and landmarks in stored pixels).
- ``/verify/capture/run``: static ``AnyValues`` ``wall_s`` and ``stream_s`` (realtime_fraction = stream_s / wall_s).
"""

import io
import itertools
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.catalog import served
from dagkit.runners import run_threaded
from dagkit.sources import SegmentRef
from dagkit.types import cells, flat
from dagkit.verify import Case, Phase, load_card, run_counts
from datafusion import col, lit
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from PIL import Image
from rerun.catalog import DatasetEntry
from scipy.optimize import linear_sum_assignment
from simplecv.rerun_log_utils import RerunTyroConfig

from mamma.apis.capture import PACKAGE_DIR, Config, build, engine_plan

PRED_ROOT: str = "/world/mamma/people"
GT_ROOT: str = "/world/gt/people"
BODY_JOINTS: int = 22
"""MPJPE joints: the SMPL-X body joints regressed from the mesh (hands and face excluded)."""
UNMATCHED_PVE_M: float = 1e3
"""Matching cost of a GT person left unmatched, or paired with a prediction it shares no frame with."""


def case_config(case: Case) -> Config:
    """The App Config of a case: ``run`` and the gradio app's examples (``example_cases``) run the same inputs."""
    height, width = case.params["resize_hw"]
    return Config(
        RerunTyroConfig(headless=True), case.segment_ref(), cameras=tuple(case.params["cameras"]), resize_hw=(height, width),
        preset=case.params["preset"], subjects=case.params["subjects"],
    )  # fmt: skip


def example_cases() -> list[Config]:
    """The gradio app's examples, the ``VERIFY.toml`` cases: ``short`` (the first 10 s), ``full`` (one person) and ``contact``
    (two people). They never register."""
    cases: list[Case] = [case for block in load_card(PACKAGE_DIR / "VERIFY.toml").graph for case in block.cases]
    return [replace(case_config(case), register="off") for case in sorted(cases, key=lambda case: case.params["subjects"])]


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The capture graph (``apis/capture.py``, threaded, lossless) on the case's segment window; every output goes into ``recording``.

    Nothing is registered and no layer file is written, so the shared catalog is only read. ``phase`` does not change what runs.
    """
    report = run_threaded(build(case_config(case), recording), mode="lossless")
    recording.log("/verify/capture/run", rr.AnyValues(wall_s=report.wall_s, stream_s=report.stream_s, trt_engine=engine_plan() is not None), static=True)
    print(f"{case.name} {phase}: {report.nodes['track'].ticks} framesets in {report.wall_s:.1f} s; " + ", ".join(f"{n} {s.step_ms_mean:.0f} ms" for n, s in report.nodes.items() if s.ticks))


def _mask(cell: pa.Scalar) -> Bool[ndarray, "h w"]:
    blob: bytes = bytes(cell.as_py()) if not isinstance(cell.as_py(), list) else bytes(cell.as_py()[0])
    return np.asarray(Image.open(io.BytesIO(blob))) > 0


def _people(table: pa.Table, root: str) -> list[str]:
    return sorted({c.removeprefix(f"{root}/").split("/")[0] for c in table.column_names if c.startswith(f"{root}/person_")})


def score(case: Case, golden: Path) -> dict[str, float]:
    """MPJPE / PVE of the fitted meshes against the ``mosh_gt`` meshes, person-mask IoU at the GT mask size, fit coverage and
    realtime fraction. Persons are matched to GT persons by the smallest mean PVE (one-to-one)."""
    segment_ref: SegmentRef = case.segment_ref()
    cameras: list[str] = case.params["cameras"]
    start_ns, stop_ns = segment_ref.window_ns or (-(2**62), 2**62)
    time = col("video_time").cast(pa.int64())
    window = (time >= lit(start_ns)) & (time < lit(stop_ns))  # the GT reads transfer only the case's window
    mesh_paths: list[str] = ["/world/mamma/people/**"]
    mask_paths: list[str] = [f"/world/{camera}/pinhole/mamma/masks/**" for camera in cameras]
    with served(golden) as golden_dataset:
        pred_meshes: pa.Table = golden_dataset.filter_contents(mesh_paths).reader(index="video_time").to_arrow_table()
        pred_masks: pa.Table = golden_dataset.filter_contents(mask_paths).reader(index="video_time").to_arrow_table()
        counts: dict[str, float] = run_counts(golden_dataset, "capture")
    source: DatasetEntry = segment_ref.entry()
    segment = source.filter_segments(segment_ref.segment)
    gt_mesh_paths: list[str] = sorted(
        {
            c.entity_path
            for c in source.schema().component_columns()
            if c.entity_path.startswith(f"{GT_ROOT}/") and c.entity_path.endswith("/body/mesh")
        }
    )
    gt_meshes: pa.Table = segment.filter_contents(gt_mesh_paths).reader(index="video_time").filter(window).to_arrow_table()
    gt_mask_paths: list[str] = [f"/world/{camera}/pinhole/masks/**" for camera in cameras]
    gt_masks: pa.Table = segment.filter_contents(gt_mask_paths).reader(index="video_time").filter(window).to_arrow_table()

    J_regressor: Float32[ndarray, "j v"] = np.load(PACKAGE_DIR / "data/body_models/smplx/SMPLX_NEUTRAL.npz")["J_regressor"][:BODY_JOINTS].astype(
        np.float32
    )
    mesh: str = "body/mesh:Mesh3D:vertex_positions"
    gt: dict[str, dict[int, pa.Scalar]] = {p: cells(gt_meshes, f"{GT_ROOT}/{p}/{mesh}", "video_time") for p in _people(gt_meshes, GT_ROOT)}
    gt_people: list[str] = [p for p in gt if gt[p]]  # the dataset schema also names persons of other segments
    pred_people: list[str] = _people(pred_meshes, PRED_ROOT)
    pred: dict[str, dict[int, pa.Scalar]] = {p: cells(pred_meshes, f"{PRED_ROOT}/{p}/{mesh}", "video_time") for p in pred_people}
    errors: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (gt, pred) -> per frame (pve_m, mpjpe_m)
    for g, p in itertools.product(gt_people, pred_people):
        pairs: list[tuple[float, float]] = []
        for t in sorted(gt[g].keys() & pred[p].keys()):
            gt_vertices, pred_vertices = (flat(cell, np.float32).reshape(-1, 3) for cell in (gt[g][t], pred[p][t]))
            pve: float = float(np.linalg.norm(pred_vertices - gt_vertices, axis=1).mean())
            pairs.append((pve, float(np.linalg.norm(J_regressor @ pred_vertices - J_regressor @ gt_vertices, axis=1).mean())))
        errors[g, p] = pairs
    cost: Float64[ndarray, "g c"] = np.full((len(gt_people), len(pred_people) + len(gt_people)), UNMATCHED_PVE_M)  # + one unmatched column per GT person
    for (i, g), (j, p) in itertools.product(enumerate(gt_people), enumerate(pred_people)):
        if errors[g, p]:
            cost[i, j] = np.mean([e[0] for e in errors[g, p]])
    columns: Int64[ndarray, " g"] = linear_sum_assignment(cost)[1]  # every GT row is assigned, in row order
    matching: list[str | None] = [pred_people[j] if j < len(pred_people) else None for j in columns.tolist()]
    matched: list[tuple[float, float]] = [e for g, p in zip(gt_people, matching, strict=True) if p for e in errors[g, p]]
    print(
        f"  matching gt->pred {dict(zip(gt_people, matching, strict=True))}; scored {len(matched)} of {sum(len(v) for v in gt.values())} GT mesh frames"
    )

    ious: list[float] = []
    for (g, p), camera in itertools.product(zip(gt_people, matching, strict=True), cameras):
        gt_cells: dict[int, pa.Scalar] = cells(gt_masks, f"/world/{camera}/pinhole/masks/{g}:EncodedImage:blob", "video_time")
        pred_cells: dict[int, pa.Scalar] = cells(pred_masks, f"/world/{camera}/pinhole/mamma/masks/{p}:EncodedImage:blob", "video_time") if p else {}
        for t, cell in gt_cells.items():
            target: Bool[ndarray, "h w"] = _mask(cell)
            if t in pred_cells:  # nearest sample of the engine-size mask at the GT pixel centres
                full: Bool[ndarray, "hh ww"] = _mask(pred_cells[t])
                rows: Int64[ndarray, " h"] = ((np.arange(target.shape[0]) + 0.5) * full.shape[0] / target.shape[0]).astype(np.int64)
                cols: Int64[ndarray, " w"] = ((np.arange(target.shape[1]) + 0.5) * full.shape[1] / target.shape[1]).astype(np.int64)
                guess: Bool[ndarray, "h w"] = full[rows][:, cols]
            else:
                guess = np.zeros_like(target)
            union: int = int((target | guess).sum())
            if union:
                ious.append(float((target & guess).sum()) / union)
    return {
        "mpjpe_mm": float(np.mean([e[1] for e in matched]) * 1000.0) if matched else float("nan"),
        "pve_mm": float(np.mean([e[0] for e in matched]) * 1000.0) if matched else float("nan"),
        "mask_iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "fit_coverage": len(matched) / max(1, sum(len(v) for v in gt.values())),
        "realtime_fraction": counts["stream_s"] / counts["wall_s"],
    }
