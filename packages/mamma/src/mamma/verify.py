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
from pathlib import Path

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.sources import SegmentRef
from dagkit.verify import Case, Phase
from jaxtyping import Bool, Float32, Int64
from numpy import ndarray
from PIL import Image
from rerun.catalog import DatasetEntry

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/mamma``: weights, body models and the TensorRT cache live under it."""
PRED_ROOT: str = "/world/mamma/people"
GT_ROOT: str = "/world/gt/people"
BODY_JOINTS: int = 22
"""MPJPE joints: the SMPL-X body joints regressed from the mesh (hands and face excluded)."""


def _window_ns(case: Case) -> tuple[int, int]:
    return (round(case.window_s[0] * 1e9), round(case.window_s[1] * 1e9)) if case.window_s is not None else (-(2**62), 2**62)


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """The capture graph (``apis/capture.py``, threaded, lossless) on the case's segment window; every output goes into ``recording``.

    Nothing is registered and no layer file is written, so the shared catalog is only read. ``phase`` does not change what runs.
    """
    from dagkit.runners import run_threaded
    from simplecv.rerun_log_utils import RerunTyroConfig

    from mamma.apis.capture import Config, build, engine_plan

    assert case.dataset is not None and case.segment is not None, "mamma cases are catalog segments"
    window: tuple[int, int] | None = _window_ns(case) if case.window_s is not None else None
    height, width = (int(x) for x in case.params["resize_hw"].split(","))
    config = Config(
        RerunTyroConfig(headless=True), SegmentRef(case.dataset, case.segment, window_ns=window), tuple(case.params["cameras"].split(",")),
        (height, width), "quality" if case.params["preset"] == "quality" else "fast", int(case.params["subjects"]),
    )  # fmt: skip
    report = run_threaded(build(config, recording), mode="lossless")
    recording.log("/verify/capture/run", rr.AnyValues(wall_s=report.wall_s, stream_s=report.stream_s, trt_engine=engine_plan() is not None), static=True)
    print(f"{case.name} {phase}: {report.nodes['track'].ticks} framesets in {report.wall_s:.1f} s; " + ", ".join(f"{n} {s.step_ms_mean:.0f} ms" for n, s in report.nodes.items() if s.ticks))


def _series(table: pa.Table, column: str) -> dict[int, pa.Scalar]:
    """Non-null cells of one column keyed by ``video_time`` (ns)."""
    if column not in table.column_names:
        return {}
    times: Int64[ndarray, " r"] = table.column("video_time").cast(pa.int64()).to_numpy()
    cells: pa.ChunkedArray = table.column(column)
    return {int(t): cells[i] for i, t in enumerate(times) if cells[i].is_valid}


def _vertices(cell: pa.Scalar) -> Float32[ndarray, "v 3"]:
    return np.asarray(cell.values.flatten(), dtype=np.float32).reshape(-1, 3)


def _mask(cell: pa.Scalar) -> Bool[ndarray, "h w"]:
    blob: bytes = bytes(cell.as_py()) if not isinstance(cell.as_py(), list) else bytes(cell.as_py()[0])
    return np.asarray(Image.open(io.BytesIO(blob))) > 0


def _people(table: pa.Table, root: str) -> list[str]:
    return sorted({c.removeprefix(f"{root}/").split("/")[0] for c in table.column_names if c.startswith(f"{root}/person_")})


def score(case: Case, golden: Path) -> dict[str, float]:
    """MPJPE / PVE of the fitted meshes against the ``mosh_gt`` meshes, person-mask IoU at the GT mask size, fit coverage and
    realtime fraction. Persons are matched to GT persons by the smallest mean PVE (one-to-one)."""
    assert case.dataset is not None and case.segment is not None, "mamma cases are catalog segments"
    cameras: list[str] = case.params["cameras"].split(",")
    start_ns, stop_ns = _window_ns(case)
    mesh_paths: list[str] = ["/world/mamma/people/**"]
    mask_paths: list[str] = [f"/world/{camera}/pinhole/mamma/masks/**" for camera in cameras]
    with rr.server.Server(host="127.0.0.1", datasets={"golden": [str(golden)]}) as server:
        golden_dataset: DatasetEntry = server.client().get_dataset("golden")  # an entry must outlive its queries
        pred_meshes: pa.Table = golden_dataset.filter_contents(mesh_paths).reader(index="video_time").to_arrow_table()
        pred_masks: pa.Table = golden_dataset.filter_contents(mask_paths).reader(index="video_time").to_arrow_table()
        report: dict = golden_dataset.filter_contents(["/verify/capture/run"]).reader(index=None).to_arrow_table().to_pylist()[0]
    source: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()
    segment = source.filter_segments(case.segment)
    gt_mesh_paths: list[str] = sorted(
        {
            c.entity_path
            for c in source.schema().component_columns()
            if c.entity_path.startswith(f"{GT_ROOT}/") and c.entity_path.endswith("/body/mesh")
        }
    )
    gt_meshes: pa.Table = segment.filter_contents(gt_mesh_paths).reader(index="video_time").to_arrow_table()
    gt_masks: pa.Table = (
        segment.filter_contents([f"/world/{camera}/pinhole/masks/**" for camera in cameras]).reader(index="video_time").to_arrow_table()
    )

    J_regressor: Float32[ndarray, "j v"] = np.load(PACKAGE_DIR / "data/body_models/smplx/SMPLX_NEUTRAL.npz")["J_regressor"][:BODY_JOINTS].astype(
        np.float32
    )
    mesh: str = "body/mesh:Mesh3D:vertex_positions"
    gt: dict[str, dict[int, pa.Scalar]] = {
        p: {t: c for t, c in _series(gt_meshes, f"{GT_ROOT}/{p}/{mesh}").items() if start_ns <= t < stop_ns} for p in _people(gt_meshes, GT_ROOT)
    }
    gt_people: list[str] = [p for p in gt if gt[p]]  # the dataset schema also names persons of other segments
    pred_people: list[str] = _people(pred_meshes, PRED_ROOT)
    pred: dict[str, dict[int, pa.Scalar]] = {p: _series(pred_meshes, f"{PRED_ROOT}/{p}/{mesh}") for p in pred_people}
    errors: dict[tuple[str, str], list[tuple[float, float]]] = {}  # (gt, pred) -> per frame (pve_m, mpjpe_m)
    for g, p in itertools.product(gt_people, pred_people):
        pairs: list[tuple[float, float]] = []
        for t in sorted(gt[g].keys() & pred[p].keys()):
            gt_vertices, pred_vertices = _vertices(gt[g][t]), _vertices(pred[p][t])
            pve: float = float(np.linalg.norm(pred_vertices - gt_vertices, axis=1).mean())
            pairs.append((pve, float(np.linalg.norm(J_regressor @ pred_vertices - J_regressor @ gt_vertices, axis=1).mean())))
        errors[g, p] = pairs
    candidates: list[tuple[str | None, ...]] = list(itertools.permutations([*pred_people, *[None] * len(gt_people)], len(gt_people)))
    matching: tuple[str | None, ...] = min(
        candidates, key=lambda m: sum(np.mean([e[0] for e in errors[g, p]]) if p and errors[g, p] else 1e3 for g, p in zip(gt_people, m, strict=True))
    )
    matched: list[tuple[float, float]] = [e for g, p in zip(gt_people, matching, strict=True) if p for e in errors[g, p]]
    print(
        f"  matching gt->pred {dict(zip(gt_people, matching, strict=True))}; scored {len(matched)} of {sum(len(v) for v in gt.values())} GT mesh frames"
    )

    ious: list[float] = []
    for (g, p), camera in itertools.product(zip(gt_people, matching, strict=True), cameras):
        gt_cells: dict[int, pa.Scalar] = _series(gt_masks, f"/world/{camera}/pinhole/masks/{g}:EncodedImage:blob")
        pred_cells: dict[int, pa.Scalar] = _series(pred_masks, f"/world/{camera}/pinhole/mamma/masks/{p}:EncodedImage:blob") if p else {}
        for t, cell in gt_cells.items():
            if not start_ns <= t < stop_ns:
                continue
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
        "realtime_fraction": float(report["/verify/capture/run:stream_s"][0]) / float(report["/verify/capture/run:wall_s"][0]),
    }
