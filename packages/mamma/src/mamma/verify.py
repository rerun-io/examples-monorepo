"""Graph-tier check of mamma, driven by ``VERIFY.toml`` through ``python -m dagkit.verify``.

``run`` writes a golden rrd; ``score`` reads only that golden plus the segment's GT (``mosh_gt`` meshes, person masks)
from the catalog. Golden contract (the ported graph writes the same paths on the segment's ``video_time``):

- ``/world/mamma/people/person_NN/body/mesh``: ``Mesh3D`` vertex positions (SMPL-X, world metres) per fitted frame;
  ``triangle_indices`` static. ``body/smplx``: ``AnyValues`` pose (165 axis-angle), betas, trans per fitted frame.
- ``/world/mamma/people/person_NN/landmarks3d``: triangulated landmarks (``Points3D``) per frame.
- ``/world/<camera>/pinhole/mamma/masks/person_NN``: palette PNG (index 1 = the person) at engine size per frame, static
  ``Transform3D`` scale to the stored video size; ``.../mamma/landmarks/person_NN`` (``Points2D`` in stored-video pixels
  + visibility) and ``.../mamma/persons`` (``Boxes2D``, class id = person + 1).
- ``/verify/capture/run``: static ``AnyValues`` ``wall_s`` and ``stream_s`` (realtime_fraction = stream_s / wall_s).
"""

import io
import itertools
import time
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import rerun as rr
from dagkit.sources import SegmentRef
from dagkit.types import read_column
from dagkit.verify import Case, Phase
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from PIL import Image
from rerun.catalog import DatasetEntry

PACKAGE_DIR: Path = Path(__file__).resolve().parents[2]
"""``packages/mamma``: weights, body models and the TensorRT cache live under it."""
PRED_ROOT: str = "/world/mamma/people"
GT_ROOT: str = "/world/gt/people"
BODY_JOINTS: int = 22
"""MPJPE joints: the SMPL-X body joints regressed from the mesh (hands and face excluded)."""
MASK_RGB: tuple[int, int, int] = (255, 140, 0)


def _window_ns(case: Case) -> tuple[int, int]:
    return (round(case.window_s[0] * 1e9), round(case.window_s[1] * 1e9)) if case.window_s is not None else (-(2**62), 2**62)


def _png(mask: Bool[ndarray, "h w"]) -> bytes:
    image: Image.Image = Image.fromarray(mask.astype(np.uint8), mode="P")
    image.putpalette([0, 0, 0, *MASK_RGB])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", transparency=0)
    return buffer.getvalue()


def _export(case: Case, cameras: Sequence[str], out: Path) -> tuple[Int64[ndarray, " n"], list[str]]:
    """The segment's AV1 packets (keyframe before the window start → window end) wrapped as mp4 without re-encoding, plus its
    calibration as the ``ma_cap`` npz contract the old pipeline reads. Returns each wrapped frame's ``video_time`` and the camera names."""
    from simplecv.catalog_video_codec import catalog_codec_name, wrap_mp4
    from simplecv.rerun_dataloader import read_segment_packets
    from simplecv.types.rig import RigCameras

    from mamma.calibration.npz_contract import CameraCalibration, dump_camera_npz

    assert case.dataset is not None and case.segment is not None, "mamma cases are catalog segments"
    dataset: DatasetEntry = SegmentRef(case.dataset, case.segment).entry()  # an entry must outlive its queries
    rig_table: pa.Table = read_column(dataset, case.segment, RigCameras, "/world", "video_time", static=True)
    rig: RigCameras = next(RigCameras.from_layer(rig_table, "/world", "video_time"))[1].select(cameras)
    start_ns, stop_ns = _window_ns(case)
    names: list[str] = [camera.replace("/", "_") for camera in cameras]
    (out / "meta").mkdir(parents=True)
    (out / "videos").mkdir()
    clip_times: Int64[ndarray, " n"] | None = None
    first: int = 0
    for name, camera, K, cam_T_world, wh in zip(names, cameras, rig.intrinsics, rig.cam_T_world, rig.resolution_wh, strict=True):
        packets: tuple = read_segment_packets(dataset, case.segment, f"world/{camera}/pinhole/video", "video_time")
        times: Int64[ndarray, " m"] = np.asarray(packets[0]).astype("timedelta64[ns]").astype(np.int64)
        keyframes: Bool[ndarray, " m"] = np.asarray(packets[2], dtype=bool)
        inside: Int64[ndarray, " k"] = np.flatnonzero((times >= start_ns) & (times < stop_ns))
        begin: int = int(np.flatnonzero(keyframes[: inside[0] + 1])[-1])
        if clip_times is None:
            clip_times, first = times[begin : inside[-1] + 1], int(inside[0] - begin)
        elif not np.array_equal(times[begin : inside[-1] + 1], clip_times):
            raise ValueError(f"{camera}: frame times differ from {cameras[0]}; the old pipeline needs index-aligned cameras")
        mp4: bytes = wrap_mp4(packets[1][begin : inside[-1] + 1], keyframes[begin : inside[-1] + 1].tolist(), 30, catalog_codec_name(packets[3]))
        (out / "videos" / f"{name}.mp4").write_bytes(mp4)
        calibration = CameraCalibration(name=name, k_matrix=K, world_to_cam=cam_T_world, width=int(wh[0]), height=int(wh[1]))
        dump_camera_npz(calibration, out / "meta" / f"{name}.npz")
    assert clip_times is not None
    fps: float = 1e9 / float(np.median(np.diff(clip_times)))
    np.savez(out / "meta" / "global.npz", seq_name=case.segment, fps=fps, frame_start=first, frame_end=clip_times.size)
    return clip_times, names


def run(case: Case, phase: Phase, recording: rr.RecordingStream) -> None:
    """Pre-port: the streaming pipeline (``engine/pipeline.py``) on the segment's exported packets + calibration.

    Its own Rerun logging (``StreamLogger``) goes to a scratch file as in a ``--rr-config.save`` run; the outputs are
    collected per tick (validation mode, as ``tools/dump_artifacts.py``) and logged into ``recording`` on the segment's
    ``video_time`` after the run. Wall time spans the stream loop plus the head/tail drains, not model construction.
    """
    import tensorrt as trt
    import torch
    from simplecv.video_io import TorchCodecVideoReader

    from mamma.datasets.mamma_npz import load_mamma_sequence
    from mamma.engine.pipeline import ResultCollector, build_streaming_pipeline
    from mamma.engine.presets import PipelinePreset, get_preset
    from mamma.fitting.stage import TickFitOutput
    from mamma.fitting.window_fitter import FitResult

    torch.set_float32_matmul_precision("high")  # as every old tool's entry point
    cameras: list[str] = case.params["cameras"].split(",")
    subjects: int = int(case.params["subjects"])
    resize_hw: tuple[int, int] = (int(case.params["resize_hw"].split(",")[0]), int(case.params["resize_hw"].split(",")[1]))
    preset: PipelinePreset = get_preset("quality" if case.params["preset"] == "quality" else "fast", expected_subjects=subjects)
    if subjects > 1:
        preset.fitter.use_cuda_graph = False  # per-person graph captures deadlock (tools/dump_artifacts.py)
    sm: str = "sm{}{}".format(*torch.cuda.get_device_capability())
    engine: Path = PACKAGE_DIR / ".trt_cache" / f"mammanet_b4_fp16_trt{trt.__version__.replace('.', '')}_{sm}.plan"

    class Collector(ResultCollector):
        def __init__(self) -> None:
            super().__init__()
            self.ticks: list[tuple[int, dict, dict, dict]] = []  # frame, {(cam, obj): mask bits + box}, {(cam, obj): lm}, {obj: tri}
            self.fit_frames: dict[int, dict[int, FitResult]] = {}

        def collect(self, frame_idx: int, tracks: list | None, landmarks: list | None, fit_output: TickFitOutput | None = None) -> None:
            if frame_idx >= 0:
                masks = {(c, o): (t.mask.cpu().numpy(), t.bbox_xyxy) for c, cam in enumerate(tracks or []) for o, t in cam.items()}
                lms = {
                    (c, o): (r.joints2d[:, :2].cpu().numpy(), r.visibility.cpu().numpy())
                    for c, cam in enumerate(landmarks or [])
                    for o, r in cam.items()
                }
                tri = {o: (p[v].cpu().numpy()) for o, (p, v) in (fit_output.triangulated.items() if fit_output else [])}
                self.ticks.append((frame_idx, masks, lms, tri))
            for obj, fit in fit_output.fits.items() if fit_output else []:
                self.fit_frames.setdefault(obj, {})[fit.frame_idx] = fit

    with TemporaryDirectory() as tmp:
        times_ns, names = _export(case, cameras, Path(tmp) / "capture")
        sequence = load_mamma_sequence(Path(tmp) / "capture", camera_names=names)
        viz = rr.RecordingStream("mamma-verify-viz", send_properties=False)
        viz.save(Path(tmp) / "viz.rrd")
        previous: rr.RecordingStream | None = rr.set_global_data_recording(viz)
        collector = Collector()
        pipeline = build_streaming_pipeline(
            sequence, resize_hw=resize_hw, device="cuda", tracker_config=preset.tracker, fitter_config=preset.fitter,
            mammanet_weights=PACKAGE_DIR / "data/weights/ma_2d/mamma_mask_full_cvpr.safetensors",
            trt_engine=engine if engine.exists() else None, collector=collector, hires_crops=preset.hires_crops,
        )  # fmt: skip
        # Decode fix (broken pre-port pipeline): the reader's NVDEC decoders are created on this thread but decode on pool
        # threads; torchcodec CUDA decoders are thread-affine and fail from the third 32-frame chunk on. Smallest fix: the
        # same readers, each decoding on its own thread (recorded in the golden as decode_fix).
        pipeline.reader._video_readers = [
            TorchCodecVideoReader(path, device="cuda", resize_hw=pipeline.reader.resize_hw, thread_owned=True) for path in sequence.video_paths
        ]
        start: float = time.perf_counter()
        stats = pipeline.run()
        assert pipeline.fitting is not None
        for tail in pipeline.fitting.drain_head() + pipeline.fitting.drain():
            collector.collect(-1, None, None, tail)
        wall_s: float = time.perf_counter() - start
        viz.flush()
        viz.disconnect()
        if previous is not None:
            rr.set_global_data_recording(previous)
        faces: Int64[ndarray, "f 3"] = np.asarray(pipeline.fitting.faces, dtype=np.int64)
    ticks: list[int] = [tick[0] for tick in collector.ticks]
    print(f"{case.name} {phase}: {stats.ticks} ticks in {wall_s:.1f}s; fits {[(o, len(f)) for o, f in sorted(collector.fit_frames.items())]}")

    scale: tuple[float, float] = (sequence.cameras[0].width / resize_hw[1], sequence.cameras[0].height / resize_hw[0])
    recording.log(
        "/verify/capture/run",
        rr.AnyValues(
            wall_s=wall_s,
            stream_s=(times_ns[ticks[-1]] - times_ns[ticks[0]]) / 1e9,
            trt_engine=engine.exists(),
            decode_fix="thread-owned NVDEC readers",
        ),
        static=True,
    )
    assert all((cam.width / resize_hw[1], cam.height / resize_hw[0]) == scale for cam in sequence.cameras), "one engine scale for every camera"
    for obj in sorted(collector.fit_frames):
        root: str = f"{PRED_ROOT}/person_{obj:02d}"
        recording.log(f"{root}/body/mesh", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[1.0, 0.55, 0.0, 0.8]), static=True)
        for frame, fit in sorted(collector.fit_frames[obj].items()):
            recording.set_time("video_time", duration=np.timedelta64(int(times_ns[frame]), "ns"))
            recording.log(f"{root}/body/mesh", rr.Mesh3D.from_fields(vertex_positions=fit.vertices))
            recording.log(f"{root}/body/smplx", rr.AnyValues(pose=fit.pose, betas=fit.betas, trans=fit.trans))
    logged_scale: set[str] = set()
    for frame, masks, lms, tri in collector.ticks:
        recording.set_time("video_time", duration=np.timedelta64(int(times_ns[frame]), "ns"))
        for obj, points in tri.items():
            recording.log(f"{PRED_ROOT}/person_{obj:02d}/landmarks3d", rr.Points3D(points, radii=0.008, colors=MASK_RGB))
        for (c, obj), (mask, _) in masks.items():
            entity: str = f"/world/{cameras[c]}/pinhole/mamma/masks/person_{obj:02d}"
            if entity not in logged_scale:
                recording.log(entity, rr.Transform3D(scale=[scale[0], scale[1], 1.0]), static=True)
                logged_scale.add(entity)
            recording.log(entity, rr.EncodedImage(contents=_png(mask), media_type="image/png", opacity=0.5, draw_order=11.0))
        for c in range(len(cameras)):
            boxes = [(obj, box) for (cc, obj), (_, box) in masks.items() if cc == c and box is not None]
            if boxes:
                xyxy: Float64[ndarray, "b 4"] = np.stack([b for _, b in boxes]).astype(np.float64) * np.array([*scale, *scale])
                recording.log(
                    f"/world/{cameras[c]}/pinhole/mamma/persons",
                    rr.Boxes2D(array=xyxy, array_format=rr.Box2DFormat.XYXY, class_ids=[o + 1 for o, _ in boxes]),
                )
        for (c, obj), (xy, visibility) in lms.items():
            colors: list[tuple[int, int, int]] = [MASK_RGB if v > 0.5 else (90, 90, 90) for v in visibility]
            recording.log(
                f"/world/{cameras[c]}/pinhole/mamma/landmarks/person_{obj:02d}",
                rr.Points2D(xy * np.array(scale, dtype=np.float32), radii=1.5, colors=colors),
                rr.AnyValues(visibility=visibility),
            )


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
