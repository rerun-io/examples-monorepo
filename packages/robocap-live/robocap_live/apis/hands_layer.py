"""Track one robocap catalog segment's hands with robocap-live's Rust pipeline and replace its ``hands`` layer.

The segment is read from the catalog (``robocap_live.catalog_segment``) and every frameset goes through ``robocap_live._core``:
robocap-live's own scheduler as ``robocap-live --source replay --slam reference`` runs it (the 640x360 images, DetNet and
KeyNet on ONNX Runtime, the perspective KeyNet crops, the ROBUST tracker on handfit, the UmeTrack mesh, the logger's
hands-layer writer). The finished ``.rrd`` is registered as the segment's ``hands`` layer, replacing an earlier one. The file
is written under a temporary name and renamed into place, so a layer the catalog server has open is never truncated under it.
"""

import importlib.util
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from serde import serde
from serde.json import to_json

from robocap_live import _core
from robocap_live.catalog_segment import CatalogSegment, Frameset, open_segment

LAYER: str = "hands"
"""Catalog layer name of the hands layer."""
DATASET: str = "robocap"
"""The dataset prefix of the segments this tool reads (``robocap__<device>__s<session>``)."""


@dataclass(frozen=True, slots=True)
class Config:
    """Track one robocap catalog segment's hands and register (or replace) its ``hands`` layer."""

    segment: str
    """Full segment id, e.g. ``robocap__<device>__s00000010``."""
    output_dir: Path
    """The dataset's hands-layer directory, visible at the same path to this host and to the catalog server."""
    catalog: str
    """Catalog URL, ``rerun+http://<host>:<port>``."""
    models_dir: Path
    """The ONNX models: ``detnet_full.onnx`` and ``keynet.onnx`` (packages/robocap-live/models/MODELS.md)."""
    device: Literal["auto", "cpu", "cuda"] = "auto"
    """ONNX Runtime device: CUDA on GPU 0 when its execution provider registers (auto), or forced."""
    ort_dylib: Path | None = None
    """``libonnxruntime.so``; default ``ORT_DYLIB_PATH``, else the library of this env's ``onnxruntime`` package."""
    ort_threads: int = 0
    """ONNX Runtime intra-op threads per session (0 = its default)."""
    overlays: Literal["fit", "debug", "verbose"] = "debug"
    """How much of the hand pipeline the camera panes show."""
    decode_threads: int = 2
    """libavcodec threads per camera decoder (six run in lockstep)."""
    max_framesets: int | None = None
    """Track only the first N framesets: a smoke run, written as ``<segment>.first<N>.rrd`` and never registered."""
    register: bool = True
    """Register the layer on the catalog (replacing the segment's ``hands`` layer). A run that does not register refuses to
    overwrite an existing file."""
    timing_json: Path | None = None
    """Also write the run's stage timings here."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class LayerRun:
    """One run's counts and stage timings (seconds). The decode thread and the pipeline's stages run in parallel, so the stage
    sums overlap each other and the loop."""

    segment: str
    output: str
    nets: str
    framesets: int
    with_pose: int
    held_pose: int
    """Framesets without their own pose, tracked on the newest earlier one (the runtime's pose store rule)."""
    tracked: list[int]
    reported: list[int]
    scale: float
    scale_final: bool
    catalog_read_s: float
    """Rig, encoded videos and poses, one bulk query each."""
    nets_load_s: float
    """ONNX Runtime and the two sessions, the tracker, the open layer file and the pipeline's threads."""
    decode_busy_s: float
    """The decode thread's own time (libavcodec, six cameras)."""
    loop_wait_s: float
    """The main thread waiting for decoded framesets (the loop is decode-bound by this much)."""
    push_wait_s: float
    """The main thread blocked in ``push`` (the loop is pipeline-bound by this much)."""
    downsample_s: float
    hands_s: float
    """The hands stage's tracker steps; split below."""
    detnet_s: float
    crops_s: float
    keynet_s: float
    fit_s: float
    tracker_s: float
    output_s: float
    """The output stage: handing records to the layer writer (its worker writes beside the stages)."""
    loop_s: float
    """Decode + push, wall."""
    finish_s: float
    """Draining the pipeline and closing the file."""
    register_s: float
    """Rename into place and register (0 when not registered)."""
    total_s: float
    loop_framesets_per_s: float
    total_framesets_per_s: float


def default_ort_dylib() -> Path:
    """``ORT_DYLIB_PATH``, else the ``libonnxruntime.so`` of the env's ``onnxruntime`` package, found without importing it.

    Importing the package would load its own copy of ONNX Runtime beside the one the core loads.

    Raises:
        ValueError: If neither exists.
    """
    from_env: str | None = os.environ.get("ORT_DYLIB_PATH")
    if from_env:
        return Path(from_env)
    spec = importlib.util.find_spec("onnxruntime")
    if spec is not None and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            libraries: list[Path] = sorted((Path(location) / "capi").glob("libonnxruntime.so*"))
            if libraries:
                return libraries[-1]
    raise ValueError("no ONNX Runtime library: pass ort_dylib or set ORT_DYLIB_PATH (or run in the robocap-live env)")


def segment_layers(dataset: DatasetEntry, segment_id: str) -> list[str]:
    """The layer names registered for one segment.

    Raises:
        ValueError: If the segment is not in the dataset.
    """
    table = dataset.segment_table().select("rerun_segment_id", "rerun_layer_names").to_arrow_table()
    for row in table.to_pylist():
        if row["rerun_segment_id"] == segment_id:
            return list(row["rerun_layer_names"])
    raise ValueError(f"{segment_id}: absent from the catalog dataset {dataset.name}")


def main(config: Config) -> LayerRun:
    """Read, track and write the whole segment before anything is registered."""
    started: float = time.perf_counter()
    if Path(config.segment).name != config.segment or not config.segment.startswith(f"{DATASET}__"):
        raise ValueError(f"segment {config.segment!r}: expected a {DATASET}__<device>__<session> id")
    if config.max_framesets is not None and config.max_framesets < 2:
        raise ValueError("max_framesets must be at least 2")
    dataset: DatasetEntry = rr.catalog.CatalogClient(config.catalog).get_dataset(DATASET)
    layers: list[str] = segment_layers(dataset, config.segment)
    if "slam_rs" not in layers:
        raise ValueError(f"{config.segment}: no slam_rs layer (layers {layers}); register the SLAM poses first (slam-rs-catalog-layer)")
    segment: CatalogSegment = open_segment(dataset, config.segment, source=f"catalog {config.catalog} {DATASET}", max_framesets=config.max_framesets)
    with_pose: int = int(np.isfinite(segment.world_from_rig).all(axis=(1, 2)).sum())
    print(f"{config.segment}: {len(segment.index.t_ns)} framesets, {with_pose} with a slam_rs pose, layers {layers}; "
          f"catalog read {segment.read_s:.1f} s", flush=True)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    registering: bool = config.register and config.max_framesets is None
    name: str = config.segment if config.max_framesets is None else f"{config.segment}.first{config.max_framesets}"
    output: Path = config.output_dir.resolve() / f"{name}.rrd"
    if not registering and output.exists():
        # The catalog may serve that file: replacing it without registering would leave the catalog's index of it stale.
        raise ValueError(f"{output} exists and this run does not register: pass another output_dir, or register")
    # Unique per run, so two runs never write (or publish) each other's unfinished file. Runs on one segment still race to
    # publish: run one at a time per segment.
    partial: Path = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    loaded: float = time.perf_counter()
    core: _core.HandsLayer = _core.HandsLayer(
        segment.rig, partial, config.segment, segment.index.t_ns, segment.world_from_rig, models_dir=config.models_dir,
        device=config.device, ort_dylib=config.ort_dylib or default_ort_dylib(), ort_threads=config.ort_threads,
        overlays=config.overlays,
    )
    nets_load_s: float = time.perf_counter() - loaded
    print(f"nets {core.nets} (loaded in {nets_load_s:.1f} s)", flush=True)

    loop_started: float = time.perf_counter()
    wait_s: float = 0.0
    push_s: float = 0.0
    framesets = segment.framesets(decode_threads=config.decode_threads)
    try:
        while True:
            waiting: float = time.perf_counter()
            frameset: Frameset | None = next(framesets, None)
            pushing: float = time.perf_counter()
            wait_s += pushing - waiting
            if frameset is None:
                break
            index: int = core.push(frameset.t_ns, frameset.luma, frameset.cam_t_ns)
            push_s += time.perf_counter() - pushing
            if index % 300 == 0:
                print(f"  frameset {index}: {(index + 1) / (time.perf_counter() - loop_started):.1f} framesets/s", flush=True)
        loop_s: float = time.perf_counter() - loop_started
        finishing: float = time.perf_counter()
        summary: _core.LayerSummary = core.finish()
        finish_s: float = time.perf_counter() - finishing
    except BaseException:
        # Stop the decode thread and the pipeline now, not when the traceback is collected, and leave no partial file.
        framesets.close()
        core.abort()
        partial.unlink(missing_ok=True)
        raise
    # The file at the layer's URI is complete from here on. A registration that fails after this leaves the catalog's index
    # of it stale (the server keeps serving the old file it holds open): rerun the tool.
    os.replace(partial, output)

    register_s: float = 0.0
    if registering:
        register_started: float = time.perf_counter()
        dataset.register([output.as_uri()], layer_name=LAYER, on_duplicate=OnDuplicateSegmentLayer.REPLACE).wait()
        register_s = time.perf_counter() - register_started
    total_s: float = time.perf_counter() - started
    stages: _core.HandStageTimes = summary.hand_stages
    run: LayerRun = LayerRun(
        segment=config.segment, output=str(output), nets=core.nets, framesets=summary.framesets, with_pose=summary.with_pose,
        held_pose=summary.held_pose, tracked=list(summary.tracked), reported=list(summary.reported), scale=summary.scale,
        scale_final=summary.scale_final, catalog_read_s=segment.read_s, nets_load_s=nets_load_s, decode_busy_s=segment.decode.busy_s,
        loop_wait_s=wait_s, push_wait_s=push_s, downsample_s=summary.stage_total_ms("downsample") / 1e3,
        hands_s=summary.stage_total_ms("hands") / 1e3, detnet_s=stages.detnet_ms / 1e3, crops_s=stages.crops_ms / 1e3,
        keynet_s=stages.keynet_ms / 1e3, fit_s=stages.fit_ms / 1e3, tracker_s=stages.tracker_ms / 1e3,
        output_s=summary.stage_total_ms("output") / 1e3, loop_s=loop_s, finish_s=finish_s, register_s=register_s, total_s=total_s,
        loop_framesets_per_s=summary.framesets / loop_s, total_framesets_per_s=summary.framesets / total_s,
    )
    print(f"{config.segment}: {run.framesets} framesets, reported left {run.reported[0]} / right {run.reported[1]}, "
          f"scale {run.scale:.4f}{'' if run.scale_final else ' (not calibrated)'} -> {output}"
          f"{'' if register_s else ' (not registered)'}")
    for label, seconds in [("catalog read", run.catalog_read_s), ("nets load", run.nets_load_s), ("decode (thread)", run.decode_busy_s),
                           ("loop wait for decode", run.loop_wait_s), ("loop wait in push", run.push_wait_s),
                           ("downsample stage", run.downsample_s), ("hands stage", run.hands_s), ("  detnet", run.detnet_s),
                           ("  crops", run.crops_s), ("  keynet", run.keynet_s), ("  fit", run.fit_s), ("  tracker", run.tracker_s),
                           ("output stage", run.output_s), ("loop (wall)", run.loop_s), ("drain + close", run.finish_s),
                           ("register", run.register_s), ("total (wall)", run.total_s)]:
        print(f"  {label:<22}{seconds:8.2f} s  {1e3 * seconds / max(run.framesets, 1):7.2f} ms/frameset")
    print(f"  {run.loop_framesets_per_s:.1f} framesets/s in the loop, {run.total_framesets_per_s:.1f} end to end")
    if register_s:
        print(dataset.segment_url(config.segment))
    if config.timing_json is not None:
        config.timing_json.write_text(to_json(run))
    return run
