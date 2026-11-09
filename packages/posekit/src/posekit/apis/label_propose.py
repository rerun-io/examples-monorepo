"""Hand labeler proposer: EgoHand boxes on the 5 fps grid of every camera -> ``hand_proposals`` layer -> catalog.

For each segment: read all camera videos from the catalog, decode the grid frames camera by camera on NVDEC,
run the EgoHand RT-DETR detector (TensorRT; ONNX Runtime has no CUDA provider on aarch64), assign hand
identities frame by frame, write
``<output_dir>/<segment>.rrd`` with the base layer's application id and ``recording_id = segment``, and register
it on the dataset as layer ``hand_proposals`` (REPLACE). One GPU job: segments run one after another.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry

from posekit.labeler.frames import GRID_PERIOD_NS, FrameSource
from posekit.labeler.proposals import ProposalsByTime
from posekit.labeler.propose import CameraTiming, base_application_id, detect_camera, detections_to_proposals, register_layer, write_proposal_layer
from posekit.labeler.schema import PROPOSAL_LAYER
from posekit.models.egohand import EgoHandDetector, EgoHandDetectorConfig
from posekit.runtimes import TensorRtBackendConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class LabelProposeConfig:
    """Write and register the ``hand_proposals`` layer of the labeling segments."""

    catalog_url: str = "rerun+http://127.0.0.1:9897"
    """Catalog serving the labeling dataset."""
    dataset: str = "robocap-labeling"
    """Dataset whose segments get proposals."""
    segments: tuple[str, ...] = ()
    """Segment ids; empty runs every segment of the dataset."""
    output_dir: Path = Path("/mnt/nas/ssd-8tb/scratch/hand-labeler/layers") / PROPOSAL_LAYER
    """Where ``<segment>.rrd`` goes (a path the catalog server can read)."""
    detector: EgoHandDetectorConfig = field(
        default_factory=lambda: EgoHandDetectorConfig(backend=TensorRtBackendConfig(max_batch_size=32, opt_batch_size=16), gpu_preprocessing=True)
    )
    """EgoHand RT-DETR hand detector."""
    batch_size: int = 32
    """Grid frames per decoded GPU batch."""
    max_frames: int | None = None
    """Walk only the first ``max_frames`` grid times (smoke runs); None walks the whole segment."""
    register: bool = True
    """Register the written layer on the dataset (REPLACE)."""


def propose_segment(config: LabelProposeConfig, dataset: DatasetEntry, detector: EgoHandDetector, segment_id: str) -> Path:
    """Detect, write and (optionally) register one segment's proposals; prints its throughput."""
    started: float = time.perf_counter()
    source: FrameSource = FrameSource.from_catalog(dataset, segment_id)
    read_s: float = time.perf_counter() - started
    proposals: ProposalsByTime = {t: [] for t in source.times_ns[: config.max_frames]}
    timings: list[CameraTiming] = []
    for cam in source.cams:
        times, detections, timing = detect_camera(detector, source.iter_grid([cam.cam], batch_size=config.batch_size, device="cuda", max_frames=config.max_frames))
        for t_ns, found in detections_to_proposals(detections, times, cam).items():
            proposals[t_ns].extend(found)
        timings.append(timing)
        print(f"  cam_{cam.cam:02d} {cam.name:<11} {timing.frames} frames, {detections.num_detections} boxes")
    path: Path = config.output_dir / f"{segment_id}.rrd"
    write_proposal_layer(
        path,
        application_id=base_application_id(dataset, segment_id),
        segment_id=segment_id,
        cams=source.cams,
        proposals=proposals,
        properties={
            "model": "egohand",
            "detector": "nvidia/RT-DETR-Hand-Detector-v1",
            "detector_revision": config.detector.hf_detector_revision,
            "threshold": config.detector.threshold,
            "grid_period_ms": GRID_PERIOD_NS // 1_000_000,
        },
    )
    if config.register:
        register_layer(dataset, path)
    frames: int = sum(timing.frames for timing in timings)
    decode_s: float = sum(timing.decode_s for timing in timings)
    detect_s: float = sum(timing.detect_s for timing in timings)
    boxes: int = sum(len(found) for found in proposals.values())
    wall_s: float = time.perf_counter() - started
    print(
        f"{segment_id}: {len(source.times_ns)} grid times x {len(source.cams)} cams = {frames} frames, {boxes} boxes | "
        f"catalog read {read_s:.1f} s, decode {frames / max(decode_s, 1e-9):.0f} fps, detect {frames / max(detect_s, 1e-9):.0f} fps, "
        f"end to end {frames / wall_s:.0f} fps ({wall_s:.0f} s) -> {path}"
    )
    return path


def main(config: LabelProposeConfig) -> None:
    """Run the proposer on every requested segment, one after another."""
    dataset: DatasetEntry = CatalogClient(config.catalog_url).get_dataset(config.dataset)
    segments: list[str] = list(config.segments) or sorted(dataset.segment_ids())
    missing: set[str] = set(segments) - set(dataset.segment_ids())
    if missing:
        raise ValueError(f"segments absent from {config.dataset}: {sorted(missing)}")
    detector: EgoHandDetector = config.detector.setup()
    for segment_id in segments:
        propose_segment(config, dataset, detector, segment_id)
