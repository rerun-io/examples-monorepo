"""Ingest Apple's CA-1M laser ground truth as stackable ARKitScenes layers."""

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from beartype.roar import BeartypeException
from jaxtyping import Float64, Int64
from rerun.experimental import RrdReader
from scipy.spatial.transform import Rotation

from arkitscenes_download.ca1m.alignment import ClockDiagnostics, RigidAlignment, diagnose_clock, rigid_umeyama
from arkitscenes_download.ca1m.archive import Ca1mFrame, parse_archive
from arkitscenes_download.download_dataset import download_file
from arkitscenes_download.ingest.blueprint import DEPTH_RANGE_MM
from arkitscenes_download.ingest.paths import CAM_WIDE, GT, GT_DEPTH, GT_PINHOLE_WIDE, GT_RIG, RIG, TIMELINE
from arkitscenes_download.ingest.recording import atomic_recording
from arkitscenes_download.schema import GT_DEPTH_LAYER, GT_POSES_LAYER

DEFAULT_DATASET_ROOT: Path = Path("/mnt/nas/datasets/arkitscenes/arkitscenes.2026.07.22")
DEFAULT_SCRATCH: Path = Path("/var/tmp/ca1m-scratch")
MANIFEST_URLS: dict[str, str] = {
    "train": "https://raw.githubusercontent.com/apple/ml-cubifyanything/main/data/train.txt",
    "val": "https://raw.githubusercontent.com/apple/ml-cubifyanything/main/data/val.txt",
}
PROVENANCE: str = "ca1m-v1"
DEGENERATE_EXTENT2_M: float = 0.05
"""Second principal trajectory extent below which the Umeyama roll is weakly observable."""


@dataclass(frozen=True, slots=True)
class Config:
    """Configuration for concurrent CA-1M ingestion."""

    output: Path
    """Layer-major output root; ``gt_poses`` and ``gt_depth`` are created beneath it."""
    dataset_root: Path = DEFAULT_DATASET_ROOT
    """Existing ARKitScenes layer-major root containing ``base`` and ``calibration``."""
    scratch: Path = DEFAULT_SCRATCH
    """Manifest and resumable tar download/cache directory."""
    video_ids: list[str] | None = None
    """Capture subset; all manifested captures with a base layer when omitted."""
    workers: int = 4
    """Maximum number of captures downloaded and converted concurrently."""
    force: bool = False
    """Rewrite captures even when both output layers already exist."""
    keep_tars: bool = False
    """Keep downloaded or pre-populated tar files after successful conversion."""


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """One manifested CA-1M capture download."""

    video_id: str
    """ARKitScenes capture identifier."""
    split: str
    """CA-1M split, ``train`` or ``val``."""
    url: str
    """Tar download URL."""


@dataclass(frozen=True, slots=True)
class CalibrationTrajectory:
    """Existing ARKit world-from-rig trajectory."""

    video_times_s: Float64[np.ndarray, "n"]
    """Capture-relative trajectory times in seconds."""
    translations_xyz: Float64[np.ndarray, "n 3"]
    """ARKit-world rig translations in metres."""


@dataclass(frozen=True, slots=True)
class GtCoverage:
    """Where CA-1M's frames sit inside the capture's video timeline.

    Apple only releases frames whose laser registration succeeded, so captures
    can miss GT at the head/tail (e.g. 42898570 starts at +16.4 s) or have
    interior holes. This is a property of the CA-1M release, not of ingestion.
    """

    gt_start_s: float
    """video_time of the first written GT frame."""
    gt_end_s: float
    """video_time of the last written GT frame."""
    video_end_s: float
    """End of the capture's calibration timeline (video_time seconds)."""
    max_interior_gap_s: float
    """Largest gap between consecutive GT frames."""


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Successful capture conversion and audit fields."""

    video_id: str
    """ARKitScenes capture identifier."""
    frames: int
    """Number of CA-1M frames written."""
    clock_delta_ms_median: float
    """Median nearest-calibration timestamp delta in milliseconds."""
    clock_delta_ms_max: float
    """Maximum nearest-calibration timestamp delta in milliseconds."""
    clock_fraction_over_10ms: float
    """Fraction of timestamp matches farther than ten milliseconds."""
    umeyama_rms_mm: float
    """Rigid-alignment RMS residual in millimetres."""
    umeyama_pairs: int
    """Number of correspondences within ten milliseconds used for alignment."""
    seconds: float
    """Capture conversion wall time in seconds."""
    coverage: GtCoverage
    """GT frame span vs the video timeline (release property, see GtCoverage)."""

    def summary_line(self) -> str:
        """Return the one-line capture audit summary."""
        return (
            f"id={self.video_id} frames={self.frames} "
            f"clock_delta_ms_median={self.clock_delta_ms_median:.3f} clock_delta_ms_max={self.clock_delta_ms_max:.3f} "
            f"umeyama_rms_mm={self.umeyama_rms_mm:.3f} umeyama_pairs={self.umeyama_pairs} "
            f"gt_span={self.coverage.gt_start_s:.1f}..{self.coverage.gt_end_s:.1f}/{self.coverage.video_end_s:.1f}s "
            f"max_gap={self.coverage.max_interior_gap_s:.1f}s seconds={self.seconds:.2f}"
        )


@dataclass(frozen=True, slots=True)
class CaptureFailure:
    """One capture failure that must not stop other workers."""

    video_id: str
    """ARKitScenes capture identifier."""
    message: str
    """Human-readable exception summary."""
    seconds: float
    """Wall time before the failure in seconds."""


def _manifest_paths(config: Config) -> dict[str, Path]:
    """Fetch each pinned manifest once into scratch."""
    manifest_root: Path = config.scratch / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {split: manifest_root / f"{split}.txt" for split in MANIFEST_URLS}
    for split, path in paths.items():
        if download_file(MANIFEST_URLS[split], path) is None:
            raise RuntimeError(f"failed to download the CA-1M {split} manifest from {MANIFEST_URLS[split]}")
    return paths


def load_capture_specs(config: Config) -> list[CaptureSpec]:
    """Load manifested captures and apply explicit or base-layer selection."""
    specs_by_id: dict[str, CaptureSpec] = {}
    for split, manifest_path in _manifest_paths(config).items():
        for line in manifest_path.read_text().splitlines():
            url: str = line.strip()
            if not url or url.startswith("#"):
                continue
            filename: str = url.rsplit("/", 1)[-1]
            prefix: str = f"ca1m-{split}-"
            if not filename.startswith(prefix) or not filename.endswith(".tar"):
                raise ValueError(f"unexpected {split} CA-1M manifest entry: {url}")
            video_id: str = filename[len(prefix) : -len(".tar")]
            if not video_id.isdigit():
                raise ValueError(f"unexpected CA-1M video_id in manifest entry: {url}")
            if video_id in specs_by_id:
                raise ValueError(f"duplicate CA-1M video_id in manifests: {video_id}")
            specs_by_id[video_id] = CaptureSpec(video_id, split, url)

    if config.video_ids is None:
        selected_ids: list[str] = sorted(video_id for video_id in specs_by_id if (config.dataset_root / "base" / f"{video_id}.rrd").is_file())
    else:
        selected_ids = list(dict.fromkeys(config.video_ids))
    missing_ids: list[str] = [video_id for video_id in selected_ids if video_id not in specs_by_id]
    if missing_ids:
        raise ValueError(f"video_ids absent from CA-1M manifests: {missing_ids}")
    return [specs_by_id[video_id] for video_id in selected_ids]


def read_capture_epoch(base_rrd: Path) -> float:
    """Read the ARKit uptime at ``video_time == 0`` from a base-layer RRD."""
    reader: RrdReader = RrdReader(base_rrd)
    for chunk in reader.stream():
        if str(chunk.entity_path).rsplit("/", 1)[-1] != "capture":
            continue
        batch: Any = chunk.to_record_batch()
        if "uptime_epoch_seconds" not in batch.schema.names:
            continue
        values: list[Any] = batch.column("uptime_epoch_seconds").to_pylist()
        if values and values[0] is not None and len(values[0]):
            return float(values[0][0])
    raise ValueError(f"base layer {base_rrd} is missing the capture:uptime_epoch_seconds property")


def _component_array(batch: Any, name: str, width: int) -> Float64[np.ndarray, "n width"]:
    """Flatten a singleton-list Rerun vector component into a numeric array."""
    values: list[Any] = batch.column(name).to_pylist()
    array: np.ndarray = np.asarray(values, dtype=np.float64)
    return array.reshape(len(values), width)


def read_calibration_trajectory(calibration_rrd: Path) -> CalibrationTrajectory:
    """Read world-from-rig samples, requiring the wide camera to be the rig reference."""
    times_parts: list[np.ndarray] = []
    translation_parts: list[np.ndarray] = []
    cam_translation_xyz: Float64[np.ndarray, "3"] | None = None
    cam_rotation_xyzw: Float64[np.ndarray, "4"] | None = None
    reader: RrdReader = RrdReader(calibration_rrd)
    for chunk in reader.stream():
        entity_path: str = str(chunk.entity_path)
        batch: Any = chunk.to_record_batch()
        if entity_path == f"/{RIG}" and not chunk.is_static:
            required: set[str] = {TIMELINE, "Transform3D:translation"}
            if not required.issubset(batch.schema.names):
                continue
            times_ns: np.ndarray = np.asarray(batch.column(TIMELINE)).astype("timedelta64[ns]").astype(np.int64)
            times_parts.append(times_ns.astype(np.float64) / 1e9)
            translation_parts.append(_component_array(batch, "Transform3D:translation", 3))
        elif entity_path == f"/{CAM_WIDE}" and chunk.is_static:
            if "Transform3D:translation" in batch.schema.names:
                cam_translation_xyz = _component_array(batch, "Transform3D:translation", 3)[0]
            if "Transform3D:quaternion" in batch.schema.names:
                cam_rotation_xyzw = _component_array(batch, "Transform3D:quaternion", 4)[0]
    if not times_parts or not translation_parts:
        raise ValueError(f"calibration layer {calibration_rrd} has no world/rig_00 trajectory")
    if cam_translation_xyz is None or cam_rotation_xyzw is None:
        raise ValueError(f"calibration layer {calibration_rrd} has no static cam_00 transform")

    # CA-1M RT.json is FARO-world-from-camera; treating it as FARO-world-from-rig
    # is only valid while cam_00 IS the rig reference (identity extrinsic), which
    # our own ingest guarantees. Fail loudly if that schema assumption ever breaks.
    rig_from_cam00_44: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    rig_from_cam00_44[:3, :3] = Rotation.from_quat(cam_rotation_xyzw).as_matrix()
    rig_from_cam00_44[:3, 3] = cam_translation_xyz
    if not np.allclose(rig_from_cam00_44, np.eye(4), rtol=0.0, atol=1e-6):
        raise ValueError(f"calibration layer {calibration_rrd}: cam_00 is not the rig reference; CA-1M poses assume camera == rig")

    video_times_s: Float64[np.ndarray, "n"] = np.concatenate(times_parts).astype(np.float64)
    translations_xyz: Float64[np.ndarray, "n 3"] = np.concatenate(translation_parts).astype(np.float64)
    order: Int64[np.ndarray, "n"] = np.argsort(video_times_s).astype(np.int64)
    return CalibrationTrajectory(video_times_s[order], translations_xyz[order])


def _write_pose_layer(
    output_path: Path,
    video_id: str,
    video_times_s: Float64[np.ndarray, "n"],
    faro_from_rig_n44: Float64[np.ndarray, "n 4 4"],
    alignment: RigidAlignment,
    coverage: GtCoverage,
) -> None:
    """Write aligned GT poses plus the typed ``gt`` recording-property group.

    Provenance, alignment quality, and coverage become ``property:gt:<field>``
    segment-table columns so segments are sortable and filterable in the
    catalog without an entity reader. Deeper diagnostics stay in the audit log.
    """
    with atomic_recording(output_path, video_id, send_properties=False) as recording:
        recording.send_property(
            "gt",
            rr.AnyValues(
                provenance=PROVENANCE,
                umeyama_rms_m=float(alignment.rms_m),
                start_s=float(coverage.gt_start_s),
                end_s=float(coverage.gt_end_s),
                max_interior_gap_s=float(coverage.max_interior_gap_s),
            ),
        )
        recording.log(GT, rr.Transform3D(translation=alignment.translation_xyz, mat3x3=alignment.rotation_33), static=True)
        rr.send_columns(
            GT_RIG,
            indexes=[rr.TimeColumn(TIMELINE, duration=video_times_s)],
            columns=rr.Transform3D.columns(translation=faro_from_rig_n44[:, :3, 3], mat3x3=faro_from_rig_n44[:, :3, :3]),
            recording=recording,
        )


def _write_depth_layer(output_path: Path, video_id: str, video_times_s: Float64[np.ndarray, "n"], frames: list[Ca1mFrame]) -> None:
    """Write unchanged depth PNG columns and timestamped pinhole calibration (K varies per frame)."""
    intrinsics_n33: Float64[np.ndarray, "n 3 3"] = np.stack([frame.intrinsics_33 for frame in frames])
    resolutions_wh: list[tuple[int, int]] = [frame.resolution_wh for frame in frames]
    depth_pngs: list[bytes] = [frame.depth_png for frame in frames]
    with atomic_recording(output_path, video_id, send_properties=False) as recording:
        rr.send_columns(
            GT_PINHOLE_WIDE,
            indexes=[rr.TimeColumn(TIMELINE, duration=video_times_s)],
            columns=rr.Pinhole.columns(
                image_from_camera=intrinsics_n33,
                resolution=resolutions_wh,
                camera_xyz=[rr.ViewCoordinates.RDF] * len(frames),
            ),
            recording=recording,
        )
        rr.send_columns(
            GT_DEPTH,
            indexes=[rr.TimeColumn(TIMELINE, duration=video_times_s)],
            columns=rr.EncodedDepthImage.columns(
                blob=depth_pngs,
                media_type=["image/png"] * len(frames),
                meter=[1000.0] * len(frames),
                depth_range=[DEPTH_RANGE_MM] * len(frames),
            ),
            recording=recording,
        )


def ingest_capture(spec: CaptureSpec, config: Config) -> CaptureResult:
    """Download/cache, align, and atomically write both layers for one capture."""
    started: float = time.perf_counter()
    tar_path: Path = config.scratch / f"ca1m-{spec.split}-{spec.video_id}.tar"
    # download_file is atomic (.part + rename), resumable, and retrying, so a
    # tar at the final path is complete by construction and reused as-is.
    if download_file(spec.url, tar_path) is None:
        raise RuntimeError(f"failed to download the CA-1M tar from {spec.url}")

    epoch_seconds: float = read_capture_epoch(config.dataset_root / "base" / f"{spec.video_id}.rrd")
    calibration: CalibrationTrajectory = read_calibration_trajectory(config.dataset_root / "calibration" / f"{spec.video_id}.rrd")
    frames: list[Ca1mFrame] = parse_archive(tar_path, expected_video_id=spec.video_id)
    timestamps_ns: Int64[np.ndarray, "n"] = np.asarray([frame.timestamp_ns for frame in frames], dtype=np.int64)
    # parse_archive sorts by timestamp, so video_times_s is ascending.
    video_times_s: Float64[np.ndarray, "n"] = timestamps_ns.astype(np.float64) / 1e9 - epoch_seconds
    if video_times_s[0] < 0.0:
        raise ValueError(f"CA-1M frames precede video_time zero (first at {video_times_s[0]:.3f}s); never seen in the corpus")

    diagnostics: ClockDiagnostics = diagnose_clock(video_times_s, calibration.video_times_s)
    if diagnostics.should_warn:
        print(
            f"WARN id={spec.video_id}: CA-1M clock assumption may be broken; median={diagnostics.median_delta_s * 1000.0:.3f}ms "
            f"over_10ms={diagnostics.fraction_over_10ms:.3%}"
        )
    faro_from_rig_n44: Float64[np.ndarray, "n 4 4"] = np.stack([frame.faro_from_camera_44 for frame in frames])
    pair_mask: np.ndarray = diagnostics.deltas_s <= 0.010
    num_pairs: int = int(np.count_nonzero(pair_mask))
    if num_pairs < 3:
        raise ValueError(f"only {num_pairs} CA-1M/calibration pairs are within 10 ms")
    alignment: RigidAlignment = rigid_umeyama(
        faro_from_rig_n44[pair_mask, :3, 3],
        calibration.translations_xyz[diagnostics.nearest_indices[pair_mask]],
    )
    if alignment.source_extent2_m < DEGENERATE_EXTENT2_M:
        print(
            f"WARN id={spec.video_id}: near-collinear trajectory (second extent "
            f"{alignment.source_extent2_m * 100.0:.1f} cm) — Umeyama roll is weakly observable; rms alone cannot be trusted"
        )

    coverage: GtCoverage = GtCoverage(
        gt_start_s=float(video_times_s[0]),
        gt_end_s=float(video_times_s[-1]),
        video_end_s=float(calibration.video_times_s.max()),
        max_interior_gap_s=float(np.diff(video_times_s).max()) if len(video_times_s) > 1 else 0.0,
    )

    poses_dir: Path = config.output / GT_POSES_LAYER
    depth_dir: Path = config.output / GT_DEPTH_LAYER
    poses_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    _write_pose_layer(poses_dir / f"{spec.video_id}.rrd", spec.video_id, video_times_s, faro_from_rig_n44, alignment, coverage)
    _write_depth_layer(depth_dir / f"{spec.video_id}.rrd", spec.video_id, video_times_s, frames)
    if not config.keep_tars:
        tar_path.unlink()
    return CaptureResult(
        video_id=spec.video_id,
        frames=len(frames),
        clock_delta_ms_median=diagnostics.median_delta_s * 1000.0,
        clock_delta_ms_max=diagnostics.max_delta_s * 1000.0,
        clock_fraction_over_10ms=diagnostics.fraction_over_10ms,
        umeyama_rms_mm=alignment.rms_m * 1000.0,
        umeyama_pairs=num_pairs,
        seconds=time.perf_counter() - started,
        coverage=coverage,
    )


def _ingest_capture_safely(spec: CaptureSpec, config: Config) -> CaptureResult | CaptureFailure:
    """Convert one capture while preserving beartype failures for the coordinator."""
    started: float = time.perf_counter()
    try:
        return ingest_capture(spec, config)
    except BeartypeException:
        raise
    except Exception as error:
        return CaptureFailure(spec.video_id, f"{type(error).__name__}: {error}", time.perf_counter() - started)


def _append_audit_log(log_path: Path, result: CaptureResult) -> None:
    """Append one successful capture result as a JSON line."""
    with log_path.open("a", encoding="utf-8") as audit_file:
        audit_file.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def run(config: Config) -> list[str]:
    """Run the selected concurrent batch and return failed capture identifiers."""
    config.scratch.mkdir(parents=True, exist_ok=True)
    config.output.mkdir(parents=True, exist_ok=True)
    selected_specs: list[CaptureSpec] = load_capture_specs(config)
    specs: list[CaptureSpec] = []
    for spec in selected_specs:
        poses_path: Path = config.output / GT_POSES_LAYER / f"{spec.video_id}.rrd"
        depth_path: Path = config.output / GT_DEPTH_LAYER / f"{spec.video_id}.rrd"
        if not config.force and poses_path.is_file() and depth_path.is_file():
            print(f"id={spec.video_id} skipped=both_outputs_exist")
        else:
            specs.append(spec)

    failed_ids: list[str] = []
    audit_path: Path = config.output / "ca1m_ingest_log.jsonl"
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures: list[Future[CaptureResult | CaptureFailure]] = [pool.submit(_ingest_capture_safely, spec, config) for spec in specs]
        for future in as_completed(futures):
            # BeartypeException propagates from future.result() and kills the batch:
            # a type-contract violation is a code bug, not a per-capture failure.
            result: CaptureResult | CaptureFailure = future.result()
            if isinstance(result, CaptureFailure):
                failed_ids.append(result.video_id)
                print(f"FAILED id={result.video_id} seconds={result.seconds:.2f} error={result.message}")
            else:
                print(result.summary_line())
                _append_audit_log(audit_path, result)
    failed_ids.sort()
    if failed_ids:
        print(f"failed_ids={failed_ids}")
    return failed_ids


def main(config: Config) -> None:
    """Run CA-1M ingestion and exit nonzero when any capture failed."""
    failed_ids: list[str] = run(config)
    raise SystemExit(1 if failed_ids else 0)
