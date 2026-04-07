"""Batch calibration for processing multiple episodes directly.

Bypasses Gradio API overhead by invoking ExoOnlyCalibService directly.
This module contains the core logic; CLI wrapper is in tools/batch_exo_calib_client.py.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Float, Float32, Int, UInt8
from numpy import ndarray
from simplecv.apis.view_exoego import (
    LogPaths,
    SceneSetupResult,
    log_environment_mesh,
    log_exoego_batch,
    setup_scene,
)
from simplecv.data.exo.base_exo import BaseExoSequence
from simplecv.data.exoego.base_exoego import BaseExoEgoSequence
from simplecv.data.exoego.rrd_exoego import RRDExoEgoConfig
from simplecv.rerun_log_utils import log_pinhole
from simplecv.video_io import MultiVideoReader, TorchCodecMultiVideoReader

from mv_api.api.exo_only_calibration import (
    ExoCalibResult,
    ExoOnlyCalibService,
    ExoOnlyCalibServiceConfig,
    create_exo_ego_blueprint,
    get_frame_timestamps_from_reader,
    get_target_frame_idx,
    set_annotation_context,
)


@dataclass
class EpisodeInfo:
    """Metadata for a single episode."""

    rrd_path: Path
    """Path to the source episode RRD file."""
    calibrated_path: Path
    """Path where calibrated RRD will be saved."""
    sequence_id: str
    """Parent sequence identifier."""

    @property
    def is_calibrated(self) -> bool:
        """Check if calibrated output already exists."""
        return self.calibrated_path.exists()


@dataclass
class ManifestEntry:
    """Status entry for a single episode in the manifest."""

    episode_path: str
    calibrated_path: str
    status: str  # 'pending', 'success', 'error'
    error_message: str | None = None
    calibrated_at: str | None = None


@dataclass
class SequenceManifest:
    """Progress tracking manifest for a sequence."""

    sequence_id: str
    created_at: str
    updated_at: str
    episodes: dict[str, ManifestEntry] = field(default_factory=dict)

    def save(self, manifest_path: Path) -> None:
        """Write manifest to disk."""
        data: dict[str, Any] = {
            "sequence_id": self.sequence_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "episodes": {
                k: {
                    "episode_path": v.episode_path,
                    "calibrated_path": v.calibrated_path,
                    "status": v.status,
                    "error_message": v.error_message,
                    "calibrated_at": v.calibrated_at,
                }
                for k, v in self.episodes.items()
            },
        }
        manifest_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load_or_create(cls, manifest_path: Path, sequence_id: str) -> SequenceManifest:
        """Load existing manifest or create new one."""
        now: str = datetime.now().isoformat()
        if manifest_path.exists():
            data: dict[str, Any] = json.loads(manifest_path.read_text())
            episodes: dict[str, ManifestEntry] = {}
            for k, v in data.get("episodes", {}).items():
                episodes[k] = ManifestEntry(
                    episode_path=v["episode_path"],
                    calibrated_path=v["calibrated_path"],
                    status=v["status"],
                    error_message=v.get("error_message"),
                    calibrated_at=v.get("calibrated_at"),
                )
            return cls(
                sequence_id=data["sequence_id"],
                created_at=data["created_at"],
                updated_at=now,
                episodes=episodes,
            )
        return cls(sequence_id=sequence_id, created_at=now, updated_at=now)


@dataclass
class BatchCalibConfig:
    """Configuration for batch exo calibration."""

    cut_root: Path | None = None
    """Root directory containing all sequences (processes everything)."""

    sequence_path: Path | None = None
    """Path to a single sequence to process (alternative to cut_root)."""

    skip_tsdf_fusion: bool = True
    """Skip TSDF mesh fusion for faster processing (extrinsics only)."""

    frame_selection: Literal["middle", "first", "last"] | int = "middle"
    """Which frame to process for calibration."""

    dry_run: bool = False
    """If True, discover episodes but don't actually calibrate."""

    inplace: bool = True
    """Append calibration to original RRD (fast) vs create new -calibrated.rrd (safe)."""


ProgressCallback = Callable[[str], None]
"""Callback for progress messages: (message: str) -> None."""


def discover_episodes_in_sequence(sequence_path: Path) -> list[EpisodeInfo]:
    """Find all episode RRD files in a sequence directory."""
    episodes_dir: Path = sequence_path / "episodes"
    if not episodes_dir.exists():
        return []

    episode_infos: list[EpisodeInfo] = []
    sequence_id: str = sequence_path.name

    for episode_dir in sorted(episodes_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        rrd_path: Path = episode_dir / f"{episode_dir.name}.rrd"
        if not rrd_path.exists():
            continue
        calibrated_path: Path = episode_dir / f"{episode_dir.name}-calibrated.rrd"
        episode_infos.append(
            EpisodeInfo(
                rrd_path=rrd_path,
                calibrated_path=calibrated_path,
                sequence_id=sequence_id,
            )
        )

    return episode_infos


def discover_all_sequences(cut_root: Path) -> list[Path]:
    """Find all sequence directories under cut root."""
    sequences: list[Path] = []
    # Structure: cut/{date}/{sequence_id}/episodes/
    for date_dir in sorted(cut_root.iterdir()):
        if not date_dir.is_dir():
            continue
        for sequence_dir in sorted(date_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue
            if (sequence_dir / "episodes").exists():
                sequences.append(sequence_dir)
    return sequences


@rr.thread_local_stream("batch_exo_calib")
def calibrate_episode_direct(
    service: ExoOnlyCalibService,
    episode: EpisodeInfo,
    config: BatchCalibConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, str | None]:
    """Calibrate a single episode directly using the service.

    This bypasses Gradio API overhead by invoking the service directly.

    Args:
        service: Pre-loaded calibration service.
        episode: Episode metadata.
        config: Calibration configuration.
        progress_callback: Optional callback for progress messages.

    Returns:
        Tuple of (success, error_message). error_message is None on success.
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    try:
        _log(f"  Loading: {episode.rrd_path.name}")

        # Create temp file for output
        with tempfile.NamedTemporaryFile(prefix="exo_calib_", suffix=".rrd", delete=False) as temp_file:
            output_path: Path = Path(temp_file.name)

        # Configure Rerun to save to temp file
        rr.save(str(output_path))

        parent_log_path: Path = Path("world")
        timeline: str = "video_time"

        # Load dataset
        dataset_cfg: RRDExoEgoConfig = RRDExoEgoConfig(rrd_path=episode.rrd_path)
        exoego_sequence: BaseExoEgoSequence = dataset_cfg.setup()
        exo_sequence_obj: object | None = exoego_sequence.exo_sequence
        if exo_sequence_obj is None:
            return False, "Dataset setup failed to provide an exo sequence."
        exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)

        # Setup Rerun
        rr.log("/", exoego_sequence.world_coordinate_system, static=True)
        set_annotation_context(recording=None)

        # Setup scene
        scene_setup_result: SceneSetupResult = setup_scene(
            exoego_sequence,
            parent_log_path=parent_log_path,
            timeline=timeline,
            log_ego=True,
            log_exo=True,
        )
        log_paths: LogPaths = scene_setup_result.log_paths
        shortest_timestamp: Int[ndarray, "n_frames"] = scene_setup_result.shortest_timestamp

        # Log GT data if available
        if exoego_sequence.exoego_labels is not None:
            log_exoego_batch(
                exoego_sequence,
                parent_log_path=parent_log_path,
                timeline=timeline,
                shortest_timestamp=shortest_timestamp,
                log_ego=True,
                log_exo=False,
                log_mano=True,
            )

        if exoego_sequence.environment_mesh is not None:
            log_environment_mesh(exoego_sequence, parent_log_path)

        # Send blueprint
        exo_video_log_paths: list[Path] = log_paths.exo_video_log_paths or []
        ego_video_log_paths: list[Path] = log_paths.ego_video_log_paths or []
        blueprint: rrb.Blueprint = create_exo_ego_blueprint(
            exo_video_log_paths=exo_video_log_paths if exo_video_log_paths else None,
            ego_video_log_paths=ego_video_log_paths if ego_video_log_paths else None,
        )
        rr.send_blueprint(blueprint)

        _log("  Extracting frames...")

        # Extract frames
        exo_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = exo_sequence.exo_video_readers
        exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
            get_frame_timestamps_from_reader(reader) for reader in exo_mv_reader.video_readers
        ]
        min_exo_ts: Int[ndarray, "num_frames"] = min(
            exo_frame_timestamps_list, key=lambda arr: int(arr.shape[0])
        )
        total_frames: int = len(min_exo_ts)
        frame_idx: int = get_target_frame_idx(config.frame_selection, total_frames)
        timestamp_ns: int = int(min_exo_ts[frame_idx])

        # Load BGR frames
        bgr_list: list[UInt8[ndarray, "H W 3"]] = []
        for reader in exo_mv_reader.video_readers:
            frame_obj: Any = reader[frame_idx]
            if frame_obj is None:
                return False, f"Missing exo frame at index {frame_idx}."
            frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
            bgr_list.append(frame_array)

        # Extract GT keypoints
        gt_xyzc: Float32[ndarray, "133 4"] | None = None
        if exoego_sequence.exoego_labels is not None:
            gt_xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_sequence.exoego_labels.xyzc_stack
            if frame_idx < len(gt_xyzc_stack):
                gt_xyzc = gt_xyzc_stack[frame_idx].astype(np.float32, copy=True)

        rr.set_time(timeline=timeline, duration=np.timedelta64(timestamp_ns, "ns"))

        _log("  Calibrating...")

        # Run calibration
        result: ExoCalibResult = service(
            bgr_list=bgr_list,
            gt_xyzc=gt_xyzc,
            skip_tsdf_fusion=config.skip_tsdf_fusion,
            align_to_ego=True,
        )

        # Log camera poses
        for pinhole, exo_log_path in zip(result.pinhole_list, exo_video_log_paths, strict=True):
            cam_log_path: Path = exo_log_path.parent.parent
            log_pinhole(camera=pinhole, cam_log_path=cam_log_path, image_plane_distance=0.1, static=True)

        # Move temp file to final location
        import shutil

        shutil.move(str(output_path), str(episode.calibrated_path))
        _log(f"  ✓ Saved: {episode.calibrated_path.name}")

        return True, None

    except Exception as e:
        error_msg: str = f"{type(e).__name__}: {e}"
        _log(f"  ✗ Error: {error_msg}")
        return False, error_msg


def calibrate_episode_inplace(
    service: ExoOnlyCalibService,
    episode: EpisodeInfo,
    config: BatchCalibConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, ExoCalibResult | str | None]:
    """Calibrate and append extrinsics to original RRD (in-place).

    This is ~3x faster than full relay because it only writes extrinsics
    (~100KB) rather than the entire dataset (~300MB).

    Args:
        service: Pre-loaded calibration service.
        episode: Episode metadata.
        config: Calibration configuration.
        progress_callback: Optional callback for progress messages.

    Returns:
        Tuple of (success, ExoCalibResult if success else error_message).
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    try:
        _log(f"  Loading: {episode.rrd_path.name}")

        # Load original RRD to get app_id and recording_id
        original_recording: Any = rr.recording.load_recording(episode.rrd_path)
        app_id: str = original_recording.application_id()
        rec_id: str = original_recording.recording_id()

        # Load dataset for calibration
        dataset_cfg: RRDExoEgoConfig = RRDExoEgoConfig(rrd_path=episode.rrd_path)
        exoego_sequence: BaseExoEgoSequence = dataset_cfg.setup()
        exo_sequence_obj: object | None = exoego_sequence.exo_sequence
        if exo_sequence_obj is None:
            return False, "Dataset setup failed to provide an exo sequence."
        exo_sequence: BaseExoSequence = cast(BaseExoSequence, exo_sequence_obj)

        # Get exo camera paths from dataset video names (works even without calibration)
        exo_video_names: list[str] = exo_sequence.exo_video_names
        exo_cam_paths: list[Path] = [Path("world/exo") / name for name in exo_video_names]

        # Extract frames for calibration
        _log("  Extracting frames...")
        exo_mv_reader: MultiVideoReader | TorchCodecMultiVideoReader = exo_sequence.exo_video_readers
        exo_frame_timestamps_list: list[Int[ndarray, "num_frames"]] = [
            get_frame_timestamps_from_reader(reader) for reader in exo_mv_reader.video_readers
        ]
        min_exo_ts: Int[ndarray, "num_frames"] = min(
            exo_frame_timestamps_list, key=lambda arr: int(arr.shape[0])
        )
        total_frames: int = len(min_exo_ts)
        frame_idx: int = get_target_frame_idx(config.frame_selection, total_frames)

        # Load BGR frames
        bgr_list: list[UInt8[ndarray, "H W 3"]] = []
        for reader in exo_mv_reader.video_readers:
            frame_obj: Any = reader[frame_idx]
            if frame_obj is None:
                return False, f"Missing exo frame at index {frame_idx}."
            frame_array: UInt8[ndarray, "H W 3"] = np.asarray(frame_obj, dtype=np.uint8)
            bgr_list.append(frame_array)

        # Extract GT keypoints for alignment
        gt_xyzc: Float32[ndarray, "133 4"] | None = None
        if exoego_sequence.exoego_labels is not None:
            gt_xyzc_stack: Float[ndarray, "n_frames 133 4"] = exoego_sequence.exoego_labels.xyzc_stack
            if frame_idx < len(gt_xyzc_stack):
                gt_xyzc = gt_xyzc_stack[frame_idx].astype(np.float32, copy=True)

        _log("  Calibrating...")

        # Run calibration
        result: ExoCalibResult = service(
            bgr_list=bgr_list,
            gt_xyzc=gt_xyzc,
            skip_tsdf_fusion=config.skip_tsdf_fusion,
            align_to_ego=True,
        )

        # Create patch RRD with only extrinsics
        with tempfile.NamedTemporaryFile(prefix="exo_calib_patch_", suffix=".rrd", delete=False) as temp_file:
            patch_path: Path = Path(temp_file.name)

        # Initialize new recording with SAME app_id and recording_id
        rec: rr.RecordingStream = rr.RecordingStream(application_id=app_id, recording_id=rec_id)
        rec.save(str(patch_path))

        # Log only extrinsics (static Transform3D at each camera)
        for pinhole, cam_path in zip(result.pinhole_list, exo_cam_paths, strict=True):
            log_pinhole(camera=pinhole, cam_log_path=cam_path, image_plane_distance=0.1, static=True, recording=rec)

        # Flush the recording
        del rec

        # Append patch to original RRD file
        import shutil

        with open(episode.rrd_path, "ab") as orig_file, open(patch_path, "rb") as patch_file:
            shutil.copyfileobj(patch_file, orig_file)

        # Clean up patch file
        patch_path.unlink()

        _log(f"  ✓ Appended: {episode.rrd_path.name}")
        return True, result  # Return result for propagation

    except Exception as e:
        error_msg: str = f"{type(e).__name__}: {e}"
        _log(f"  ✗ Error: {error_msg}")
        return False, None


def propagate_extrinsics_to_episode(
    calibration_result: ExoCalibResult,
    exo_cam_paths: list[Path],
    episode: EpisodeInfo,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, str | None]:
    """Propagate existing calibration extrinsics to an episode (no recalibration).

    This is much faster than calibrate_episode_inplace because it skips:
    - Dataset loading
    - Video frame extraction
    - MV calibration (VGGT+MoGe)
    - Pose estimation

    Only loads RRD metadata and appends extrinsics patch.

    Args:
        calibration_result: Previously computed calibration result.
        exo_cam_paths: Camera entity paths (e.g., ['world/exo/Dragon-413A7CC1', ...]).
        episode: Target episode to patch.
        progress_callback: Optional callback for progress messages.

    Returns:
        Tuple of (success, error_message).
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    try:
        _log(f"  Propagating: {episode.rrd_path.name}")

        # Load original RRD to get app_id and recording_id
        original_recording: Any = rr.recording.load_recording(episode.rrd_path)
        app_id: str = original_recording.application_id()
        rec_id: str = original_recording.recording_id()

        # Create patch RRD with extrinsics
        with tempfile.NamedTemporaryFile(prefix="exo_calib_patch_", suffix=".rrd", delete=False) as temp_file:
            patch_path: Path = Path(temp_file.name)

        rec: rr.RecordingStream = rr.RecordingStream(application_id=app_id, recording_id=rec_id)
        rec.save(str(patch_path))

        for pinhole, cam_path in zip(calibration_result.pinhole_list, exo_cam_paths, strict=True):
            log_pinhole(camera=pinhole, cam_log_path=cam_path, image_plane_distance=0.1, static=True, recording=rec)

        del rec

        # Append to original
        import shutil

        with open(episode.rrd_path, "ab") as orig_file, open(patch_path, "rb") as patch_file:
            shutil.copyfileobj(patch_file, orig_file)

        patch_path.unlink()

        _log(f"  ✓ Propagated: {episode.rrd_path.name}")
        return True, None

    except Exception as e:
        error_msg: str = f"{type(e).__name__}: {e}"
        _log(f"  ✗ Error: {error_msg}")
        return False, error_msg


def process_sequence(
    service: ExoOnlyCalibService,
    sequence_path: Path,
    config: BatchCalibConfig,
    progress_callback: ProgressCallback | None = None,
) -> tuple[int, int]:
    """Process all episodes in a sequence.

    Args:
        service: Pre-loaded calibration service.
        sequence_path: Path to sequence directory.
        config: Batch calibration configuration.
        progress_callback: Optional callback for progress messages.

    Returns:
        Tuple of (success_count, error_count).
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    sequence_id: str = sequence_path.name
    _log(f"\n{'='*60}")
    _log(f"Processing sequence: {sequence_id}")
    _log(f"{'='*60}")

    episodes: list[EpisodeInfo] = discover_episodes_in_sequence(sequence_path)
    if not episodes:
        _log("  No episodes found.")
        return 0, 0

    _log(f"  Found {len(episodes)} episode(s)")
    if config.inplace:
        _log("  Mode: calibrate-once-propagate (fast)")

    # Load or create manifest
    manifest_path: Path = sequence_path / "calibration_manifest.json"
    manifest: SequenceManifest = SequenceManifest.load_or_create(manifest_path, sequence_id)

    success_count: int = 0
    error_count: int = 0

    # For calibrate-once-propagate: store the result and cam paths from first episode
    sequence_calib_result: ExoCalibResult | None = None
    sequence_exo_cam_paths: list[Path] | None = None

    for i, episode in enumerate(episodes):
        episode_name: str = episode.rrd_path.stem

        # Skip if already calibrated (check file exists OR manifest says success)
        if episode.is_calibrated:
            existing_entry: ManifestEntry | None = manifest.episodes.get(episode_name)
            if existing_entry and existing_entry.status == "success":
                _log(f"  Skipping (already calibrated): {episode_name}")
                success_count += 1
                continue

        # Initialize manifest entry
        manifest.episodes[episode_name] = ManifestEntry(
            episode_path=str(episode.rrd_path),
            calibrated_path=str(episode.calibrated_path),
            status="pending",
        )

        if config.dry_run:
            action: str = "calibrate" if i == 0 or not config.inplace else "propagate"
            _log(f"  [DRY RUN] Would {action}: {episode_name}")
            continue

        # Calibrate or propagate
        success: bool = False
        error_msg: str | None = None

        if config.inplace:
            # Calibrate-once-propagate strategy
            if sequence_calib_result is None or sequence_exo_cam_paths is None:
                # First episode: run full calibration and capture cam paths
                # Load dataset to get camera paths
                dataset_cfg: RRDExoEgoConfig = RRDExoEgoConfig(rrd_path=episode.rrd_path)
                exoego_sequence: BaseExoEgoSequence = dataset_cfg.setup()
                exo_seq: BaseExoSequence | None = exoego_sequence.exo_sequence
                if exo_seq is not None:
                    sequence_exo_cam_paths = [Path("world/exo") / n for n in exo_seq.exo_video_names]

                success_result: tuple[bool, ExoCalibResult | str | None] = calibrate_episode_inplace(
                    service, episode, config, progress_callback=progress_callback
                )
                success = success_result[0]
                if success and isinstance(success_result[1], ExoCalibResult):
                    sequence_calib_result = success_result[1]
                elif not success:
                    error_msg = str(success_result[1]) if success_result[1] else "Unknown error"
            else:
                # Subsequent episodes: propagate extrinsics only
                prop_result: tuple[bool, str | None] = propagate_extrinsics_to_episode(
                    sequence_calib_result, sequence_exo_cam_paths, episode, progress_callback=progress_callback
                )
                success = prop_result[0]
                error_msg = prop_result[1]
        else:
            # Full relay mode: calibrate every episode
            success, error_msg = calibrate_episode_direct(
                service, episode, config, progress_callback=progress_callback
            )

        # Update manifest
        now: str = datetime.now().isoformat()
        if success:
            manifest.episodes[episode_name].status = "success"
            manifest.episodes[episode_name].calibrated_at = now
            success_count += 1
        else:
            manifest.episodes[episode_name].status = "error"
            manifest.episodes[episode_name].error_message = error_msg
            error_count += 1

        manifest.updated_at = now
        manifest.save(manifest_path)

    return success_count, error_count


class BatchCalibService:
    """Service for batch calibrating multiple episodes.

    Loads ExoOnlyCalibService once at construction, then processes
    multiple episodes without reloading models.
    """

    config: BatchCalibConfig
    """Batch calibration configuration."""
    _calib_service: ExoOnlyCalibService
    """Cached calibration service with loaded models."""

    def __init__(
        self,
        config: BatchCalibConfig,
        service_config: ExoOnlyCalibServiceConfig | None = None,
    ) -> None:
        """Initialize batch service with cached models.

        Args:
            config: Batch calibration configuration.
            service_config: Optional calibration service configuration.
        """
        self.config = config
        if service_config is None:
            from monopriors.apis.multiview_calibration import MultiViewCalibratorConfig

            from mv_api.multiview_pose_estimator import MultiviewBodyTrackerConfig

            service_config = ExoOnlyCalibServiceConfig(
                calib_config=MultiViewCalibratorConfig(segment_people=False),
                tracker_config=MultiviewBodyTrackerConfig(),
            )
        print("[BatchCalibService] Loading models...")
        self._calib_service = ExoOnlyCalibService(service_config)
        print("[BatchCalibService] Ready.")

    def run(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[int, int]:
        """Run batch calibration on configured sequences.

        Args:
            progress_callback: Optional callback for progress messages.

        Returns:
            Tuple of (total_success, total_error).
        """

        def _log(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                print(msg)

        # Determine which sequences to process
        sequences: list[Path]
        if self.config.sequence_path is not None:
            sequences = [self.config.sequence_path]
        elif self.config.cut_root is not None:
            sequences = discover_all_sequences(self.config.cut_root)
        else:
            msg: str = "Must specify either sequence_path or cut_root"
            raise ValueError(msg)

        _log(f"Found {len(sequences)} sequence(s) to process")

        if self.config.dry_run:
            _log("[DRY RUN MODE - no calibrations will be performed]")

        # Process each sequence
        total_success: int = 0
        total_error: int = 0

        for sequence_path in sequences:
            success: int
            error: int
            success, error = process_sequence(
                self._calib_service,
                sequence_path,
                self.config,
                progress_callback=progress_callback,
            )
            total_success += success
            total_error += error

        # Summary
        _log(f"\n{'='*60}")
        _log("BATCH CALIBRATION COMPLETE")
        _log(f"{'='*60}")
        _log(f"  Successful: {total_success}")
        _log(f"  Errors:     {total_error}")
        _log(f"{'='*60}")

        return total_success, total_error


def run_batch_calibration(config: BatchCalibConfig) -> None:
    """CLI entry point for batch calibration.

    Args:
        config: Batch calibration configuration from CLI.
    """
    service: BatchCalibService = BatchCalibService(config)
    total_success: int
    total_error: int
    total_success, total_error = service.run()

    # Exit with error code if any failures
    if total_error > 0:
        import sys

        sys.exit(1)
