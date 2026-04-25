"""Orchestrate ingestion of one VSLAM-LAB sequence into a single ``.rrd`` file.

Scene-graph (only paths whose modality applies are populated):

    /world/body                              Transform3D STREAM (world_T_body from GT;
                                             static identity fallback when GT is empty)
    /world/body/cam_0                        Transform3D (body -> rgb_0 extrinsic, static)
    /world/body/cam_0/pinhole                Pinhole intrinsics + distortion (static)
    /world/body/cam_0/pinhole/video          VideoStream (H.264, rgb_0)
    /world/body/cam_0/pinhole/depth          DepthImage stream (rgbd; registered to rgb_0)
    /world/body/cam_1/...                    Same tree for rgb_1 when stereo
    /world/body/cam_1/pinhole/depth          DepthImage stream (OPENLORIS-style stereo+rgbd)
    /imu/gyro, /imu/accel                    Scalars streams when *-vi

The ground-truth pose is logged directly as the body entity's transform
rather than on a separate ``/world/gt`` path — this makes every sensor
under ``/world/body`` ride the GT trajectory automatically via the scene
graph, matching simplecv's convention of logging extrinsics at the entity
itself (see ``simplecv.rerun_log_utils.log_pinhole``).

Rationale for shared pinhole: VSLAM-LAB's ``calibration.yaml`` already encodes
``cam_type: rgb+depth, depth_name: depth_0`` on a single camera entry, meaning
the depth stream is pre-registered to the RGB sensor. Mirroring
``packages/monoprior/monopriors/rr_logging_utils.py:24-47``, RGB and depth
sit under the same pinhole so the viewer hover-syncs them at the pixel level.

Properties (``/__properties``):
    info              dataset, sequence, slug, modality, counts, fps, duration
    calibration       num_cameras, cam0 intrinsics summary, depth_factor
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import rerun as rr
from jaxtyping import Int64
from numpy import ndarray

from slam_evals.blueprint import build_blueprint
from slam_evals.data.parse import parse_calibration, parse_groundtruth, parse_imu, parse_rgb_csv
from slam_evals.data.types import Calibration, Sequence
from slam_evals.ingest.calibration import log_calibration_static
from slam_evals.ingest.columns import log_groundtruth_columns, log_imu_columns, trajectory_length_m
from slam_evals.ingest.properties import send_sequence_properties
from slam_evals.ingest.video import encode_and_log_video


def _fps_from_timestamps(ts_ns: Int64[ndarray, "n"]) -> float | None:
    if ts_ns.shape[0] < 2:
        return None
    dt_ns = float(np.median(np.diff(ts_ns)))
    if dt_ns <= 0:
        return None
    return 1e9 / dt_ns


def _log_depth_stream(
    *,
    seq_root: Path,
    paths: tuple[str, ...],
    timestamps_ns: Int64[ndarray, "n"],
    t0_ns: int,
    entity_path: str,
    depth_factor: float | None,
    recording: rr.RecordingStream,
    timeline: str = "video_time",
) -> int:
    """Stream depth frames straight from disk into ``EncodedDepthImage``.

    VSLAM-LAB depth files are uniformly 16-bit single-channel PNG (audited
    across all 29 rgbd / rgbd-vi sequences). Since rerun's RRD format
    accepts pre-encoded image bytes, we ship the on-disk PNG bytes verbatim
    instead of decoding to a ndarray and re-encoding via
    ``DepthImage.compress(...)``. Saves both libpng round-trips —
    ~12 ms/frame becomes ~0.07 ms/frame.

    Tradeoff: source PNGs use looser filter selection than rerun's libpng
    path would, so the RRD is ~28% larger on average than re-compressed
    output. Acceptable for a benchmark catalog where ingest is rare and
    visualisation is the hot path.
    """
    meter: float | None = float(depth_factor) if depth_factor and depth_factor > 0 else None
    count = 0
    for i, rel in enumerate(paths):
        path = seq_root / rel
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        t_rel_s = (int(timestamps_ns[i]) - t0_ns) * 1e-9
        rr.set_time(timeline, duration=t_rel_s, recording=recording)
        depth = rr.archetypes.EncodedDepthImage.from_fields(
            blob=blob,
            media_type="image/png",
            meter=meter,
        )
        rr.log(entity_path, depth, recording=recording)
        count += 1
    return count


def ingest_sequence(
    sequence: Sequence,
    out_rrd_path: Path,
    *,
    application_id: str = "slam-evals",
) -> Path:
    """Ingest ``sequence`` into a fresh RRD at ``out_rrd_path``. Returns the path."""
    out_rrd_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_csv = parse_rgb_csv(sequence.root / "rgb.csv")
    gt = parse_groundtruth(sequence.root / "groundtruth.csv")
    imu = parse_imu(sequence.root / "imu_0.csv") if sequence.modality.has_imu else None
    calib: Calibration | None = parse_calibration(sequence.root / "calibration.yaml")

    # Fresh, isolated stream per sequence. Using it as a context manager makes
    # it the active recording for any default-recording calls inside; helper
    # functions that accept ``recording=`` still get it passed explicitly for
    # clarity.
    recording = rr.RecordingStream(
        application_id=application_id,
        recording_id=f"{sequence.recording_id}__{uuid4().hex[:8]}",
    )

    with recording:
        # Embed the default blueprint into the RRD so `rerun <file>.rrd` opens
        # with the right 3D / 2D / timeseries layout for any modality.
        rr.send_blueprint(build_blueprint(), make_active=True, make_default=True, recording=recording)

        rr.log("/", rr.ViewCoordinates.RDF, static=True, recording=recording)  # SLAM CV convention
        # Body frame: time-varying world_T_body from GT (logged further down).
        # When GT is empty we still need an anchor so child sensor entities
        # resolve, so fall back to a static identity transform.
        if gt.ts_ns.shape[0] == 0:
            rr.log(
                "/world/body",
                rr.Transform3D(translation=[0.0, 0.0, 0.0]),
                static=True,
                recording=recording,
            )

        # Static per-camera calibration. depth_0 / depth_1 are *not* given
        # their own entity paths — depth is pre-registered to rgb_0 / rgb_1
        # and logged as a sibling of the RGB VideoStream under the shared
        # pinhole. See module docstring for the rationale.
        log_calibration_static(
            calib,
            cam_entity_paths={
                "rgb_0": "/world/body/cam_0",
                "rgb_1": "/world/body/cam_1",
            },
            recording=recording,
        )

        t0_ns: int = int(rgb_csv.ts_rgb_0_ns[0])

        rgb_0_paths = [sequence.root / p for p in rgb_csv.path_rgb_0]
        fps_rgb = _fps_from_timestamps(rgb_csv.ts_rgb_0_ns)
        encode_and_log_video(
            entity_path="/world/body/cam_0/pinhole/video",
            image_paths=rgb_0_paths,
            timestamps_ns=rgb_csv.ts_rgb_0_ns,
            recording=recording,
            fps_hint=fps_rgb,
        )

        if sequence.modality.has_stereo and rgb_csv.path_rgb_1 is not None and rgb_csv.ts_rgb_1_ns is not None:
            rgb_1_paths = [sequence.root / p for p in rgb_csv.path_rgb_1]
            encode_and_log_video(
                entity_path="/world/body/cam_1/pinhole/video",
                image_paths=rgb_1_paths,
                timestamps_ns=rgb_csv.ts_rgb_1_ns,
                recording=recording,
                fps_hint=fps_rgb,
            )

        # Depth streams (when present). Shared pinhole tree: depth_0 sits under
        # cam_0's pinhole, depth_1 under cam_1's.
        if rgb_csv.path_depth_0 is not None and rgb_csv.ts_depth_0_ns is not None:
            depth_factor_0 = next(
                (c.depth_factor for c in (calib.cameras if calib else ()) if c.cam_name in ("rgb_0", "depth_0")),
                None,
            )
            _log_depth_stream(
                seq_root=sequence.root,
                paths=rgb_csv.path_depth_0,
                timestamps_ns=rgb_csv.ts_depth_0_ns,
                t0_ns=t0_ns,
                entity_path="/world/body/cam_0/pinhole/depth",
                depth_factor=depth_factor_0,
                recording=recording,
            )

        if rgb_csv.path_depth_1 is not None and rgb_csv.ts_depth_1_ns is not None:
            depth_factor_1 = next(
                (c.depth_factor for c in (calib.cameras if calib else ()) if c.cam_name in ("rgb_1", "depth_1")),
                None,
            )
            _log_depth_stream(
                seq_root=sequence.root,
                paths=rgb_csv.path_depth_1,
                timestamps_ns=rgb_csv.ts_depth_1_ns,
                t0_ns=t0_ns,
                entity_path="/world/body/cam_1/pinhole/depth",
                depth_factor=depth_factor_1,
                recording=recording,
            )

        # Some sequences (HAMLYN/*, HILTI2026/floor_3_*) ship an empty
        # groundtruth.csv. The static identity transform logged above stands in
        # so every sensor entity still has a parent pose.
        if gt.ts_ns.shape[0] > 0:
            log_groundtruth_columns(gt, t0_ns=t0_ns, recording=recording)

        num_imu = 0
        if imu is not None:
            log_imu_columns(imu, t0_ns=t0_ns, recording=recording)
            num_imu = int(imu.ts_ns.shape[0])

        duration_s = float((int(rgb_csv.ts_rgb_0_ns[-1]) - t0_ns) * 1e-9)
        send_sequence_properties(
            sequence=sequence,
            num_rgb_frames=int(rgb_csv.ts_rgb_0_ns.shape[0]),
            num_gt_poses=int(gt.ts_ns.shape[0]),
            trajectory_len_m=trajectory_length_m(gt.translation),
            duration_s=duration_s,
            fps_rgb=fps_rgb,
            num_imu_samples=num_imu,
            calibration=calib,
            recording=recording,
        )

    recording.save(str(out_rrd_path))
    return out_rrd_path
