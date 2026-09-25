"""HO-Cap base and hand-pose writers using the shared hand and video writers."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pyarrow as pa
import rerun as rr
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters

from dataforge import hands, schema, writing
from dataforge.datasets.hocap_source import (
    CLOCK_SOURCE,
    EGO_RIG,
    FRAME_RATE,
    MANO_SIDES,
    SOURCE_REVISION,
    HandLabels,
    HocapCamera,
    SequenceData,
    posed_rows,
    present_rows,
    read_labels,
)
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    log_camera_node,
    log_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, parallel_clips


def write_base(
    recording: rr.RecordingStream, scene: SequenceData, subject: ZipFile, key: str, identity: SequenceIdentity, timer: SequenceTimer, work_root: Path
) -> None:
    """Write native color streams, calibrated rigs and a dense PV pose track."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)
    rr.log("/", annotation_context(), static=True, recording=recording)

    def encode(source: HocapCamera, clip: Path) -> None:
        # Python 3.12 ZipFile serialises shared reads with _SharedFile's lock.
        count: int = encode_frames_to_mp4(
            (subject.read(f"{key}/{source.serial}/color_{index:06d}.jpg") for index in range(scene.count)),
            clip,
            source=FrameSource("jpeg"),
            fps=FRAME_RATE,
            gop=AV1_GOP,
            cq=AV1_CQ,
        )
        if count != scene.count:
            raise ValueError(f"{key}/{source.serial}: encoded {count} frames, expected {scene.count}")

    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="hocap-", dir=work_root) as work:
        clips: list[Path] = [Path(work) / f"{source.serial}.mp4" for source in scene.cameras]
        jobs = [(clip, partial(encode, source, clip)) for source, clip in zip(scene.cameras, clips, strict=True)]
        with parallel_clips(jobs, timer) as encoded:
            for source, color, clip in zip(scene.cameras, scene.intrinsics, encoded, strict=True):
                rig: int = source.rig
                serial: str = source.serial
                log_rig_node(recording, rig, reference="cam_00" if source.kind == "ego" else None, num_cameras=1, name=serial, kind=source.kind)
                transform: Float64[ndarray, "4 4"] = np.eye(4) if source.kind == "ego" else scene.world_T_cam[rig]
                camera: PinholeParameters = PinholeParameters(
                    name=serial,
                    extrinsics=Extrinsics(world_R_cam=transform[:3, :3], world_t_cam=transform[:3, 3]),
                    intrinsics=Intrinsics.from_focal_principal_point(
                        camera_conventions="RDF", fl_x=color.fx, fl_y=color.fy, cx=color.ppx, cy=color.ppy, width=color.width, height=color.height
                    ),
                )
                log_camera_node(recording, rig, 0, camera, name=serial, kind="rgb", image_plane_distance=0.05)
                rr.log(
                    schema.cam_path(rig, 0),
                    rr.AnyValues(source_width=color.width, source_height=color.height, video_codec="av1", cq=AV1_CQ, gop=AV1_GOP),
                    static=True,
                    recording=recording,
                )
                log_video_stream(recording, clip, schema.video_path(rig, 0), times_ns=scene.times_ns, frame_indices=scene.frame_indices)
                clip.unlink()
    posed: Bool[ndarray, "n"] = posed_rows(scene.pv)
    log_pose_track(
        recording,
        schema.rig_path(EGO_RIG),
        times_ns=scene.times_ns,
        frame_indices=scene.frame_indices,
        translations_xyz=np.where(posed[:, None], scene.pv[:, 4:], np.float32(np.nan)),
        quaternions_xyzw=np.where(posed[:, None], scene.pv[:, :4], np.float32(np.nan)),
    )
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=scene.count,
        num_cameras=len(scene.cameras),
        source_revision=SOURCE_REVISION,
        clock_source=CLOCK_SOURCE,
        source_num_frames=scene.meta.num_frames,
        source_resolution=pa.array(
            [f"{source.serial}:{color.width}x{color.height}" for source, color in zip(scene.cameras, scene.intrinsics, strict=True)]
        ),
    )
    recording.send_property(
        "episode",
        rr.AnyValues(
            subject_id=scene.meta.subject_id,
            object_ids=pa.array(scene.meta.object_ids),
            task_id=scene.meta.task_id,
            mano_sides=pa.array(scene.meta.mano_sides),
        ),
    )


def write_hands(recording: rr.RecordingStream, scene: SequenceData, archive: ZipFile, key: str, *, members: frozenset[str]) -> None:
    """Write shipped 3D/2D labels and every MANO parameter, including missing rows."""
    labels: HandLabels = read_labels(archive, key, scene, members=members)
    hands.log_keypoints3d(recording, times_ns=scene.times_ns, frame_indices=scene.frame_indices, positions=labels.xyz, confidence=None)
    for camera in scene.exo:
        rig: int = camera.rig
        hands.log_keypoints2d(
            recording, rig, 0, times_ns=scene.times_ns, frame_indices=scene.frame_indices, positions=labels.uv[rig], confidence=None
        )
    for side, mano in zip(MANO_SIDES, scene.mano, strict=True):
        parameters: Float32[ndarray, "n 51"] = np.where(present_rows(mano)[:, None], mano, np.float32(np.nan))
        path: str = schema.hand_mano_path(side)
        rr.log(
            path,
            rr.AnyValues(betas=scene.betas, use_pca=True, source="poses_m.npy", present=side in scene.meta.mano_sides),
            static=True,
            recording=recording,
        )
        rr.send_columns(
            path,
            indexes=[time_column(scene.times_ns), frame_index_column(scene.frame_indices)],
            columns=rr.AnyValues.columns(
                global_orient=pa.FixedSizeListArray.from_arrays(pa.array(parameters[:, :3].reshape(-1)), 3),
                pca_coefficients=pa.FixedSizeListArray.from_arrays(pa.array(parameters[:, 3:48].reshape(-1)), 45),
                translation=pa.FixedSizeListArray.from_arrays(pa.array(parameters[:, 48:].reshape(-1)), 3),
            ),
            recording=recording,
        )

