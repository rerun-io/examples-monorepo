"""Layer writers for the EgoHumans capture layout (EgoHumans and Harmony4D): base, body_pose, body_mesh and projections.

Rigs: exo camera ``camNN`` is ``rig_NN`` (one camera); the Aria glasses ``ariaNN`` are ``rig_(90+NN)`` with ``cam_00`` rgb,
``cam_01`` slam-left and ``cam_02`` slam-right, the rig frame being the rgb camera's device-orientation frame. People are
``person_(NN-1)`` after the glasses key ``ariaNN``. Every stream is on ``video_time`` = (frame - first) / 20 fps beside the
release frame number as ``frame_index``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Float64, Int64, UInt32
from numpy import ndarray
from simplecv.camera_parameters import Extrinsics, Fisheye62Parameters, Fisheye624Parameters, Intrinsics, KannalaBrandtDistortion
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch

from dataforge import blueprints, hands, meshes, schema, writing
from dataforge.datasets.egohumans_source import ARIA_MODES, ARIA_STREAM_IDS, COCO17, FPS, AriaGlasses, ExoCamera, PersonTrack, Sequence, SequenceFiles
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import (
    annotation_context,
    frame_index_column,
    layout_frustum_m,
    log_camera_node,
    log_camera_source,
    log_dense_pose_track,
    log_rig_node,
    log_video_stream,
    time_column,
)
from dataforge.timing import SequenceTimer
from dataforge.video_encoding import AV1_CQ, AV1_GOP, FrameSource, encode_frames_to_mp4, parallel_clips, work_dir

EGO_RIG_OFFSET: int = 90
"""``ariaNN`` is ``rig_(90+NN)``, so exo ``camNN`` keeps ``rig_NN`` whether a take has glasses or not."""
MAX_PIXELS: int = 3_200_000
"""Videos above this many pixels are stored at half size, repeatedly (Pablo's ~3 MP cap, as MAMMA): the 3840x2160 GoPros at
1920x1080 through libjpeg's DCT scaling; the Aria streams stay native."""
MESH_STRIDE: int = FPS // 10
"""body_mesh keeps every second 20 fps frame (10 Hz), the display rate of every dataforge body mesh (``meshes.BODY_MESH_STRIDE``)."""
MESH_BATCH_ROWS: int = 32
"""Mesh rows per chunk: one SMPL row is 82 KB."""
ENCODE_WORKERS: int = 4
"""Cameras encoded at once; each decodes its JPEGs on 8 threads and holds one NVENC session."""
PERSON_COLOURS: list[tuple[int, int, int]] = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
"""Box colour of each person, cycled (MAMMA's mask palette)."""
BODY_SOURCE: str = "release SMPL fit (neutral, 10 betas); vertices and joints as shipped"
WORLD_FRAMES: dict[str, str] = {
    "colmap_from_aria_transforms.pkl": "aria01's SLAM world (gravity-aligned, z up, metres; origin at aria01's first head pose), "
    "reached through colmap_from_aria_transforms.pkl",
    "scale.npy": "the COLMAP world made metric and gravity-aligned by scale.npy (z up, metres; origin where COLMAP put it)",
}
"""``property:capture:world_frame`` for each way a take reaches its world (``Sequence.world_source``)."""


def exo_rig(camera: ExoCamera) -> int:
    """``cam07`` -> rig 7."""
    return camera.number


def ego_rig(glasses: AriaGlasses) -> int:
    """``aria02`` -> rig 92."""
    return EGO_RIG_OFFSET + glasses.number


def person_index(person: PersonTrack) -> int:
    """``aria02`` -> person 1."""
    return person.number - 1


def halvings(width: int, height: int) -> int:
    """How many times a ``width`` x ``height`` video is halved to fit ``MAX_PIXELS``."""
    count: int = 0
    while (width >> count) * (height >> count) > MAX_PIXELS:
        count += 1
    return count


def exo_scale(camera: ExoCamera) -> float:
    """Stored / source pixel size of an exo camera."""
    return 1.0 / (1 << halvings(camera.width, camera.height))



def smpl_faces(model_root: Path) -> UInt32[ndarray, "f 3"]:
    """SMPL's triangle list (shape-independent), from the official neutral model under ``model_root/smpl``."""
    model: SmplxLayerTorch = SmplxLayerTorch(betas=np.zeros(10, dtype=np.float32), model_type="smpl", gender="neutral", model_root_dir=model_root)
    return np.asarray(model.faces, dtype=np.uint32)


def scene_view(exo_slots: int, ego_rigs: tuple[int, ...], *, card: bool = False) -> rrb.Spatial3DView:
    """The scene, z up, for takes of up to ``exo_slots`` GoPros and the glasses on ``ego_rigs`` (rules on absent entities do nothing).

    Lens-projected keypoints and boxes stay in the panes; a card drops every video, a layout hides them (a hidden video is
    not decoded). The default grid is off: the world origin is aria01's first head pose, ~1.5 m above the floor.
    """
    slots: list[tuple[int, int]] = [(rig, 0) for rig in range(1, exo_slots + 1)] + [(rig, cam) for rig in ego_rigs for cam in range(len(ARIA_MODES))]
    return rrb.Spatial3DView(
        name="Scene",
        origin="/world",
        contents=[
            "+ /world/**",
            *(f"- {schema.coco133_uv_projected_path(rig, cam)}/**" for rig, cam in slots),
            *(f"- {schema.pinhole_path(rig, cam)}/boxes/**" for rig, cam in slots),
            *(blueprints.video_exclusions(slots) if card else []),
        ],
        overrides=None if card else {schema.video_path(rig, cam): rrb.EntityBehavior(visible=False) for rig, cam in slots},
        eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital),  # no pose: the viewer fits the eye to each take's camera ring
        line_grid=rrb.LineGrid3D(visible=False),
    )

@dataclass(frozen=True, slots=True)
class Clock:
    """The frames a conversion writes and their times."""

    frames: Int64[ndarray, "t"]
    """Release frame numbers (from 1)."""
    times_ns: Int64[ndarray, "t"]
    """``video_time``, nanoseconds from the first frame."""

    @classmethod
    def of(cls, sequence: Sequence, frame_limit: int | None) -> Clock:
        count: int = min(len(sequence.frames), frame_limit or len(sequence.frames))
        return cls(sequence.frames[:count], sequence.video_times_ns[:count])

    @property
    def indexes(self) -> list[rr.TimeColumn]:
        """Both recording clocks at every frame."""
        return [time_column(self.times_ns), frame_index_column(self.frames)]


def exo_camera(camera: ExoCamera) -> Fisheye62Parameters:
    """The GoPro at its stored size: COLMAP's OPENCV_FISHEYE is Kannala-Brandt with four radial terms."""
    scale: float = exo_scale(camera)
    fx, fy, cx, cy, k1, k2, k3, k4 = (float(value) for value in camera.lens)
    return Fisheye62Parameters(
        name=camera.name,
        extrinsics=Extrinsics(world_R_cam=camera.world_T_cam[:3, :3], world_t_cam=camera.world_T_cam[:3, 3]),
        intrinsics=Intrinsics.from_focal_principal_point(
            camera_conventions="RDF",
            fl_x=fx * scale,
            fl_y=fy * scale,
            cx=cx * scale,
            cy=cy * scale,
            width=round(camera.width * scale),
            height=round(camera.height * scale),
        ),
        distortion=KannalaBrandtDistortion(k1=k1, k2=k2, k3=k3, k4=k4),
    )


def aria_camera(glasses: AriaGlasses, index: int) -> Fisheye62Parameters:
    """One Aria camera upright at the first frame, posed on its rig (thin-prism terms dropped from the logged lens)."""
    upright: Fisheye624Parameters = glasses.cameras[index].calibration(0)
    rig_T_upright: Float64[ndarray, "4 4"] = glasses.rig_T_cam(ARIA_MODES[index]) @ upright.rig_T_cam.matrix()
    return upright.to_fisheye62(rig_T_upright)


def encode_job(files: SequenceFiles, directory: str, frames: Int64[ndarray, "t"], clip: Path, scaling: tuple[int, int]) -> Callable[[], None]:
    """Encode ``directory/NNNNN.jpg`` of ``frames`` into ``clip`` (AV1, 20 fps), decoding at ``scaling``."""

    def encode() -> None:
        # Imported here: PyTurboJPEG is a dataforge-env dependency, and the registry imports this module everywhere.
        from dataforge.jpeg import decode_jpeg_frames, jpeg_frame_source

        images: Iterator[bytes] = (files.read(f"{directory}/{frame:05d}.jpg") for frame in frames)
        first: bytes = next(images)
        source: FrameSource = jpeg_frame_source(first, scaling_factor=scaling)
        count: int = encode_frames_to_mp4(
            decode_jpeg_frames(chain([first], images), source=source, scaling_factor=scaling),
            clip,
            source=source,
            fps=FPS,
            gop=AV1_GOP,
            cq=AV1_CQ,
            filter_threads=1,
        )
        if count != len(frames):
            raise ValueError(f"{directory}: encoded {count} frames, expected {len(frames)}")

    return encode


def write_base(
    recording: rr.RecordingStream,
    identity: SequenceIdentity,
    files: SequenceFiles,
    sequence: Sequence,
    clock: Clock,
    timer: SequenceTimer,
    *,
    source_revision: str,
    episode: dict[str, str],
) -> None:
    """Every camera (lens at the stored size, pose in the anchor world), every video re-encoded to AV1, the glasses' poses."""
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)  # the anchor Aria's SLAM world, z up
    rr.log("/", annotation_context(), static=True, recording=recording)
    frustum_m: float = layout_frustum_m(np.array([camera.world_T_cam[:2, 3] for camera in sequence.exo]))
    resolutions: list[str] = []
    jobs: list[tuple[Path, Callable[[], None]]] = []
    slots: list[tuple[int, int]] = []
    with work_dir("egohumans-") as work:
        for camera in sequence.exo:
            rig: int = exo_rig(camera)
            stored: Fisheye62Parameters = exo_camera(camera)
            log_rig_node(recording, rig, reference=None, num_cameras=1, name=camera.name, kind="exo")
            log_camera_node(recording, rig, 0, stored, name=camera.name, kind="rgb", image_plane_distance=frustum_m, camera_model="OPENCV_FISHEYE")
            resolutions.append(
                log_camera_source(
                    recording,
                    rig,
                    0,
                    name=camera.name,
                    source_width=camera.width,
                    source_height=camera.height,
                    stored_width=None if stored.intrinsics.width == camera.width else stored.intrinsics.width,
                    stored_height=None if stored.intrinsics.height == camera.height else stored.intrinsics.height,
                    video_codec="av1",
                    cq=AV1_CQ,
                    gop=AV1_GOP,
                )
            )
            clip: Path = work / f"{camera.name}.mp4"
            jobs.append((clip, encode_job(files, f"exo/{camera.name}/images", clock.frames, clip, (1, 1 << halvings(camera.width, camera.height)))))
            slots.append((rig, 0))
        for glasses in sequence.glasses:
            rig = ego_rig(glasses)
            log_rig_node(recording, rig, reference=None, num_cameras=len(ARIA_MODES), name=glasses.name, kind="ego")
            for index, mode in enumerate(ARIA_MODES):
                stored_aria: Fisheye62Parameters = aria_camera(glasses, index)
                log_camera_node(
                    recording,
                    rig,
                    index,
                    stored_aria,
                    name=f"{glasses.name}-{mode}",
                    kind="rgb" if mode == "rgb" else "grayscale",
                    image_plane_distance=frustum_m / 2,
                    camera_model="FISHEYE624 (thin-prism omitted)",
                    image_rotation_cw_deg=90,
                )
                resolutions.append(
                    log_camera_source(
                        recording,
                        rig,
                        index,
                        name=f"{glasses.name}-{mode}",
                        source_width=glasses.cameras[index].width,
                        source_height=glasses.cameras[index].height,
                        stream_id=ARIA_STREAM_IDS[mode],
                        video_codec="av1",
                        cq=AV1_CQ,
                        gop=AV1_GOP,
                    )
                )
                clip = work / f"{glasses.name}-{mode}.mp4"
                jobs.append((clip, encode_job(files, f"ego/{glasses.name}/images/{mode}", clock.frames, clip, (1, 1))))
                slots.append((rig, index))
            log_dense_pose_track(
                recording, schema.rig_path(rig), times_ns=clock.times_ns, frame_indices=clock.frames, transforms=glasses.world_T_rig[: len(clock.frames)]
            )
        with parallel_clips(jobs, timer, workers=ENCODE_WORKERS) as encoded:
            for (rig, cam), clip in zip(slots, encoded, strict=True):
                with timer.stage("write:video"):
                    log_video_stream(recording, clip, schema.video_path(rig, cam), times_ns=clock.times_ns, frame_indices=clock.frames)
    writing.send_capture_properties(
        recording,
        identity,
        num_frames=len(clock.frames),
        num_cameras=len(slots),
        source_num_frames=len(sequence.frames),
        fps=FPS,
        source_revision=source_revision,
        clock_source=f"release frame number / {FPS} fps (the release ships frames, no capture timestamps)",
        world_frame=WORLD_FRAMES[sequence.world_source],
        source_resolution=pa.array(resolutions),
        video_source=f"AV1 NVENC re-encode (CQ {AV1_CQ}) of the release's JPEG frames; videos above {MAX_PIXELS / 1e6:.1f} MP stored at half size",
    )
    values: dict[str, Any] = {
        **episode,
        "num_people": len(sequence.people),
        "people": pa.array([person.name for person in sequence.people]),
        "num_exo_cameras": len(sequence.exo),
        "glasses": pa.array([glasses.name for glasses in sequence.glasses], type=pa.string()),
    }
    recording.send_property("episode", rr.AnyValues(**values))


def coco133(person: PersonTrack, rows: int) -> tuple[Float32[ndarray, "t 133 3"], Float32[ndarray, "t 133"]]:
    """The shipped COCO-17 body keypoints in COCO-133's first 17 slots, the rest NaN with confidence 0."""
    positions: Float32[ndarray, "t 133 3"] = np.full((rows, 133, 3), np.nan, dtype=np.float32)
    confidence: Float32[ndarray, "t 133"] = np.zeros((rows, 133), dtype=np.float32)
    positions[:, :COCO17] = person.coco17[:rows]
    confidence[:, :COCO17] = person.coco17_confidence[:rows]
    return positions, confidence


def write_body_pose(recording: rr.RecordingStream, sequence: Sequence, clock: Clock) -> None:
    """Every person's SMPL parameters, SMPL-45 joints and fit loss at full rate, and their COCO-133 keypoints."""
    rows: int = len(clock.frames)
    for person in sequence.people:
        index: int = person_index(person)
        rr.log(
            schema.person_path(index),
            rr.AnyValues(name=person.name, source=BODY_SOURCE, model="SMPL neutral", keypoints_source=sequence.keypoints_source),
            static=True,
            recording=recording,
        )
        path: str = schema.person_body_path(index, "smpl")
        rr.log(
            path,
            rr.AnyValues(
                pose_layout="72 axis-angle: global_orient (3), body_pose (69: 23 joints)",
                translation_pivot="root joint (smplx transl)",
                joints_layout="SMPL-45: 24 SMPL joints, then smplx's vertex_joint_selector extras",
            ),
            static=True,
            recording=recording,
        )

        def column(values: Float32[ndarray, "t k"]) -> pa.FixedSizeListArray:
            return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1)), values.shape[1])

        rr.send_columns(
            path,
            indexes=clock.indexes,
            columns=rr.AnyValues.columns(
                global_orient=column(person.global_orient[:rows]),
                body_pose=column(person.body_pose[:rows]),
                betas=column(person.betas[:rows]),
                transl=column(person.transl[:rows]),
                joints=column(person.joints[:rows].reshape(rows, -1)),
                fit_loss=pa.array(person.fit_loss[:rows]),
            ),
            recording=recording,
        )
        positions, confidence = coco133(person, rows)
        hands.log_keypoints3d(
            recording,
            times_ns=clock.times_ns,
            frame_indices=clock.frames,
            positions=positions,
            confidence=confidence,
            path=schema.person_coco133_xyz_path(index),
        )


def write_body_mesh(recording: rr.RecordingStream, sequence: Sequence, clock: Clock, faces: UInt32[ndarray, "f 3"]) -> None:
    """Every person's shipped SMPL vertices at 10 Hz (``MESH_STRIDE``), an empty mesh where the person has no fit."""
    keep: Int64[ndarray, "k"] = np.flatnonzero((clock.frames - clock.frames[0]) % MESH_STRIDE == 0)
    for person in sequence.people:
        path: str = schema.person_body_path(person_index(person), "mesh")
        meshes.log_mesh_static(recording, path, albedo_factor=hands.BODY_ALBEDO)
        for start in range(0, len(keep), MESH_BATCH_ROWS):
            rows: Int64[ndarray, "b"] = keep[start : start + MESH_BATCH_ROWS]
            present: Bool[ndarray, "b"] = person.present[rows]
            meshes.log_mesh_batch(
                recording,
                path,
                times_ns=clock.times_ns[rows],
                frame_indices=clock.frames[rows],
                vertices=person.vertices[rows[present]],
                trusted=present.tolist(),
                topology=faces,
            )


def projected(sequence: Sequence, person: PersonTrack, clock: Clock) -> Iterator[tuple[tuple[int, int], Float32[ndarray, "t 133 2"]]]:
    """``person``'s COCO-133 keypoints through every camera's full lens, in stored pixels, one camera at a time."""
    rows: int = len(clock.frames)
    positions: Float64[ndarray, "t 133 3"] = coco133(person, rows)[0].astype(np.float64)
    for camera in sequence.exo:
        pixels: Float64[ndarray, "m 2"] = camera.project(positions.reshape(-1, 3)) * exo_scale(camera)
        yield (exo_rig(camera), 0), pixels.reshape(rows, 133, 2).astype(np.float32)
    for glasses in sequence.glasses:
        for index, aria in enumerate(glasses.cameras):
            # Loop locals stay unannotated: beartype would rebuild a jaxtyping checker per frame in the dev env.
            frames = np.stack([aria.project(row, positions[row]) for row in range(rows)])
            yield (ego_rig(glasses), index), frames.astype(np.float32)


def write_projections(recording: rr.RecordingStream, sequence: Sequence, clock: Clock) -> None:
    """Each person's COCO-133 keypoints through every lens, and the release's person boxes as shipped."""
    recording.send_property(
        "projections",
        rr.AnyValues(
            derived_from="coco133_xyz",
            camera_model="OPENCV_FISHEYE (exo), FISHEYE624 per frame (Aria, turned upright)",
            calibration_source="COLMAP cameras/images.txt and the per-frame Aria calib files",
        ),
    )
    for person in sequence.people:
        for (rig, cam), pixels in projected(sequence, person, clock):
            hands.log_keypoints2d(
                recording,
                schema.person_projection_path(rig, cam, person_index(person)),
                times_ns=clock.times_ns,
                frame_indices=clock.frames,
                positions=pixels,
                confidence=None,  # 1 where the lens put the point, 0 where it is NaN
            )
    slots: dict[str, tuple[int, int, float]] = {camera.name: (exo_rig(camera), 0, exo_scale(camera)) for camera in sequence.exo}
    slots |= {f"{glasses.name}/{mode}": (ego_rig(glasses), index, 1.0) for glasses in sequence.glasses for index, mode in enumerate(ARIA_MODES)}
    rows: int = len(clock.frames)
    for camera_boxes in sequence.boxes:
        rig, cam, scale = slots[camera_boxes.camera]
        for person in sequence.people:
            # As shipped, scaled to the stored image: the Aria boxes keep the release's corner order (x2 < x1 where it turned
            # each corner on its own) and its wrapped values; docs list it as a label fault.
            xyxy: Float32[ndarray, "t 4"] = camera_boxes.boxes[person.name][:rows] * np.float32(scale)
            present: Bool[ndarray, "t"] = np.isfinite(xyxy).all(axis=1)
            centres: Float32[ndarray, "k 2"] = (xyxy[present, :2] + xyxy[present, 2:]) / 2
            half_sizes: Float32[ndarray, "k 2"] = (xyxy[present, 2:] - xyxy[present, :2]) / 2
            colour: tuple[int, int, int] = PERSON_COLOURS[person_index(person) % len(PERSON_COLOURS)]
            path: str = schema.boxes_path(rig, cam, f"person_{person_index(person):02d}")
            rr.log(path, rr.Boxes2D.from_fields(colors=[colour], labels=[person.name], show_labels=False), static=True, recording=recording)
            lengths: list[int] = present.astype(int).tolist()
            rr.send_columns(
                path,
                indexes=clock.indexes,
                columns=rr.Boxes2D.columns(centers=centres, half_sizes=half_sizes).partition(lengths),
                recording=recording,
            )
