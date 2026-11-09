"""GSI Egocentric Six-View, derived layers: ``pose``, ``hand_mesh`` and ``projections`` from one ``Episode``, and the trail overrides.

The hands live in GSI's rig frame; the world-frame layers move them by each frame's SLAM pose (NaN on a dropout), and
the projections go through each view's shipped pinhole, which needs no SLAM. See docs/gsi-six-view.md.
"""

from collections.abc import Iterator

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Bool, Float32, Float64
from numpy import ndarray
from rerun.encodings import EntityPathLike
from simplecv.sensors.camera.base_camera import pixels_in_image

from dataforge import blueprints, hands, meshes, paths, schema
from dataforge.datasets.gsi_six_view_source import FLU_R_RDF, RIG, RUN_SOURCE, SIDES, SOURCE_SIZE, VIEWS, Episode
from dataforge.logging_toolkit import (
    log_dense_pose_track,
    time_column,
)

MESH_BATCH: int = 256
"""Frames per hand-mesh chunk."""
POSE_LAYER: str = "pose"
"""Hand keypoints, wrists, confidence and MANO records; the name DAS-Ego's pose layer uses."""


def to_world(world_T_rig: Float64[ndarray, "f 4 4"], points: Float32[ndarray, "f k 3"]) -> Float32[ndarray, "f k 3"]:
    """Rig-frame points of every frame in the world frame."""
    moved: Float64[ndarray, "f k 3"] = np.einsum("fij,fkj->fki", world_T_rig[:, :3, :3], points.astype(np.float64)) + world_T_rig[:, None, :3, 3]
    return moved.astype(np.float32)


def coco133(joints: Float32[ndarray, "f 2 21 d"]) -> Float32[ndarray, "f 133 d"]:
    """GSI's joints in their COCO-133 hand slots (91-111 left, 112-132 right); nothing derived."""
    return np.stack([hands.coco133_from_coco_hands(frame) for frame in joints])


def project(rig_T_cam: Float64[ndarray, "4 4"], intrinsics: Float64[ndarray, "3 3"], points: Float32[ndarray, "f k 3"]) -> Float32[ndarray, "f k 2"]:
    """Rig-frame points through one view's shipped pinhole; NaN behind the camera or off the image."""
    cam_T_rig: Float64[ndarray, "4 4"] = np.linalg.inv(rig_T_cam)
    in_cam: Float64[ndarray, "f k 3"] = points.astype(np.float64) @ cam_T_rig[:3, :3].T + cam_T_rig[:3, 3]
    optical: Float64[ndarray, "f k 3"] = in_cam @ FLU_R_RDF
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels: Float64[ndarray, "f k 2"] = (optical @ intrinsics.T)[..., :2] / optical[..., 2:]
    flat: Float64[ndarray, "n 2"] = pixels.reshape(-1, 2)
    inside: Bool[ndarray, "n"] = pixels_in_image(flat, *SOURCE_SIZE) & (optical.reshape(-1, 3)[:, 2] > 0.0)
    return np.where(inside[:, None], flat, np.nan).reshape(pixels.shape).astype(np.float32)


def write_pose_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``pose`` layer: COCO-133 hand keypoints and both wrists in the world, each hand's confidence and GSI's MANO record."""
    world_keypoints: Float32[ndarray, "f 2 21 3"] = to_world(episode.world_T_rig, episode.rig_keypoints.reshape(len(episode.frames), -1, 3)).reshape(-1, 2, 21, 3)
    hands.log_keypoints3d(recording, times_ns=episode.times_ns, frame_indices=episode.frames, positions=coco133(world_keypoints), confidence=None)
    indexes: list[rr.TimeColumn] = [time_column(episode.times_ns), rr.TimeColumn(schema.FRAME_INDEX, sequence=episode.frames)]
    for side_index, side in enumerate(SIDES):
        hands.log_hand_confidence(recording, side, times_ns=episode.times_ns, frame_indices=episode.frames, confidence=episode.usable[:, side_index].astype(np.float64))
        rig_R_wrist: Float64[ndarray, "f 3 3"] = np.asarray(episode.mano[f"global_orient_{side}"].flatten(), dtype=np.float64).reshape(-1, 3, 3)
        world_T_wrist: Float64[ndarray, "f 4 4"] = np.tile(np.eye(4), (len(episode.frames), 1, 1))
        world_T_wrist[:, :3, :3] = episode.world_T_rig[:, :3, :3] @ rig_R_wrist
        world_T_wrist[:, :3, 3] = world_keypoints[:, side_index, 0]
        log_dense_pose_track(recording, schema.hand_wrist_path(side), times_ns=episode.times_ns, frame_indices=episode.frames, transforms=world_T_wrist)
        rr.send_columns(
            schema.hand_mano_path(side),
            indexes=indexes,
            columns=rr.AnyValues.columns(**{name: episode.mano[f"{name}_{side}"] for name in ("hand_pose", "betas", "global_orient", "transl", "is_detected", "is_interpolated", "is_usable")}),
            recording=recording,
        )
    recording.send_property(
        POSE_LAYER,
        rr.AnyValues(
            keypoints="GSI's 21 joints per hand in COCO-133 slots 91-132 (GSI ships them in COCO-WholeBody hand order); no derived slots",
            keypoint_confidence="1.0 for every finite joint: GSI ships no keypoint score",
            hand_confidence="GSI's is_usable flag",
            wrist="translation = joint 0 in the world, rotation = world_R_rig · MANO global_orient",
            mano_frame="as shipped: global_orient and transl in GSI's rig frame (left_front, FLU), not the world",
            coverage_left=float(episode.usable[:, 0].mean()),
            coverage_right=float(episode.usable[:, 1].mean()),
        ),
    )


def write_hand_mesh_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``hand_mesh`` layer: GSI's skinned MANO meshes, moved to the world frame, on the frames GSI marks usable."""
    for side_index, side in enumerate(SIDES):
        path: str = schema.hand_mesh_path(side)
        meshes.log_mesh_static(recording, path, albedo_factor=hands.HAND_ALBEDO[side])
        for start in range(0, len(episode.frames), MESH_BATCH):
            batch: slice = slice(start, start + MESH_BATCH)
            trusted: Bool[ndarray, "n"] = episode.usable[batch, side_index] & episode.posed[batch]
            world_vertices: Float32[ndarray, "n 778 3"] = to_world(episode.world_T_rig[batch], episode.rig_vertices[batch, side_index])
            meshes.log_mesh_batch(
                recording, path, times_ns=episode.times_ns[batch], frame_indices=episode.frames[batch],
                vertices=world_vertices[trusted], trusted=trusted.tolist(), topology=episode.faces[side_index],
            )  # fmt: skip
    recording.send_property(
        paths.HAND_MESH_LAYER, rr.AnyValues(model="MANO vertices as GSI skinned them, moved from its rig frame to the world by the SLAM pose; none on a SLAM dropout")
    )


def write_projections_layer(recording: rr.RecordingStream, episode: Episode) -> None:
    """The ``projections`` layer: the COCO-133 hand keypoints through each view's shipped pinhole (GSI's videos are undistorted)."""
    rig_coco: Float32[ndarray, "f 133 3"] = coco133(episode.rig_keypoints)
    confidence: Float32[ndarray, "f 133"] = np.isfinite(rig_coco).all(axis=-1).astype(np.float32)

    def pixels_by_camera() -> Iterator[tuple[str, Float32[ndarray, "f 133 2"]]]:
        for cam in range(len(VIEWS)):
            yield schema.coco133_uv_projected_path(RIG, cam), project(episode.rig_T_cams[cam], episode.intrinsics[cam], rig_coco)

    hands.log_projections_property(recording, camera_model="pinhole", calibration_source="GSI's Pinhole per view (= /meta/calibration K)")
    hands.log_projections(recording, pixels_by_camera(), times_ns=episode.times_ns, frame_indices=episode.frames, confidence=confidence)


def trail_overrides() -> dict[EntityPathLike, rrb.VisibleTimeRanges | rr.LineStrips3D]:
    """The SLAM trail through the shared cursor window over the whole path, drawn thin and dim as in ``blueprints.rig_blueprint``'s follow view."""
    return {
        schema.trail_path(RUN_SOURCE): blueprints.trail_window_range(),
        schema.trajectory_path(RUN_SOURCE): rr.LineStrips3D.from_fields(
            colors=blueprints.DIM_TRAJECTORY_COLOR, radii=rr.Radius.ui_points(blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS)
        ),
    }


