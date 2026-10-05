"""Test helpers: a synthetic four-camera rig with ground truth, and a fake ``SegmentTrack`` made from ground truth.

The fake track is what the Rerun layers were built and screenshotted with before the tracker existed:
the predicted pose is the ground-truth pose plus small noise, boxes come from the projected ground
truth, presence is 0.9, and the hands are dropped for a stretch in the middle and re-acquired by DetNet.
"""

from dataclasses import replace

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int8, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from handtrack.geometry.camera import CameraRig
from handtrack.rerun_layers import GroundTruth, camera_pixels, enclosing_squares, generic_numpy_model, gt_landmarks, inside_images, skinned_landmarks
from handtrack.results import BoxSource, SegmentTrack, TrackMetadata

FAKE_PRESENCE: float = 0.9


def synthetic_rig() -> CameraRig:
    """Four 636x480 equidistant fisheyes looking along +z, toed out left/right and tilted down/up like UmeTrack's headset."""
    cam_from_rig: Float32[ndarray, "4 4 4"] = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    for index, (yaw, pitch) in enumerate([(-0.5, 0.0), (-0.15, 0.35), (0.15, 0.35), (0.5, 0.0)]):
        cam_from_rig[index, :3, :3] = Rotation.from_euler("yx", [yaw, pitch]).as_matrix()  # R_x(pitch) @ R_y(yaw)
        cam_from_rig[index, :3, 3] = [0.03 * (index - 1.5), 0.0, 0.0]
    return CameraRig(
        names=tuple(f"/world/rig_00/cam_{index:02}" for index in range(4)),
        image_size=torch.tensor([[636.0, 480.0]] * 4),
        cam_from_rig=torch.from_numpy(cam_from_rig),
        focal=torch.tensor([[240.0, 240.0]] * 4),
        principal=torch.tensor([[318.0, 240.0]] * 4),
        fisheye62=torch.zeros(4, 8),
    )


def synthetic_truth(num_frames: int, *, segment: str = "umetrack__synthetic__test__user_00__recording_00") -> GroundTruth:
    """Both hands in front of a still headset at 0.3 m, waving; the right hand leaves the views for the last fifth."""
    t: Float32[ndarray, "f"] = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    translation: Float32[ndarray, "f 2 3"] = np.zeros((num_frames, 2, 3), dtype=np.float32)
    translation[:, 0] = np.stack([-0.08 + 0.03 * np.sin(6 * t), 0.08 + 0.0 * t, 0.3 + 0.0 * t], axis=-1)
    translation[:, 1] = np.stack([0.08 - 0.03 * np.sin(6 * t), 0.08 + 0.0 * t, 0.3 + 0.0 * t], axis=-1)
    rotation: Float32[ndarray, "f 2 3 3"] = np.tile(Rotation.from_euler("x", -1.2).as_matrix().astype(np.float32), (num_frames, 2, 1, 1))
    joint_angles: Float32[ndarray, "f 2 22"] = np.tile(np.linspace(0.0, 0.4, 22, dtype=np.float32), (num_frames, 2, 1))
    present: Bool[ndarray, "f 2"] = np.ones((num_frames, 2), dtype=bool)
    present[int(num_frames * 0.8) :, 1] = False
    return GroundTruth(
        segment=segment,
        rig=synthetic_rig(),
        video_time_ns=np.arange(num_frames, dtype=np.int64) * 33_333_333,
        frame_index=np.arange(num_frames, dtype=np.int64),
        world_from_rig=np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1)),
        rotation=rotation,
        translation=translation,
        joint_angles=joint_angles,
        present=present,
        model=generic_numpy_model(),
    )


def fake_track(truth: GroundTruth, *, seed: int = 0, drop: tuple[float, float] = (0.4, 0.5)) -> SegmentTrack:
    """A ``SegmentTrack`` from ground truth: GT pose + noise (3 mm, 0.03 rad), GT hand boxes, presence 0.9.

    Hands are untracked while absent and during the ``drop`` fraction of the frames; the first tracked frame
    of every run is a DetNet acquisition on the round-robin camera, the rest are tracked boxes.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    num_frames: int = len(truth.video_time_ns)
    dropped: Bool[ndarray, "f"] = np.zeros(num_frames, dtype=bool)
    dropped[int(num_frames * drop[0]) : int(num_frames * drop[1])] = True
    tracked: Bool[ndarray, "f 2"] = truth.present & ~dropped[:, None]
    translation: Float32[ndarray, "f 2 3"] = truth.translation + rng.normal(0.0, 0.003, truth.translation.shape).astype(np.float32)
    joint_angles: Float32[ndarray, "f 2 22"] = truth.joint_angles + rng.normal(0.0, 0.03, truth.joint_angles.shape).astype(np.float32)
    landmarks: Float32[ndarray, "f 2 21 3"] = skinned_landmarks(truth.model, truth.rotation, translation, joint_angles, tracked)
    gt_pixels: Float32[ndarray, "f 4 2 21 2"] = camera_pixels(truth.rig, truth.world_from_rig, gt_landmarks(truth))
    pred_pixels: Float32[ndarray, "f 4 2 21 2"] = camera_pixels(truth.rig, truth.world_from_rig, landmarks)
    inside: Bool[ndarray, "f 4 2 21"] = inside_images(truth.rig, gt_pixels)
    counts: Int64[ndarray, "f 4 2"] = inside.sum(-1)
    box: Float32[ndarray, "f 4 2 4"] = enclosing_squares(gt_pixels, np.isfinite(gt_pixels).all(-1))
    seen: Bool[ndarray, "f 4 2"] = tracked[:, None, :] & (counts > 0)
    box[~seen] = np.nan
    acquired: Bool[ndarray, "f 2"] = tracked & ~np.concatenate([np.zeros((1, 2), dtype=bool), tracked[:-1]])
    detnet_camera: Int8[ndarray, "f"] = np.where((~tracked).any(axis=1) | acquired.any(axis=1), np.arange(num_frames) % 4, -1).astype(np.int8)
    box_source: Int8[ndarray, "f 4 2"] = np.where(seen, np.int8(BoxSource.TRACKED), np.int8(BoxSource.NONE)).astype(np.int8)
    for frame, side in zip(*np.nonzero(acquired), strict=True):
        camera: int = int(detnet_camera[frame])
        box_source[frame, :, side] = BoxSource.NONE
        box[frame, :, side] = np.nan
        if counts[frame, camera, side] > 0:
            box_source[frame, camera, side] = BoxSource.DETNET
            box[frame, camera, side] = enclosing_squares(gt_pixels[frame, camera, side], np.isfinite(gt_pixels[frame, camera, side]).all(-1))
    # KeyNet runs on the (at most) two views with the most keypoints inside.
    order: Int64[ndarray, "f 4 2"] = np.argsort(-np.where(box_source > 0, counts, -1), axis=1, kind="stable")
    ran: Bool[ndarray, "f 4 2"] = np.zeros((num_frames, 4, 2), dtype=bool)
    np.put_along_axis(ran, order[:, :2], True, axis=1)
    ran &= box_source > 0
    keypoints_2d: Float32[ndarray, "f 4 2 21 2"] = np.where(ran[..., None, None], pred_pixels, np.nan).astype(np.float32)
    presence: Float32[ndarray, "f 4 2"] = np.where(ran, np.float32(FAKE_PRESENCE), np.nan).astype(np.float32)
    detnet_presence: Float32[ndarray, "f 2"] = np.where(detnet_camera[:, None] >= 0, np.where(acquired, FAKE_PRESENCE, 0.1), np.nan).astype(np.float32)
    # results.py: the pose is NaN on untracked frames.
    rotation: Float32[ndarray, "f 2 3 3"] = np.where(tracked[..., None, None], truth.rotation, np.float32(np.nan)).astype(np.float32)
    translation = np.where(tracked[..., None], translation, np.float32(np.nan)).astype(np.float32)
    joint_angles = np.where(tracked[..., None], joint_angles, np.float32(np.nan)).astype(np.float32)
    return SegmentTrack(
        meta=TrackMetadata(segment=truth.segment, detnet_sha256="fake", keynet_sha256="fake", hand_mode="known", hand_scale=1.0, timings_s={}),
        video_time_ns=truth.video_time_ns,
        frame_index=truth.frame_index,
        tracked=tracked,
        rotation=rotation,
        translation=translation,
        joint_angles=joint_angles,
        landmarks=landmarks,
        box=box,
        box_source=box_source,
        keypoints_2d=keypoints_2d,
        presence=presence,
        detnet_camera=detnet_camera,
        detnet_presence=detnet_presence,
        fit_energy=np.where(tracked, rng.uniform(1.0, 5.0, tracked.shape), np.nan).astype(np.float32),
    )


def fake_detnet(truth: GroundTruth, *, seed: int = 1) -> SegmentTrack:
    """DetNet alone on every frame and camera: the GT hand box wherever a hand has a keypoint inside, presence 0.9 (0.1 elsewhere)."""
    base: SegmentTrack = fake_track(truth, seed=seed, drop=(0.0, 0.0))
    gt_pixels: Float32[ndarray, "f 4 2 21 2"] = camera_pixels(truth.rig, truth.world_from_rig, gt_landmarks(truth))
    inside: Bool[ndarray, "f 4 2 21"] = inside_images(truth.rig, gt_pixels)
    seen: Bool[ndarray, "f 4 2"] = inside.any(-1)
    box: Float32[ndarray, "f 4 2 4"] = enclosing_squares(gt_pixels, np.isfinite(gt_pixels).all(-1))
    box[~seen] = np.nan
    return replace(
        base,
        meta=replace(base.meta, kind="detnet_alone"),
        box=box,
        box_source=np.where(seen, np.int8(BoxSource.DETNET), np.int8(BoxSource.NONE)).astype(np.int8),
        presence=np.where(seen, np.float32(FAKE_PRESENCE), np.float32(0.1)).astype(np.float32),
        keypoints_2d=np.full_like(base.keypoints_2d, np.nan),
        detnet_camera=np.full_like(base.detnet_camera, -1),
    )
