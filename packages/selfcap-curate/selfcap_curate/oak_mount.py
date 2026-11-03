"""Robust OAK gyro rotation and session RGB mount from hand correspondences."""

from dataclasses import dataclass

import cv2
import numpy as np
from dataforge.datasets.selfcap_raw import FactoryCalibration, Intrinsic, oak_stereo
from jaxtyping import Bool, Float64
from numpy import ndarray
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from selfcap_curate.device_tables import RotationTable
from selfcap_curate.factory import intrinsic_matrix, transform
from selfcap_curate.quest_fit import hand_neighbors
from selfcap_curate.records import (
    HAND_BASE,
    HAND_LATENCY_MS,
    MAPPED,
    OAK_CAMERAS,
    GyroFit,
    OakDetections,
    OakFit,
    SessionEpisode,
    gyro_head_offsets,
    oak_quest_offset_ms,
)


def gyro_mount(session: str, episodes: list[SessionEpisode]) -> GyroFit:
    """30 ms smoothing, 0.15–5 rad/s, Kabsch, discard worst 10%, refit."""
    targets, sources, rotations = [], [], []
    for episode in episodes:
        clocks = gyro_head_offsets(episode.sync)
        if clocks is None or min(len(episode.hands.head_times_s), len(episode.gyro_times)) < 100:
            continue
        qp, oi = clocks.quest_pose_ms, clocks.oak_imu_ms
        ht = episode.hands.head_times_s + qp / 1000.0
        keep = np.r_[True, np.diff(ht) > 0.0]
        ht, quaternions = ht[keep], episode.hands.head_quaternions[keep]
        it = episode.gyro_times + oi / 1000.0
        rot = Rotation.from_quat(quaternions)
        dt = np.diff(ht)
        omega = gaussian_filter1d((rot[:-1].inv() * rot[1:]).as_rotvec() / dt[:, None], 0.03 / np.median(dt), axis=0)
        gyro = gaussian_filter1d(episode.gyro, 0.03 / np.median(np.diff(it)), axis=0)
        qt, speed = (ht[:-1] + ht[1:]) / 2.0, np.linalg.norm(omega, axis=1)
        mask = (qt > max(it[0], qt[0]) + 0.2) & (qt < min(it[-1], qt[-1]) - 0.2) & (speed > 0.15) & (speed < 5.0)
        if mask.sum() < 50:
            continue
        target = omega[mask]
        source = np.column_stack([np.interp(qt[mask], it, gyro[:, k]) for k in range(3)])
        rotations.append(Rotation.align_vectors(target, source)[0])
        targets.append(target)
        sources.append(source)
    if not targets:
        return GyroFit(session, False)
    target, source = np.concatenate(targets), np.concatenate(sources)
    rotation = Rotation.align_vectors(target, source)[0]
    error = np.linalg.norm(rotation.apply(source) - target, axis=1)
    good = error <= np.percentile(error, 90)
    rotation = Rotation.align_vectors(target[good], source[good])[0]
    return GyroFit(session, True, rotation.as_matrix(), max(float(np.degrees((rotation.inv() * r).magnitude())) for r in rotations))


Observation = tuple[Float64[ndarray, "3"], Float64[ndarray, "2"], str, float]
"""Quest-rig joint, image pixel, ``episode/frame/side`` group and detection score."""


def hand_observations(
    episode_index: int, episode: SessionEpisode, detections: OakDetections, *, require_video: bool = False, max_image_speed: float | None = None,
) -> list[Observation]:
    """Mapped Quest joints in the Quest rig frame at each detection instant, paired with their detected pixels."""
    qp = oak_quest_offset_ms(episode.sync)
    if qp is None:
        return []
    head = episode.hands
    ht, hp = head.head_times_s, head.head_positions
    vt = detections.video_time
    if require_video and len(vt) < 2:
        return []
    image_speed: Float64[ndarray, "n 133"] = (
        np.linalg.norm(np.gradient(detections.xy, vt, axis=0), axis=-1) if max_image_speed is not None else np.zeros_like(detections.scores)
    )
    tq = vt - qp / 1000.0
    i = np.clip(np.searchsorted(ht, tq), 1, len(ht) - 1)
    a = np.clip((tq - ht[i - 1]) / np.maximum(ht[i] - ht[i - 1], 1e-9), 0.0, 1.0)
    position = (1.0 - a[:, None]) * hp[i - 1] + a[:, None] * hp[i]
    rotation = Slerp(ht, Rotation.from_quat(head.head_quaternions))(np.clip(tq, ht[0], ht[-1]))
    okh = (tq >= ht[0]) & (tq <= ht[-1]) & (ht[i] - ht[i - 1] < 0.06)
    observations: list[Observation] = []
    for side, base in HAND_BASE.items():
        hand = episode.quest[side]
        j, alpha, valid = hand_neighbors(hand, tq + HAND_LATENCY_MS / 1000.0)
        valid = okh & valid
        joints = (1.0 - alpha[:, None, None]) * hand.joints[j - 1] + alpha[:, None, None] * hand.joints[j]
        for frame in np.flatnonzero(valid):
            local = rotation[frame].inv().apply(joints[frame] - position[frame])
            for k in MAPPED:
                if (
                    detections.scores[frame, base + k] >= 0.3
                    and np.isfinite(detections.xy[frame, base + k]).all()
                    and (max_image_speed is None or image_speed[frame, base + k] <= max_image_speed)
                ):
                    observations.append((local[k], detections.xy[frame, base + k], f"{episode_index}/{frame}/{side}/{k}", detections.scores[frame, base + k]))
    return observations


def solve_oak(
    session: str, episodes: list[SessionEpisode], intrinsic: Intrinsic, device: str, fixed_rotation: Float64[ndarray, "3 3"] | None = None,
    *, weighted: bool = False, gate_image_speed: bool = True,
) -> OakFit:
    """Fit mapped fingers; optionally use dense EgoHand scores, a 300 px/s gate and one full-lens trimming pass."""
    observations = [
        (point, pixel, group.rsplit("/", 1)[0], score)
        for episode_index, episode in enumerate(episodes)
        for point, pixel, group, score in hand_observations(
            episode_index, episode, episode.oak, require_video=weighted, max_image_speed=300.0 if weighted and gate_image_speed else None
        )
    ]
    if len(observations) < 100:
        return OakFit(session, False, device)
    world, image, group, confidence = (np.asarray(values) for values in zip(*observations, strict=True))
    matrix = intrinsic_matrix(intrinsic)
    undistorted = cv2.undistortPoints(
        image.reshape(-1, 1, 2), matrix, np.asarray(intrinsic.distortion.coefficients, dtype=np.float64), P=matrix
    ).reshape(-1, 2)
    ok, rv, tv, _ = cv2.solvePnPRansac(world, undistorted, matrix, None, reprojectionError=25.0, iterationsCount=2000, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return OakFit(session, False, device)

    def residual(p: Float64[ndarray, "6"]) -> Float64[ndarray, "n 2"]:
        camera = world @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:]
        r = (
            cv2.projectPoints(world, p[:3], p[3:], matrix, np.asarray(intrinsic.distortion.coefficients, dtype=np.float64))[0].reshape(-1, 2) - image
            if weighted else camera[:, :2] / np.maximum(camera[:, 2:3], 1e-6) * np.diag(matrix)[:2] + matrix[:2, 2] - undistorted
        )
        r[camera[:, 2] <= 0.0] = 1e3
        return r

    p = np.r_[rv.ravel(), tv.ravel()]
    error, keep, medians = np.zeros(len(world)), np.ones(len(world), dtype=bool), {}

    if weighted:
        fixed_rv: Float64[ndarray, "3"] | None = None if fixed_rotation is None else Rotation.from_matrix(fixed_rotation.T).as_rotvec()
        weights: Float64[ndarray, "n 1"] = np.sqrt(confidence)[:, None]
        for iteration in range(2):
            sol = least_squares(
                lambda values, keep=keep: (residual(np.r_[fixed_rv, values] if fixed_rv is not None else values) * weights)[keep].ravel(),
                p[3:] if fixed_rv is not None else p, loss="cauchy", f_scale=6.0, x_scale="jac",
            )
            p = np.r_[fixed_rv, sol.x] if fixed_rv is not None else sol.x
            error = np.linalg.norm(residual(p), axis=1)
            if iteration == 0:
                keep = error <= 30.0
                if keep.sum() < 100:
                    return OakFit(session, False, device)
        medians = {g: float(np.median(error[group == g])) for g in np.unique(group)}
        return OakFit(
            session, True, device, float(np.median(error[keep])),
            np.linalg.inv(transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])),
            float(np.mean([value <= 30.0 for value in medians.values()])),
            "hands" if fixed_rotation is None else "gyro mount rotation @ device rgb_R_imu",
            observation_source="egohand",
        )

    def refine(
        initial: Float64[ndarray, "6"], fixed_rv: Float64[ndarray, "3"] | None
    ) -> tuple[Float64[ndarray, "6"], Float64[ndarray, "n"], Bool[ndarray, "n"], dict[str, float]]:
        p = initial
        keep = np.ones(len(world), dtype=bool)
        for _ in range(3):
            sol = least_squares(
                lambda q, keep=keep: residual(np.r_[fixed_rv, q] if fixed_rv is not None else q)[keep].ravel(),
                p[3:] if fixed_rv is not None else p,
                loss="cauchy",
                f_scale=6.0,
                x_scale="jac",
            )
            p = np.r_[fixed_rv, sol.x] if fixed_rv is not None else sol.x
            error = np.linalg.norm(residual(p), axis=1)
            medians = {g: float(np.median(error[group == g])) for g in np.unique(group)}
            keep = np.array([medians[g] <= 30.0 for g in group])
            if keep.sum() < 100:
                keep[:] = True
                break
        return p, error, keep, medians

    p, error, keep, medians = refine(p, None)
    if fixed_rotation is not None:
        p, error, keep, medians = refine(p, Rotation.from_matrix(fixed_rotation.T).as_rotvec())
    mount = np.linalg.inv(transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:]))
    return OakFit(
        session,
        True,
        device,
        float(np.median(error[keep])),
        mount,
        float(np.mean([v <= 30.0 for v in medians.values()])),
        "hands" if fixed_rotation is None else "gyro mount rotation @ device rgb_R_imu",
    )


def fixed_rotation(gyro: GyroFit, device: str, rotations: RotationTable) -> Float64[ndarray, "3 3"] | None:
    """Use the pooled RGB/IMU mount only with <=1.5 degree episode spread."""
    return (
        gyro.R_head_from_imu @ rotations.devices[device].rgb_R_imu.T
        if gyro.ok
        and gyro.max_episode_dev_deg is not None
        and gyro.max_episode_dev_deg <= 1.5
        and device in rotations.devices
        and gyro.R_head_from_imu is not None
        else None
    )


def fixed_rotation_reason(gyro: GyroFit, device: str, rotations: RotationTable) -> str | None:
    """Why ``fixed_rotation`` returned None, or None when the gyro mount is eligible."""
    if not gyro.ok:
        return "gyro fit not ok"
    if gyro.max_episode_dev_deg is None:
        return "gyro episode spread unavailable"
    if gyro.max_episode_dev_deg > 1.5:
        return "episode spread exceeds 1.5 degrees"
    if device not in rotations.devices:
        return "device missing from rgb_R_imu table"
    if gyro.R_head_from_imu is None:
        return "gyro rotation unavailable"
    return None


@dataclass(frozen=True, slots=True)
class OakLens:
    """One OAK camera: factory pixels, all distortion terms, and its pose relative to the RGB camera."""

    intrinsic: Float64[ndarray, "3 3"]
    distortion: Float64[ndarray, "d"]
    cam_T_rgb: Float64[ndarray, "4 4"]
    size: tuple[int, int]
    """Decoded width and height."""


def oak_lenses(calibration: FactoryCalibration) -> dict[str, OakLens] | None:
    """The three OAK cameras from the factory file, or None when its stereo extrinsics are missing."""
    stereo = oak_stereo(calibration)
    if not stereo:
        return None
    cam_T_rgb = {"rgb": np.eye(4), "left": np.linalg.inv(stereo["rgb_T_left"]), "right": np.linalg.inv(stereo["rgb_T_right"])}
    layouts = {"rgb": "center", "left": "left", "right": "right"}
    lenses = {}
    for camera in OAK_CAMERAS:
        intrinsic = next(c for c in calibration.intrinsics if c.positional_layout == layouts[camera])
        lenses[camera] = OakLens(intrinsic_matrix(intrinsic), np.asarray(intrinsic.distortion.coefficients, dtype=np.float64), cam_T_rgb[camera],
                                 (intrinsic.capture_resolution.width, intrinsic.capture_resolution.height))
    return lenses


def _kabsch(source: Float64[ndarray, "n 3"], target: Float64[ndarray, "n 3"]) -> Float64[ndarray, "4 4"]:
    """Rigid transform with ``target ≈ R @ source + t``, refitted on the best 80% four times."""
    keep = np.ones(len(source), dtype=bool)
    for _ in range(4):
        cs, ct = source[keep].mean(0), target[keep].mean(0)
        u, _, vt = np.linalg.svd((source[keep] - cs).T @ (target[keep] - ct))
        rotation = vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        error = np.linalg.norm(source @ rotation.T + ct - rotation @ cs - target, axis=1)
        keep = error <= np.percentile(error, 80)
    return transform(rotation, ct - rotation @ cs)


def solve_oak_stereo(
    session: str, episodes: list[SessionEpisode], views: list[dict[str, OakDetections]], lenses: dict[str, OakLens], device: str,
) -> OakFit:
    """Mount from EgoHand on all three OAK cameras: stereo-triangulated hands start a joint fit through every lens.

    ``views[e][camera]`` holds episode ``e``'s detections on the same sampled instants in every camera (row ``f`` is one
    instant), so the left/right rows pair up for triangulation. One RGB camera alone cannot separate a hand's depth
    from the mount's tilt; the stereo baseline can.
    """
    observations = {
        camera: [obs for index, (episode, view) in enumerate(zip(episodes, views, strict=True)) for obs in hand_observations(index, episode, view[camera])]
        for camera in OAK_CAMERAS
    }
    if sum(map(len, observations.values())) < 100:
        return OakFit(session, False, device)
    # Start: triangulate each joint seen by both stereo cameras (in the RGB frame) and align the Quest joints to them.
    right = {group: pixel for _, pixel, group, _ in observations["right"]}
    pairs = [(point, pixel, right[group]) for point, pixel, group, _ in observations["left"] if group in right]
    if len(pairs) < 50:
        return OakFit(session, False, device)
    quest, left_px, right_px = (np.asarray(values) for values in zip(*pairs, strict=True))
    normalised = {
        name: cv2.undistortPoints(px.reshape(-1, 1, 2), lenses[name].intrinsic, lenses[name].distortion).reshape(-1, 2).T
        for name, px in (("left", left_px), ("right", right_px))
    }
    points = cv2.triangulatePoints(lenses["left"].cam_T_rgb[:3], lenses["right"].cam_T_rgb[:3], normalised["left"], normalised["right"])
    points = (points[:3] / points[3]).T
    front = (points[:, 2] > 0.05) & (points[:, 2] < 2.0)
    if front.sum() < 50:
        return OakFit(session, False, device)
    start = _kabsch(quest[front], points[front])  # rgb_T_questrig
    data = {
        camera: tuple(np.asarray(values) for values in zip(*rows, strict=True))
        for camera, rows in observations.items() if rows
    }

    def offsets(q: Float64[ndarray, "6"], camera: str) -> Float64[ndarray, "n 2"]:
        """Projected minus detected pixels in one camera for ``q`` = rgb_T_questrig as (rotvec, t); 1e3 behind it."""
        point, pixel, _, _ = data[camera]
        cam_T_q = lenses[camera].cam_T_rgb @ transform(Rotation.from_rotvec(q[:3]).as_matrix(), q[3:])
        lens = lenses[camera]
        projected = cv2.projectPoints(point, cv2.Rodrigues(cam_T_q[:3, :3])[0], cam_T_q[:3, 3], lens.intrinsic, lens.distortion)[0].reshape(-1, 2)
        offset = projected - pixel
        offset[(point @ cam_T_q[:3, :3].T + cam_T_q[:3, 3])[:, 2] <= 0.0] = 1e3
        return offset

    p = np.r_[Rotation.from_matrix(start[:3, :3]).as_rotvec(), start[:3, 3]]
    keep = {camera: np.ones(len(values[0]), dtype=bool) for camera, values in data.items()}
    for iteration in range(2):  # fit, drop single points beyond 30 px, refit once
        p = least_squares(
            lambda q, keep=keep: np.concatenate([(offsets(q, c) * np.sqrt(data[c][3])[:, None])[keep[c]].ravel() for c in data]),
            p, loss="cauchy", f_scale=6.0, x_scale="jac",
        ).x
        errors = {camera: np.linalg.norm(offsets(p, camera), axis=1) for camera in data}
        if iteration == 0:
            keep = {camera: error <= 30.0 for camera, error in errors.items()}
            if sum(int(k.sum()) for k in keep.values()) < 100:
                return OakFit(session, False, device)
    # A hand group is one camera's view of one hand at one instant.
    hand_medians = [
        float(np.median(errors[camera][hands == hand]))
        for camera in data
        for hands in [np.array([group.rsplit("/", 1)[0] for group in data[camera][2]])]
        for hand in np.unique(hands)
    ]
    kept = np.concatenate([errors[camera][keep[camera]] for camera in data])
    return OakFit(
        session, True, device, float(np.median(kept)),
        np.linalg.inv(transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])),
        float(np.mean([value <= 30.0 for value in hand_medians])),
        "stereo+rgb joint",
        observation_source="egohand",
        camera_median_px={camera: float(np.median(errors[camera][keep[camera]])) for camera in data},
    )
