"""Retired unweighted solver, retained only for frozen upstream reference tests."""

import cv2
import numpy as np
from dataforge.datasets.selfcap_raw import Intrinsic
from jaxtyping import Bool, Float64
from numpy import ndarray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from selfcap_curate.factory import intrinsic_matrix, transform
from selfcap_curate.oak_mount import hand_observations
from selfcap_curate.records import OakFit, SessionEpisode


def legacy_solve_oak(
    session: str,
    episodes: list[SessionEpisode],
    intrinsic: Intrinsic,
    device: str,
    fixed_rotation: Float64[ndarray, "3 3"] | None = None,
    *,
    weighted: bool = False,
    gate_image_speed: bool = True,
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
            if weighted
            else camera[:, :2] / np.maximum(camera[:, 2:3], 1e-6) * np.diag(matrix)[:2] + matrix[:2, 2] - undistorted
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
                p[3:] if fixed_rv is not None else p,
                loss="cauchy",
                f_scale=6.0,
                x_scale="jac",
            )
            p = np.r_[fixed_rv, sol.x] if fixed_rv is not None else sol.x
            error = np.linalg.norm(residual(p), axis=1)
            if iteration == 0:
                keep = error <= 30.0
                if keep.sum() < 100:
                    return OakFit(session, False, device)
        medians = {g: float(np.median(error[group == g])) for g in np.unique(group)}
        return OakFit(
            session,
            True,
            device,
            float(np.median(error[keep])),
            np.linalg.inv(transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:])),
            float(np.mean([value <= 30.0 for value in medians.values()])),
            "hands" if fixed_rotation is None else "gyro mount rotation @ device rgb_R_imu",
            observation_source="egohand",
            head_frame="rdf",
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
        head_frame="rdf",
    )
