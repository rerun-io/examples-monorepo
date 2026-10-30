"""Exo camera models, fixed-intrinsics session fits and focal cost profiles."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import cv2
import numpy as np
from jaxtyping import Float64
from numpy import ndarray
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from serde import serde

from selfcap_curate.device_tables import GRID, FocalProfile
from selfcap_curate.quest_fit import Correspondences, pairs
from selfcap_curate.records import SessionEpisode, finite


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class Camera:
    """World-to-camera pinhole parameters; distortion is held fixed."""

    parameters: Float64[ndarray, "11"]  # rvec(3), tvec(3), f, cx, cy, k1, k2.

    def project(self, world: Float64[ndarray, "n 3"]) -> Float64[ndarray, "n 3"]:
        """Project Float64[n,3] world points to pixel u,v and camera depth."""
        p = self.parameters
        camera = world @ Rotation.from_rotvec(p[:3]).as_matrix().T + p[3:6]
        depth = camera[:, 2]
        normalized = camera[:, :2] / np.where(np.abs(depth) < 1e-6, 1e-6, depth)[:, None]
        radius = (normalized**2).sum(1)
        uv = p[6] * (1.0 + p[9] * radius + p[10] * radius**2)[:, None] * normalized + p[7:9]
        return np.c_[uv, depth]

    def residuals(self, data: Correspondences) -> Float64[ndarray, "n"]:
        """Pixel distances, with infinite cost behind the camera."""
        projected = self.project(data.world)
        return np.where(projected[:, 2] > 0.0, np.linalg.norm(projected[:, :2] - data.image, axis=1), np.inf)

    @property
    def world_T_cam(self) -> Float64[ndarray, "4 4"]:
        """Camera-to-Quest-world transform, with camera centre in metres."""
        transform: Float64[ndarray, "4 4"] = np.eye(4)
        transform[:3, :3] = Rotation.from_rotvec(self.parameters[:3]).as_matrix().T
        transform[:3, 3] = -transform[:3, :3] @ self.parameters[3:6]
        return transform


def solve_camera(data: Correspondences, focal: float, *, seed: Camera | None = None) -> Camera | None:
    """Cauchy LS (6 px scale), EPnP RANSAC seed, fixed centre and distortion."""
    if seed is None:
        matrix = np.array([[focal, 0.0, 960.0], [0.0, focal, 540.0], [0.0, 0.0, 1.0]])
        ok, rv, tv, _ = cv2.solvePnPRansac(data.world, data.image, matrix, None, reprojectionError=40.0, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        seed = Camera(np.r_[rv.ravel(), tv.ravel(), focal, 960.0, 540.0, 0.0, 0.0])
    fixed: Float64[ndarray, "5"] = np.array([focal, 960.0, 540.0, 0.0, 0.0])

    def residual(parameters: Float64[ndarray, "n"]) -> Float64[ndarray, "m"]:
        p = np.r_[parameters, fixed]
        projected = Camera(p).project(data.world)
        errors = (projected[:, :2] - data.image) * np.sqrt(data.weights)[:, None]
        errors[projected[:, 2] <= 0.0] = 1e3
        return errors.ravel()

    sol = least_squares(residual, seed.parameters[:6], loss="cauchy", f_scale=6.0, x_scale="jac", max_nfev=200)
    return Camera(np.r_[sol.x, fixed])


@dataclass(frozen=True, slots=True)
class BodyObservation:
    """One phone's slow-joint observations at a shared frame-row index."""

    xy: Float64[ndarray, "13 2"]  # COCO joints 0–6 and 11–16, in pixels.
    scores: Float64[ndarray, "13"]  # Confidence; BA accepts >=0.5.


Hands: TypeAlias = dict[str, list[tuple[str, Correspondences]]]


Body: TypeAlias = list[dict[str, BodyObservation]]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class EpisodeCalibration:
    """Episode gate and optional pose override, retaining every fallback scope."""

    n: int  # Number of 3D–2D hand pairs.
    median_px: float | None  # Residual for the selected pose; null if not finite.
    ok: bool  # Whether the episode passed the scratch gate.
    scope: Literal["session", "session-few-points", "episode", "flagged"]  # Shared pose, sparse-data fallback, re-solve, or rejected re-solve.
    world_T_cam: Float64[ndarray, "4 4"] | None = None  # Only episode re-solves override the session camera-to-world transform.


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PhoneCalibration:
    """Session camera and episode checks, or an explicit failed pass-1 fit."""

    ok: bool  # At least 60 hand points and an initialized camera.
    camera: Camera | None = None  # Pinhole parameters; absent when fitting failed.
    median_px: float | None = None  # Session median reprojection residual.
    p90_px: float | None = None  # Session 90th percentile reprojection residual.
    n: int = 0  # Total hand point count.
    episodes: dict[str, EpisodeCalibration] = field(default_factory=dict)  # Per-episode gates, including ok=false and explicit fallback scopes.
    why: str | None = None  # Failure reason when pass 1 has no usable camera.


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SessionCalibration:
    """Machine calibration record in Quest world metres with OpenCV RDF cameras."""

    session: str  # Explicit date__sid8 session id.
    phones: dict[str, PhoneCalibration]  # Full device ids, never ambiguous name prefixes.
    body_median_px: float | None = None  # Unweighted session BA body diagnostic for calibration provenance.


@dataclass(frozen=True, slots=True)
class BundleResult:
    """Optimized cameras and the profile objective used for pooling."""

    cameras: dict[str, Camera]  # Camera-to-Quest-world model parameters.
    cost: float  # Robust least-squares objective.
    body_median_px: float | None = None  # Unweighted body residual median for layer provenance.
    converged: bool = True  # Whether the optimizer terminated before its evaluation limit.


def triangulate(cameras: dict[str, Camera], views: dict[str, BodyObservation]) -> Float64[ndarray, "13 3"]:
    """DLT body points observed by >=2 cameras with score >=0.5, else NaN."""
    points: Float64[ndarray, "13 3"] = np.full((13, 3), np.nan)
    for joint in range(13):
        rows = []
        for phone, view in views.items():
            if phone not in cameras or view.scores[joint] < 0.5 or not np.isfinite(view.xy[joint]).all():
                continue
            p = cameras[phone].parameters
            matrix = np.array([[p[6], 0.0, 960.0], [0.0, p[6], 540.0], [0.0, 0.0, 1.0]]) @ np.c_[Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6]]
            rows.extend([view.xy[joint, 0] * matrix[2] - matrix[0], view.xy[joint, 1] * matrix[2] - matrix[1]])
        if len(rows) >= 4:
            homogeneous = np.linalg.svd(np.asarray(rows))[2][-1]
            points[joint] = homogeneous[:3] / homogeneous[3]
    return points


def bundle(stacked: dict[str, Correspondences], body: Body, cameras: dict[str, Camera], *, max_nfev: int = 80,
           anchors: Callable[[dict[str, Camera]], Float64[ndarray, "a"]] | None = None) -> BundleResult:
    """Optimize poses/body points with fixed intrinsics and optional optical anchors."""
    phones: list[str] = sorted(cameras)
    pi: dict[str, int] = {p: i for i, p in enumerate(phones)}
    world = {p: stacked[p].world for p in phones}
    image = {p: stacked[p].image for p in phones}
    points, observations = [], []
    for views in body:
        for joint, point in enumerate(triangulate(cameras, views)):
            if not np.isfinite(point).all():
                continue
            index = len(points)
            points.append(point)
            for phone, view in views.items():
                if phone in pi and view.scores[joint] >= 0.5 and np.isfinite(view.xy[joint]).all():
                    observations.append((index, pi[phone], *view.xy[joint], np.sqrt(view.scores[joint])))
    obs: Float64[ndarray, "n 5"] = np.asarray(observations, dtype=np.float64).reshape(-1, 5)
    nc, nh = len(phones), sum(len(world[p]) for p in phones)
    logf = np.array([np.log(cameras[p].parameters[6]) for p in phones])
    initial = np.concatenate([np.r_[cameras[p].parameters[:6], logf[pi[p]]] for p in phones] + [np.asarray(points).ravel()])
    anchor_count = 0 if anchors is None else len(anchors(cameras))

    def unpack(values: Float64[ndarray, "n"]) -> dict[str, Camera]:
        c = values[: 7 * nc].reshape(nc, 7)
        return {p: Camera(np.r_[c[i, :6], cameras[p].parameters[6:]]) for i, p in enumerate(phones)}

    def residual(values: Float64[ndarray, "n"]) -> Float64[ndarray, "m"]:
        cams = unpack(values)
        xyz = values[7 * nc :].reshape(-1, 3)
        errors = []
        for p in phones:
            uv = cams[p].project(world[p])
            r = uv[:, :2] - image[p]
            r[uv[:, 2] <= 0.0] = 1e3
            errors.append(r.ravel())
        for i, p in enumerate(phones):
            mask = obs[:, 1] == i
            uv = cams[p].project(xyz[obs[mask, 0].astype(int)])
            r = (uv[:, :2] - obs[mask, 2:4]) * obs[mask, 4:5]
            r[uv[:, 2] <= 0.0] = 1e3
            errors.append(r.ravel())
        if anchors is not None:
            errors.append(anchors(cams))
        return np.concatenate(errors)

    sparsity = lil_matrix((2 * nh + 2 * len(obs) + anchor_count, len(initial)), dtype=np.int8)
    row = 0
    for p in phones:
        n = len(world[p])
        sparsity[row : row + 2 * n, 7 * pi[p] : 7 * pi[p] + 7] = 1
        row += 2 * n
    for i in range(nc):
        for point in obs[obs[:, 1] == i, 0].astype(int):
            sparsity[row : row + 2, 7 * i : 7 * i + 7] = 1
            sparsity[row : row + 2, 7 * nc + 3 * point : 7 * nc + 3 * point + 3] = 1
            row += 2
    if anchor_count:
        sparsity[2 * nh + 2 * len(obs):, :7 * nc] = 1
    for i in range(nc):
        sparsity[:, 7 * i + 6] = 0
    result = least_squares(residual, initial, jac_sparsity=sparsity, loss="cauchy", f_scale=6.0, x_scale="jac", max_nfev=max_nfev, method="trf")
    body_error = np.linalg.norm(residual(result.x)[2 * nh : 2 * nh + 2 * len(obs)].reshape(-1, 2), axis=1) / np.maximum(obs[:, 4], 1e-6)
    return BundleResult(unpack(result.x), float(result.cost), float(np.median(body_error)) if len(body_error) else None, bool(result.success))


BODY: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16)


def gather(episodes: list[SessionEpisode]) -> tuple[Hands, Body]:
    """Use v2 hand pairs (unweighted) and body joints matched by frame row, not time."""
    hands: Hands = {}
    body: Body = []
    for episode in episodes:
        rows: dict[int, dict[str, BodyObservation]] = {}
        for phone, detections in episode.detections.items():
            fit = episode.sync.exo_fit.get(phone)
            if fit is None or fit.tau_phone_to_pose_ms is None:
                continue
            data = pairs(detections, episode.quest, fit.tau_phone_to_pose_ms / 1000.0)
            hands.setdefault(phone, []).append((episode.sync.episode, Correspondences(data.world, data.image, np.ones(len(data.world)))))
            for row in range(len(detections.times_s)):
                rows.setdefault(row, {})[phone] = BodyObservation(detections.xy[row, BODY, :], detections.scores[row, BODY])
        body.extend(views for views in rows.values() if len(views) >= 2)
    return hands, body


def calibrate(session: str, episodes: list[SessionEpisode], focals: dict[str, float]) -> SessionCalibration:
    """Hold device focal, centre and distortion.

    Phones need 60 pairs to initialize. Episode outliers (>1.6× session median
    AND >median+4 px) are re-solved; accept <=max(8 px, session median).
    Fewer than 30 pairs retain the shared pose only if nonempty and <=12 px.
    """
    hands, body = gather(episodes)
    output: dict[str, PhoneCalibration] = {}
    cameras: dict[str, Camera] = {}
    stacked: dict[str, Correspondences] = {}
    default: float = float(np.median(list(focals.values()))) if focals else 700.0
    for phone, items in hands.items():
        data = Correspondences.concat([d for _, d in items])
        stacked[phone] = data
        camera = None if len(data.world) < 60 else solve_camera(data, focals.get(phone, default))
        if camera is None:
            output[phone] = PhoneCalibration(False, n=len(data.world), why="fewer than 60 hand points" if len(data.world) < 60 else "no initial pose")
        else:
            cameras[phone] = camera
    body_median: float | None = None
    if len(cameras) >= 2:
        optimized = bundle(stacked, body, cameras)
        cameras, body_median = optimized.cameras, optimized.body_median_px
    for phone, camera in cameras.items():
        residual = camera.residuals(stacked[phone])
        median = float(np.median(residual))
        checked: dict[str, EpisodeCalibration] = {}
        for episode, data in hands[phone]:
            error = float(np.median(camera.residuals(data))) if len(data.world) else None
            record = EpisodeCalibration(len(data.world), finite(error), True, "session")
            if len(data.world) < 30:
                record = EpisodeCalibration(len(data.world), finite(error), error is not None and error <= 12.0, "session-few-points")
            elif error is not None and error > max(1.6 * median, median + 4.0):
                solved = solve_camera(data, float(camera.parameters[6]), seed=camera)
                refined = float(np.median(solved.residuals(data))) if solved is not None else None
                accepted = refined is not None and refined <= max(8.0, median)
                record = EpisodeCalibration(
                    len(data.world),
                    finite(refined if accepted else error),
                    accepted,
                    "episode" if accepted else "flagged",
                    solved.world_T_cam if accepted and solved is not None else None,
                )
            checked[episode] = record
        finite_residual = residual[np.isfinite(residual)]
        output[phone] = PhoneCalibration(
            True, camera, finite(median), float(np.percentile(finite_residual, 90)) if len(finite_residual) else None, len(residual), checked
        )
    return SessionCalibration(session, output, body_median)


def profile_session(session: str, episodes: list[SessionEpisode], f_ref: dict[str, float] | None = None) -> FocalProfile:
    """Nine fixed-focal BA costs per phone, with the other phones at f_ref or 700 px."""
    hands, body = gather(episodes)
    phones = sorted(p for p in hands if sum(len(d.world) for _, d in hands[p]) >= 60)
    if len(phones) < 2:
        return FocalProfile(session, error="fewer than two phones")
    stacked = {p: Correspondences.concat([d for _, d in hands[p]]) for p in phones}
    base = {p: (f_ref or {}).get(p, 700.0) for p in phones}
    curves = {}
    for phone in phones:
        costs = []
        for focal in GRID:
            fs = {**base, phone: focal}
            cameras = {p: solve_camera(stacked[p], fs[p]) for p in phones}
            if any(c is None for c in cameras.values()):
                raise ValueError(f"{session}: no initial pose for focal profile")
            costs.append(bundle(stacked, body, {p: c for p, c in cameras.items() if c is not None}, max_nfev=40).cost)
        curves[phone] = tuple(costs)
    return FocalProfile(session, curves)
