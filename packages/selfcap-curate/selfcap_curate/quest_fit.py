"""CPU geometric/time fit, ported numerically from frozen questlib.py.

The cost, joint subset, confidence threshold and robust weighting retain v2's
semantics. No guessed confidence is assigned to the catalog's raw Quest hands.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray

from selfcap_curate.catalog_io import Segment, read_catalog_hands, read_gyro
from selfcap_curate.records import HAND_BASE, MAPPED, CatalogHands, HandTrack, PhoneDetections, SessionEpisode, Side, TimeSync


def load_quest_hands(source: CatalogHands) -> dict[Side, HandTrack]:
    """Reject jumps, freezes and hands outside a 60-degree median-centred cone.

    Neighbour suppression around >4 m/s jumps guards against tracking resets.
    Repeated snapshots guard against lost tracking. The median-centred cone
    guards against differing head-axis conventions between capture batches.
    The base carries 21 mapped joints and a left-eye rig origin; the scratch
    loader had all 26 joints and a two-eye average, which cannot be recovered
    from unlogged raw data and is not silently invented here.
    """
    if len(source.head_times_s) == 0:
        raise ValueError("Quest hands require the base's capture-time head pose")
    result: dict[Side, HandTrack] = {}
    directions: dict[Side, Float64[ndarray, "n 3"]] = {}
    for side, base in HAND_BASE.items():
        joints = np.nan_to_num(source.positions[:, base : base + 21], nan=0.0)
        wrist = joints[:, 0]
        valid = np.abs(joints).sum((1, 2)) > 0.0
        speed = np.zeros(len(source.times_s), dtype=np.float64)
        speed[1:] = np.linalg.norm(np.diff(wrist, axis=0), axis=1) / np.maximum(np.diff(source.times_s), 1e-3)
        jump = speed > 4.0
        valid &= ~(jump | np.roll(jump, 1) | np.roll(jump, -1))
        frozen = np.r_[False, np.abs(np.diff(joints, axis=0)).sum((1, 2)) == 0.0]
        valid &= ~(frozen & np.roll(frozen, -1))
        hi = np.clip(np.searchsorted(source.head_times_s, source.times_s), 0, len(source.head_times_s) - 1)
        q = source.head_quaternions[hi] * np.array([-1.0, -1.0, -1.0, 1.0])
        v = wrist - source.head_positions[hi]
        cross = 2.0 * np.cross(q[:, :3], v)
        rotated = v + q[:, 3:4] * cross + np.cross(q[:, :3], cross)
        directions[side] = rotated / np.maximum(np.linalg.norm(rotated, axis=1, keepdims=True), 1e-6)
        result[side] = HandTrack(source.times_s, joints, valid, speed)
    pooled = np.concatenate([directions[side][result[side].valid] for side in result])
    if len(pooled):
        median = np.median(pooled, axis=0)
        median /= max(np.linalg.norm(median), 1e-6)
        for side in result:
            result[side].valid[:] &= directions[side] @ median > np.cos(np.radians(60.0))
    return result


@dataclass(frozen=True, slots=True)
class Correspondences:
    """Weighted 3D-to-2D pairs, in the same joint-major order as questlib."""

    world: Float64[ndarray, "n 3"]
    """Quest world points in metres."""
    image: Float64[ndarray, "n 2"]
    """Phone image points in pixels."""
    weights: Float64[ndarray, "n"]
    """Motion and robust-fit weights."""

    @staticmethod
    def concat(parts: list["Correspondences"]) -> "Correspondences":
        """Stack parts in order, keeping each part's joint-major pair order."""
        return Correspondences(
            np.concatenate([p.world for p in parts]), np.concatenate([p.image for p in parts]), np.concatenate([p.weights for p in parts])
        )


def hand_neighbors(hand: HandTrack, times: Float64[ndarray, "n"]) -> tuple[Int64[ndarray, "n"], Float64[ndarray, "n"], Bool[ndarray, "n"]]:
    """Share the 1e-6 denominator and strict 40 ms tracked-neighbor gate only."""
    i = np.clip(np.searchsorted(hand.times_s, times), 1, len(hand.times_s) - 1)
    alpha = np.clip((times - hand.times_s[i - 1]) / np.maximum(hand.times_s[i] - hand.times_s[i - 1], 1e-6), 0.0, 1.0)
    valid = (
        (times >= hand.times_s[0]) & (times <= hand.times_s[-1]) & hand.valid[i] & hand.valid[i - 1] & (hand.times_s[i] - hand.times_s[i - 1] < 0.04)
    )
    return i, alpha, valid


def pairs(detections: PhoneDetections, quest: dict[Side, HandTrack], tau_s: float) -> Correspondences:
    """Pair presented phone PTS + tau with raw Quest capture stamps, as v2 did.

    No MP4 start is subtracted: original torchcodec pts_seconds and patched
    video capture_time both include it. A 40 ms interpolation-gap limit guards
    tracking gaps; score >=0.3 guards weak detections. Clipped wrist-speed weights
    prefer informative motion without discarding slow frames entirely.
    """
    worlds = []
    images = []
    weights = []
    for side, base in HAND_BASE.items():
        q = quest[side]
        if len(q.times_s) < 2:
            continue
        tq = detections.times_s + tau_s
        i, a, valid = hand_neighbors(q, tq)
        speed = (1.0 - a) * q.speed[i - 1] + a * q.speed[i]
        motion_weight = np.clip(speed / 0.3, 0.1, 1.0)
        for joint in MAPPED:
            world = (1.0 - a)[:, None] * q.joints[i - 1, joint] + a[:, None] * q.joints[i, joint]
            image = detections.xy[:, base + joint]
            keep = valid & (detections.scores[:, base + joint] >= 0.3) & np.isfinite(image).all(1)
            worlds.append(world[keep])
            images.append(image[keep])
            weights.append(motion_weight[keep])
    if not worlds:
        return Correspondences(np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0))
    return Correspondences(np.concatenate(worlds), np.concatenate(images), np.concatenate(weights))


def dlt(data: Correspondences) -> Float64[ndarray, "3 4"]:
    """Weighted Hartley-normalised DLT; normalization guards numerical scale imbalance."""
    world, image, weights = data.world, data.image, data.weights
    cx, c_world = image.mean(0), world.mean(0)
    sx = np.sqrt(2.0) / np.mean(np.linalg.norm(image - cx, axis=1))
    sw = np.sqrt(3.0) / np.mean(np.linalg.norm(world - c_world, axis=1))
    xn, wn = (image - cx) * sx, (world - c_world) * sw
    homogeneous = np.hstack([wn, np.ones((len(wn), 1))])
    zero = np.zeros_like(homogeneous)
    design = np.vstack([np.hstack([homogeneous, zero, -xn[:, :1] * homogeneous]), np.hstack([zero, homogeneous, -xn[:, 1:2] * homogeneous])])
    weight = np.concatenate([weights, weights])
    _, vectors = np.linalg.eigh((design * weight[:, None]).T @ design)
    camera = vectors[:, 0].reshape(3, 4)
    t_image = np.array([[sx, 0.0, -sx * cx[0]], [0.0, sx, -sx * cx[1]], [0.0, 0.0, 1.0]])
    t_world = np.diag([sw, sw, sw, 1.0])
    t_world[:3, 3] = -sw * c_world
    return np.linalg.inv(t_image) @ camera @ t_world


def residuals(camera: Float64[ndarray, "3 4"], data: Correspondences) -> Float64[ndarray, "n"]:
    """Reprojection distance; opposite-depth points cost 1000 px to reject flips."""
    projected = np.hstack([data.world, np.ones((len(data.world), 1))]) @ camera.T
    depth = projected[:, 2:3]
    distance = np.linalg.norm(projected[:, :2] / np.where(np.abs(depth) < 1e-9, 1e-9, depth) - data.image, axis=1)
    return np.where(depth[:, 0] * np.sign(np.median(depth)) > 0.0, distance, 1e3)


def robust_fit(data: Correspondences, iters: int = 3, delta: float = 8.0) -> Float64[ndarray, "3 4"] | None:
    """Require 12 pairs, then trim large residual influence with v2's Huber weights."""
    if len(data.world) < 12:
        return None
    camera = dlt(data)
    for _ in range(iters):
        distance = residuals(camera, data)
        camera = dlt(Correspondences(data.world, data.image, data.weights * np.minimum(1.0, delta / np.maximum(distance, 1e-6))))
    return camera


def cost(camera: Float64[ndarray, "3 4"] | None, data: Correspondences) -> float:
    """Cap residuals at 30 px so isolated detections cannot choose the clock."""
    if camera is None or len(data.world) < 12:
        return float("inf")
    return float((data.weights * np.minimum(residuals(camera, data), 30.0)).sum() / data.weights.sum())


@dataclass(frozen=True, slots=True)
class FitMetrics:
    """One episode's temporal minimum and confidence diagnostics."""

    tau_ms: float | None
    """Phone-to-Quest hand-clock lag."""
    cmin_px: float | None
    """Minimum camera cost."""
    basin_halfwidth_ms: float | None
    """Width of the near-minimum region."""
    points: int = 0
    """Number of paired points."""
    edge: bool = False
    """The fine or coarse search hit an edge."""


def curve_metrics(taus: Float64[ndarray, "n"], costs: Float64[ndarray, "n"]) -> FitMetrics:
    """Parabolic sub-grid refinement and 5% basin guard against ambiguous minima."""
    if not np.isfinite(costs).any():
        return FitMetrics(None, None, None)
    values = np.where(np.isfinite(costs), costs, np.nanmax(np.where(np.isfinite(costs), costs, np.nan)) * 2.0)
    j = int(np.argmin(values))
    tau = float(taus[j])
    if 0 < j < len(values) - 1:
        denominator = values[j - 1] - 2.0 * values[j] + values[j + 1]
        if denominator > 0.0:
            tau += float(0.5 * (values[j - 1] - values[j + 1]) / denominator * (taus[1] - taus[0]))
    inside = taus[values <= 1.05 * values[j]]
    return FitMetrics(round(tau * 1000.0, 1), round(float(values[j]), 2), round(float(inside.max() - inside.min()) * 500.0, 1))


def read_session_episode(segment: Segment, sync: TimeSync) -> SessionEpisode:
    """Read one named segment's raw hands/pose/gyro, never the shared segment table."""
    layout = segment.layout
    times, gyro = read_gyro(segment)
    hands = read_catalog_hands(segment)
    return SessionEpisode(sync, hands, times, gyro, load_quest_hands(hands), layout)
