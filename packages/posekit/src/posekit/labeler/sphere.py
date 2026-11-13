"""A hand as a 3D sphere: label it in one or two views, derive its box in every other view through the KB4 rig.

MEgATrack's DetNet box is the circle around the projected hand keypoints, so a sphere around the hand is the simplest 3D
shape whose projection is that box. A sphere lives in the rig frame (metres); its box in a camera is the bounding box of its
silhouette (the tangent cone, sampled and projected through the KB4 lens), clipped to the image.

- Two or more labeled views (``triangulate``): the centre starts as the least-squares point of the box-centre rays, the
  radius as the median of ``distance * sin(half angle)`` over the views; Levenberg-Marquardt then fits centre and radius to
  the box edges in pixels.
- One labeled view (``lift_box``): the box fixes the ray and the angular size, so only the depth is free. The depth is the
  one whose sphere best overlaps the other views' proposals (EgoHand boxes of any side, because its side label flips); with
  no overlap at least ``MIN_SUPPORT_IOU`` the hand-size prior ``HAND_RADIUS_M`` sets it. ``confidence`` is the best overlap,
  0 for the prior: the UI asks for a second view when it is low.

Not modelled: occlusion (a sphere projects into a view even when a body part hides the hand there) and the hand's shape (a
flat, open hand is wider than its sphere in some views).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float64
from numpy import ndarray

from posekit.labeler.rig import Kb4Rig
from posekit.labeler.schema import Box, SolveMode

HAND_RADIUS_M: float = 0.095
"""Radius of the sphere around an adult hand (the circle around its keypoints), m."""
HAND_RADIUS_SIGMA_M: float = 0.03
"""Spread of the hand-size prior, m: it only breaks ties between depths the proposals support equally."""
SILHOUETTE_SAMPLES: int = 72
"""Points sampled on the sphere's tangent circle."""
DEPTH_RANGE_M: tuple[float, float] = (0.12, 1.5)
"""Distances from the labeled camera that the one-view search tries, m."""
DEPTH_STEPS: int = 160
MIN_SUPPORT_IOU: float = 0.3
"""A view supports a depth when the sphere's box overlaps one of its proposals at least this much."""
MIN_BOX_SIDE_PX: float = 4.0
"""Smallest side of a derived box after clipping, px."""
LM_ITERATIONS: int = 30
FULL_CONFIDENCE_ANGLE_DEG: float = 10.0
"""Angle between two labeled views' rays at the hand from which depth counts as fixed: RoboCap's stacked front/eye
cameras (2 cm apart) see a hand at 40 cm under about 3 degrees, the left/right pairs (9-10 cm) under about 14."""


@dataclass(frozen=True, slots=True)
class HandSphere:
    """A hand in the rig frame."""

    centre: Float64[ndarray, "3"]
    """Rig-frame centre, m."""
    radius: float
    """m."""


@dataclass(frozen=True, slots=True)
class DepthChoice:
    """The one-view lift: the sphere along the labeled ray and how well the other views back it."""

    sphere: HandSphere
    mode: Literal["proposals", "prior"]
    confidence: float
    """Best overlap of the derived box with another view's proposal, 0 for the prior."""
    support: tuple[int, ...]
    """Views whose proposals back the chosen depth (overlap at least ``MIN_SUPPORT_IOU``)."""


@dataclass(frozen=True, slots=True)
class ViewSolve:
    """One hand side at one time: the sphere and its box in every camera."""

    sphere: HandSphere
    mode: SolveMode
    confidence: float
    """Triangulated: the widest angle between two labeled views' rays at the centre over ``FULL_CONFIDENCE_ANGLE_DEG``,
    at most 1; one view: ``DepthChoice.confidence``."""
    labeled: tuple[int, ...]
    boxes: dict[int, Box | None]
    """Every camera: the person's box in a labeled view, the sphere's box elsewhere (None when it is not in the image)."""


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    w: float = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    h: float = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter: float = w * h
    union: float = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0.0 else 0.0


def max_theta(rig: Kb4Rig, cam: int) -> float:
    """Widest angle off the optical axis that the image still shows (its corners), rad."""
    w, h = (float(v) for v in rig.size_wh[cam])
    corners: Float64[ndarray, "4 2"] = np.array([[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]])
    return float(np.max(np.arccos(np.clip(rig.unproject(cam, corners)[:, 2], -1.0, 1.0))))


def sphere_box(rig: Kb4Rig, cam: int, sphere: HandSphere) -> Box | None:
    """The sphere's box in camera ``cam``: the bounding box of its projected silhouette, clipped to the image; None when the
    camera is inside the sphere or no part of it falls in the image."""
    centre: Float64[ndarray, "3"] = rig.to_cam(cam, sphere.centre[None])[0]
    distance: float = float(np.linalg.norm(centre))
    if distance <= sphere.radius * 1.01:
        return None
    axis: Float64[ndarray, "3"] = centre / distance
    helper: Float64[ndarray, "3"] = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1: Float64[ndarray, "3"] = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2: Float64[ndarray, "3"] = np.cross(axis, e1)
    half: float = math.asin(sphere.radius / distance)
    angles: Float64[ndarray, "s"] = np.linspace(0.0, 2.0 * math.pi, SILHOUETTE_SAMPLES, endpoint=False)
    rays: Float64[ndarray, "s 3"] = math.cos(half) * axis + math.sin(half) * (np.cos(angles)[:, None] * e1 + np.sin(angles)[:, None] * e2)
    # The silhouette is a circle of rays; any ray beyond the lens' field of view is cut (its pixel would fold back).
    seen: Bool[ndarray, "s"] = np.arccos(np.clip(rays[:, 2], -1.0, 1.0)) <= max_theta(rig, cam)
    if not seen.any():
        return None
    pixels: Float64[ndarray, "k 2"] = rig.project(cam, rays[seen])
    w, h = (float(v) for v in rig.size_wh[cam])
    x0, y0 = np.clip(pixels.min(axis=0), 0.0, [w, h])
    x1, y1 = np.clip(pixels.max(axis=0), 0.0, [w, h])
    if x1 - x0 < MIN_BOX_SIDE_PX or y1 - y0 < MIN_BOX_SIDE_PX:
        return None
    return Box(float(x0), float(y0), float(x1), float(y1))


def box_ray(rig: Kb4Rig, cam: int, box: Box) -> tuple[Float64[ndarray, "3"], float]:
    """Rig-frame unit ray through the box centre and the box's angular half size (mean of its width and height), rad."""
    cx, cy = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    pixels: Float64[ndarray, "5 2"] = np.array([[cx, cy], [box.x0, cy], [box.x1, cy], [cx, box.y0], [cx, box.y1]])
    rays: Float64[ndarray, "5 3"] = rig.unproject(cam, pixels)
    width: float = math.acos(float(np.clip(rays[1] @ rays[2], -1.0, 1.0)))
    height: float = math.acos(float(np.clip(rays[3] @ rays[4], -1.0, 1.0)))
    return rig.rays_to_rig(cam, rays[:1])[0], (width + height) / 4.0


def box_residuals(rig: Kb4Rig, boxes: Mapping[int, Box], sphere: HandSphere) -> Float64[ndarray, "r"]:
    """Edge differences between the sphere's boxes and the labeled boxes, px (a whole image when it has no box)."""
    out: list[float] = []
    for cam, want in boxes.items():
        got: Box | None = sphere_box(rig, cam, sphere)
        w, h = (float(v) for v in rig.size_wh[cam])
        have: Box = got if got is not None else Box(0.0, 0.0, w, h)
        out += [have.x0 - want.x0, have.y0 - want.y0, have.x1 - want.x1, have.y1 - want.y1]
    return np.asarray(out)


def triangulate(rig: Kb4Rig, boxes: Mapping[int, Box]) -> HandSphere:
    """The sphere whose boxes best match two or more labeled views.

    Raises:
        ValueError: With fewer than two views.
    """
    if len(boxes) < 2:
        raise ValueError(f"triangulate needs two or more views, got {sorted(boxes)}")
    # Closest point to the box-centre rays: sum (I - d d^T) x = sum (I - d d^T) o.
    normal: Float64[ndarray, "3 3"] = np.zeros((3, 3))
    rhs: Float64[ndarray, "3"] = np.zeros(3)
    halves: list[tuple[Float64[ndarray, "3"], float]] = []
    for cam, box in boxes.items():
        ray, half = box_ray(rig, cam, box)
        projector: Float64[ndarray, "3 3"] = np.eye(3) - np.outer(ray, ray)
        normal += projector
        rhs += projector @ rig.centre(cam)
        halves.append((rig.centre(cam), half))
    centre: Float64[ndarray, "3"] = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    radius: float = float(np.median([np.linalg.norm(centre - origin) * math.sin(half) for origin, half in halves]))
    params: Float64[ndarray, "4"] = np.append(centre, radius)

    def cost(p: Float64[ndarray, "4"]) -> Float64[ndarray, "r"]:
        return box_residuals(rig, boxes, HandSphere(p[:3], float(max(p[3], 1e-3))))

    # Levenberg-Marquardt with a forward-difference Jacobian (4 unknowns, a few dozen residuals).
    damping: float = 1e-3
    residual: Float64[ndarray, "r"] = cost(params)
    for _ in range(LM_ITERATIONS):
        jacobian: Float64[ndarray, "r 4"] = np.stack([(cost(params + step) - residual) / 1e-5 for step in np.eye(4) * 1e-5], axis=-1)
        gram: Float64[ndarray, "4 4"] = jacobian.T @ jacobian
        update: Float64[ndarray, "4"] = np.linalg.solve(gram + damping * np.diag(np.diag(gram) + 1e-9), -jacobian.T @ residual)
        trial: Float64[ndarray, "r"] = cost(params + update)
        if trial @ trial < residual @ residual:
            params, residual, damping = params + update, trial, damping * 0.3
            if np.linalg.norm(update) < 1e-6:
                break
        else:
            damping *= 10.0
    return HandSphere(params[:3], float(max(params[3], 1e-3)))


def lift_box(rig: Kb4Rig, cam: int, box: Box, proposals: Mapping[int, Sequence[Box]]) -> DepthChoice:
    """The sphere along one labeled box: the depth whose boxes best overlap the other views' proposals, else the prior."""
    ray, half = box_ray(rig, cam, box)
    origin: Float64[ndarray, "3"] = rig.centre(cam)
    others: list[int] = [other for other in rig.cams if other != cam and proposals.get(other)]
    best: tuple[float, float, tuple[int, ...], float] | None = None  # (score, distance, support, confidence)
    for distance in np.geomspace(*DEPTH_RANGE_M, DEPTH_STEPS):
        sphere: HandSphere = HandSphere(origin + float(distance) * ray, float(distance) * math.sin(half))
        overlaps: dict[int, float] = {}
        for other in others:
            got: Box | None = sphere_box(rig, other, sphere)
            if got is not None:
                overlaps[other] = max(box_iou(got, proposal) for proposal in proposals[other])
        support: tuple[int, ...] = tuple(other for other, value in overlaps.items() if value >= MIN_SUPPORT_IOU)
        prior: float = math.exp(-0.5 * ((sphere.radius - HAND_RADIUS_M) / HAND_RADIUS_SIGMA_M) ** 2)
        score: float = sum(overlaps[other] for other in support) + 0.1 * prior
        if support and (best is None or score > best[0]):
            best = (score, float(distance), support, max(overlaps[other] for other in support))
    if best is None:
        distance = HAND_RADIUS_M / math.sin(half)
        return DepthChoice(HandSphere(origin + distance * ray, HAND_RADIUS_M), "prior", 0.0, ())
    _, distance, support, confidence = best
    return DepthChoice(HandSphere(origin + distance * ray, distance * math.sin(half)), "proposals", confidence, support)


def solve_views(rig: Kb4Rig, labeled: Mapping[int, Box], proposals: Mapping[int, Sequence[Box]]) -> ViewSolve:
    """One hand side: the person's boxes in ``labeled`` (one or more views) and the derived box in every other camera.

    Raises:
        ValueError: With no labeled view.
    """
    if not labeled:
        raise ValueError("solve_views needs at least one labeled view")
    cams: tuple[int, ...] = tuple(sorted(labeled))
    if len(cams) >= 2:
        sphere: HandSphere = triangulate(rig, labeled)
        mode: SolveMode = "triangulated"
        rays: list[Float64[ndarray, "3"]] = [(sphere.centre - rig.centre(cam)) / np.linalg.norm(sphere.centre - rig.centre(cam)) for cam in cams]
        widest: float = max(math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))) for a, b in combinations(rays, 2))
        confidence: float = min(1.0, widest / FULL_CONFIDENCE_ANGLE_DEG)
    else:
        choice: DepthChoice = lift_box(rig, cams[0], labeled[cams[0]], {cam: boxes for cam, boxes in proposals.items() if cam != cams[0]})
        sphere, mode, confidence = choice.sphere, choice.mode, choice.confidence
    boxes: dict[int, Box | None] = {cam: labeled[cam] if cam in labeled else sphere_box(rig, cam, sphere) for cam in rig.cams}
    return ViewSolve(sphere, mode, confidence, cams, boxes)
