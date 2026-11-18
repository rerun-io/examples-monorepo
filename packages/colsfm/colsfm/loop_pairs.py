"""Which image pairs a loop measurement will ask for, before any of it runs.

The enumeration `colsfm.loop_closure.plan_loop_search` needs and
`colsfm.loop_pose.RigPoseEstimator` obeys, in one place so the two cannot drift:
`local_map_partners` is the estimator's own partner rule, `loop_camera_pairing`
decides once which target cameras each source anchor is matched against, and
`required_image_pairs` walks both over a shortlist.

This is what replaced the discovery run. The pipeline used to learn its pair list
by *executing* the whole loop search against a matcher that recorded requests and
returned nothing -- so pair scheduling depended on driving the geometric
verification path with deliberately false input, and every future change to
verification became a change to scheduling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeAlias

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from colsfm.database import Keypoints
from colsfm.pairs import normalise_pair
from colsfm.retrieval import RetrievalIndex
from colsfm.rig_geometry import RigFrameIndex, RigGeometry

CameraPairing: TypeAlias = Mapping[tuple[int, int, int], tuple[int, ...]]
"""`(source rig id, target rig id, source camera id)` -> the target cameras that source anchor is matched against,
primary first, then the primary's stereo partner. One entry per source camera that has an image in the source rig frame."""


def local_map_partners(
    index: RigFrameIndex, geometry: RigGeometry, neighbour_span: int, rig_id: int, camera_id: int
) -> list[int]:
    """Images whose rays join one anchor image's in a rig frame's local map.

    A free function rather than only a method, because two callers need it and they
    must not drift: `RigPoseEstimator` matches these pairs, and `required_image_pairs`
    tells the pipeline which pairs to match *before* an estimator exists.

    Args:
        index: The rig-frame index.
        geometry: The rig's cameras and their stereo declarations.
        neighbour_span: Temporal neighbours contributing rays; 0 uses the stereo pair alone.
        rig_id: The source rig frame.
        camera_id: The anchor image's camera.

    Returns:
        Image ids of the stereo partner inside the rig frame and of the same camera in
        every temporal neighbour, in a stable order.
    """
    partners: list[int] = []
    partner_camera: int | None = geometry.stereo_partner.get(camera_id)
    if partner_camera is not None:
        stereo_image: int | None = index.keyframe(rig_id, partner_camera)
        if stereo_image is not None:
            partners.append(stereo_image)
    for neighbour in index.neighbours(rig_id, neighbour_span):
        neighbour_image: int | None = index.keyframe(neighbour, camera_id)
        if neighbour_image is not None:
            partners.append(neighbour_image)
    return partners


def loop_camera_pairing(
    index: RigFrameIndex,
    geometry: RigGeometry,
    retrieval: RetrievalIndex,
    keypoints: Mapping[int, Keypoints],
    rig_pairs: Iterable[tuple[int, int]],
    by_prior: bool,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Choose, once per shortlisted rig pair and source camera, the target cameras a loop measurement matches.

    Each source anchor is matched against one primary target camera and that camera's stereo partner. The primary is
    the target camera that sees the same scene, chosen by one of two scores:

    * by default, the retrieval score of the anchor against each target camera's image (`RetrievalIndex.score`): the
      appearance chooses the camera, with no prior;
    * with `by_prior`, the cosine between the anchor camera's optical axis and each target camera's, under the prior
      relative rotation of the two rig frames: for a profile that trusts its metric prior.

    Tie rule: the source's own camera stays the primary unless a camera that is not its stereo partner scores strictly
    higher. So a mono rig and a stereo rig always get the same-camera pairs of the old rule, and a rig whose cameras look in several
    directions gets the camera that faces the source anchor on a revisit walked another way (a cube face pairs with the
    opposite face after a 180 deg turn). A target camera without an image, or without keypoints, cannot be chosen. The
    rule chooses a camera; it never drops one.

    Args:
        index: The rig-frame index (images and prior poses).
        geometry: The rig's cameras, extrinsics and stereo declarations.
        retrieval: The index the shortlist came from; only its per-pair score is read.
        keypoints: Keypoint pixels per image id.
        rig_pairs: The shortlisted `(source rig id, target rig id)` pairs, directed as the measurement walks them.
        by_prior: Score the target cameras by their prior viewing direction instead of by retrieval.

    Returns:
        The table `CameraPairing` describes.
    """
    indexed: frozenset[int] = frozenset(retrieval.image_ids)
    optical_axes: Float64[ndarray, "n_cameras 3"] = np.stack([pose.rotation.matrix()[2] for pose in geometry.cams_from_rig])
    pairing: dict[tuple[int, int, int], tuple[int, ...]] = {}
    for source_rig_id, target_rig_id in rig_pairs:
        source_R_target: Float64[ndarray, "3 3"] = (index.world_T_rig[source_rig_id].inverse() * index.world_T_rig[target_rig_id]).rotation.matrix()
        for camera_id in geometry.camera_ids:
            anchor: int | None = index.keyframe(source_rig_id, camera_id)
            if anchor is None:
                continue
            anchor_axis_in_target: Float64[ndarray, "3"] = source_R_target.T @ optical_axes[geometry.position_of(camera_id)]
            scores: dict[int, float] = {}
            for target_camera in geometry.camera_ids:
                observed: int | None = index.keyframe(target_rig_id, target_camera)
                if observed is None or observed not in keypoints or target_camera == geometry.stereo_partner.get(camera_id):
                    continue
                if by_prior:
                    scores[target_camera] = float(optical_axes[geometry.position_of(target_camera)] @ anchor_axis_in_target)
                elif anchor in indexed and observed in indexed:
                    scores[target_camera] = retrieval.score(anchor, observed)
            primary: int = camera_id
            best: int | None = max(scores, key=scores.__getitem__, default=None)
            if best is not None and scores[best] > scores.get(camera_id, -np.inf):
                primary = best
            partner_camera: int | None = geometry.stereo_partner.get(primary)
            pairing[(source_rig_id, target_rig_id, camera_id)] = (primary,) if partner_camera is None else (primary, partner_camera)
    return pairing


def required_image_pairs(
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    neighbour_span: int,
    rig_pairs: Iterable[tuple[int, int]],
    pairing: CameraPairing,
) -> tuple[tuple[int, int], ...]:
    """Every image pair the estimator will ask `match_fn` for, before any of it runs.

    This is what replaces the discovery run. The pipeline used to learn the pair list
    by executing the whole loop search against a matcher that returned nothing, then
    throwing the result away — so pair *scheduling* depended on running the geometric
    verification path with deliberately false input. The two sources of pairs are
    small and entirely determined by the rig geometry and the shortlist:

    * the **local map** of each source rig frame — each anchor image against its
      stereo partner and against the same camera in each temporal neighbour
      (`local_map_partners`, asked for unconditionally by `_anchor_tracks`);
    * the **observations** of each shortlisted rig pair — each source anchor against
      the target cameras `pairing` chose for it (`loop_camera_pairing`).

    Args:
        index: The rig-frame index.
        geometry: The rig's cameras, extrinsics and stereo declarations.
        keypoints: Keypoint pixels per image id; an image without them is skipped
            exactly where the estimator skips it.
        neighbour_span: Temporal neighbours contributing rays to a local map;
            `RigPoseConfig.neighbour_span` is what the estimator passes.
        rig_pairs: The shortlisted `(source rig id, target rig id)` pairs, directed as
            the measurement will walk them.
        pairing: The plan's camera pairing; it holds every pair of `rig_pairs`.

    Returns:
        The image pairs, normalised low id first, deduplicated and sorted.
    """
    targets_by_source: dict[int, list[int]] = {}
    for source_rig_id, target_rig_id in rig_pairs:
        targets_by_source.setdefault(source_rig_id, []).append(target_rig_id)

    pairs: set[tuple[int, int]] = set()
    for source_rig_id, target_rig_ids in targets_by_source.items():
        for camera_id in geometry.camera_ids:
            anchor: int | None = index.keyframe(source_rig_id, camera_id)
            if anchor is None:
                continue
            # `local_map` only anchors an image it has keypoints for; `_observations`
            # matches from that anchor whether or not it does.
            if anchor in keypoints:
                for partner in local_map_partners(index, geometry, neighbour_span, source_rig_id, camera_id):
                    pairs.add(normalise_pair(anchor, partner))
            for target_rig_id in target_rig_ids:
                for target_camera in pairing[(source_rig_id, target_rig_id, camera_id)]:
                    observed: int | None = index.keyframe(target_rig_id, target_camera)
                    if observed is None or observed not in keypoints:
                        continue
                    pairs.add(normalise_pair(anchor, observed))
    return tuple(sorted(pairs))
