"""Which image pairs a loop measurement will ask for, before any of it runs.

The enumeration `colsfm.loop_closure.plan_loop_search` needs and
`colsfm.loop_pose.RigPoseEstimator` obeys, in one place so the two cannot drift:
`local_map_partners` is the estimator's own partner rule, and
`required_image_pairs` walks it plus the observation walk over a shortlist.

This is what replaced the discovery run. The pipeline used to learn its pair list
by *executing* the whole loop search against a matcher that recorded requests and
returned nothing -- so pair scheduling depended on driving the geometric
verification path with deliberately false input, and every future change to
verification became a change to scheduling.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from colsfm.database import Keypoints
from colsfm.pairs import normalise_pair
from colsfm.rig_geometry import RigFrameIndex, RigGeometry


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


@dataclass(frozen=True, slots=True)
class ObservationPair:
    """One image pair the measurement of a rig pair matches: a source anchor against a target image."""

    anchor: int
    """Image id of the source rig frame's anchor."""
    observed: int
    """Image id of the target rig frame's image: the anchor's own camera or its stereo partner."""
    anchor_camera_id: int
    """The anchor's camera."""
    observed_camera_id: int
    """The observed image's camera: `anchor_camera_id` or its stereo partner."""


def observation_pairs(
    index: RigFrameIndex, geometry: RigGeometry, keypoints: Mapping[int, Keypoints], rig_pair: tuple[int, int]
) -> list[ObservationPair]:
    """The observation walk of one shortlisted rig pair, in the order `RigPoseEstimator` matches it.

    Each source anchor against the target rig's same camera and that camera's stereo partner. One enumeration for the
    estimator (`colsfm.loop_pose`), the plan (`required_image_pairs`) and the mapper's pose check
    (`colsfm.loop_mapper_check`).

    Args:
        index: The rig-frame index.
        geometry: The rig's cameras and their stereo declarations.
        keypoints: Keypoint pixels per image id; a target image without them is skipped where the estimator skips it.
        rig_pair: `(source rig id, target rig id)`, directed as the measurement walks it.

    Returns:
        The pairs in camera order, anchor first.
    """
    source_rig_id, target_rig_id = rig_pair
    pairs: list[ObservationPair] = []
    for camera_id in geometry.camera_ids:
        anchor: int | None = index.keyframe(source_rig_id, camera_id)
        if anchor is None:
            continue
        target_cameras: list[int] = [camera_id]
        partner_camera: int | None = geometry.stereo_partner.get(camera_id)
        if partner_camera is not None:
            target_cameras.append(partner_camera)
        for target_camera in target_cameras:
            observed: int | None = index.keyframe(target_rig_id, target_camera)
            if observed is None or observed not in keypoints:
                continue
            pairs.append(ObservationPair(anchor=anchor, observed=observed, anchor_camera_id=camera_id, observed_camera_id=target_camera))
    return pairs


def required_image_pairs(
    index: RigFrameIndex,
    geometry: RigGeometry,
    keypoints: Mapping[int, Keypoints],
    neighbour_span: int,
    rig_pairs: Iterable[tuple[int, int]],
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
      the target rig's same camera and its stereo partner.

    Args:
        index: The rig-frame index.
        geometry: The rig's cameras, extrinsics and stereo declarations.
        keypoints: Keypoint pixels per image id; an image without them is skipped
            exactly where the estimator skips it.
        neighbour_span: Temporal neighbours contributing rays to a local map;
            `RigPoseConfig.neighbour_span` is what the estimator passes.
        rig_pairs: The shortlisted `(source rig id, target rig id)` pairs, directed as
            the measurement will walk them.

    Returns:
        The image pairs, normalised low id first, deduplicated and sorted.
    """
    rig_pairs = tuple(rig_pairs)
    pairs: set[tuple[int, int]] = set()
    for source_rig_id in dict.fromkeys(source for source, _ in rig_pairs):
        for camera_id in geometry.camera_ids:
            anchor: int | None = index.keyframe(source_rig_id, camera_id)
            # `local_map` only anchors an image it has keypoints for; `_observations`
            # matches from that anchor whether or not it does.
            if anchor is not None and anchor in keypoints:
                for partner in local_map_partners(index, geometry, neighbour_span, source_rig_id, camera_id):
                    pairs.add(normalise_pair(anchor, partner))
    pairs |= observed_image_pairs(index, geometry, keypoints, rig_pairs)
    return tuple(sorted(pairs))


def observed_image_pairs(
    index: RigFrameIndex, geometry: RigGeometry, keypoints: Mapping[int, Keypoints], rig_pairs: Iterable[tuple[int, int]]
) -> set[tuple[int, int]]:
    """The image pairs of the observation walks of some rig pairs (`observation_pairs`), normalised low id first.

    Args:
        index: The rig-frame index.
        geometry: The rig's cameras and their stereo declarations.
        keypoints: Keypoint pixels per image id; a target image without them is skipped where the estimator skips it.
        rig_pairs: `(source rig id, target rig id)` pairs, directed as the measurement walks them.

    Returns:
        The image pairs, deduplicated.
    """
    return {
        normalise_pair(observation.anchor, observation.observed)
        for rig_pair in rig_pairs
        for observation in observation_pairs(index, geometry, keypoints, rig_pair)
    }
