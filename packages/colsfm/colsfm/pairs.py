"""Which image pairs get matched — the Python replacement for `feature_matcher_task_builder_main`.

With the shipped isaac config (`search_method: POSE_GRAPH_ASSOCIATION`) the task
builder is a pure filter over pose-graph edges, and the pose graph in that mode
holds exactly three kinds of constraint (feature_matcher_task_builder_main.md
§4.2, pose_graph_main.md §6.3):

| constraint | pair rule |
|---|---|
| `CONSECUTIVE` | same camera, this rig frame to each of the next `connected_keyframe_num` rig frames |
| `EXTRINSIC` | the two cameras of a **declared** stereo pair inside one rig frame |
| `LOOP` | supplied by the loop-closure stage; passed in as `loop_pairs` |

So the graph never has to be built to get the pair list: the metadata's
`synced_sample_id` grouping and its `stereo_pair` declarations decide the first
two categories outright. Measured on the shipped 226-keyframe Galileo run, that
rule reproduces 331 of the blob's 339 task pairs (218 consecutive + 113 stereo)
with no extras, the remaining eight being LOOP edges; on the 32-keyframe run it
reproduces all 40 exactly.

Pairs are normalised to `(min, max)`, deduplicated and sorted, which is what
`UpdateMatchingTaskAndSave` guarantees downstream (§4.3). The blob emits them in
pose-graph order instead; only byte-identical task files would notice, and the
matcher is order independent.

Those rules cannot pair two cameras beyond a declared stereo baseline, and loop
closure pairs a camera only with itself. A rig whose cameras never share a view inside
one frame (the six faces of a 360 cube) then has no cross-camera pair at all, so its
intrinsics and rig cannot be estimated, and its long-range links are few (Mission
office: 0 of 6,470 verified pairs cross-face; ~2,860 pairs more than 10 rig frames
apart against Spirula's ~23,000). `prior_overlap_pairs` is colsfm's fourth rule, off by
default: it pairs any two images whose *prior* views overlap, at a fixed budget per image.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Float64, Int64
from numpy import ndarray

from colsfm.config import PriorOverlapConfig
from colsfm.database import ImagePair, image_ids_by_name
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame

FEATURE_MATCHING_TASK: str = "protos.visual.cusfm.FeatureMatchingTask"
"""The message `feature_matcher_task_builder_main` writes to `matching_task_%03zu.pb`."""


def _keyframe_by_rig_and_camera(frames_meta: FramesMeta) -> dict[tuple[int, int], int]:
    """Index keyframe ids by their rig frame and camera.

    Args:
        frames_meta: The collection to index.

    Returns:
        Keyframe id keyed by `(synced_sample_id, camera_params_id)`. A camera
        firing twice inside one rig frame keeps its lowest keyframe id, matching
        the reference-frame rule in `FramesMeta.rig_frames`.
    """
    index: dict[tuple[int, int], int] = {}
    for keyframe in sorted(frames_meta.keyframes, key=lambda entry: entry.keyframe_id):
        index.setdefault((keyframe.synced_sample_id, keyframe.camera_params_id), keyframe.keyframe_id)
    return index


def consecutive_pairs(frames_meta: FramesMeta, connected_keyframe_num: int) -> list[ImagePair]:
    """Link every camera to itself across the next rig frames in time order.

    Args:
        frames_meta: The collection, whose `synced_sample_id` order is time order.
        connected_keyframe_num: How many following rig frames each rig frame links
            to; the shipped isaac `pose_graph_config.pb.txt` uses 1.

    Returns:
        Normalised, deduplicated, sorted pairs.

    Raises:
        ValueError: When `connected_keyframe_num` is below 1, which `RunPoseGraph` rejects.
    """
    if connected_keyframe_num < 1:
        raise ValueError(f"connected_keyframe_num is {connected_keyframe_num}, which should be at least 1")
    index: dict[tuple[int, int], int] = _keyframe_by_rig_and_camera(frames_meta)
    rig_frames: tuple[RigFrame, ...] = frames_meta.rig_frames()
    keyframe_by_id: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    pairs: set[ImagePair] = set()
    for position, rig_frame in enumerate(rig_frames):
        for keyframe_id in rig_frame.keyframe_ids:
            camera_params_id: int = keyframe_by_id[keyframe_id].camera_params_id
            for offset in range(1, connected_keyframe_num + 1):
                if position + offset >= len(rig_frames):
                    break
                successor_id: int | None = index.get((rig_frames[position + offset].synced_sample_id, camera_params_id))
                if successor_id is not None:
                    pairs.add(normalise_pair(keyframe_id, successor_id))
    return sorted(pairs)


def stereo_pairs(frames_meta: FramesMeta) -> list[ImagePair]:
    """Pair the two cameras of every declared stereo rig inside every rig frame.

    Only pairs declared in the metadata's `stereo_pair` list survive the blob's
    `EXTRINSIC` filter; a rig frame's other camera combinations are dropped.

    Args:
        frames_meta: The collection, whose `stereo_pairs` declare the rigs.

    Returns:
        Normalised, deduplicated, sorted pairs.
    """
    index: dict[tuple[int, int], int] = _keyframe_by_rig_and_camera(frames_meta)
    pairs: set[ImagePair] = set()
    for rig_frame in frames_meta.rig_frames():
        for stereo_pair in frames_meta.stereo_pairs:
            left_id: int | None = index.get((rig_frame.synced_sample_id, stereo_pair.left_camera_params_id))
            right_id: int | None = index.get((rig_frame.synced_sample_id, stereo_pair.right_camera_params_id))
            if left_id is not None and right_id is not None:
                pairs.add(normalise_pair(left_id, right_id))
    return sorted(pairs)


def prior_overlap_proposals(frames_meta: FramesMeta, config: PriorOverlapConfig, connected_keyframe_num: int) -> dict[int, list[int]]:
    """The partners each image proposes from its prior view overlap.

    Candidates are the images of any camera in another rig frame whose prior optical
    axis is within `max_axis_angle_deg` and whose centre is within
    `max_centre_distance_m`, less the same-camera images CONSECUTIVE already pairs.
    Each image takes the best-aligned candidates first, skipping any candidate within
    `suppression_rig_frames` of one it already took, until it holds `pairs_per_image`.
    The suppression is what keeps long-range links: without it the rig frames next to
    an image, whose views overlap it most, would take the whole budget.

    Args:
        frames_meta: The collection, with prior `world_T_cam` poses.
        config: The rule.
        connected_keyframe_num: CONSECUTIVE's lookahead; those same-camera pairs are left out.

    Returns:
        Partner keyframe ids per keyframe id, best-aligned first.
    """
    keyframes: list[KeyframeMeta] = sorted(frames_meta.keyframes, key=lambda entry: entry.keyframe_id)
    position: dict[int, int] = {rig_frame.synced_sample_id: index for index, rig_frame in enumerate(frames_meta.rig_frames())}
    axes: Float64[ndarray, "num_images 3"] = np.array([keyframe.world_T_cam.rotation.matrix()[:, 2] for keyframe in keyframes])
    centres: Float64[ndarray, "num_images 3"] = np.array([np.asarray(keyframe.world_T_cam.translation) for keyframe in keyframes])
    frames: Int64[ndarray, " num_images"] = np.array([position[keyframe.synced_sample_id] for keyframe in keyframes], dtype=np.int64)
    cameras: Int64[ndarray, " num_images"] = np.array([keyframe.camera_params_id for keyframe in keyframes], dtype=np.int64)
    cos_limit: float = float(np.cos(np.radians(config.max_axis_angle_deg)))
    proposals: dict[int, list[int]] = {}
    for row, keyframe in enumerate(keyframes):
        cosines: Float64[ndarray, " num_images"] = axes @ axes[row]
        gaps: Int64[ndarray, " num_images"] = np.abs(frames - frames[row])
        consecutive: np.ndarray = (cameras == cameras[row]) & (gaps <= connected_keyframe_num)
        near: np.ndarray = np.linalg.norm(centres - centres[row], axis=1) <= config.max_centre_distance_m
        candidates: Int64[ndarray, " num_candidates"] = np.flatnonzero((cosines >= cos_limit) & near & (gaps > 0) & ~consecutive)
        taken_frames: list[int] = []
        partners: list[int] = []
        for candidate in candidates[np.argsort(-cosines[candidates], kind="stable")]:
            if len(partners) == config.pairs_per_image:
                break
            if any(abs(int(frames[candidate]) - taken) < config.suppression_rig_frames for taken in taken_frames):
                continue
            taken_frames.append(int(frames[candidate]))
            partners.append(keyframes[candidate].keyframe_id)
        proposals[keyframe.keyframe_id] = partners
    return proposals


def prior_overlap_pairs(frames_meta: FramesMeta, config: PriorOverlapConfig, connected_keyframe_num: int) -> list[ImagePair]:
    """Pair images of any two cameras whose prior views overlap; see `prior_overlap_proposals`.

    Args:
        frames_meta: The collection, with prior `world_T_cam` poses.
        config: The rule.
        connected_keyframe_num: CONSECUTIVE's lookahead; those same-camera pairs are left out.

    Returns:
        Normalised, deduplicated, sorted pairs; at most `pairs_per_image` times the image count.
    """
    proposals: dict[int, list[int]] = prior_overlap_proposals(frames_meta, config, connected_keyframe_num)
    return sorted({normalise_pair(keyframe_id, partner) for keyframe_id, partners in proposals.items() for partner in partners})


def select_pairs(
    frames_meta: FramesMeta, connected_keyframe_num: int = 1, prior_overlap: PriorOverlapConfig | None = None
) -> list[ImagePair]:
    """Build the whole matching pair list for a collection.

    Loop pairs are deliberately absent: they only exist after retrieval and
    verification, and `colsfm.pipeline` matches them into the same database
    afterwards rather than re-running pair selection (see that module's
    "Pair selection cannot see loop pairs").

    Args:
        frames_meta: The keyframes to match, already filtered by keyframe selection.
        connected_keyframe_num: Rig-frame lookahead for same-camera links.
        prior_overlap: The prior-overlap rule, or None for cuSFM's two rules alone.

    Returns:
        Normalised, deduplicated, sorted `(image_id1, image_id2)` pairs.

    Raises:
        ValueError: When `connected_keyframe_num` is below 1.
    """
    pairs: set[ImagePair] = set(consecutive_pairs(frames_meta, connected_keyframe_num))
    pairs.update(stereo_pairs(frames_meta))
    if prior_overlap is not None:
        pairs.update(prior_overlap_pairs(frames_meta, prior_overlap, connected_keyframe_num))
    return sorted(pairs)


LONG_RANGE_RIG_FRAMES: int = 10
"""`pair_mix` counts a pair as long-range above this many rig frames apart (the sfm-root yardstick)."""


def pair_mix(frames_meta: FramesMeta, pairs: Sequence[ImagePair]) -> str:
    """Say how many pairs join two cameras and how many span more than `LONG_RANGE_RIG_FRAMES`.

    Args:
        frames_meta: The collection the pairs index.
        pairs: Image pairs.

    Returns:
        A short phrase for the run log.
    """
    keyframes: dict[int, KeyframeMeta] = frames_meta.keyframe_by_id()
    position: dict[int, int] = {rig_frame.synced_sample_id: index for index, rig_frame in enumerate(frames_meta.rig_frames())}
    cross: int = sum(keyframes[first].camera_params_id != keyframes[second].camera_params_id for first, second in pairs)
    long_range: int = sum(
        abs(position[keyframes[first].synced_sample_id] - position[keyframes[second].synced_sample_id]) > LONG_RANGE_RIG_FRAMES
        for first, second in pairs
    )
    return f"{cross} cross-camera, {long_range} more than {LONG_RANGE_RIG_FRAMES} rig frames apart"


def normalise_pair(image_id1: int, image_id2: int) -> ImagePair:
    """Order a pair as `(min, max)`.

    Public because the canonical form is the package's, not this module's: loop pair
    enumeration, the rig-pair shortlist and the walkthrough's revisit heat map all key
    dictionaries and sets by an unordered pair, and each used to inline `(min, max)`
    with no self-pair check at all.

    Args:
        image_id1: One id — an image id, or a rig id where the pair is a rig pair.
        image_id2: The other id.

    Returns:
        The canonical pair.

    Raises:
        ValueError: When both ids are the same image.
    """
    if image_id1 == image_id2:
        raise ValueError(f"Cannot pair image {image_id1} with itself")
    return (min(image_id1, image_id2), max(image_id1, image_id2))


def write_pair_list(pair_list_path: Path, pairs: Sequence[ImagePair], database_path: Path) -> None:
    """Write the `image_name1 image_name2` file pycolmap's pair matchers read.

    `match_image_pairs` addresses images by name, so the
    names are looked up in the database rather than trusted from the caller.

    Args:
        pair_list_path: Destination text file; parent directories are created.
        pairs: Image-id pairs to write, in the order given.
        database_path: Database holding the images.

    Raises:
        KeyError: When a pair names an image the database does not hold.
    """
    name_by_id: dict[int, str] = {image_id: name for name, image_id in image_ids_by_name(database_path).items()}
    lines: list[str] = []
    for image_id1, image_id2 in pairs:
        for image_id in (image_id1, image_id2):
            if image_id not in name_by_id:
                raise KeyError(f"Image {image_id} is not in {database_path}")
        lines.append(f"{name_by_id[image_id1]} {name_by_id[image_id2]}")
    pair_list_path.parent.mkdir(parents=True, exist_ok=True)
    pair_list_path.write_text("\n".join(lines) + "\n")


