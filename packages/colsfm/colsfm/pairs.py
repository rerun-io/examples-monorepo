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

Note the stage cannot invent cross-camera pairs beyond the stereo baseline: all
other cross-camera connectivity comes from loop closure.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from colsfm.database import ImagePair, image_ids_by_name
from colsfm.frames_meta import FramesMeta, KeyframeMeta, RigFrame
from colsfm.schema import CusfmSchema, load_schema

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


def select_pairs(frames_meta: FramesMeta, connected_keyframe_num: int = 1) -> list[ImagePair]:
    """Build the whole matching pair list for a collection.

    Loop pairs are deliberately absent: they only exist after retrieval and
    verification, and `colsfm.pipeline` matches them into the same database
    afterwards rather than re-running pair selection (see that module's
    "Pair selection cannot see loop pairs").

    Args:
        frames_meta: The keyframes to match, already filtered by keyframe selection.
        connected_keyframe_num: Rig-frame lookahead for same-camera links.

    Returns:
        Normalised, deduplicated, sorted `(image_id1, image_id2)` pairs.

    Raises:
        ValueError: When `connected_keyframe_num` is below 1.
    """
    pairs: set[ImagePair] = set(consecutive_pairs(frames_meta, connected_keyframe_num))
    pairs.update(stereo_pairs(frames_meta))
    return sorted(pairs)


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

    `match_image_pairs` and `verify_matches` both address images by name, so the
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


def read_task_pairs(task_path: Path, schema: CusfmSchema | None = None) -> list[ImagePair]:
    """Read a blob `matching_task_NNN.pb` back into a pair list.

    Only used to compare against cuSFM's own output; the Python pipeline passes
    pairs in memory rather than through task files.

    Args:
        task_path: A binary `protos.visual.cusfm.FeatureMatchingTask`.
        schema: Loaded cuSFM schema; the vendored one by default.

    Returns:
        Normalised, deduplicated, sorted pairs.
    """
    resolved_schema: CusfmSchema = load_schema() if schema is None else schema
    task = resolved_schema.message_class(FEATURE_MATCHING_TASK)()
    task.ParseFromString(task_path.read_bytes())
    return sorted({normalise_pair(int(pair.source_frame_id), int(pair.target_frame_id)) for pair in task.frame_pairs})
