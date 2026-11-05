"""COCO-133 <- Goliath-308 keypoint projection for Sapiens2 pose output."""

from functools import lru_cache
from typing import Any

from posekit.models.sapiens_net.pose import parse_pose_metainfo
from sapiens2_lite import keypoints308

COCO133_GOLIATH_ALIASES: dict[str, str] = {
    **{f"{side}_hand_root": f"{side}_wrist" for side in ("left", "right")},
    **{
        f"{side}_{finger}1": f"{side}_{finger}_third_joint"
        for side in ("left", "right")
        for finger in ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
    },
}
"""COCO WholeBody hand joints Goliath predicts under another name: the hand root is the wrist, a finger's first
joint is its ``third_joint`` (the base, which Goliath's skeleton links straight to the wrist). The upstream table
matches names only, so without these the 2D hands lose their roots and finger bases."""


@lru_cache(maxsize=1)
def coco133_goliath_pairs() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The mapped COCO-133 joints and the Goliath-308 heatmap each reads, in COCO order.

    The fast export keeps exactly these heatmaps, in this order, so its output channel ``i`` is COCO joint ``coco[i]``.
    """
    meta: dict[str, Any] = parse_pose_metainfo(keypoints308.dataset_info)
    coco_keypoint_info: dict[Any, Any] = keypoints308.coco_wholebody_info["keypoint_info"]  # untyped upstream config dict
    coco_ids: dict[str, int] = {str(info["name"]): int(info["id"]) for info in coco_keypoint_info.values()}
    goliath_ids: dict[str, int] = {str(name): int(index) for name, index in meta["keypoint_name2id"].items()}
    mapping: dict[Any, Any] = {
        **meta["coco_wholebody_to_goliath_mapping"],
        **{coco_ids[coco]: goliath_ids[goliath] for coco, goliath in COCO133_GOLIATH_ALIASES.items()},
    }
    pairs: list[tuple[int, int]] = sorted((int(coco), int(goliath)) for coco, goliath in mapping.items() if 0 <= int(coco) < 133)
    return tuple(coco for coco, _ in pairs), tuple(goliath for _, goliath in pairs)
