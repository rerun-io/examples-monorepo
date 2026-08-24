"""Keypoint metadata and schema projection helpers for Sapiens2 pose output."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

from sapiens2_pose.sapiens_lite.pose import parse_pose_metainfo

PACKAGE_DIR: Path = Path(__file__).resolve().parents[1]
CONFIGS_DIR: Path = PACKAGE_DIR / "assets" / "configs"
KEYPOINTS_308_PATH: Path = CONFIGS_DIR / "_base_" / "keypoints308.py"

KeypointSchemaName = Literal["coco133", "sapiens308"]


@dataclass(frozen=True, slots=True)
class PoseSchema:
    """Rerun-facing pose schema metadata."""

    name: KeypointSchemaName
    """Stable schema identifier used by CLI and Gradio controls."""
    class_label: str
    """Human-readable label for the Rerun annotation class."""
    keypoint_ids: list[int]
    """Ordered keypoint IDs expected by Rerun Points2D logs."""
    id2name: dict[int, str]
    """Mapping from keypoint ID to display name."""
    keypoint_connections: list[tuple[int, int]]
    """Skeleton edges expressed in schema-local keypoint IDs."""


_keypoint_module_cache: Any | None = None
_sapiens_metainfo_cache: dict[str, Any] | None = None
_pose_schema_cache: dict[KeypointSchemaName, PoseSchema] = {}


def _load_keypoint_module() -> Any:
    """Load the bundled Sapiens2 keypoint metadata module."""
    global _keypoint_module_cache
    if _keypoint_module_cache is None:
        spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location(
            "_sapiens2_keypoints308_api",
            KEYPOINTS_308_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load keypoint metadata from {KEYPOINTS_308_PATH}")
        module: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _keypoint_module_cache = module
    return _keypoint_module_cache


def get_sapiens_metainfo() -> dict[str, Any]:
    """Return parsed Sapiens/Goliath 308-keypoint metadata."""
    global _sapiens_metainfo_cache
    if _sapiens_metainfo_cache is None:
        _sapiens_metainfo_cache = parse_pose_metainfo({"from_file": str(KEYPOINTS_308_PATH)})
    return _sapiens_metainfo_cache


def _build_sapiens_schema() -> PoseSchema:
    """Build the native Sapiens/Goliath 308-keypoint schema."""
    meta: dict[str, Any] = get_sapiens_metainfo()
    keypoint_ids: list[int] = list(range(int(meta["num_keypoints"])))
    id2name: dict[int, str] = {int(idx): str(meta["keypoint_id2name"][idx]) for idx in keypoint_ids}
    keypoint_connections: list[tuple[int, int]] = [(int(a), int(b)) for a, b in meta["skeleton_links"]]
    return PoseSchema(
        name="sapiens308",
        class_label="Sapiens2 308",
        keypoint_ids=keypoint_ids,
        id2name=id2name,
        keypoint_connections=keypoint_connections,
    )


def _build_coco133_schema() -> PoseSchema:
    """Build the COCO WholeBody 133-keypoint schema from bundled metadata."""
    module: Any = _load_keypoint_module()
    coco_info: dict[str, Any] = module.coco_wholebody_info
    keypoint_ids: list[int] = sorted(int(idx) for idx in coco_info["keypoint_info"])
    id2name: dict[int, str] = {
        int(idx): str(coco_info["keypoint_info"][idx]["name"])
        for idx in keypoint_ids
    }
    name2id: dict[str, int] = {name: idx for idx, name in id2name.items()}
    keypoint_connections: list[tuple[int, int]] = []
    for skeleton_info in coco_info["skeleton_info"].values():
        link: tuple[str, str] = tuple(skeleton_info["link"])
        if link[0] in name2id and link[1] in name2id:
            keypoint_connections.append((name2id[link[0]], name2id[link[1]]))

    return PoseSchema(
        name="coco133",
        class_label="COCO WholeBody 133",
        keypoint_ids=keypoint_ids,
        id2name=id2name,
        keypoint_connections=keypoint_connections,
    )


def get_pose_schema(name: KeypointSchemaName) -> PoseSchema:
    """Return cached pose schema metadata by name."""
    if name not in _pose_schema_cache:
        match name:
            case "coco133":
                _pose_schema_cache[name] = _build_coco133_schema()
            case "sapiens308":
                _pose_schema_cache[name] = _build_sapiens_schema()
    return _pose_schema_cache[name]


def project_keypoints_to_coco133(
    keypoints: Float32[ndarray, "sapiens_k 2"],
    scores: Float32[ndarray, "sapiens_k"],
) -> tuple[Float32[ndarray, "133 2"], Float32[ndarray, "133"]]:
    """Project native Sapiens/Goliath keypoints into COCO WholeBody 133 order.

    Args:
        keypoints: Native Sapiens/Goliath keypoints in image coordinates.
        scores: Native Sapiens/Goliath confidence scores.

    Returns:
        COCO WholeBody keypoints and scores. Missing mappings are filled with
        NaN coordinates and zero confidence.
    """
    meta: dict[str, Any] = get_sapiens_metainfo()
    mapping: dict[int, int] = {int(k): int(v) for k, v in meta["coco_wholebody_to_goliath_mapping"].items()}

    coco_keypoints: Float32[ndarray, "133 2"] = np.full((133, 2), np.nan, dtype=np.float32)
    coco_scores: Float32[ndarray, "133"] = np.zeros((133,), dtype=np.float32)
    keypoints_f32: Float32[ndarray, "sapiens_k 2"] = np.asarray(keypoints, dtype=np.float32)
    scores_f32: Float32[ndarray, "sapiens_k"] = np.asarray(scores, dtype=np.float32).reshape(-1)

    for coco_idx, sapiens_idx in mapping.items():
        if 0 <= coco_idx < 133 and 0 <= sapiens_idx < keypoints_f32.shape[0] and sapiens_idx < scores_f32.shape[0]:
            coco_keypoints[coco_idx] = keypoints_f32[sapiens_idx]
            coco_scores[coco_idx] = scores_f32[sapiens_idx]

    return coco_keypoints, coco_scores


def project_instances_to_schema(
    keypoints: list[Float32[ndarray, "sapiens_k 2"]],
    scores: list[Float32[ndarray, "sapiens_k"]],
    schema_name: KeypointSchemaName,
) -> tuple[list[Float32[ndarray, "schema_k 2"]], list[Float32[ndarray, "schema_k"]]]:
    """Project per-person keypoints and scores into the selected schema."""
    if schema_name == "sapiens308":
        keypoints_out: list[Float32[ndarray, "schema_k 2"]] = [np.asarray(kpts, dtype=np.float32) for kpts in keypoints]
        scores_out: list[Float32[ndarray, "schema_k"]] = [np.asarray(scr, dtype=np.float32).reshape(-1) for scr in scores]
        return keypoints_out, scores_out

    projected_keypoints: list[Float32[ndarray, "schema_k 2"]] = []
    projected_scores: list[Float32[ndarray, "schema_k"]] = []
    for kpts, scr in zip(keypoints, scores, strict=False):
        coco_keypoints: Float32[ndarray, "133 2"]
        coco_scores: Float32[ndarray, "133"]
        coco_keypoints, coco_scores = project_keypoints_to_coco133(kpts, scr)
        projected_keypoints.append(coco_keypoints)
        projected_scores.append(coco_scores)
    return projected_keypoints, projected_scores


def log_annotation_context(schema: PoseSchema, *, recording: rr.RecordingStream | None = None) -> None:
    """Log a Rerun annotation context for a pose schema."""
    rr.log(
        "/",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label=schema.class_label, color=(0, 255, 0)),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=idx, label=schema.id2name[idx])
                        for idx in schema.keypoint_ids
                    ],
                    keypoint_connections=schema.keypoint_connections,
                )
            ]
        ),
        static=True,
        recording=recording,
    )
