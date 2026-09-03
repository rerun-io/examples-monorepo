"""Helpers shared by the catalog-dataloader and raw-disk ARKitScenes PromptDA tools."""

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
from arkitscenes_download.ingest.depth import ArkitDepthConfidence
from beartype.roar import BeartypeException
from einops import rearrange
from jaxtyping import Float, Float32, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import CatalogClient
from scipy.spatial.transform import Rotation
from simplecv.ops.depth import quantize_depth_m_to_mm
from torch import Tensor

from rerun_prompt_da.trt_predictor import PromptDATrtPredictor, postprocess_depth, preprocess_batch

ARKITSCENES_DATASET = "arkitscenes"
NATIVE_FPS = 60.0


def stride_for(native_fps: float, target_fps: float) -> int:
    """Return the closest positive whole-frame stride for a target rate."""
    if target_fps <= 0.0:
        raise ValueError("target_fps must be greater than zero")
    return max(1, round(native_fps / target_fps))


def world_t_cam_from_pose(
    translation: Float[ndarray, "3"], quaternion_xyzw: Float[ndarray, "4"]
) -> Float[ndarray, "4 4"]:
    """Build a camera-to-world transform from an ARKit pose.

    Args:
        translation: World-space camera translation with shape ``(3,)``.
        quaternion_xyzw: Camera rotation as an xyzw quaternion with shape ``(4,)``.

    Returns:
        Homogeneous ``world_T_cam`` transform with shape ``(4, 4)``.
    """
    world_t_cam_44: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    world_t_cam_44[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    world_t_cam_44[:3, 3] = translation
    return world_t_cam_44


def filter_depth_for_fusion(
    depth_mm: UInt16[ndarray, "h w"], confidence: UInt8[ndarray, "h2 w2"], max_depth_meter: float
) -> UInt16[ndarray, "h w"]:
    """Mask low-confidence and out-of-range predictions before fusion.

    Args:
        depth_mm: Predicted uint16 depth in millimetres with shape ``(H, W)``.
        confidence: ARKit uint8 confidence values with shape ``(H2, W2)``.
        max_depth_meter: Furthest depth retained for fusion, in metres.

    Returns:
        Filtered uint16 depth with the same shape as ``depth_mm``.
    """
    confidence_hw: UInt8[ndarray, "h w"] = np.asarray(
        cv2.resize(confidence, (depth_mm.shape[1], depth_mm.shape[0]), interpolation=cv2.INTER_NEAREST), dtype=np.uint8
    )
    filtered_depth_mm: UInt16[ndarray, "h w"] = depth_mm.copy()
    filtered_depth_mm[confidence_hw < ArkitDepthConfidence.MEDIUM] = 0
    filtered_depth_mm[depth_mm > max_depth_meter * 1000.0] = 0
    return filtered_depth_mm


def connect_catalog(catalog_url: str, dataset_name: str) -> CatalogClient:
    """Connect to the local ARKitScenes catalog or terminate with setup guidance."""
    try:
        client: CatalogClient = CatalogClient(catalog_url)
        dataset_names: list[str] = client.dataset_names()
    except BeartypeException:
        raise
    except Exception as error:
        raise SystemExit(f"catalog not reachable at {catalog_url} — start it with `pixi run arkitscenes-download-serve`") from error
    if dataset_name not in dataset_names:
        raise SystemExit(f"dataset {dataset_name!r} is absent — create it with `pixi run arkitscenes-download-register`")
    return client


def segments_to_process(
    rows: list[dict[str, Any]],
    video_id: str | None,
    process_all: bool,
    layer_name: str,
    is_processable: Callable[[str], bool] | None = None,
) -> list[str]:
    """Select catalog segment ids according to the requested execution mode.

    An explicit ``video_id`` is always processed (replacing any existing layer);
    ``process_all`` selects segments still missing ``layer_name``, additionally
    filtered by ``is_processable`` when the tool needs local data to exist.
    """
    if (video_id is None) == (not process_all):
        raise SystemExit("give exactly one of --video-id or --process-all")
    available_ids: list[str] = [str(row["rerun_segment_id"]) for row in rows]
    if video_id is not None:
        if video_id not in available_ids:
            raise SystemExit(f"video id {video_id!r} is absent; available ids: {', '.join(available_ids)}")
        return [video_id]
    selected: list[str] = []
    for row in rows:
        segment_id: str = str(row["rerun_segment_id"])
        if layer_name in (row.get("rerun_layer_names") or []):
            continue
        if is_processable is not None and not is_processable(segment_id):
            continue
        selected.append(segment_id)
    return selected


def run_promptda_batch(
    predictor: PromptDATrtPredictor,
    rgb_bhw3: UInt8[Tensor, "b h w 3"],
    prompt_bhw: Float32[Tensor, "b 192 256"],
    output_hw: tuple[int, int],
) -> tuple[UInt16[ndarray, "b oh ow"], UInt16[ndarray, "b nh nw"]]:
    """Run one PromptDA batch and convert both predictions to uint16 millimetres.

    Args:
        predictor: Cached TensorRT PromptDA predictor.
        rgb_bhw3: Landscape RGB batch with shape ``(B, H, W, 3)``.
        prompt_bhw: Prompt depth in metres with shape ``(B, 192, 256)``.
        output_hw: Full ``(H, W)`` resolution for the logged depth.

    Returns:
        Full-resolution depth for logging and network-resolution depth for
        fusion, both in millimetres.
    """
    image_b3hw: Float32[Tensor, "b 3 nh nw"]
    prompt_b1hw: Float32[Tensor, "b 1 192 256"]
    image_b3hw, prompt_b1hw = preprocess_batch(rgb_bhw3, prompt_bhw, predictor.image_hw)
    depth_model_b1hw: Float32[Tensor, "b 1 nh nw"] = predictor.runtime({"image": image_b3hw, "prompt_depth": prompt_b1hw})["depth"]
    depth_bhw: Float32[Tensor, "b oh ow"] = postprocess_depth(depth_model_b1hw, output_hw)
    # Scale and cast on the GPU: halves the device-to-host copy and drops two full-size host passes.
    depth_mm_bhw: UInt16[ndarray, "b oh ow"] = quantize_depth_m_to_mm(depth_bhw).cpu().numpy()
    depth_model_mm_bhw: UInt16[ndarray, "b nh nw"] = (
        quantize_depth_m_to_mm(
            rearrange(depth_model_b1hw, "b 1 h w -> b h w")
        )
        .cpu()
        .numpy()
    )
    return depth_mm_bhw, depth_model_mm_bhw
