"""WiLoR 2D hand keypoints for one known hand box, on posekit's WiLoR model.

mv-api refines body-model hand keypoints with WiLoR: it already knows each
hand's box and side, so it skips the hand detector. Its crops keep the
geometry of the former ``wilor_nano.hand_keypoints`` (integer centre and size,
float image); posekit runs the network and the full-image camera lift.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from jaxtyping import Float, Float32, UInt8
from numpy import ndarray
from posekit.models.wilor import CROP_SIZE, WilorHandPose3d, WilorHandPoseConfig, WilorHands
from posekit.models.wilor_net.geometry import generate_image_patch_cv2
from posekit.predictions import BoxDetections, HandDetections, HandSide
from skimage.filters import gaussian
from torch import Tensor


@dataclass(frozen=True, slots=True)
class WilorKeypointRefinerConfig:
    """WiLoR model used to refine one hand at a time."""

    pose: WilorHandPoseConfig = field(default_factory=WilorHandPoseConfig)
    """posekit WiLoR configuration (torch fp16 reference by default)."""


def hand_patch(rgb_hw3: UInt8[ndarray, "h w 3"], xyxy: Float32[ndarray, "4"], side: HandSide, rescale_factor: float) -> Float[ndarray, "256 256 3"]:
    """WiLoR's square hand crop: integer centre and size, blurred first when it downsamples, mirrored for left hands.

    Only the source window the crop samples is converted to float and blurred: its bilinear
    neighbours plus the Gaussian's reach (scipy ``truncate=4``) around them, so every sampled
    pixel equals the full-frame blur. A 1080p frame took ~55 ms per hand before.

    Args:
        rgb_hw3: Source RGB image.
        xyxy: Hand box in source pixels.
        side: Anatomical side; left hands are mirrored.
        rescale_factor: Box enlargement before the crop.

    Returns:
        The 256x256 float crop (float64 after a blur, float32 otherwise).
    """
    center: Float32[ndarray, "2"] = (xyxy[2:4] + xyxy[0:2]) / 2.0
    box_size: float = float((rescale_factor * (xyxy[2:4] - xyxy[0:2])).max())
    center_x, center_y, size = int(center[0]), int(center[1]), int(box_size)
    # Blur before downsampling to avoid aliasing, as WiLoR's demo does.
    downsampling: float = box_size / CROP_SIZE / 2.0
    sigma: float = (downsampling - 1) / 2 if downsampling > 1.1 else 0.0
    margin: int = int(4.0 * sigma + 0.5) + 2
    height, width = rgb_hw3.shape[:2]
    x0, y0 = max(center_x - size // 2 - margin, 0), max(center_y - size // 2 - margin, 0)
    x1, y1 = min(center_x + size // 2 + margin + 1, width), min(center_y + size // 2 + margin + 1, height)
    image: Float[ndarray, "wh ww 3"] = rgb_hw3[y0:y1, x0:x1].astype(np.float32)
    if sigma > 0.0:
        image = gaussian(image, sigma=sigma, channel_axis=2, preserve_range=True)
    patch, _ = generate_image_patch_cv2(
        image, center_x - x0, center_y - y0, size, size, CROP_SIZE, CROP_SIZE, side == "left", 1.0, 0, border_mode=cv2.BORDER_CONSTANT
    )
    return patch


class WilorKeypointRefiner:
    """RGB frame + hand box + side -> 21 WiLoR keypoints in source pixels (MediaPipe order)."""

    def __init__(self, config: WilorKeypointRefinerConfig | None = None) -> None:
        self.config: WilorKeypointRefinerConfig = config or WilorKeypointRefinerConfig()
        self.model: WilorHandPose3d = self.config.pose.setup()

    @torch.inference_mode()
    def __call__(self, rgb_hw3: UInt8[ndarray, "h w 3"], xyxy: Float32[ndarray, "4"], side: HandSide) -> Float32[ndarray, "21 2"]:
        """Project WiLoR's 3D hand for the hand inside ``xyxy``.

        Args:
            rgb_hw3: Source RGB image.
            xyxy: Hand box in source pixels.
            side: Anatomical side; left hands run mirrored.

        Returns:
            The 21 hand keypoints in source pixels.
        """
        patch: Float[ndarray, "256 256 3"] = hand_patch(rgb_hw3, xyxy, side, self.config.pose.rescale_factor)
        # float32, or float64 after the blur; the model casts it to fp16 once, as before the port.
        crops: Float[Tensor, "1 256 256 3"] = torch.from_numpy(np.ascontiguousarray(patch)[None]).cuda()
        detections = HandDetections(
            BoxDetections(
                torch.from_numpy(xyxy[None].astype(np.float32)).cuda(),
                torch.ones((1,), dtype=torch.float32, device="cuda"),
                torch.zeros((1,), dtype=torch.int64, device="cuda"),
            ),
            (side,),
        )
        hands: WilorHands = self.model.hands_from_crops(crops, detections, (int(rgb_hw3.shape[1]), int(rgb_hw3.shape[0])))
        return hands.keypoints.xy[0].cpu().numpy()


def best_hand_per_side(detections: HandDetections, frame: int) -> dict[HandSide, Float32[ndarray, "4"]]:
    """The most confident box of each side in one frame (single-person tools assume one left, one right).

    Args:
        detections: Hand detections of a frame batch.
        frame: Batch index of the frame to select from.

    Returns:
        ``xyxy`` per side that has at least one box in ``frame``.
    """
    boxes: Float32[ndarray, "n 4"] = detections.boxes.xyxy_numpy()
    scores: Float32[ndarray, "n"] = detections.boxes.scores.cpu().numpy().astype(np.float32)
    frames: ndarray = detections.boxes.frame_indices.cpu().numpy()
    best: dict[HandSide, Float32[ndarray, "4"]] = {}
    for side in ("left", "right"):
        rows: ndarray = np.flatnonzero((frames == frame) & np.asarray([s == side for s in detections.sides], dtype=bool))
        if len(rows):
            best[side] = boxes[rows[np.argmax(scores[rows])]]
    return best


class RtmPoseHandScorer:
    """Per-keypoint hand confidences from RTMPose-m hand5 (rtmlib, ONNX Runtime), which WiLoR does not predict."""

    URL: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
    """The OpenMMLab SDK zip the former wilor_nano.hand_keypoints used."""

    def __init__(self) -> None:
        import onnxruntime as ort
        from rtmlib.tools.pose_estimation.rtmpose import RTMPose

        device: str = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        self.model = RTMPose(onnx_model=self.URL, model_input_size=(256, 256), device=device)

    def __call__(self, rgb_hw3: UInt8[ndarray, "h w 3"], xyxy: Float32[ndarray, "4"]) -> Float32[ndarray, "21"]:
        """Confidence of each of the 21 hand keypoints inside ``xyxy``."""
        scores: ndarray = self.model(rgb_hw3, bboxes=[xyxy.tolist()])[1]
        return np.asarray(scores[0], dtype=np.float32)
