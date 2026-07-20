"""RTMPose / RTMW top-down 2D pose estimators (SimCC decode).

ONNX artifacts are the OpenMMLab SDK deploy zips rtmlib uses, run GPU-resident
through ONNX Runtime IOBinding or a cached TensorRT engine. Preprocessing
replicates rtmlib exactly (cv2 top-down affine, BGR input, rtmlib's RGB-ordered
mean/std applied to BGR channels), so results match the reference pipeline.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from posekit.artifacts import fetch_openmmlab_onnx
from posekit.models.base import TopDownPose2d
from posekit.ops.crops import IMAGENET_MEAN_255, IMAGENET_STD_255, CropBatch, CropSpec, crop_coords_to_image, crop_frames
from posekit.ops.decode import decode_simcc
from posekit.predictions import BoxDetections, Keypoints2d
from posekit.runtimes import OnnxBackendConfig, OnnxOrTrtBackendConfig, TensorRuntime, create_runtime_from_onnx
from posekit.runtimes.base import run_chunked
from posekit.skeletons import COCO_17, COCO_133, KeypointSkeleton

RtmPoseVariant = Literal["rtmpose-m-coco17", "rtmpose-x-coco17", "rtmw-x-coco133"]

RTMPOSE_ONNX_ZIP_URLS: dict[RtmPoseVariant, str] = {
    "rtmpose-m-coco17": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    "rtmpose-x-coco17": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
    "rtmw-x-coco133": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip",
}

RTMPOSE_SKELETONS: dict[RtmPoseVariant, KeypointSkeleton] = {
    "rtmpose-m-coco17": COCO_17,
    "rtmpose-x-coco17": COCO_17,
    "rtmw-x-coco133": COCO_133,
}



@dataclass(frozen=True, slots=True)
class RtmPoseConfig:
    """RTMPose/RTMW top-down 2D pose configuration."""

    variant: RtmPoseVariant = "rtmw-x-coco133"
    """Model checkpoint to run (defines skeleton and crop size)."""
    backend: OnnxOrTrtBackendConfig = field(default_factory=OnnxBackendConfig)
    """Accelerated backend running the pose graph."""
    padding: float = 1.25
    """Bbox padding multiplier before cropping."""

    def setup(self) -> "RtmPose2d":
        """Fetch artifacts, build/load the runtime, and return a ready estimator."""
        return RtmPose2d(self)


class RtmPose2d(TopDownPose2d):
    """Batched GPU RTMPose/RTMW estimator over an ONNX/TensorRT runtime."""

    def __init__(self, config: RtmPoseConfig) -> None:
        """Fetch the ONNX artifact and create the backend runtime.

        Args:
            config: Variant, backend, and crop padding.
        """
        self.config: RtmPoseConfig = config
        self.skeleton = RTMPOSE_SKELETONS[config.variant]
        self.runtime: TensorRuntime = create_runtime_from_onnx(fetch_openmmlab_onnx(RTMPOSE_ONNX_ZIP_URLS[config.variant]), config.backend)
        input_shape: tuple[int, ...] = self.runtime.spec.inputs[0].shape
        # rtmlib applies the RGB-ordered ImageNet constants to BGR crops; replicated verbatim.
        self.crop_spec: CropSpec = CropSpec(
            input_size=(int(input_shape[2]), int(input_shape[1])),
            padding=config.padding,
            align="cv2",
            bgr=True,
            mean_rgb=IMAGENET_MEAN_255,
            std_rgb=IMAGENET_STD_255,
        )
        output_names: tuple[str, ...] = tuple(spec.name for spec in self.runtime.spec.outputs)
        if "simcc_x" in output_names and "simcc_y" in output_names:
            self._simcc_x_name, self._simcc_y_name = "simcc_x", "simcc_y"
        else:
            # Fall back to bin-count heuristics: x bins track crop width.
            by_bins = sorted(self.runtime.spec.outputs, key=lambda spec: spec.shape[-1])
            self._simcc_y_name, self._simcc_x_name = by_bins[0].name, by_bins[1].name

    @torch.inference_mode()
    def __call__(self, frames_rgb: UInt8[Tensor, "b h w 3"], detections: BoxDetections) -> Keypoints2d:
        """Estimate keypoints for every detection.

        Args:
            frames_rgb: uint8 RGB NHWC frame batch on the inference device.
            detections: Instance boxes referencing ``frames_rgb`` by index.

        Returns:
            Image-space keypoints, one instance per detection row.
        """
        crop_batch: CropBatch = crop_frames(
            frames_rgb,
            frame_indices=detections.frame_indices,
            bboxes_xyxy=detections.xyxy,
            spec=self.crop_spec,
            dtype=self.runtime.spec.inputs[0].dtype,
        )
        if int(crop_batch.inputs.shape[0]) == 0:
            return Keypoints2d.empty(self.skeleton, frames_rgb.device)
        outputs: dict[str, Tensor] = run_chunked(self.runtime, {self.runtime.spec.inputs[0].name: crop_batch.inputs})
        xy_crop, scores = decode_simcc(outputs[self._simcc_x_name], outputs[self._simcc_y_name], simcc_split_ratio=2.0)
        xy_image: Float[Tensor, "n k 2"] = crop_coords_to_image(
            xy_crop, centers=crop_batch.centers, scales=crop_batch.scales, input_size=self.crop_spec.input_size
        )
        return Keypoints2d(xy_image, scores, detections.frame_indices, self.skeleton)
