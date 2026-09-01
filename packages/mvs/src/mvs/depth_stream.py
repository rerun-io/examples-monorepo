"""Depth-model queries straight from the Rerun dataloader.

One ``RerunIterableDataset`` carries every model input as a ``Field``: the AV1
video (NVDEC-decoded to CUDA frames) plus the six numeric geometry columns, all
riding the same fetch query. A stateful ``collate_fn`` composes each sample's
pose and s1 intrinsics and assembles the causal depth query — source-view
selection gathers seven scattered past keyframes, which no ``Field`` can express
(windows ship contiguous ranges only), so it lives at the collate boundary.
"""

from dataclasses import dataclass
from typing import TypedDict, cast

import torch
from einops import rearrange
from jaxtyping import Float
from kornia.geometry.liegroup import Se3
from kornia.geometry.quaternion import Quaternion
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    NoShuffle,
    NumericDecoder,
    RerunIterableDataset,
)
from simplecv.rerun_dataloader import SegmentNvdecDecoder
from torch import Tensor

from mvs.apis.live_mesh import FETCH_BLOCK_SIZE, NATIVE_FPS, TIMELINE, VIDEO_WIDE
from mvs.depth_engine import DepthInputs, preprocess_image, s1_intrinsics
from mvs.pose_stream import (
    CAM_QUATERNION,
    CAM_TRANSLATION,
    IMAGE_FROM_CAMERA,
    NUM_SOURCE_VIEWS,
    RESOLUTION,
    RIG_QUATERNION,
    RIG_TRANSLATION,
    Keyframe,
    KeyframeBuffer,
)


class DepthRow(TypedDict):
    """One raw Rerun dataloader sample; keys match ``depth_dataset``'s fields."""

    video: Tensor | None
    rig_quaternion: Tensor | None
    rig_translation: Tensor | None
    cam_quaternion: Tensor | None
    cam_translation: Tensor | None
    image_from_camera: Tensor | None
    resolution: Tensor | None


@dataclass(frozen=True, slots=True)
class DepthSample:
    """One video-grid frame with its geometry and, once warmed up, a ready model query."""

    frame_index: int
    """Zero-based video frame index within the segment."""
    timestamp_ns: int
    """Sampling-grid timestamp in nanoseconds."""
    world_T_cam_44: Float[Tensor, "4 4"]
    """CPU camera-to-world transform."""
    K_native_33: Float[Tensor, "3 3"]
    """CPU native-resolution row-major intrinsics."""
    resolution_wh: tuple[int, int]
    """Native image resolution."""
    inputs: DepthInputs | None
    """Complete device-resident depth query, or None while the keyframe buffer warms up."""


def _se3(quat_xyzw_4: Float[Tensor, "4"], translation_3: Float[Tensor, "3"]) -> Se3:
    """Build an Se3 from Rerun's xyzw quaternion convention (kornia wants wxyz)."""
    x, y, z, w = quat_xyzw_4
    quat_wxyz_14: Float[Tensor, "1 4"] = torch.stack([w, x, y, z])[None]
    return Se3(Quaternion(quat_wxyz_14).normalize(), translation_3[None])


def depth_dataset(
    dataset: DatasetEntry, segment_id: str, device: torch.device
) -> tuple[RerunIterableDataset, SegmentNvdecDecoder]:
    """Build the iterable dataset whose samples carry every depth-model input column.

    Args:
        dataset: Rerun catalog dataset entry.
        segment_id: Segment identifier to stream.
        device: CUDA device that receives decoded frames.

    Returns:
        A natural-order iterable dataset over the segment's video grid, and its video
        decoder (whose raw AV1 samples are relayable as a Rerun VideoStream).
    """
    video_decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(dataset, VIDEO_WIDE, TIMELINE, device, int(NATIVE_FPS))
    fields: dict[str, Field] = {
        "video": Field(f"/{VIDEO_WIDE}:VideoStream:sample", decode=video_decoder),
        "rig_quaternion": Field(RIG_QUATERNION, decode=NumericDecoder()),
        "rig_translation": Field(RIG_TRANSLATION, decode=NumericDecoder()),
        "cam_quaternion": Field(CAM_QUATERNION, decode=NumericDecoder()),
        "cam_translation": Field(CAM_TRANSLATION, decode=NumericDecoder()),
        "image_from_camera": Field(IMAGE_FROM_CAMERA, decode=NumericDecoder()),
        "resolution": Field(RESOLUTION, decode=NumericDecoder()),
    }
    samples: RerunIterableDataset = RerunIterableDataset(
        DataSource(dataset=dataset, segments=[segment_id]),
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=NATIVE_FPS),
        shuffle_strategy=NoShuffle(),
        fetch_block_size=FETCH_BLOCK_SIZE,
    )
    return samples, video_decoder


class DepthCollate:
    """Stateful collate_fn: raw Rerun dataloader samples in, ``DepthSample``s out.

    Holds the causal keyframe buffer and the grid position, so it requires the
    streaming-inference regime: ``batch_size=1``, ``num_workers=0``, natural order.
    Training over precomputed tuples would use a stateless batching collate instead.
    """

    def __init__(self, samples: RerunIterableDataset, device: torch.device) -> None:
        """Bind the collate to one dataset's sampling grid.

        Args:
            samples: The dataset this collate consumes (provides the grid timing).
            device: Inference device for assembled queries.
        """
        segment = samples.sample_index.segments[0]
        ns_per_sample: int | None = samples.sample_index.ns_per_sample
        assert ns_per_sample is not None  # FixedRateSampling on a temporal timeline always sets it
        self._index_start_ns: int = segment.index_start
        self._ns_per_sample: int = ns_per_sample
        self._device: torch.device = device
        self._keyframes: KeyframeBuffer = KeyframeBuffer()
        self._grid_index: int = -1

    def __call__(self, batch: list[DepthRow]) -> DepthSample | None:
        """Compose one grid slot's geometry and, once warmed up, its depth query."""
        row: DepthRow = batch[0]
        self._grid_index += 1
        frame_chw: Tensor | None = row["video"]
        if frame_chw is None:
            return None
        rig_quaternion: Tensor | None = row["rig_quaternion"]
        rig_translation: Tensor | None = row["rig_translation"]
        cam_quaternion: Tensor | None = row["cam_quaternion"]
        cam_translation: Tensor | None = row["cam_translation"]
        image_from_camera: Tensor | None = row["image_from_camera"]
        resolution: Tensor | None = row["resolution"]
        if (
            rig_quaternion is None
            or rig_translation is None
            or cam_quaternion is None
            or cam_translation is None
            or image_from_camera is None
            or resolution is None
        ):
            raise ValueError(f"Grid slot {self._grid_index} is missing pose or calibration data.")

        # cast: kornia types Se3.__mul__ as a union over its operand kinds.
        world_T_cam: Se3 = cast(
            "Se3",
            _se3(rig_quaternion.reshape(4), rig_translation.reshape(3))
            * _se3(cam_quaternion.reshape(4), cam_translation.reshape(3)),
        )
        world_T_cam_44: Float[Tensor, "4 4"] = world_T_cam.matrix()[0]
        cam_T_world_44: Float[Tensor, "4 4"] = world_T_cam.inverse().matrix()[0]
        # PinholeProjection is stored column-major; consumers need row-major matrices.
        K_native_33: Float[Tensor, "3 3"] = image_from_camera.reshape(3, 3).T.to(torch.float32)
        resolution_wh: tuple[int, int] = (int(resolution.reshape(2)[0]), int(resolution.reshape(2)[1]))
        K_s1_144, invK_s1_144 = s1_intrinsics(K_native_33[None], resolution_wh)

        image_3hw: Float[Tensor, "c=3 h=384 w=512"] = preprocess_image(frame_chw)
        source_keyframes: list[Keyframe] = self._keyframes.select_sources(world_T_cam_44)
        inputs: DepthInputs | None = None
        if len(source_keyframes) == NUM_SOURCE_VIEWS:
            inputs = DepthInputs(
                cur_image_b3hw=rearrange(image_3hw, "c h w -> 1 c h w"),
                src_image_bm3hw=rearrange(
                    torch.stack([keyframe.image_3hw for keyframe in source_keyframes]), "m c h w -> 1 m c h w"
                ),
                src_K_bm44=torch.stack([keyframe.K_s1_44 for keyframe in source_keyframes])[None].to(self._device),
                cur_invK_b44=invK_s1_144.to(self._device),
                src_cam_T_world_bm44=torch.stack([keyframe.cam_T_world_44 for keyframe in source_keyframes])[None].to(
                    self._device
                ),
                cur_world_T_cam_b44=world_T_cam_44[None].to(self._device),
            )
        self._keyframes.add(Keyframe(self._grid_index, world_T_cam_44, cam_T_world_44, K_s1_144[0], image_3hw))
        return DepthSample(
            frame_index=self._grid_index,
            timestamp_ns=self._index_start_ns + self._grid_index * self._ns_per_sample,
            world_T_cam_44=world_T_cam_44,
            K_native_33=K_native_33,
            resolution_wh=resolution_wh,
            inputs=inputs,
        )
