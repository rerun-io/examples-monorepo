"""PromptDA model queries streamed straight from the Rerun dataloader.

One ``RerunIterableDataset`` carries every model input as a ``Field``: the AV1
video (NVDEC-decoded to CUDA frames by simplecv's ``SegmentNvdecDecoder``) plus
the LiDAR prompt depth and the numeric pose/calibration columns, all riding the
same fetch query. PromptDA is a per-frame model, so unlike the causal multiview
stereo pipeline in ``mvs`` the collate here is stateless apart from its position
on the sampling grid: it stacks whichever rows arrived into one device-resident
batch and hands it to the TensorRT engine.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import torch
from arkitscenes_download.ingest.paths import CONFIDENCE, DEPTH, PINHOLE_WIDE, RIG, TIMELINE, VIDEO_WIDE
from einops import rearrange
from jaxtyping import Float32, Float64, UInt8, UInt16
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.experimental.dataloader import (
    DataSource,
    Field,
    FixedRateSampling,
    ImageDecoder,
    NoShuffle,
    NumericDecoder,
    RerunIterableDataset,
)
from simplecv.rerun_dataloader import RECOMMENDED_FETCH_BLOCK_SIZE, SegmentNvdecDecoder
from torch import Tensor

from rerun_prompt_da.apis.arkitscenes_shared import world_t_cam_from_pose

NATIVE_FPS: float = 60.0
"""Frame rate of the stored wide-camera AV1 stream."""
class PromptDARow(TypedDict):
    """One raw Rerun dataloader sample; keys match ``promptda_dataset``'s fields.

    Only ``video`` is optional: a decoder returns None to signal missing data, and of
    the three decoders here only ``SegmentNvdecDecoder`` does so, for grid slots before
    a segment's first packet. ``ImageDecoder`` and ``NumericDecoder`` have no None path
    — they return an *empty* tensor when the sampling grid overshoots a segment edge, so
    the shapes below describe a populated slot and the collate guards both edges.

    The video frame is device-resident (NVDEC); the rest arrive on the CPU.
    """

    video: UInt8[Tensor, "3 h w"] | None
    depth: UInt16[Tensor, "1 prompt_h prompt_w"]
    conf: UInt8[Tensor, "n"]
    k: Float32[Tensor, "k=9"]
    pose_t: Float32[Tensor, "xyz=3"]
    pose_q: Float32[Tensor, "xyzw=4"]


@dataclass(frozen=True, slots=True)
class PromptDABatch:
    """One batch of complete PromptDA queries, oriented landscape for the network."""

    frame_indices: list[int]
    """Zero-based sampling-grid index per row."""
    timestamps_ns: list[int]
    """Sampling-grid timestamp per row, in nanoseconds."""
    quarter_turns: int
    """Counter-clockwise quarter turns applied to reach landscape; undo them when logging."""
    rgb_bhw3: UInt8[Tensor, "b h w 3"]
    """Device-resident landscape RGB frames."""
    prompt_bhw: Float32[Tensor, "b prompt_h=192 prompt_w=256"]
    """Device-resident landscape LiDAR prompt depth, in metres."""
    prompt_mm_bhw: UInt16[ndarray, "b stored_prompt_h stored_prompt_w"]
    """Prompt depth in millimetres, in the stored orientation, for logging."""
    confidence_bhw: UInt8[ndarray, "b stored_prompt_h stored_prompt_w"]
    """ARKit confidence in the stored orientation, for fusion filtering."""
    K_native_b33: Float32[ndarray, "b 3 3"]
    """Row-major native intrinsics in the stored orientation."""
    world_T_cam_b44: Float64[ndarray, "b 4 4"]
    """Camera-to-world transforms."""
    stored_hw: tuple[int, int]
    """Native frame (height, width) as stored, before any orientation unbake."""


def promptda_dataset(dataset: DatasetEntry, segment_id: str, target_fps: float, device: torch.device) -> tuple[RerunIterableDataset, SegmentNvdecDecoder]:
    """Build the iterable dataset whose samples carry every PromptDA input column.

    Args:
        dataset: Rerun catalog dataset entry.
        segment_id: Segment identifier to stream.
        target_fps: Inference rate; the sampling grid is built at this rate, so
            frames between grid slots are never fetched or decoded.
        device: CUDA device that receives decoded frames.

    Returns:
        A natural-order iterable dataset over the segment's inference grid, and its
        video decoder (whose raw AV1 samples are relayable as a Rerun VideoStream).
    """
    video_decoder: SegmentNvdecDecoder = SegmentNvdecDecoder(dataset, VIDEO_WIDE, TIMELINE, device, int(NATIVE_FPS))
    fields: dict[str, Field] = {
        "video": Field(f"/{VIDEO_WIDE}:VideoStream:sample", decode=video_decoder),
        "depth": Field(f"/{DEPTH}:EncodedDepthImage:blob", decode=ImageDecoder()),
        "conf": Field(f"/{CONFIDENCE}:SegmentationImage:buffer", decode=NumericDecoder()),
        "k": Field(f"/{PINHOLE_WIDE}:Pinhole:image_from_camera", decode=NumericDecoder()),
        "pose_t": Field(f"/{RIG}:Transform3D:translation", decode=NumericDecoder()),
        "pose_q": Field(f"/{RIG}:Transform3D:quaternion", decode=NumericDecoder()),
    }
    samples: RerunIterableDataset = RerunIterableDataset(
        DataSource(dataset=dataset, segments=[segment_id]),
        index=TIMELINE,
        fields=fields,
        timeline_sampling=FixedRateSampling(rate_hz=target_fps),
        shuffle_strategy=NoShuffle(),
        fetch_block_size=RECOMMENDED_FETCH_BLOCK_SIZE,
    )
    return samples, video_decoder


class PromptDACollate:
    """Stateful collate_fn: raw Rerun dataloader samples in, one ``PromptDABatch`` out.

    The state is only the position on the sampling grid (the dataloader ships no
    index with a sample), so this requires natural order: ``num_workers=0`` with
    ``shuffle_strategy=NoShuffle()``. The NVDEC decoder is CUDA-stateful and would
    not survive worker processes either.
    """

    def __init__(
        self,
        samples: RerunIterableDataset,
        device: torch.device,
        quarter_turns: int | None = None,
        timestamp_step_ns: int | None = None,
    ) -> None:
        """Bind the collate to one dataset's sampling grid.

        Args:
            samples: The dataset this collate consumes (provides the grid timing).
            device: Inference device the assembled batch lands on.
            quarter_turns: Counter-clockwise rotation to landscape, or None to infer
                a quarter turn from a portrait frame's shape.
            timestamp_step_ns: Output timeline step, or None to use the sampling
                grid step. The register tool supplies the legacy native-grid step.
        """
        segment = samples.sample_index.segments[0]
        ns_per_sample: int | None = samples.sample_index.ns_per_sample
        assert ns_per_sample is not None  # FixedRateSampling on a temporal timeline always sets it
        self._index_start_ns: int = segment.index_start
        self._ns_per_sample: int = ns_per_sample if timestamp_step_ns is None else timestamp_step_ns
        self._device: torch.device = device
        self._quarter_turns: int | None = quarter_turns
        self._grid_index: int = -1

    def __call__(self, batch: list[PromptDARow]) -> PromptDABatch | None:
        """Stack the rows that carry every field into one device-resident query."""
        frame_indices: list[int] = []
        timestamps_ns: list[int] = []
        frames_3hw: list[UInt8[Tensor, "3 stored_h stored_w"]] = []
        prompts_hw: list[UInt16[Tensor, "stored_prompt_h stored_prompt_w"]] = []
        confidences_hw: list[UInt8[Tensor, "stored_prompt_h stored_prompt_w"]] = []
        k_matrices_33: list[Float32[ndarray, "3 3"]] = []
        world_T_cam_44: list[Float64[ndarray, "4 4"]] = []
        row: PromptDARow
        for row in batch:
            self._grid_index += 1
            # Read loose, then bind shaped: an empty tensor would fail a shape-pinned
            # annotation before the guard below could skip it (beartype checks annotated
            # locals, though not TypedDict fields). See PromptDARow for the two edges.
            video: Tensor | None = row["video"]
            depth: Tensor = row["depth"]
            conf: Tensor = row["conf"]
            k: Tensor = row["k"]
            pose_t: Tensor = row["pose_t"]
            pose_q: Tensor = row["pose_q"]
            if video is None:
                continue
            if min(video.numel(), depth.numel(), conf.numel(), k.numel(), pose_t.numel(), pose_q.numel()) == 0:
                continue
            frame_chw: UInt8[Tensor, "3 stored_h stored_w"] = video
            depth_1hw: UInt16[Tensor, "1 stored_prompt_h stored_prompt_w"] = depth
            confidence_n: UInt8[Tensor, "n"] = conf
            k_9: Float32[Tensor, "k=9"] = k
            pose_t_3: Float32[Tensor, "xyz=3"] = pose_t
            pose_q_4: Float32[Tensor, "xyzw=4"] = pose_q
            frame_indices.append(self._grid_index)
            timestamps_ns.append(self._index_start_ns + self._grid_index * self._ns_per_sample)
            frames_3hw.append(frame_chw)
            prompts_hw.append(rearrange(depth_1hw, "1 h w -> h w"))
            confidence_hw: UInt8[Tensor, "stored_prompt_h stored_prompt_w"] = confidence_n.reshape(depth_1hw.shape[1:])
            confidences_hw.append(confidence_hw)
            # Rerun stores Pinhole image_from_camera flattened column-major.
            k_matrices_33.append(np.asarray(k_9.numpy()).reshape(3, 3).T)
            world_T_cam_44.append(world_t_cam_from_pose(np.asarray(pose_t_3.numpy()), np.asarray(pose_q_4.numpy())))

        if not frames_3hw:
            return None
        return assemble_promptda_batch(
            frame_indices,
            timestamps_ns,
            frames_3hw,
            prompts_hw,
            confidences_hw,
            k_matrices_33,
            world_T_cam_44,
            device=self._device,
            quarter_turns=self._quarter_turns,
        )


def assemble_promptda_batch(
    frame_indices: list[int],
    timestamps_ns: list[int],
    frames_3hw: list[UInt8[Tensor, "3 stored_h stored_w"]],
    prompts_hw: list[UInt16[Tensor, "stored_prompt_h stored_prompt_w"]],
    confidences_hw: list[UInt8[Tensor, "stored_prompt_h stored_prompt_w"]],
    k_matrices_33: list[Float32[ndarray, "3 3"]],
    world_T_cam_44: list[Float64[ndarray, "4 4"]],
    *,
    device: torch.device,
    quarter_turns: int | None,
) -> PromptDABatch:
    """Stack per-frame inputs into one landscape-oriented, device-resident batch.

    Args:
        frame_indices: Source index per row (sampling-grid slot or packet index).
        timestamps_ns: Output timeline timestamp per row, in nanoseconds.
        frames_3hw: Decoded RGB frames in stored orientation.
        prompts_hw: uint16 millimetre prompt depth per row, stored orientation.
        confidences_hw: ARKit confidence per row, stored orientation.
        k_matrices_33: Row-major native intrinsics per row.
        world_T_cam_44: Camera-to-world transform per row.
        device: Inference device the RGB and prompt tensors land on.
        quarter_turns: Counter-clockwise turns to landscape, or None to infer one
            quarter turn from a tall frame shape.
    """
    stored_hw: tuple[int, int] = (int(frames_3hw[0].shape[1]), int(frames_3hw[0].shape[2]))
    prompt_stored_bhw: UInt16[Tensor, "b stored_prompt_h stored_prompt_w"] = torch.stack(prompts_hw).to(device, non_blocking=True)
    confidence_stored_bhw: UInt8[Tensor, "b stored_prompt_h stored_prompt_w"] = torch.stack(confidences_hw)
    # A tall frame would have its aspect ratio squashed into the engine's fixed
    # landscape input. The frame shape decides this, so no orientation property is
    # read — those are spelled differently across dataset generations.
    turns: int = quarter_turns if quarter_turns is not None else (1 if stored_hw[0] > stored_hw[1] else 0)
    rgb_b3hw: UInt8[Tensor, "b 3 h w"] = torch.rot90(torch.stack(frames_3hw).to(device), turns, dims=(2, 3))
    prompt_bhw: UInt16[Tensor, "b prompt_h=192 prompt_w=256"] = torch.rot90(prompt_stored_bhw, turns, dims=(1, 2))
    return PromptDABatch(
        frame_indices=frame_indices,
        timestamps_ns=timestamps_ns,
        quarter_turns=turns,
        rgb_bhw3=rearrange(rgb_b3hw, "b c h w -> b h w c"),
        prompt_bhw=prompt_bhw.float() / 1000.0,
        prompt_mm_bhw=prompt_stored_bhw.cpu().numpy().astype(np.uint16),
        confidence_bhw=confidence_stored_bhw.cpu().numpy().astype(np.uint8),
        K_native_b33=np.stack(k_matrices_33),
        world_T_cam_b44=np.stack(world_T_cam_44),
        stored_hw=stored_hw,
    )
