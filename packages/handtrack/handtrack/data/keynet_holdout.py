"""The frozen KeyNet evaluation set: perspective crops cut from catalog frames fixed in code (``keynet_holdout_cuts``).

Segments: UmeTrack / SHOW3D validation users, every labelled HOT3D Quest 3 scene (never trained on), and the frozen detector holdout
of GSI and Gen-HumanEgo (``configs/detector_holdout.json``). Each chunk is cut twice with ``stream.evaluation_augment`` (ground-truth
crop cameras, the clean extrapolated prior, no jitter, presence negatives kept): variant 0 as deployed, variant 1 rolled by a uniform
angle on the whole circle. Every model is then scored on the same crops, cut anew from the catalog at the start of each run.
"""

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Int64, UInt8
from torch import Tensor

from handtrack.data.batches import KeyNetBatch
from handtrack.data.catalog import list_segments, select_split
from handtrack.data.keynet_holdout_cuts import HOLDOUT_CUTS, HOLDOUT_SEGMENTS
from handtrack.data.keynet_rerun import NVDEC_SLICE, KeyNetCutter, keynet_finish
from handtrack.data.luma import grey_from_rgb
from handtrack.data.rerun_source import DATASET_SPECS, NOMINAL_FPS
from handtrack.data.stream import KeyNetAugment, KeyNetSamples, evaluation_augment, sample_count, select_samples
from handtrack.train.source import KeyNetValidation

HOLDOUT_DATASETS: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d", "dataforge-hot3d-quest3", "gsi-six-view", "dataforge-gen-humanego-subset")
VARIANTS: tuple[str, ...] = ("deployed", "rotated")


def eval_segments(name: str, held_out: dict[str, list[str]], catalog_url: str) -> list[str]:
    """The segments a dataset is scored on (see the module docstring)."""
    import rerun as rr

    if name in held_out:
        return sorted(held_out[name])
    dataset = rr.catalog.CatalogClient(catalog_url).get_dataset(name)
    split = "test" if name == "dataforge-hot3d-quest3" else "val"
    return [info.segment_id for info in select_split(list_segments(dataset, name), split)]


@dataclass(frozen=True, slots=True)
class KeyNetHoldout:
    """The evaluation crops (CPU tensors) and where each came from."""

    samples: KeyNetSamples
    dataset_names: tuple[str, ...]
    """``samples.dataset`` indexes this tuple."""
    camera: Int64[Tensor, "n"]
    """Camera index within its dataset (``rerun_source.DATASET_SPECS``)."""
    variant: Int64[Tensor, "n"]
    """Index into ``VARIANTS``."""


def holdout_augments() -> list[KeyNetAugment]:
    """``VARIANTS``: the evaluation recipe as deployed, then the same rolled by a uniform angle on the whole circle."""
    base: KeyNetAugment = evaluation_augment(KeyNetAugment(negative_margin=0.25))
    return [base, dataclasses.replace(base, max_rotation=math.pi)]


def load_holdout(catalog_url: str, device: torch.device) -> KeyNetHoldout:
    """Cut the frozen holdout from the catalog: decode every (segment, camera)'s ``HOLDOUT_CUTS`` frames once, then replay the cuts in
    their recorded order on the build's generator (CUDA, seed 0), so the random draws (negatives, roll angles) fall as they did."""
    import rerun as rr
    from simplecv.rerun_dataloader import open_segment_decoder

    if device.type != "cuda":  # a CPU generator draws another sequence from the same seed: another benchmark
        raise ValueError(f"the frozen KeyNet holdout is cut on CUDA (NVDEC, and the build's CUDA generator), not {device}")

    client = rr.catalog.CatalogClient(catalog_url)
    parts: list[KeyNetSamples] = []
    cameras: list[Tensor] = []
    variants: list[Tensor] = []
    for index, name in enumerate(HOLDOUT_DATASETS):
        dataset = client.get_dataset(name)
        segments: tuple[str, ...] = HOLDOUT_SEGMENTS[name]
        wanted: dict[tuple[str, int], list[int]] = {}
        for cut in HOLDOUT_CUTS[name]:
            wanted.setdefault((segments[cut.segment], cut.camera), []).extend(cut.frames)
        images: dict[tuple[str, int, int], Tensor] = {}
        for (segment, camera), frames in wanted.items():  # as SegmentNvdecDecoder.decode: frame_index -> sample, NVDEC_SLICE frames a call
            entity: str = f"{DATASET_SPECS[name][0][camera]}/pinhole/video".lstrip("/")
            times, _, _, decoder = open_segment_decoder(dataset, segment, entity, "frame_index", device, NOMINAL_FPS)
            values: list[int] = sorted(set(frames))
            positions: list[int] = (np.searchsorted(times, values, side="right") - 1).tolist()
            for first in range(0, len(values), NVDEC_SLICE):
                grey: Tensor = grey_from_rgb(decoder.get_frames_at(positions[first : first + NVDEC_SLICE]).data)
                images.update({(segment, camera, value): image for value, image in zip(values[first : first + NVDEC_SLICE], grey.cpu(), strict=True)})
        cutter = KeyNetCutter(dataset, name, index, holdout_augments(), device, torch.Generator(device=device).manual_seed(0))
        for cut in HOLDOUT_CUTS[name]:
            segment = segments[cut.segment]
            rows: Int64[Tensor, "m"] = torch.from_numpy(cutter.truth(segment).row_of_frame[list(cut.frames)])
            chunk: UInt8[Tensor, "m h w"] = torch.stack([images[(segment, cut.camera, value)] for value in cut.frames])
            for samples, camera, variant in cutter.cut(segment, cut.camera, chunk, rows):
                parts.append(KeyNetSamples(**{f.name: getattr(samples, f.name).cpu() for f in dataclasses.fields(samples)}))
                cameras.append(torch.full((sample_count(samples),), camera, dtype=torch.int64))
                variants.append(torch.full((sample_count(samples),), variant, dtype=torch.int64))
        print(f"holdout {name}: {len(HOLDOUT_CUTS[name])} cuts from {len(segments)} segments", flush=True)
    merged: KeyNetSamples = KeyNetSamples(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(parts[0])})
    return KeyNetHoldout(merged, HOLDOUT_DATASETS, torch.cat(cameras), torch.cat(variants))


class HoldoutSource:
    """A validation ``BatchSource`` over one variant of a ``KeyNetHoldout``, with exact metadata (``KeyNetValidationSource``)."""

    def __init__(self, holdout: KeyNetHoldout, batch_size: int, device: torch.device | str, variant: int = 0) -> None:
        chosen: Int64[Tensor, "m"] = (holdout.variant == variant).nonzero()[:, 0]
        # A seeded permutation: the file is one dataset after another, so any prefix (a capped validation pass) covers every dataset.
        order: Int64[Tensor, "m"] = chosen[torch.randperm(len(chosen), generator=torch.Generator().manual_seed(0))]
        self.samples: KeyNetSamples = select_samples(holdout.samples, order)
        self.samples = KeyNetSamples(**{f.name: getattr(self.samples, f.name).to(device) for f in dataclasses.fields(self.samples)})
        self.batch_size: int = batch_size
        self.position: int = 0
        self.last: KeyNetSamples | None = None
        self.finish = keynet_finish(None)
        self.generator: torch.Generator = torch.Generator(device=device).manual_seed(0)

    def start_epoch(self, epoch: int) -> None:
        del epoch
        self.position = 0

    def next_keynet_batch(self) -> KeyNetBatch | None:
        total: int = sample_count(self.samples)
        if self.position >= total:
            return None
        index: Int64[Tensor, "b"] = torch.arange(self.position, min(total, self.position + self.batch_size), device=self.samples.crops.device)
        self.position += self.batch_size
        self.last = select_samples(self.samples, index)
        return self.finish(self.last, self.generator)

    def next_detnet_batch(self) -> None:
        raise RuntimeError("the KeyNet holdout feeds KeyNet only")

    def keynet_validation(self) -> KeyNetValidation:
        if self.last is None:
            raise RuntimeError("keynet_validation before the first batch")
        return KeyNetValidation(crop_from_net=self.last.crop_from_net, points_crop=self.last.points_crop, distance_mm=self.last.d_rel_mm)

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        pass
