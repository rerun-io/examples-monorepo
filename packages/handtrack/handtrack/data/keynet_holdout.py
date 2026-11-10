"""The frozen KeyNet evaluation set: perspective crops from each dataset's held-out segments, written once as one Parquet table.

Segments: UmeTrack and SHOW3D validation users/subjects (``catalog.select_split(..., "val")``), every labelled HOT3D Quest 3 scene
(``"test"``; never trained on), and the frozen detector holdout of GSI and Gen-HumanEgo (``configs/detector_holdout.json``). Each
chunk of decoded frames is cut twice with the evaluation recipe (``stream.evaluation_augment``: ground-truth crop cameras, the clean
extrapolated prior, no jitter): variant 0 as deployed (the dataset's crop roll), variant 1 rolled by a uniform angle on the whole
circle (robustness to cameras cut without a roll). The presence negatives stay. Every model is then scored on the same bytes.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Int64
from numpy import ndarray
from torch import Tensor

from handtrack.data.array_table import read_arrays, write_arrays
from handtrack.data.batches import CropKind, KeyNetBatch
from handtrack.data.keynet_rerun import KEYNET_DATASETS, keynet_finish, stream_cuts
from handtrack.data.stream import KeyNetAugment, KeyNetSamples, evaluation_augment, sample_count, select_samples
from handtrack.train.source import KeyNetValidation

HOLDOUT_DATASETS: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d", "dataforge-hot3d-quest3", "gsi-six-view", "dataforge-gen-humanego-subset")
VARIANTS: tuple[str, ...] = ("deployed", "rotated")


def eval_segments(name: str, held_out: dict[str, list[str]], catalog_url: str) -> list[str]:
    """The segments a dataset is scored on (see the module docstring)."""
    import rerun as rr

    from handtrack.data.catalog import list_segments, select_split

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
    """Camera index within its dataset (``KEYNET_DATASETS[name].cameras``)."""
    variant: Int64[Tensor, "n"]
    """Index into ``VARIANTS``."""


def build_holdout(datasets: tuple[str, ...], held_out: dict[str, list[str]], catalog_url: str, device: torch.device, crops_per_dataset: int,
                  seed: int = 0, negative_margin: float = 0.25) -> KeyNetHoldout:
    """Decode each dataset's evaluation segments in small shuffled blocks (many segments early) until ``crops_per_dataset`` crops of
    variant 0 exist; both variants of the same frames are kept."""
    base: KeyNetAugment = evaluation_augment(KeyNetAugment(negative_margin=negative_margin))
    augments: list[KeyNetAugment] = [base, dataclasses.replace(base, max_rotation=math.pi)]
    parts: list[KeyNetSamples] = []
    cameras: list[Tensor] = []
    variants: list[Tensor] = []
    for index, name in enumerate(datasets):
        segments: list[str] = eval_segments(name, held_out, catalog_url)
        cuts = stream_cuts(name, index, segments, catalog_url=catalog_url, device=device, augments=augments, seed=seed, endless=False, fetch_block_size=8, chunk=8)
        count: int = 0
        for samples, camera, variant in cuts(seed):
            cpu: KeyNetSamples = KeyNetSamples(**{f.name: getattr(samples, f.name).cpu() for f in dataclasses.fields(samples)})
            parts.append(cpu)
            cameras.append(torch.full((sample_count(cpu),), camera, dtype=torch.int64))
            variants.append(torch.full((sample_count(cpu),), variant, dtype=torch.int64))
            count += sample_count(cpu) if variant == 0 else 0
            if count >= crops_per_dataset and variant == len(augments) - 1:
                break
        print(f"holdout {name}: {count} crops per variant from {len(segments)} segments", flush=True)
    merged: KeyNetSamples = KeyNetSamples(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(parts[0])})
    return KeyNetHoldout(merged, datasets, torch.cat(cameras), torch.cat(variants))


def write_holdout(path: Path, holdout: KeyNetHoldout) -> None:
    arrays: dict[str, ndarray] = {f.name: getattr(holdout.samples, f.name).numpy() for f in dataclasses.fields(holdout.samples)}
    arrays["camera"] = holdout.camera.numpy()
    arrays["variant"] = holdout.variant.numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial: Path = path.with_suffix(".partial.parquet")
    write_arrays(partial, arrays)
    partial.replace(path)
    path.with_suffix(".datasets.txt").write_text("\n".join(holdout.dataset_names) + "\n")


def read_holdout(path: Path) -> KeyNetHoldout:
    arrays: dict[str, ndarray] = read_arrays(path)
    names: tuple[str, ...] = tuple(path.with_suffix(".datasets.txt").read_text().split())
    unknown: list[str] = [name for name in names if name not in KEYNET_DATASETS]
    if unknown:
        raise ValueError(f"{path}: unknown datasets {unknown}")
    fields: list[str] = [f.name for f in dataclasses.fields(KeyNetSamples)]
    samples: KeyNetSamples = KeyNetSamples(**{name: torch.from_numpy(np.ascontiguousarray(arrays[name])) for name in fields})
    return KeyNetHoldout(samples, names, torch.from_numpy(arrays["camera"]), torch.from_numpy(arrays["variant"]))


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


def positive_count(holdout: KeyNetHoldout) -> dict[str, int]:
    """Positive crops of variant 0 per dataset (a report line)."""
    positive: Bool[Tensor, "n"] = (holdout.samples.kind == int(CropKind.POSITIVE)) & (holdout.variant == 0)
    return {name: int((positive & (holdout.samples.dataset == i)).sum()) for i, name in enumerate(holdout.dataset_names)}
