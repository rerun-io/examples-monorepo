"""The frozen KeyNet evaluation set: perspective crops from each dataset's held-out segments, written once as one Parquet table.

Segments: UmeTrack / SHOW3D validation users, every labelled HOT3D Quest 3 scene (never trained on), and the frozen detector holdout
of GSI and Gen-HumanEgo (``configs/detector_holdout.json``). Each chunk is cut twice with ``stream.evaluation_augment`` (ground-truth
crop cameras, the clean extrapolated prior, no jitter, presence negatives kept): variant 0 as deployed, variant 1 rolled by a uniform
angle on the whole circle. Every model is then scored on the same bytes.
"""

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from jaxtyping import Int64
from numpy import ndarray
from torch import Tensor

from handtrack.data.array_table import read_arrays, write_arrays
from handtrack.data.batches import KeyNetBatch
from handtrack.data.catalog import list_segments, select_split
from handtrack.data.keynet_rerun import keynet_finish, stream_cuts
from handtrack.data.rerun_source import DATASET_SPECS
from handtrack.data.stream import KeyNetAugment, KeyNetSamples, evaluation_augment, sample_count, select_samples
from handtrack.train.source import KeyNetValidation

HOLDOUT_DATASETS: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d", "dataforge-hot3d-quest3", "gsi-six-view", "dataforge-gen-humanego-subset")
VARIANTS: tuple[str, ...] = ("deployed", "rotated")
DATASETS_KEY: bytes = b"keynet_datasets"
"""Schema metadata of a holdout (and of ``export_keynet``'s table): the dataset names ``dataset`` indexes, one per line."""


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


def build_holdout(datasets: tuple[str, ...], held_out: dict[str, list[str]], catalog_url: str, device: torch.device, crops_per_dataset: int,
                  seed: int = 0) -> KeyNetHoldout:
    """Decode each dataset's evaluation segments in small shuffled blocks (many segments early) until ``crops_per_dataset`` crops of
    variant 0 exist; both variants of the same frames are kept."""
    base: KeyNetAugment = evaluation_augment(KeyNetAugment(negative_margin=0.25))
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
    write_arrays(partial, arrays, {DATASETS_KEY: "\n".join(holdout.dataset_names).encode()})
    partial.replace(path)


def read_holdout(path: Path) -> KeyNetHoldout:
    arrays: dict[str, ndarray] = read_arrays(path)
    names: tuple[str, ...] = tuple((pq.read_schema(path).metadata or {}).get(DATASETS_KEY, b"").decode().split())
    unknown: list[str] = [name for name in names if name not in DATASET_SPECS]
    if unknown or not names:
        raise ValueError(f"{path}: datasets {names} in its {DATASETS_KEY.decode()} schema metadata ({unknown} unknown)")
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
