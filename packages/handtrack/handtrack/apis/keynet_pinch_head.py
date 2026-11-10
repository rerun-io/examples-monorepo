"""Tyro entry point: fit KeyNet-F's pinch head on frozen weights, from the catalog stream's mesh-contact labels.

The deployed pinch head was a linear probe fitted after training (2026-09-30): Linear(160, 1) on the pooled fused features of a frozen
KeyNet-F, BCE on the mesh-contact distance (``labels.pinch``'s band masked, positives weighted 4x). Its crop caches are gone, so this
cuts crops anew (``stream_cuts(pinch=True)``) from UmeTrack and SHOW3D training segments and scores on their validation users, against
the weights' carried head and optionally a reference net's own head. Every other weight is unchanged.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Float32
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor
from torch.nn import functional as F

from handtrack.apis.train import DetectorHoldout, training_segments
from handtrack.data.batches import CropKind
from handtrack.data.catalog import CATALOG_URL
from handtrack.data.keynet_holdout import eval_segments
from handtrack.data.keynet_rerun import stream_cuts
from handtrack.data.stream import KeyNetAugment, evaluation_augment
from handtrack.labels.pinch import CONTACT_CLOSED_MM, CONTACT_OPEN_MM
from handtrack.models.keynet import KeyNetF, keynet_from_state
from handtrack.pipeline import read_state
from handtrack.train.checkpoint import export_weights


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PinchScores:
    crops: int
    pinch_share: float
    precision: float
    recall: float
    f1: float


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PinchHeadReport:
    weights: str
    train_crops: int
    new_head: PinchScores
    carried_head: PinchScores | None
    reference: PinchScores | None
    """The reference net and its own head on its own features (``Config.reference``)."""


@dataclass(frozen=True, slots=True)
class Config:
    weights: Path
    """Trained KeyNet-F weights (``*.weights.pt`` with its ``.sha256``); a pinch head may already be there (it is replaced)."""
    out: Path
    """Output directory: ``best.weights.pt`` (+ ``.sha256``, the input state with the new ``pinch_head.*``) and ``pinch_head.json``."""
    reference: Path | None = None
    """Also score this net with its own pinch head on the same validation crops (e.g. the deployed KeyNet)."""
    datasets: tuple[str, ...] = ("dataforge-umetrack", "dataforge-show3d")
    train_crops: int = 60_000
    """Labelled positive crops (outside the 5-15 mm band) to fit on, split evenly across ``datasets``."""
    val_crops: int = 6_000
    epochs: int = 3
    lr: float = 0.01
    pos_weight: float = 4.0
    holdout: Path = Path("configs/detector_holdout.json")
    catalog_url: str = CATALOG_URL
    device: str = "cuda"
    seed: int = 0


def labelled_crops(segments: dict[str, list[str]], count: int, augment: KeyNetAugment, config: Config, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """(crops uint8, keypoint inputs, contact mm) of positive crops with a label outside the ambiguous band, ``count`` split over datasets."""
    crops: list[Tensor] = []
    inputs: list[Tensor] = []
    contacts: list[Tensor] = []
    share: int = math.ceil(count / len(segments))
    for index, (name, names) in enumerate(segments.items()):
        have: int = 0
        cuts = stream_cuts(name, index, names, catalog_url=config.catalog_url, device=device, augments=[augment], seed=config.seed, fetch_block_size=64, chunk=16, pinch=True)
        for samples, _, _ in cuts(config.seed):
            contact: Float32[Tensor, "n"] = samples.pinch_contact_mm
            keep: Bool[Tensor, "n"] = (samples.kind == int(CropKind.POSITIVE)) & torch.isfinite(contact) & ((contact < CONTACT_CLOSED_MM) | (contact > CONTACT_OPEN_MM))
            crops.append(samples.crops[keep].cpu())
            inputs.append(samples.keypoints[keep].cpu())
            contacts.append(contact[keep].cpu())
            have += int(keep.sum())
            if have >= share:
                break
        print(f"pinch crops {name}: {have}", flush=True)
    return torch.cat(crops)[:count], torch.cat(inputs)[:count], torch.cat(contacts)[:count]


def features(model: KeyNetF, crops: Tensor, inputs: Tensor, device: torch.device, batch: int = 256) -> Float32[Tensor, "n 160"]:
    """The pooled fused features every head reads (``KeyNetF.forward`` up to the heads)."""
    out: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, crops.shape[0], batch):
            image: Tensor = model.image((crops[start : start + batch].to(device).float() / 255.0)[:, None])
            keys: Tensor = model.keypoints(inputs[start : start + batch].to(device)).reshape(-1, 32, 12, 12)
            out.append(model.fused(torch.cat((image, keys), dim=1)).mean(dim=(2, 3)).float())
    return torch.cat(out)


def scores(logits: Float32[Tensor, "n"], contact: Float32[Tensor, "n"]) -> PinchScores:
    truth: Bool[Tensor, "n"] = contact < CONTACT_CLOSED_MM
    said: Bool[Tensor, "n"] = logits >= 0.0
    tp: int = int((truth & said).sum())
    fp: int = int((~truth & said).sum())
    fn: int = int((truth & ~said).sum())
    return PinchScores(crops=int(truth.numel()), pinch_share=float(truth.float().mean()), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                       f1=2 * tp / max(2 * tp + fp + fn, 1))


def main(config: Config) -> None:
    device: torch.device = torch.device(config.device)
    held: DetectorHoldout = from_json(DetectorHoldout, config.holdout.read_text())
    model: KeyNetF = keynet_from_state(read_state(config.weights)[0]).to(device).eval()
    train_segments: dict[str, list[str]] = {name: training_segments(name, held, config.catalog_url) for name in config.datasets}
    val_segments: dict[str, list[str]] = {name: eval_segments(name, held.segments, config.catalog_url) for name in config.datasets}
    train_crops, train_inputs, train_contact = labelled_crops(train_segments, config.train_crops, KeyNetAugment(negative_margin=0.25), config, device)
    val_crops, val_inputs, val_contact = labelled_crops(val_segments, config.val_crops, evaluation_augment(KeyNetAugment(negative_margin=0.25)), config, device)
    train_features: Float32[Tensor, "n 160"] = features(model, train_crops, train_inputs, device)
    val_features: Float32[Tensor, "v 160"] = features(model, val_crops, val_inputs, device)
    torch.manual_seed(config.seed)
    head: torch.nn.Linear = torch.nn.Linear(160, 1).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=1e-4)
    target: Float32[Tensor, "n 1"] = (train_contact < CONTACT_CLOSED_MM).float()[:, None].to(device)
    weight: Tensor = torch.tensor([config.pos_weight], device=device)
    for epoch in range(config.epochs):
        order: Tensor = torch.randperm(train_features.shape[0], device=device)
        for start in range(0, len(order), 1024):
            chosen: Tensor = order[start : start + 1024]
            loss: Tensor = F.binary_cross_entropy_with_logits(head(train_features[chosen]), target[chosen], pos_weight=weight)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        for group in optimiser.param_groups:
            group["lr"] *= 0.3
        print(f"pinch head epoch {epoch}: loss {float(loss):.4f}", flush=True)
    with torch.inference_mode():
        new: PinchScores = scores(head(val_features)[:, 0].cpu(), val_contact)
        carried: PinchScores | None = None if model.pinch_head is None else scores(model.pinch_head(val_features)[:, 0].cpu(), val_contact)
    reference: PinchScores | None = None
    if config.reference is not None:
        other: KeyNetF = keynet_from_state(read_state(config.reference)[0]).to(device).eval()
        if other.pinch_head is not None:
            with torch.inference_mode():
                reference = scores(other.pinch_head(features(other, val_crops, val_inputs, device))[:, 0].cpu(), val_contact)
    report: PinchHeadReport = PinchHeadReport(str(config.weights), int(train_features.shape[0]), new, carried, reference)
    config.out.mkdir(parents=True, exist_ok=True)
    (config.out / "pinch_head.json").write_text(to_json(report))
    model.pinch_head = head
    export_weights(config.out / "best.weights.pt", model.cpu())
    print(to_json(report), flush=True)
