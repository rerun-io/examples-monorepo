"""Tyro entry point: score KeyNet weights on the frozen holdout (``data.keynet_holdout``), per dataset, camera, variant and prior.

Every model sees the same crops twice: with the stored clean extrapolated prior (tracked) and with zeros (acquisition, as after a DetNet
detection). Geometric scores use positive crops: mean keypoint error in crop pixels (the crop side is 96, the hand about 80 of it),
PCK at 5 % and 10 % of the crop side, the error in net-frame pixels (640-wide, through the crop's local affine, comparable with the
trainer's ``keynet/error_px``) and the d_rel error in mm. Presence uses every crop (threshold 0.5). Results go to a JSON report (one
row per model x dataset x camera-or-all x variant x prior) and, optionally, an .rrd of example crops with ground truth and predictions.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Bool, Float32, Int64
from serde import serde
from serde.json import to_json
from torch import Tensor

from handtrack.data.batches import CropKind
from handtrack.data.keynet_holdout import VARIANTS, KeyNetHoldout, read_holdout
from handtrack.data.keynet_rerun import KEYNET_DATASETS
from handtrack.data.stream import KeyNetSamples, select_samples
from handtrack.labels.heatmaps import decode_distance, decode_heatmaps
from handtrack.models.keynet import KeyNetF, keynet_for_state
from handtrack.pipeline import read_state

SHORT: dict[str, str] = {"dataforge-umetrack": "UmeTrack", "dataforge-show3d": "SHOW3D", "dataforge-hot3d-quest3": "HOT3D", "gsi-six-view": "GSI",
                         "dataforge-gen-humanego-subset": "Gen-HumanEgo"}


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ScoreRow:
    """One model's scores on one slice of the holdout."""

    model: str
    dataset: str
    camera: str
    """Camera entity, or ``all``."""
    variant: str
    prior: str
    """``tracked`` (the clean extrapolated prior) or ``zero`` (acquisition)."""
    positives: int
    crops: int
    error_crop_px: float | None
    pck05: float | None
    pck10: float | None
    error_net_px: float | None
    d_rel_mm: float | None
    presence_accuracy: float
    presence_recall: float | None
    presence_precision: float | None


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ScoreReport:
    holdout: str
    models: dict[str, str]
    """Model name -> weights path."""
    rows: list[ScoreRow]


@dataclass(frozen=True, slots=True)
class Config:
    holdout: Path
    """The frozen evaluation set (``tools/keynet_holdout.py``)."""
    weights: tuple[Path, ...]
    """``*.weights.pt`` files (each with its ``.sha256`` sidecar)."""
    names: tuple[str, ...]
    """One display name per weights file."""
    out: Path
    """The JSON report to write."""
    rrd: Path | None = None
    """Also write example crops (ground truth, every model's tracked and zero-prior keypoints) to this .rrd."""
    examples_per_dataset: int = 12
    device: str = "cuda"
    batch: int = 1024


@dataclass(frozen=True, slots=True)
class Predictions:
    points: Float32[Tensor, "n 21 2"]
    d_rel: Float32[Tensor, "n 21"]
    presence: Float32[Tensor, "n"]


def _load(path: Path, device: torch.device) -> KeyNetF:
    state, _ = read_state(path)
    model: KeyNetF = keynet_for_state(state)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict(model: KeyNetF, samples: KeyNetSamples, zero_prior: bool, device: torch.device, batch: int) -> Predictions:
    points: list[Tensor] = []
    distances: list[Tensor] = []
    presence: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, samples.crops.shape[0], batch):
            crops: Float32[Tensor, "b 1 96 96"] = (samples.crops[start : start + batch].to(device).float() / 255.0)[:, None]
            keypoints: Float32[Tensor, "b 63"] = samples.keypoints[start : start + batch].to(device)
            output = model(crops, torch.zeros_like(keypoints) if zero_prior else keypoints)
            points.append(decode_heatmaps(output.heatmaps.float())[0].cpu())
            distances.append(decode_distance(output.distance.float()).cpu())
            presence.append(output.presence_logit.float().sigmoid().cpu())
    return Predictions(torch.cat(points), torch.cat(distances), torch.cat(presence))


def score(name: str, dataset: str, camera: str, variant: str, prior: str, samples: KeyNetSamples, predicted: Predictions, chosen: Bool[Tensor, "n"]) -> ScoreRow:
    positive: Bool[Tensor, "n"] = chosen & (samples.kind == int(CropKind.POSITIVE))
    count: int = int(positive.sum())
    error: Float32[Tensor, "p 21"] = (predicted.points[positive] - samples.points_crop[positive]).norm(dim=-1)
    inverse: Float32[Tensor, "p 3 3"] = torch.linalg.inv(samples.crop_from_net[positive])
    net_delta: Float32[Tensor, "p 21 2"] = torch.einsum("bij,bkj->bki", inverse[:, :2, :2], predicted.points[positive] - samples.points_crop[positive])
    said: Bool[Tensor, "n"] = predicted.presence >= 0.5
    truth: Bool[Tensor, "n"] = samples.presence > 0.5
    true_positive: int = int((said & truth & chosen).sum())
    return ScoreRow(
        model=name, dataset=dataset, camera=camera, variant=variant, prior=prior, positives=count, crops=int(chosen.sum()),
        error_crop_px=float(error.mean()) if count else None, pck05=float((error < 0.05 * 96).float().mean()) if count else None,
        pck10=float((error < 0.10 * 96).float().mean()) if count else None, error_net_px=float(net_delta.norm(dim=-1).mean()) if count else None,
        d_rel_mm=float((predicted.d_rel[positive] - samples.d_rel_mm[positive]).abs().mean()) if count else None,
        presence_accuracy=float(((said == truth) & chosen).sum() / chosen.sum().clamp_min(1)),
        presence_recall=true_positive / int((truth & chosen).sum()) if int((truth & chosen).sum()) else None,
        presence_precision=true_positive / int((said & chosen).sum()) if int((said & chosen).sum()) else None,
    )


def main(config: Config) -> None:
    if len(config.weights) != len(config.names):
        raise ValueError("one name per weights file")
    device: torch.device = torch.device(config.device)
    holdout: KeyNetHoldout = read_holdout(config.holdout)
    samples: KeyNetSamples = holdout.samples
    rows: list[ScoreRow] = []
    predictions: dict[tuple[str, str], Predictions] = {}
    for name, path in zip(config.names, config.weights, strict=True):
        model: KeyNetF = _load(path, device)
        for prior in ("tracked", "zero"):
            predicted: Predictions = predict(model, samples, prior == "zero", device, config.batch)
            predictions[(name, prior)] = predicted
            for d, dataset in enumerate(holdout.dataset_names):
                for v, variant in enumerate(VARIANTS):
                    base: Bool[Tensor, "n"] = (samples.dataset == d) & (holdout.variant == v)
                    rows.append(score(name, SHORT.get(dataset, dataset), "all", variant, prior, samples, predicted, base))
                    if v == 0 and prior == "tracked":
                        for c, camera in enumerate(KEYNET_DATASETS[dataset].cameras):
                            rows.append(score(name, SHORT.get(dataset, dataset), camera, variant, prior, samples, predicted, base & (holdout.camera == c)))
        del model
    report: ScoreReport = ScoreReport(str(config.holdout), {n: str(p) for n, p in zip(config.names, config.weights, strict=True)}, rows)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    config.out.write_text(to_json(report))
    for row in rows:
        if row.camera == "all":
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dataclasses.asdict(row).items()}), flush=True)
    if config.rrd is not None:
        log_examples(config, holdout, predictions)


def log_examples(config: Config, holdout: KeyNetHoldout, predictions: dict[tuple[str, str], Predictions]) -> None:
    """Example positives of each dataset (both variants): the crop, ground truth (cyan) and each model's tracked keypoints."""
    import rerun as rr

    rec = rr.RecordingStream("handtrack-keynet-eval", recording_id="handtrack-keynet-eval")
    rec.save(config.rrd)
    palette: list[list[int]] = [[255, 80, 80], [120, 255, 120], [255, 200, 0], [200, 120, 255]]
    samples: KeyNetSamples = holdout.samples
    for d, dataset in enumerate(holdout.dataset_names):
        for v, variant in enumerate(VARIANTS):
            chosen: Int64[Tensor, "k"] = ((samples.dataset == d) & (holdout.variant == v) & (samples.kind == int(CropKind.POSITIVE))).nonzero()[:, 0]
            picked: Int64[Tensor, "k"] = chosen[torch.linspace(0, len(chosen) - 1, min(config.examples_per_dataset, len(chosen))).long()] if len(chosen) else chosen
            for k, index in enumerate(picked.tolist()):
                path: str = f"{SHORT.get(dataset, dataset)}/{variant}/{k}"
                one: KeyNetSamples = select_samples(samples, torch.tensor([index]))
                rec.log(path, rr.Image(one.crops[0].numpy()), static=True)
                rec.log(f"{path}/gt", rr.Points2D(one.points_crop[0].numpy(), radii=1.0, colors=[0, 220, 255]), static=True)
                for m, name in enumerate(config.names):
                    rec.log(f"{path}/{name}", rr.Points2D(predictions[(name, "tracked")].points[index].numpy(), radii=0.8, colors=palette[m % len(palette)]), static=True)
    rec.flush()
