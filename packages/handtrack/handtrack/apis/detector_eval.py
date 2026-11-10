"""Score DetNet-F checkpoints side by side on held-out frames: each frame is decoded once (NVDEC) and every model sees it.

Sets: the frozen holdout episodes of the 2D-keypoint datasets (``configs/detector_holdout.json``) and a seeded sample of
the base datasets' held-out splits (UmeTrack test, real domain; SHOW3D val; HOT3D-Quest3, never trained on). Labels and
letterboxes are the training stream's (``handtrack.data.rerun_source``), unaugmented. Writes ``detector_eval.json``
(``DetectorEvalReport``) and ``misses.rrd``: hands the first model misses and its alarms on hand-free frames, with every
model's boxes.
"""
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Bool, Float32, Int64
from serde import serde
from serde.json import from_json, to_json
from torch import Tensor

from handtrack.apis.train import DetectorHoldout
from handtrack.data.catalog import CATALOG_URL, SplitName, list_segments, select_split
from handtrack.data.rerun_source import nvdec_producer, source_spec
from handtrack.data.stream import DetNetSamples
from handtrack.eval.boxes import MIN_IOU, NET_SCALE, PRESENCE_THRESHOLD, BoxCounts, box_counts, square_iou
from handtrack.models.detnet import DetNetF
from handtrack.pipeline import load_weights

BASE_SPLITS: dict[str, SplitName] = {"dataforge-umetrack": "test", "dataforge-show3d": "val", "dataforge-hot3d-quest3": "test"}
"""Base datasets: the catalog split each one is scored on (none of them is trained on)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetectorEvalConfig:
    """Score DetNet-F weights files (``*.weights.pt`` with ``.sha256`` sidecars) on the detector's held-out frames."""

    names: tuple[str, ...]
    """One label per model, in report order; the first model picks the misses."""
    weights: tuple[Path, ...]
    """Model-only state_dicts, one per name."""
    out_dir: Path
    """Receives detector_eval.json and misses.rrd."""
    datasets: tuple[str, ...] = ("gsi-six-view", "dataforge-gen-humanego-subset", "dataforge-umetrack", "dataforge-show3d", "dataforge-hot3d-quest3")
    holdout: Path = Path("configs/detector_holdout.json")
    base_segments: int = 48
    """Seeded sample size of each base dataset's split (the holdout datasets use every held-out episode)."""
    seed: int = 0
    misses_per_dataset: int = 48
    """Examples of each kind (missed hand, false alarm) kept for misses.rrd."""
    upright: bool = False
    """Score on upright frames (``catalog.UPRIGHT_TURNS``); False is the legacy input of repro-1 and the deployed DetNet-F. Each
    model is scored in the convention it was trained with, so a run compares models of one convention."""
    catalog_url: str = CATALOG_URL
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.weights) or len(set(self.names)) != len(self.names):
            raise ValueError("need one unique name per weights file")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SideCounts:
    both: BoxCounts = field(default_factory=BoxCounts)
    left: BoxCounts = field(default_factory=BoxCounts)
    right: BoxCounts = field(default_factory=BoxCounts)

    def __add__(self, other: "SideCounts") -> "SideCounts":
        return SideCounts(self.both + other.both, self.left + other.left, self.right + other.right)


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DatasetResult:
    segments: list[str]
    images: int
    models: dict[str, SideCounts]


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetectorEvalReport:
    weights: dict[str, str]
    """Model name -> weights path and sha256 ("<path> <sha256>")."""
    datasets: dict[str, DatasetResult]


def eval_segments(name: str, holdout: DetectorHoldout, catalog_url: str, base_segments: int, seed: int) -> list[str]:
    """The held-out episodes of a 2D-keypoint dataset, or a seeded sample of a base dataset's held-out split."""
    if name in holdout.segments:
        return sorted(holdout.segments[name])
    if name not in BASE_SPLITS:
        raise ValueError(f"{name}: neither in the holdout file nor a base dataset ({sorted(BASE_SPLITS)})")
    import rerun as rr

    infos = select_split(list_segments(rr.catalog.CatalogClient(catalog_url).get_dataset(name), name), BASE_SPLITS[name])
    ids: list[str] = sorted(info.segment_id for info in infos if name != "dataforge-umetrack" or info.domain == "real")
    return sorted(random.Random(seed).sample(ids, min(base_segments, len(ids))))


def side_counts(predicted: Float32[Tensor, "n 2 3"], probability: Float32[Tensor, "n 2"], samples: DetNetSamples) -> SideCounts:
    """Both sides together, then each side alone (a side's swaps need the other side, so they appear in ``both`` only)."""
    scored: Bool[Tensor, "n 2"] = samples.presence_mask
    one: list[BoxCounts] = []
    for side in (0, 1):
        only: Bool[Tensor, "n 2"] = torch.zeros_like(scored)
        only[:, side] = scored[:, side]
        one.append(box_counts(predicted, probability, samples.circle, samples.presence, only))
    return SideCounts(box_counts(predicted, probability, samples.circle, samples.presence, scored), one[0], one[1])


@dataclass(slots=True)
class Examples:
    """Missed hands and hand-free false alarms of each model (kind ``missed/<model>``, ``false_alarm/<model>``), with every model's
    prediction on the same image."""

    pooled: list[Tensor] = field(default_factory=list)
    truth: list[Tensor] = field(default_factory=list)
    present: list[Tensor] = field(default_factory=list)
    predicted: list[Tensor] = field(default_factory=list)
    """[models, 2, 3] circles."""
    probability: list[Tensor] = field(default_factory=list)
    kind: list[str] = field(default_factory=list)
    camera: list[int] = field(default_factory=list)

    def take(self, kind: str, rows: Int64[Tensor, "k"], samples: DetNetSamples, predicted: Float32[Tensor, "m n 2 3"],
             probability: Float32[Tensor, "m n 2"], limit: int) -> None:
        for row in rows[: max(limit - self.kind.count(kind), 0)].tolist():
            self.pooled.append(samples.pooled[row].cpu())
            self.truth.append(samples.circle[row].cpu())
            self.present.append((samples.presence_mask[row] & (samples.presence[row] > 0.5)).cpu())
            self.predicted.append(predicted[:, row].cpu())
            self.probability.append(probability[:, row].cpu())
            self.kind.append(kind)
            self.camera.append(int(samples.camera[row]))


def log_examples(path: Path, names: tuple[str, ...], examples: dict[str, Examples]) -> None:
    """misses.rrd: one image per example on the ``example`` timeline; ground truth green, models in fixed colours."""
    import rerun as rr
    import rerun.blueprint as rrb

    colours: list[tuple[int, int, int]] = [(255, 80, 80), (80, 160, 255), (255, 200, 0), (200, 120, 255), (0, 220, 220)]
    rec = rr.RecordingStream("hand-detector-eval")
    rec.save(path)
    pooled_scale: Float32[Tensor, "3"] = torch.tensor((160.0, 120.0, 160.0))
    for dataset, found in examples.items():
        for index in range(len(found.kind)):
            rec.set_time("example", sequence=found.kind[:index].count(found.kind[index]))  # numbered within its kind
            base: str = f"{dataset}/{found.kind[index]}"
            rec.log(f"{base}/image", rr.Image(found.pooled[index].numpy(), color_model="L"))
            truth: Tensor = found.truth[index] * pooled_scale
            shown: list[int] = [s for s in (0, 1) if bool(found.present[index][s])]
            rec.log(f"{base}/image/truth", rr.Boxes2D(centers=truth[shown, :2].numpy(), half_sizes=truth[shown, 2:].repeat(1, 2).numpy(),
                                                      labels=[f"truth {'LR'[s]}" for s in shown], colors=[(0, 220, 0)]))
            for m, name in enumerate(names):
                circle: Tensor = found.predicted[index][m] * pooled_scale
                keep: list[int] = [s for s in (0, 1) if float(found.probability[index][m, s]) > PRESENCE_THRESHOLD]
                rec.log(f"{base}/image/{name}", rr.Boxes2D(centers=circle[keep, :2].numpy(), half_sizes=circle[keep, 2:].repeat(1, 2).numpy(),
                                                           labels=[f"{name} {'LR'[s]} {float(found.probability[index][m, s]):.2f}" for s in keep],
                                                           colors=[colours[m % len(colours)]]))
            rec.log(f"{base}/camera", rr.TextDocument(f"camera id {found.camera[index]}"))
    views = [rrb.Spatial2DView(origin=f"{dataset}/{kind}/image", name=f"{dataset}: {kind}") for dataset, found in examples.items()
             for kind in sorted(set(found.kind))]
    rec.send_blueprint(rrb.Blueprint(rrb.Grid(*views), collapse_panels=True))


@torch.no_grad()
def main(config: DetectorEvalConfig) -> DetectorEvalReport:
    device: torch.device = torch.device(config.device)
    holdout: DetectorHoldout = from_json(DetectorHoldout, config.holdout.read_text())
    models: list[DetNetF] = []
    weights: dict[str, str] = {}
    for name, path in zip(config.names, config.weights, strict=True):
        model: DetNetF = DetNetF()
        weights[name] = f"{path} {load_weights(model, path)}"
        models.append(model.to(device).eval())
    results: dict[str, DatasetResult] = {}
    examples: dict[str, Examples] = {}
    for index, dataset in enumerate(config.datasets):
        segments: list[str] = eval_segments(dataset, holdout, config.catalog_url, config.base_segments, config.seed)
        totals: dict[str, SideCounts] = {name: SideCounts() for name in config.names}
        found: Examples = Examples()
        images: int = 0
        for samples in nvdec_producer(source_spec(dataset, index, config.upright), segments, catalog_url=config.catalog_url, device=device)(0):
            images += int(samples.pooled.shape[0])
            frames: Float32[Tensor, "n 1 120 160"] = samples.pooled[:, None].float() / 255.0
            predicted: list[Tensor] = []
            probability: list[Tensor] = []
            for name, model in zip(config.names, models, strict=True):
                out = model.forward_pooled(frames)
                predicted.append(torch.cat([out.center, out.radius[..., None]], dim=-1).float())
                probability.append(torch.sigmoid(out.presence_logit.float()))
                totals[name] = totals[name] + side_counts(predicted[-1], probability[-1], samples)
            stacked: Float32[Tensor, "m n 2 3"] = torch.stack(predicted)
            probabilities: Float32[Tensor, "m n 2"] = torch.stack(probability)
            present: Bool[Tensor, "n 2"] = samples.presence_mask & (samples.presence > 0.5)
            scale: Float32[Tensor, "3"] = torch.tensor(NET_SCALE, device=device)
            hand_free: Bool[Tensor, "n"] = (samples.presence_mask & ~present).all(-1)
            for m, name in enumerate(config.names):
                positive: Bool[Tensor, "n 2"] = probabilities[m] > PRESENCE_THRESHOLD
                hit: Bool[Tensor, "n 2"] = positive & (square_iou(stacked[m] * scale, samples.circle * scale) >= MIN_IOU)
                missed: Int64[Tensor, "k"] = (present & ~hit).any(-1).nonzero().flatten()
                alarms: Int64[Tensor, "k"] = (hand_free & positive.any(-1)).nonzero().flatten()
                found.take(f"missed/{name}", missed[torch.randperm(missed.numel(), device=missed.device)], samples, stacked, probabilities,
                           config.misses_per_dataset)
                found.take(f"false_alarm/{name}", alarms[torch.randperm(alarms.numel(), device=alarms.device)], samples, stacked, probabilities,
                           config.misses_per_dataset)
        results[dataset] = DatasetResult(segments, images, totals)
        examples[dataset] = found
        print(f"{dataset}: {len(segments)} segments, {images} images", flush=True)
        for name in config.names:
            both: BoxCounts = totals[name].both
            print(f"  {name:>12}: recall {both.recall} precision {both.precision} side {both.side_accuracy} "
                  f"median IoU {both.median_iou} FA/1k {both.false_alarms_per_1k} recall@0.1/0.3 {both.recall_at(0.1)}/{both.recall_at(0.3)} "
                  f"true-radius recall {both.detected_true_radius / both.present if both.present else None} "
                  f"r ratio p10/50/90 {[both.radius_ratio_percentile(q) for q in (0.1, 0.5, 0.9)]}", flush=True)
    report: DetectorEvalReport = DetectorEvalReport(weights, results)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "detector_eval.json").write_text(to_json(report))
    log_examples(config.out_dir / "misses.rrd", config.names, examples)
    return report
