"""Run the pipeline over a whole split, restartably, and aggregate the per-segment scores into tables.

A segment is skipped when every output it should have exists and verifies: each ``<segment>.metrics.json`` parses and
its ``track_sha256`` equals the sha256 of the npz beside it. Anything else is recomputed. ``--run.shard``/``--run.shards``
split the segments over parallel processes; the tables always cover every selected segment and list the missing ones.

Tables (``<run>/summary.md`` and ``summary.json``): per hand mode and group (all, ``separate_hand``, ``hand_hand``) the
pooled MKPE, MKA and MKA GT, tracking coverage, acquisition and drop delays and DetNet P/R with tracking per camera; and
DetNet-alone P/R per camera per group (also alone in ``<run>/detnet_metrics.json``). With ``--run.hand-modes`` empty
only the DetNet-alone pass runs.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import torch
from rerun.catalog import DatasetEntry
from serde import SerdeError, serde
from serde.json import from_json, to_json

from handtrack.apis.run_pipeline import (
    DETNET_DIR,
    Networks,
    RunConfig,
    RunRecord,
    file_sha256,
    load_networks,
    metrics_path,
    run_segment,
    select_segments,
)
from handtrack.data.catalog import UMETRACK, SegmentInfo
from handtrack.eval.segment import DetectionScore, DetNetAloneMetrics, PositionScore, SegmentMetrics, combine_detections, combine_positions
from handtrack.results import track_paths

GROUPS: tuple[str, ...] = ("all", "separate_hand", "hand_hand")


@dataclass(frozen=True, slots=True)
class EvaluateConfig:
    """A split-wide run: the pipeline options, plus whether to only rebuild the tables."""

    run: RunConfig = field(default_factory=lambda: RunConfig(name="test-real", hand_modes=("known", "unknown")))
    aggregate_only: bool = False
    """Skip tracking; rebuild the tables from the outputs on disk."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class GroupSummary:
    """One hand mode and group of segments, pooled."""

    hand_mode: str
    group: str
    segments: int
    position: PositionScore
    visible_frames: int
    visible_tracked_frames: int
    visible_tracked_fraction: float | None
    appearances: int
    never_acquired: int
    mean_acquire_frames: float | None
    disappearances: int
    not_dropped: int
    """Disappearances the track outlived (right-censored)."""
    mean_drop_frames: float | None
    tracked_without_hand: int
    tracked_absent: int
    detnet_with_tracking: list[DetectionScore]
    """§5.4 as written."""
    detnet_with_tracking_crop: list[DetectionScore]
    """Diagnostic: containment in the x1.2 crop box."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class DetNetSummary:
    group: str
    segments: int
    per_camera: list[DetectionScore]
    """§5.4 as written."""
    per_camera_crop: list[DetectionScore]
    """Diagnostic: containment in the x1.2 crop box."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class RunSummary:
    """``<run>/summary.json``."""

    name: str
    selected: int
    missing: list[str]
    """``<hand mode or detnet>/<segment>`` outputs that are absent or fail verification."""
    groups: list[GroupSummary]
    detnet_alone: list[DetNetSummary]


def _verified[MetricsT: (SegmentMetrics, DetNetAloneMetrics)](directory: Path, segment: str, kind: type[MetricsT]) -> MetricsT | None:
    """The segment's scores if they parse and match the npz on disk, else None."""
    path: Path = metrics_path(directory, segment)
    npz: Path = track_paths(directory, segment)[0]
    if not path.exists() or not npz.exists():
        return None
    try:
        metrics: MetricsT = from_json(kind, path.read_text())
    except (SerdeError, ValueError):
        return None
    return metrics if metrics.track_sha256 == file_sha256(npz) else None


def _complete(config: RunConfig, segment: str) -> bool:
    root: Path = config.output_root / config.name
    tracked: bool = all(_verified(root / mode, segment, SegmentMetrics) is not None for mode in config.hand_modes)
    alone: bool = not (config.detnet_alone and config.detector == "detnet") or _verified(root / DETNET_DIR, segment, DetNetAloneMetrics) is not None
    return tracked and alone


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def group_summary(mode: str, group: str, metrics: list[SegmentMetrics]) -> GroupSummary:
    hands = [hand for m in metrics for hand in m.hands]
    visible: int = sum(hand.tracking.visible_frames for hand in hands)
    visible_tracked: int = sum(hand.tracking.visible_tracked_frames for hand in hands)
    acquire: list[int | None] = [delay for hand in hands for delay in hand.tracking.acquire_frames]
    drop: list[int | None] = [delay for hand in hands for delay in hand.tracking.drop_frames]
    return GroupSummary(
        hand_mode=mode,
        group=group,
        segments=len(metrics),
        position=combine_positions([m.position for m in metrics]),
        visible_frames=visible,
        visible_tracked_frames=visible_tracked,
        visible_tracked_fraction=visible_tracked / visible if visible else None,
        appearances=len(acquire),
        never_acquired=sum(delay is None for delay in acquire),
        mean_acquire_frames=_mean([delay for delay in acquire if delay is not None]),
        disappearances=len(drop),
        not_dropped=sum(delay is None for delay in drop),
        mean_drop_frames=_mean([delay for delay in drop if delay is not None]),
        tracked_without_hand=sum(hand.tracking.tracked_without_hand for hand in hands),
        tracked_absent=sum(hand.tracking.tracked_absent for hand in hands),
        detnet_with_tracking=combine_detections([m.detnet_with_tracking for m in metrics]),
        detnet_with_tracking_crop=combine_detections([m.detnet_with_tracking_crop for m in metrics]),
    )


def _in_group(interaction: str, group: str) -> bool:
    return group == "all" or interaction == group


def summarize(config: RunConfig, segments: tuple[SegmentInfo, ...]) -> RunSummary:
    root: Path = config.output_root / config.name
    missing: list[str] = []
    groups: list[GroupSummary] = []
    for mode in config.hand_modes:
        scored: list[SegmentMetrics] = []
        for info in segments:
            metrics: SegmentMetrics | None = _verified(root / mode, info.segment_id, SegmentMetrics)
            if metrics is None:
                missing.append(f"{mode}/{info.segment_id}")
            else:
                scored.append(metrics)
        groups.extend(group_summary(mode, group, [m for m in scored if _in_group(m.interaction, group)]) for group in GROUPS)
    alone: list[DetNetSummary] = []
    if config.detnet_alone and config.detector == "detnet":
        detnet: list[DetNetAloneMetrics] = []
        for info in segments:
            result: DetNetAloneMetrics | None = _verified(root / DETNET_DIR, info.segment_id, DetNetAloneMetrics)
            if result is None:
                missing.append(f"{DETNET_DIR}/{info.segment_id}")
            else:
                detnet.append(result)
        for group in GROUPS:
            chosen: list[DetNetAloneMetrics] = [m for m in detnet if _in_group(m.interaction, group)]
            alone.append(
                DetNetSummary(
                    group, len(chosen), combine_detections([m.per_camera for m in chosen]), combine_detections([m.per_camera_crop for m in chosen])
                )
            )
    return RunSummary(name=config.name, selected=len(segments), missing=missing, groups=groups, detnet_alone=alone)


def _number(value: float | None, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _pr(scores: list[DetectionScore]) -> str:
    return " · ".join(f"{_number(score.precision, 3)} / {_number(score.recall, 3)}" for score in scores)


def markdown(summary: RunSummary) -> str:
    lines: list[str] = [f"# handtrack run `{summary.name}`", "", f"{summary.selected} segments selected; {len(summary.missing)} outputs missing.", ""]
    lines += [
        "## Full pipeline",
        "",
        "| hand | group | segments | MKPE mm | MKA | MKA GT | tracked / visible | acquire (mean frames, never) | drop (mean frames, not dropped) | tracked w/o hand | DetNet+track P / R per camera (§5.4) | same, containment in x1.2 crop (diagnostic) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in summary.groups:
        lines.append(
            f"| {g.hand_mode} | {g.group} | {g.segments} | {_number(g.position.mkpe_mm)} | {_number(g.position.mka_mm, 2)} | {_number(g.position.mka_gt_mm, 2)}"
            f" | {_number(g.visible_tracked_fraction, 3)} ({g.visible_tracked_frames}/{g.visible_frames})"
            f" | {_number(g.mean_acquire_frames, 2)}, {g.never_acquired}/{g.appearances} | {_number(g.mean_drop_frames, 2)}, {g.not_dropped}/{g.disappearances}"
            f" | {g.tracked_without_hand} | {_pr(g.detnet_with_tracking)} | {_pr(g.detnet_with_tracking_crop)} |"
        )
    if summary.detnet_alone:
        lines += [
            "",
            "## DetNet alone (every frame, every camera)",
            "",
            "| group | segments | P / R per camera (§5.4) | same, containment in x1.2 crop (diagnostic) |",
            "|---|---|---|---|",
        ]
        lines += [f"| {d.group} | {d.segments} | {_pr(d.per_camera)} | {_pr(d.per_camera_crop)} |" for d in summary.detnet_alone]
    if summary.missing:
        lines += ["", "## Missing", ""] + [f"- {name}" for name in summary.missing]
    return "\n".join(lines) + "\n"


def main(config: EvaluateConfig) -> None:
    run: RunConfig = config.run
    torch.set_num_threads(run.cpu_threads)
    device: torch.device = torch.device(run.device)
    entry: DatasetEntry = rr.catalog.CatalogClient(run.catalog_url).get_dataset(UMETRACK)
    root: Path = run.output_root / run.name
    root.mkdir(parents=True, exist_ok=True)
    everything: tuple[SegmentInfo, ...] = select_segments(dataclasses.replace(run, shard=0, shards=1), entry)
    if not config.aggregate_only:
        networks: Networks = load_networks(run, device)
        (root / f"config.shard{run.shard}.json").write_text(to_json(RunRecord.from_config(run, networks)))
        for info in select_segments(run, entry):
            if _complete(run, info.segment_id):
                print(f"skip {info.segment_id} (verified)", flush=True)
                continue
            for metrics in run_segment(run, entry, info, networks, device):
                print(
                    f"{metrics.segment} {metrics.hand_mode}: MKPE {_number(metrics.position.mkpe_mm)} mm, {metrics.timings_s['total']:.1f} s",
                    flush=True,
                )
    summary: RunSummary = summarize(run, everything)
    (root / "summary.json").write_text(to_json(summary))
    (root / "summary.md").write_text(markdown(summary))
    if summary.detnet_alone:
        (root / "detnet_metrics.json").write_text(to_json(summary.detnet_alone))
    print(markdown(summary))
