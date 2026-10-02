"""Run the pipeline over a whole split, restartably, and aggregate the per-segment scores into tables.

A segment is skipped when every output it should have exists and verifies: each ``<segment>.metrics.json`` parses and
its ``track_sha256`` equals the sha256 of the npz beside it and its metadata sidecar, with the same immutable
run identity in both JSON records. Anything else is recomputed. ``--run.shard``/``--run.shards``
split the segments over parallel processes. Only ``--aggregate-only`` publishes tables, after shards finish;
the tables cover every selected segment and list the missing ones.

Tables (``<run>/summary.md`` and ``summary.json``): per hand mode and group (all, ``separate_hand``, ``hand_hand``) the
pooled MKPE, MKA and MKA GT, tracking coverage, acquisition and drop delays and DetNet P/R with tracking per camera; and
DetNet-alone P/R per camera per group (also alone in ``<run>/detnet_metrics.json``). With ``--run.hand-modes`` empty
only the DetNet-alone pass runs.
"""

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import rerun as rr
import torch
import tyro
from rerun.catalog import DatasetEntry
from serde import SerdeError, serde
from serde.json import from_json, to_json

from handtrack.apis.run_pipeline import (
    DETNET_DIR,
    Networks,
    RunConfig,
    RunRecord,
    ensure_run_identity,
    file_sha256,
    format_number,
    load_networks,
    metrics_path,
    run_segment,
    select_segments,
)
from handtrack.data.catalog import SegmentInfo
from handtrack.eval.segment import (
    DetectionScore,
    DetNetAloneMetrics,
    HandScore,
    PositionScore,
    SegmentMetrics,
    combine_detections,
    combine_positions,
)
from handtrack.results import TrackMetadata, track_paths
from handtrack.train.checkpoint import atomic_write

GROUPS: tuple[str, ...] = ("all", "separate_hand", "hand_hand")


@dataclass(frozen=True, slots=True)
class EvaluateConfig:
    """A split-wide run: the pipeline options, plus whether to only rebuild the tables."""

    run: RunConfig = field(default_factory=lambda: RunConfig(name="test-real", hand_modes=("known", "unknown"), output_root=tyro.MISSING))
    """``--run.output-root`` is required."""
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


def _verified[MetricsT: (SegmentMetrics, DetNetAloneMetrics)](directory: Path, segment: str, kind: type[MetricsT], expected_identity: str | None = None) -> MetricsT | None:
    """The segment's scores if they parse and match the npz on disk, else None."""
    path: Path = metrics_path(directory, segment)
    npz, sidecar = track_paths(directory, segment)
    identity_path: Path = directory.parent / "identity.json"
    if not all(item.exists() for item in (path, npz, sidecar, identity_path)):
        return None
    try:
        metrics: MetricsT = from_json(kind, path.read_text())
        meta: TrackMetadata = from_json(TrackMetadata, sidecar.read_text())
        identity: str = hashlib.sha256(to_json(from_json(RunRecord, identity_path.read_text())).encode()).hexdigest()
    except (SerdeError, ValueError):
        return None
    valid: bool = (
        metrics.track_sha256 == meta.track_sha256 == file_sha256(npz)
        and metrics.run_identity_sha256 == meta.run_identity_sha256 == identity
        and (expected_identity is None or identity == expected_identity)
        and metrics.segment == meta.segment == segment
        and ((kind is DetNetAloneMetrics and meta.kind == "detnet_alone")
             or (kind is SegmentMetrics and meta.kind == "tracker" and meta.hand_mode == directory.name))
    )
    return metrics if valid else None


def _complete(config: RunConfig, segment: str, identity: str | None = None) -> bool:
    if identity is None:
        identity = ensure_run_identity(config)
    root: Path = config.output_root / config.name
    tracked: bool = all(_verified(root / mode, segment, SegmentMetrics, identity) is not None for mode in config.hand_modes)
    alone: bool = not (config.detnet_alone and config.detector == "detnet") or _verified(root / DETNET_DIR, segment, DetNetAloneMetrics, identity) is not None
    return tracked and alone


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def group_summary(mode: str, group: str, metrics: list[SegmentMetrics]) -> GroupSummary:
    hands: list[HandScore] = [hand for m in metrics for hand in m.hands]
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
    ensure_run_identity(config)
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


def _pr(scores: list[DetectionScore]) -> str:
    return " · ".join(f"{format_number(score.precision, 3)} / {format_number(score.recall, 3)}" for score in scores)


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
            f"| {g.hand_mode} | {g.group} | {g.segments} | {format_number(g.position.mkpe_mm)} | {format_number(g.position.mka_mm, 2)} | {format_number(g.position.mka_gt_mm, 2)}"
            f" | {format_number(g.visible_tracked_fraction, 3)} ({g.visible_tracked_frames}/{g.visible_frames})"
            f" | {format_number(g.mean_acquire_frames, 2)}, {g.never_acquired}/{g.appearances} | {format_number(g.mean_drop_frames, 2)}, {g.not_dropped}/{g.disappearances}"
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
    identity: str = ensure_run_identity(run)
    torch.set_num_threads(run.cpu_threads)
    device: torch.device = torch.device(run.device)
    entry: DatasetEntry = rr.catalog.CatalogClient(run.catalog_url).get_dataset(run.dataset)
    root: Path = run.output_root / run.name
    root.mkdir(parents=True, exist_ok=True)
    everything: tuple[SegmentInfo, ...] = select_segments(replace(run, shard=0, shards=1), entry)
    if not config.aggregate_only:
        networks: Networks = load_networks(run, device)
        ensure_run_identity(run, networks)
        (root / f"config.shard{run.shard}.json").write_text(to_json(RunRecord.from_config(run, networks)))
        for info in select_segments(run, entry):
            if _complete(run, info.segment_id, identity):
                print(f"skip {info.segment_id} (verified)", flush=True)
                continue
            for metrics in run_segment(run, entry, info, networks, device):
                print(
                    f"{metrics.segment} {metrics.hand_mode}: MKPE {format_number(metrics.position.mkpe_mm)} mm, {metrics.timings_s['total']:.1f} s",
                    flush=True,
                )
        return
    summary: RunSummary = summarize(run, everything)
    atomic_write(root / "summary.json", to_json(summary).encode())
    atomic_write(root / "summary.md", markdown(summary).encode())
    if summary.detnet_alone:
        atomic_write(root / "detnet_metrics.json", to_json(summary.detnet_alone).encode())
    print(markdown(summary))
