import importlib
import importlib.metadata
import math
import os
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import rerun as rr
import tyro
from serde import SerdeError, coerce, serde
from serde.toml import from_toml

from dagkit.sources import SegmentRef
from dagkit.types import Producer, log_producer

Phase: TypeAlias = Literal["pre", "pre2", "post"]


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class MetricBand:
    name: str
    unit: str
    better: Literal["lower", "higher"]
    floor: float  # smallest ABSOLUTE tolerance (a relative floor is written as the number computed from the measured pre)


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class Case:
    name: str
    clip: Literal["short", "full"]
    reference: Literal["gt", "pre-port"]
    dataset: str | None = None  # catalog input (with `segment`) ...
    segment: str | None = None
    input: str | None = None  # ... or a repo-relative local input
    window_s: tuple[float, float] | None = None
    params: dict[str, str] = field(default_factory=dict)  # config overrides of this case
    noise: dict[str, float] = field(default_factory=dict)  # metric -> measured |pre - pre2| for THIS case
    noise_from: str | None = None  # reuse the named case's noise (stated, never implied)
    published: dict[str, float] = field(default_factory=dict)  # metric -> a published result on this case (a paper's table)

    def segment_ref(self, catalog_url: str | None = None) -> SegmentRef:
        """The case's catalog segment, ``window_s`` as ``window_ns``; a local-input case has none (ValueError)."""
        if self.dataset is None or self.segment is None:
            raise ValueError(f"case {self.name!r} has no catalog segment (dataset and segment)")
        window: tuple[int, int] | None = None if self.window_s is None else (round(self.window_s[0] * 1e9), round(self.window_s[1] * 1e9))
        segment = SegmentRef(self.dataset, self.segment, window_ns=window)
        return segment if catalog_url is None else replace(segment, catalog_url=catalog_url)


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class Shot:
    view: str
    t_s: float


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class GraphBlock:
    name: str
    run: str  # `pkg.verify:run`, (Case, phase, RecordingStream) -> None: writes the golden outputs
    score: str  # `pkg.verify:score`, (Case, golden rrd) -> {metric: value}: reads a golden (+ GT), never the pipeline
    cases: list[Case]
    metrics: list[MetricBand]
    shots: list[Shot] = field(default_factory=list)


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class TestBlock:
    name: str
    tier: Literal["model", "op", "node", "types"]
    lane: Literal["unit", "integration", "golden"]
    tests: list[str]
    properties: list[str] = field(default_factory=list)  # test ids that own the property bounds


@serde(deny_unknown_fields=True, type_check=coerce)
@dataclass(frozen=True, slots=True)
class VerifyCard:
    package: str
    graph: list[GraphBlock] = field(default_factory=list)
    tests: list[TestBlock] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class BandResult:
    metric: str
    passed: bool
    reason: str  # why it passed or failed, with the numbers


def load_card(path: Path) -> VerifyCard:
    try:
        return from_toml(VerifyCard, path.read_text())
    except (SerdeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def check_band(band: MetricBand, pre: float | None, post: float | None, noise: Mapping[str, float], published: float | None = None) -> BandResult:
    """``post`` within ``tol`` of the base: ``pre``, or a ``published`` result when that is worse (it never tightens a band)."""
    if pre is None or post is None:
        return BandResult(band.name, False, f"missing metric (pre={pre}, post={post})")
    if not all(math.isfinite(v) for v in (pre, post, pre if published is None else published)):
        return BandResult(band.name, False, f"non-finite metric (pre={pre}, post={post}, published={published})")
    if band.name not in noise or not all(math.isfinite(v) and v >= 0.0 for v in (2.0 * noise[band.name], band.floor)):
        return BandResult(band.name, False, f"no finite, non-negative measured noise / floor (noise={noise.get(band.name)}, floor={band.floor})")
    tol: float = max(2.0 * noise[band.name], band.floor)
    base: float = pre if published is None else max(pre, published) if band.better == "lower" else min(pre, published)
    passed: bool = post <= base + tol if band.better == "lower" else post >= base - tol
    cited: str = "" if published is None else f" published {published:.4g}"
    return BandResult(band.name, passed, f"pre {pre:.4g}{cited} post {post:.4g} {band.unit} tol {tol:.4g} ({band.better} is better)")


def golden_path(package: str, block: str, case: str, phase: Phase) -> Path:
    root: str | None = os.environ.get("DAGKIT_GOLDENS")
    if not root:
        raise RuntimeError("set DAGKIT_GOLDENS to the goldens directory (never committed)")
    return Path(root) / package / block / f"{case}.{phase}.rrd"


def _resolve(target: str) -> Callable[..., object]:
    module, name = target.split(":")
    return getattr(importlib.import_module(module), name)


@dataclass(frozen=True, slots=True)
class VerifyArgs:
    card: tyro.conf.Positional[Path]  # the package's VERIFY.toml
    phase: Phase
    block: str | None = None
    clip: Literal["short", "full"] | None = None
    register: bool = False  # add catalog goldens as layer verify-<block>-<phase>-<sha7> (never a replacement)


def main(args: VerifyArgs) -> int:
    card: VerifyCard = load_card(args.card)
    sha7: str = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True, check=True, cwd=args.card.parent
    ).stdout.strip()
    version: str = f"{importlib.metadata.version(card.package)}+{sha7}"
    selected: list[tuple[GraphBlock, Case]] = [(b, c) for b in card.graph if args.block in (None, b.name) for c in b.cases if args.clip in (None, c.clip)]
    if not selected or any(not block.metrics for block, _ in selected):
        print(f"FAIL: block={args.block} clip={args.clip} selects no case, or a selected block has no metrics")
        return 1
    failed: bool = False
    for block, case in selected:
        run, score = _resolve(block.run), _resolve(block.score)
        golden: Path = golden_path(card.package, block.name, case.name, args.phase)
        golden.parent.mkdir(parents=True, exist_ok=True)
        partial: Path = golden.with_suffix(".partial")
        recording = rr.RecordingStream(f"verify-{card.package}", recording_id=case.segment or case.name, send_properties=False)
        recording.save(partial)
        log_producer(recording, f"verify-{block.name}", Producer(card.package, version, block.name, "verify", args.phase))
        run(case, args.phase, recording)
        recording.flush()
        metrics: dict[str, float] = dict(score(case, partial))  # type: ignore[call-overload]
        recording.log(f"/verify/{block.name}/metrics", rr.AnyValues(**cast(dict[str, Any], metrics)), static=True)
        recording.flush()
        recording.disconnect()
        partial.replace(golden)
        print(f"{block.name}/{case.name} {args.phase}: {metrics}")
        results: list[BandResult] = [BandResult(b.name, False, "missing or non-finite") for b in block.metrics if not math.isfinite(metrics.get(b.name, math.nan))]
        if args.phase == "post":  # noise: the case's own (noise, pre + pre2 runs), or those of its `noise_from` case (missing: fails)
            donor: Case = next((c for c in block.cases if c.name == case.noise_from), case) if case.noise_from and not case.noise else case
            runs: list[tuple[Case, Phase]] = [(case, "pre"), (donor, "pre"), (donor, "pre2")]
            scores: dict = {(c.name, ph): dict(score(c, p)) for c, ph in runs if (p := golden_path(card.package, block.name, c.name, ph)).exists()}  # type: ignore[call-overload]
            for band in block.metrics:
                noise_runs: list[float | None] = [scores.get((donor.name, ph), {}).get(band.name) for ph in ("pre", "pre2")]
                pre: float | None = scores.get((case.name, "pre"), {}).get(band.name)
                result: BandResult = check_band(band, pre, metrics.get(band.name), donor.noise, case.published.get(band.name))
                measured: bool = all(v is not None and math.isfinite(v) for v in noise_runs)  # both runs behind the noise exist and score
                results.append(BandResult(band.name, result.passed and measured, f"{result.reason}; noise runs (pre, pre2) of {donor.name!r}: {noise_runs}"))
        for result in results:
            print(f"  {'PASS' if result.passed else 'FAIL'} {result.metric}: {result.reason}")
        failed |= not all(r.passed for r in results)
        if args.register and all(r.passed for r in results) and case.window_s is None and case.dataset and case.segment:  # never failed or windowed
            registered: Path = golden.with_name(f"{case.name}.{args.phase}.{sha7}.{uuid.uuid4().hex[:8]}.rrd")  # immutable: the golden path is reused
            shutil.copy2(golden, registered)
            dataset = case.segment_ref().entry()
            dataset.register([registered.resolve().as_uri()], layer_name=f"verify-{block.name}-{args.phase}-{sha7}").wait()
    return 1 if failed else 0


if __name__ == "__main__":  # parse with the imported module's classes: a second `Case` here fails beartype in package callbacks
    module = importlib.import_module("dagkit.verify")
    raise SystemExit(module.main(tyro.cli(module.VerifyArgs)))
