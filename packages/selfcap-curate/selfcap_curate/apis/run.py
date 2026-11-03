"""``selfcap-run``: one dataset end to end, each step a subprocess of its pixi task, its layers swapped in as it goes.

The steps are ``STEPS``. selfcap-curate and selfcap-soma are separate pixi environments, so this module runs their tasks
and imports neither's stages.
"""

import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from statistics import median
from threading import Lock
from time import perf_counter
from typing import Literal, NoReturn, TypeAlias, get_args
from urllib.parse import unquote, urlparse

import pyarrow as pa
from dataforge import paths
from dataforge.datasets.selfcap import SOMA_METRICS, SOMA_QUALITY_PROPERTY
from dataforge.identity import SequenceIdentity
from dataforge.timing import ConvertRecord, load_records
from rerun.catalog import CatalogClient

from selfcap_curate.records import EpisodeKey

ExoMethod: TypeAlias = Literal["hands", "head"]
PixiEnvironment: TypeAlias = Literal["dataforge", "selfcap-curate", "selfcap-soma"]
StepName: TypeAlias = Literal["ingest", "register", "align", "calib", "keypoints", "publish", "soma", "soma_xlo", "preview", "blueprints"]
SegmentFile: TypeAlias = Callable[[str], Path]

MANIFEST: Path = Path(__file__).resolve().parents[4] / "pixi.toml"
REGISTRY_NAME: str = "selfcap"
"""The dataset's key in the dataforge registry: the subcommand of its tasks and the prefix of its segment ids."""
MAX_REASON_LENGTH: int = 240
SOMA_QUALITY_COLUMN: str = f"property:{SOMA_QUALITY_PROPERTY}"
STEPS: tuple[StepName, ...] = get_args(StepName)
"""Every step in run order: a step's number is its position."""


@dataclass
class Config:
    """Raw captures or as-cut bases in; one new dataset and one new run directory out."""

    run_root: Path
    """New directory for every output and log; an existing one is refused."""
    catalog_url: str
    """Catalog URL, local or on another machine; its server must read run_root at the same path."""
    dataset: str
    """New dataset on catalog_url; a name that already exists is refused, so other datasets stay safe."""
    raw_root: Path | None = None
    """Directory that holds selfcap/ raw captures, to ingest; exactly one of raw_root and base_dir."""
    base_dir: Path | None = None
    """Directory of as-cut base RRDs, which skip ingest; exactly one of raw_root and base_dir."""
    segment_ids: tuple[str, ...] = ()
    """Segment ids to run; empty runs every segment."""
    donor_catalog_url: str | None = None
    """Read-only catalog whose dataforge-selfcap holds camera-50 donors; None borrows from the run's own dataset."""
    oak_egohand_assets: Path | None = None
    """Optional local EgoHand root, passed to calibrate; None takes the pinned downloads."""
    soma_assets: Path | None = None
    """Optional local SOMA model files, passed to SOMA and soma_xlo; None takes the pinned downloads."""
    exo_method: ExoMethod = "head"
    """Phone poses from hands only, or extended with the wearer's head track."""
    jobs: int = 3
    """Align and calibrate processes at a time."""
    soma_jobs: int = 4
    """SOMA batch processes at a time; soma_xlo is one process, and keypoints runs one at a time."""
    preview_workers: int = 3
    """Segments the one preview process builds at a time."""


@dataclass(slots=True)
class StepCount:
    """One line of the final summary."""

    name: StepName
    """Step name."""
    ok: int = 0
    """Segments that passed."""
    failed: int = 0
    """Segments that failed."""
    wall_s: int = 0
    """Wall seconds of the step, without its layer swaps."""


@dataclass(frozen=True, slots=True)
class SegmentStep:
    """A step whose per-segment files prove which segments passed, and the layers it swaps in."""

    name: StepName
    """The step; also names its log files."""
    command: Callable[[list[str]], list[str]]
    """The command of one process, for the segments it takes: one unit's, or every one with ``one_process``."""
    output: SegmentFile
    """The file that proves the segment passed."""
    layers: Mapping[str, SegmentFile] = field(default_factory=dict)
    """The layers the step produces, in swap order: layer name to the segment's file."""
    superseded: Mapping[str, SegmentFile] = field(default_factory=dict)
    """This run's unregistered file a layer replaces, deleted once the catalog shows the new file for that segment."""
    unit: Callable[[str], str] = str
    """The unit of work a segment belongs to; each unit is one process, and by default each segment is its own."""
    jobs: int = 1
    """Processes at a time."""
    one_process: bool = False
    """One process takes every segment, and a segment with no file failed even if the others passed."""
    env: dict[str, str] = field(default_factory=dict)
    """Environment variables of each process."""
    detail_markers: tuple[str, ...] = ()
    """Log lines ``<marker> <segment> <detail>`` whose details join the segment's ok line."""


@dataclass(slots=True)
class Run:
    """State of one run: its resolved config, its log and its counts."""

    config: Config
    """Config with absolute paths: every task runs in its own working directory."""
    lock: Lock = field(default_factory=Lock)
    """Log lines come from worker threads."""
    counts: list[StepCount] = field(default_factory=list)
    """One entry per step started."""
    started: float = field(default_factory=perf_counter)
    """Start of the run."""

    @property
    def logs(self) -> Path:
        return self.config.run_root / "logs"

    def log(self, message: str) -> None:
        """Print one line and append it to progress.log."""
        line: str = f"[{datetime.now():%H:%M:%S}] {self.config.dataset}: {message}"
        with self.lock, (self.config.run_root / "progress.log").open("a") as progress:
            # The NAS SSD client caches new files as mode 000; chmod repairs its cached attributes.
            (self.config.run_root / "progress.log").chmod(0o644)
            print(line, flush=True)
            progress.write(line + "\n")

    def summary(self) -> None:
        """Log the per-step counts."""
        for count in self.counts:
            self.log(f"SUMMARY {count.name}: ok {count.ok}, failed {count.failed}, wall {count.wall_s} s")
        if any(count.name == "soma" and count.ok > 0 for count in self.counts):
            self.log(soma_quality_line(self.config.catalog_url, self.config.dataset))
        self.log(f"PIPELINE_DONE total {int(perf_counter() - self.started)} s")

    def stop(self, message: str) -> NoReturn:
        """End the run here with a nonzero exit."""
        self.log(message)
        self.summary()
        raise SystemExit(message)

    def require(self, segments: list[str], step: str) -> list[str]:
        """Stop the run when a step left no segment; the steps after it have nothing to read."""
        if not segments:
            self.stop(f"STOP no segments left after {step}")
        return segments


def property_cells(table: pa.Table, column: str) -> list[str | float]:
    """The column's values: a cell is a one-element list, or null; an absent column has none."""
    if column not in table.schema.names:
        return []
    cells: list[list[str | float] | None] = table[column].to_pylist()
    return [cell[0] for cell in cells if cell]


def soma_quality_line(catalog_url: str, dataset: str) -> str:
    """Pass and flag counts and the median of each per-segment residual median, from the catalog segment table."""
    client: CatalogClient = CatalogClient(catalog_url)  # the query needs the client alive until the table is read
    table: pa.Table = client.get_dataset(dataset).segment_table().to_arrow_table()
    statuses: list[str | float] = property_cells(table, f"{SOMA_QUALITY_COLUMN}:status")
    medians: list[str] = []
    for metric in SOMA_METRICS:
        values: list[str | float] = property_cells(table, f"{SOMA_QUALITY_COLUMN}:{metric}_median_mm")
        if values:
            medians.append(f"{metric} {median(float(value) for value in values):.1f}")
    return f"SOMA_QUALITY pass {statuses.count('pass')}, flag {statuses.count('flag')}, median mm: {', '.join(medians) or 'none'}"


def child_environment(parent: Mapping[str, str]) -> dict[str, str]:
    """The environment of a nested ``pixi run``, without the activation of the env that runs selfcap-run.

    pixi keeps what the new env does not set: selfcap-curate's LD_LIBRARY_PATH would load its libraries into selfcap-soma,
    and a dev env's flag would turn beartype on in the production envs of the tasks.
    """
    prefix: str | None = parent.get("CONDA_PREFIX")
    child: dict[str, str] = {}
    for name, value in parent.items():
        entries: list[str] = value.split(os.pathsep)
        kept: list[str] = entries if prefix is None else [entry for entry in entries if prefix not in entry]
        if kept and name != "PIXI_DEV_MODE":
            child[name] = os.pathsep.join(kept)
    return child


def run_command(command: list[str], env: dict[str, str], log_path: Path) -> int:
    """Run one task with its output in ``log_path`` and return its exit code."""
    with log_path.open("w") as log:
        # The NAS SSD client caches new files as mode 000; chmod repairs its cached attributes.
        log_path.chmod(0o644)
        completed: subprocess.CompletedProcess[bytes] = subprocess.run(
            command,
            env={**child_environment(os.environ), **env},
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def last_meaningful_line(log_path: Path) -> str:
    """The log's last line that holds a letter or a digit and is not pixi's banner; a Python failure ends in its exception."""
    lines: list[str] = log_path.read_text(errors="replace").splitlines()
    meaningful: list[str] = [line.strip() for line in lines if any(character.isalnum() for character in line) and "Pixi task" not in line]
    return meaningful[-1][:MAX_REASON_LENGTH] if meaningful else "its log is empty"


def missing_output_reason(log_path: Path, step: StepName, segment: str, output: Path) -> str:
    """Use this segment's explicit skip reason when a successful process wrote no output."""
    marker: str = f"{step.upper()}_SKIP"
    for line in reversed(log_path.read_text(errors="replace").splitlines()):
        fields: list[str] = line.split(maxsplit=2)
        if len(fields) == 3 and fields[0] == marker and fields[1].removeprefix("selfcap__") == segment.removeprefix("selfcap__"):
            return line.strip()[:MAX_REASON_LENGTH]
    return f"exit 0 but no {output}"


def reported_details(log_path: Path, markers: tuple[str, ...]) -> str:
    """The details of the log's marker lines, each after its marker and key, as one suffix."""
    if not markers:
        return ""
    lines: list[list[str]] = [line.split(" ", 2) for line in log_path.read_text(errors="replace").splitlines()]
    return "".join(f" {fields[2]}" for fields in lines if fields[0] in markers and len(fields) == 3)


def task(environment: PixiEnvironment, name: str, *arguments: str) -> list[str]:
    """The command of one pixi task."""
    return ["pixi", "run", "--manifest-path", str(MANIFEST), "-e", environment, "--frozen", name, *arguments]


def step_label(name: StepName) -> str:
    """The step's label in the logs, ``STEP<number>_<NAME>``."""
    return f"STEP{STEPS.index(name) + 1}_{name.upper()}"


@contextmanager
def timed_step(run: Run, name: StepName, *, summary: str = "") -> Iterator[StepCount]:
    """Announce a step, give the body its count to fill, then log and keep the count with the step's wall time."""
    run.log(f"step {STEPS.index(name) + 1}: {name}{summary}")
    started: float = perf_counter()
    count: StepCount = StepCount(name)
    yield count
    count.wall_s = int(perf_counter() - started)
    run.counts.append(count)
    run.log(f"{step_label(name)}_WALL {count.wall_s} s")


def run_each_unit(run: Run, step: SegmentStep, segments: list[str]) -> list[str]:
    """Run one process per unit of work and return the segments that passed, in order; each segment reports its own result."""
    units: dict[str, list[str]] = {}
    for segment in segments:
        units.setdefault(step.unit(segment), []).append(segment)

    def run_one(unit: str) -> list[str]:
        started: float = perf_counter()
        log_path: Path = run.logs / f"{step.name}-{unit}.log"
        code: int = run_command(step.command(units[unit]), step.env, log_path)
        elapsed: int = int(perf_counter() - started)
        passed: list[str] = []
        for segment in units[unit]:
            output: Path = step.output(segment)
            if code == 0 and output.is_file():
                run.log(f"{step.name} ok   {segment} ({elapsed} s){reported_details(log_path, step.detail_markers)}")
                passed.append(segment)
            else:
                reason: str = last_meaningful_line(log_path) if code != 0 else missing_output_reason(log_path, step.name, segment, output)
                run.log(f"{step.name} FAIL {segment} ({elapsed} s): {reason}")
        return passed

    with ThreadPoolExecutor(step.jobs) as pool:
        survivors: set[str] = {segment for passed in pool.map(run_one, units) for segment in passed}
    return [segment for segment in segments if segment in survivors]


def run_one_process(run: Run, step: SegmentStep, segments: list[str]) -> list[str]:
    """Run one process for every segment; its files are written whole or not at all, so they alone say who passed."""
    log_path: Path = run.logs / f"{step.name}.log"
    code: int = run_command(step.command(segments), step.env, log_path)
    survivors: list[str] = []
    for segment in segments:
        output: Path = step.output(segment)
        if output.is_file():
            run.log(f"{step.name} ok   {segment}")
            survivors.append(segment)
        else:
            reason: str = last_meaningful_line(log_path) if code != 0 else missing_output_reason(log_path, step.name, segment, output)
            run.log(f"{step.name} FAIL {segment}: {reason}; see {log_path}")
    return survivors


def run_segments(run: Run, step: SegmentStep, segments: list[str]) -> list[str]:
    """Run the step, swap in its layers for the segments that passed, and return those segments."""
    with timed_step(run, step.name, summary=", one process" if step.one_process else f", {step.jobs} at a time") as count:
        survivors: list[str] = (run_one_process if step.one_process else run_each_unit)(run, step, segments)
        count.ok = len(survivors)
        count.failed = len(segments) - len(survivors)
    run.require(survivors, step.name)
    for layer, file in step.layers.items():
        swap(run, layer, [file(segment) for segment in survivors])
    for layer, superseded in step.superseded.items():
        remove_superseded(run, layer, step.layers[layer], superseded, survivors)
    return survivors


def catalog_step(run: Run, name: StepName, segments: list[str], command: list[str], env: dict[str, str]) -> None:
    """Run one catalog write for all segments; a failure stops the run, because later steps read what it writes."""
    log_path: Path = run.logs / f"{name}.log"
    with timed_step(run, name) as count:
        code: int = run_command(command, env, log_path)
        count.ok = len(segments) if code == 0 else 0
        count.failed = 0 if code == 0 else len(segments)
    if code != 0:
        run.stop(f"{name} FAILED: {last_meaningful_line(log_path)}")
    run.log(f"{name} ok")


def swap(run: Run, layer: str, files: list[Path]) -> None:
    """Swap the layer to the files that exist, as a stage may write no optional layer for a segment; a failure stops the run."""
    config: Config = run.config
    present: list[Path] = [path for path in files if path.is_file()]
    if not present:
        run.log(f"swap {layer}: no files")
        return
    log_path: Path = run.logs / f"swap-{layer}.log"
    code: int = run_command(
        task("dataforge", "dataforge-swap-layer", "--layer", layer, "--files", *map(str, present), "--catalog-url", config.catalog_url, "--catalog-name", config.dataset),
        {}, log_path,
    )
    if code != 0:
        run.stop(f"swap {layer} FAILED: {last_meaningful_line(log_path)}")
    run.log(f"swapped {layer} ({len(present)} files)")


def registered_files(config: Config, layer: str) -> dict[str, Path]:
    """The file the catalog holds for each segment of the layer."""
    client: CatalogClient = CatalogClient(config.catalog_url)  # the query needs the client alive until the table is read
    rows: list[dict[str, str]] = (
        client.get_dataset(config.dataset).manifest().select("rerun_segment_id", "rerun_layer_name", "rerun_storage_url").to_arrow_table().to_pylist()
    )
    return {row["rerun_segment_id"]: Path(unquote(urlparse(row["rerun_storage_url"]).path)) for row in rows if row["rerun_layer_name"] == layer}


def remove_superseded(run: Run, layer: str, new: SegmentFile, superseded: SegmentFile, segments: list[str]) -> None:
    """Delete each segment's superseded file, or its link in a base_dir run, once the catalog shows the new file as the layer."""
    registered: dict[str, Path] = registered_files(run.config, layer)
    for segment in segments:
        file: Path = superseded(segment)
        if registered.get(segment) == new(segment).resolve():
            file.unlink(missing_ok=True)
            run.log(f"removed superseded {file}")
        else:
            run.log(f"kept {file}: the catalog does not show {new(segment)} as {layer}")


def resolved(path: Path | None) -> Path | None:
    """The absolute path: every task runs in its own working directory."""
    return None if path is None else path.resolve()


def optional_flag(flag: str, value: Path | str | None) -> list[str]:
    """The flag and its value; without one, the task takes its default."""
    return [] if value is None else [flag, str(value)]


def identity_of(segment: str) -> SequenceIdentity:
    """The segment's identity, from a recording id or an unprefixed key; a malformed one raises ValueError."""
    key: EpisodeKey = EpisodeKey.parse(segment)
    return SequenceIdentity(REGISTRY_NAME, (key.date, key.sid8, key.episode))


def as_cut_file(run_root: Path, layer: str, segment: str) -> Path:
    """The segment's layer file in run_root/rrd, the tree dataforge-register reads."""
    return paths.rrd_path(run_root / "rrd", layer=layer, identity=identity_of(segment))


def ingest(run: Run) -> list[str]:
    """Convert the raw captures into run_root/rrd and return the segments with a base."""
    config: Config = run.config
    root: Path = config.run_root / "rrd"
    env: dict[str, str] = {"DATAFORGE_RAW_ROOT": str(config.raw_root), "DATAFORGE_OUTPUT_ROOT": str(root)}
    if config.segment_ids:
        step: SegmentStep = SegmentStep(
            "ingest", lambda only: task("dataforge", "dataforge-convert", "--sequence", *only, REGISTRY_NAME), lambda segment: as_cut_file(config.run_root, paths.BASE_LAYER, segment), env=env,
        )
        return run_segments(run, step, list(config.segment_ids))
    log_path: Path = run.logs / "ingest.log"
    segments: list[str] = []
    with timed_step(run, "ingest", summary=", all") as count:
        code: int = run_command(task("dataforge", "dataforge-convert", REGISTRY_NAME), env, log_path)
        timing: Path = root / "timing/convert.jsonl"
        records: list[ConvertRecord] = load_records(timing, ConvertRecord) if timing.is_file() else []
        failed: int = 0
        for record in sorted(records, key=lambda listed: listed.recording_id):
            base: Path = as_cut_file(config.run_root, paths.BASE_LAYER, record.recording_id)
            if record.error is None and base.is_file():
                run.log(f"ingest ok   {record.recording_id}")
                segments.append(record.recording_id)
            else:
                run.log(f"ingest FAIL {record.recording_id}: {record.error or f'exit 0 but no {base}'}")
                failed += 1
        if code != 0 and not failed:
            run.log(f"ingest returned an error: {last_meaningful_line(log_path)}")
        count.ok = len(segments)
        count.failed = max(failed, int(code != 0))
    return run.require(segments, "ingest")


def link_bases(run: Run, base_dir: Path) -> list[str]:
    """Link the as-cut bases into run_root/rrd and return the segments with a base."""
    config: Config = run.config
    segments: list[str] = []
    with timed_step(run, "ingest", summary=f", as-cut bases of {base_dir}") as count:
        selected: list[str] = list(config.segment_ids) or sorted(path.stem for path in base_dir.glob(f"{REGISTRY_NAME}__*.rrd"))
        for segment in selected:
            source: Path = base_dir / f"{segment}.rrd"
            if source.is_file():
                link: Path = as_cut_file(config.run_root, paths.BASE_LAYER, segment)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(source)
                run.log(f"base ok   {segment} (0 s)")
                segments.append(segment)
            else:
                run.log(f"base FAIL {segment} (0 s): no file {source}")
        count.ok = len(segments)
        count.failed = len(selected) - len(segments)
    return run.require(segments, "base")


def main(config: Config) -> None:
    """Run every step; a failed segment is logged and skipped, and the run exits nonzero at the end."""
    if (config.raw_root is None) == (config.base_dir is None):
        raise ValueError("give exactly one of raw_root and base_dir")
    if config.jobs < 1 or config.soma_jobs < 1 or config.preview_workers < 1:
        raise ValueError("jobs, soma_jobs and preview_workers must be at least 1")
    segment_ids: tuple[str, ...] = tuple(dict.fromkeys(identity_of(segment).recording_id for segment in config.segment_ids))
    paths.require_outside([config.run_root], roots=[paths.NAS_ROOT])
    inputs: list[Path] = [assets for assets in (config.oak_egohand_assets, config.soma_assets) if assets is not None]
    if config.raw_root is not None:
        inputs.append(config.raw_root / REGISTRY_NAME)
    if config.base_dir is not None:
        inputs.append(config.base_dir)
    for directory in inputs:
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if config.run_root.exists() or config.run_root.is_symlink():
        raise FileExistsError(config.run_root)
    if config.dataset in CatalogClient(config.catalog_url).dataset_names():
        raise ValueError(f"dataset {config.dataset} already exists on {config.catalog_url}")
    config = replace(
        config,
        run_root=config.run_root.resolve(), oak_egohand_assets=resolved(config.oak_egohand_assets), soma_assets=resolved(config.soma_assets),
        raw_root=resolved(config.raw_root), base_dir=resolved(config.base_dir),
        segment_ids=segment_ids,
    )
    config.run_root.mkdir(parents=True)
    root: Path = config.run_root
    run: Run = Run(config)
    run.logs.mkdir()

    catalog: list[str] = ["--catalog-url", config.catalog_url, "--catalog-name", config.dataset]
    source: list[str] = ["--catalog-url", config.catalog_url, "--dataset", config.dataset]
    publish_blueprints: list[str] = task("dataforge", "dataforge-publish-blueprints", *catalog, "--output-root", str(root / "catalog"), REGISTRY_NAME)

    def session_of(segment: str) -> str:
        return EpisodeKey.parse(segment).session

    def aligned(segment: str) -> Path:
        return root / "align" / session_of(segment) / f"{segment}.rrd"

    def aligned_layer(segment: str, layer: str) -> Path:
        return paths.rrd_path(root / "align" / session_of(segment), layer=layer, identity=identity_of(segment))

    def calibrated(segment: str) -> Path:
        return root / "calib" / segment / f"{segment}.rrd"

    def keypoints_file(segment: str) -> Path:
        return root / "keypoints" / segment / f"{segment}.rrd"

    def soma_file(segment: str) -> Path:
        return root / "soma" / config.dataset / segment / f"{segment}.rrd"

    def soma_xlo_file(segment: str) -> Path:
        return root / "soma_xlo" / f"{segment}.rrd"

    def preview_file(segment: str) -> Path:
        return root / "preview" / f"{segment}.rrd"

    segments: list[str] = ingest(run) if config.base_dir is None else link_bases(run, config.base_dir)
    catalog_step(run, "register", segments, task("dataforge", "dataforge-register", *catalog, REGISTRY_NAME), {"DATAFORGE_OUTPUT_ROOT": str(root / "rrd")})

    segments = run_segments(
        run,
        SegmentStep(
            "align",
            lambda only: task(
                "selfcap-curate", "selfcap-align", "--base-dir", str(root / "rrd" / paths.BASE_LAYER), "--output-dir", str(aligned(only[0]).parent), *source, "--segment-ids", *only,
            ),
            aligned,
            layers={
                paths.BASE_LAYER: aligned,
                paths.GT_LAYER: lambda segment: aligned_layer(segment, paths.GT_LAYER),
                "sapiens2": lambda segment: aligned_layer(segment, "sapiens2"),
            },
            superseded={layer: partial(as_cut_file, root, layer) for layer in (paths.BASE_LAYER, paths.GT_LAYER)},
            unit=session_of,  # align pools its evidence across the episodes of a session
            jobs=config.jobs,
        ),
        segments,
    )
    segments = run_segments(
        run,
        SegmentStep(
            "calib",
            lambda only: task(
                "selfcap-curate", "selfcap-calibrate", "--output-dir", str(calibrated(only[0]).parent), "--segment-ids", *only, *source,
                "--exo-method", config.exo_method, *optional_flag("--oak-egohand-assets", config.oak_egohand_assets), *optional_flag("--donor-catalog-url", config.donor_catalog_url),
            ),
            calibrated,
            layers={"egohand": lambda segment: root / "calib" / segment / "egohand" / f"{segment}.rrd"},
            jobs=config.jobs,
            detail_markers=("HEAD_CALIBRATION",),
        ),
        segments,
    )
    segments = run_segments(
        run,
        SegmentStep(
            "keypoints",
            lambda only: task(
                "selfcap-curate", "selfcap-keypoints", "--calib-dir", str(calibrated(only[0]).parent), "--output-dir", str(keypoints_file(only[0]).parent),
                "--segment-ids", *only, *source,
            ),
            keypoints_file,
            layers={"calib": keypoints_file},
            superseded={"calib": calibrated},
            detail_markers=("KEYPOINTS_RATE", "KEYPOINTS_TIME"),
        ),
        segments,
    )
    catalog_step(run, "publish", segments, publish_blueprints, {})

    segments = run_segments(
        run,
        SegmentStep(
            "soma",
            lambda only: task(
                "selfcap-soma", "soma-batch", "--dataset", config.dataset, "--segment", *only, "--catalog-url", config.catalog_url,
                "--output", str(root / "soma"), *optional_flag("--assets", config.soma_assets),
            ),
            soma_file,
            layers={"soma": soma_file},
            jobs=config.soma_jobs,
            detail_markers=("SOMA_TIME",),
        ),
        segments,
    )
    segments = run_segments(
        run,
        SegmentStep(
            "soma_xlo",
            lambda selected: task(
                "selfcap-soma", "soma-xlo", "--soma-dir", str(root / "soma"), "--dataset", config.dataset, *optional_flag("--assets", config.soma_assets),
                "--output-dir", str(root / "soma_xlo"), "--segment", *selected,
            ),
            soma_xlo_file,
            layers={"soma_xlo": soma_xlo_file},
            one_process=True,
        ),
        segments,
    )
    segments = run_segments(
        run,
        SegmentStep(
            "preview",
            lambda selected: task(
                "selfcap-curate", "selfcap-preview", "--catalog-url", config.catalog_url, "--dataset", config.dataset, "--output-dir", str(root / "preview"),
                "--segment-ids", *selected, "--workers", str(config.preview_workers),
            ),
            preview_file,
            layers={"preview": preview_file},
            one_process=True,
        ),
        segments,
    )
    catalog_step(run, "blueprints", segments, publish_blueprints, {})
    run.summary()
    failed: int = sum(count.failed for count in run.counts)
    if failed:
        raise SystemExit(f"{failed} segment failures; see {root / 'progress.log'}")
