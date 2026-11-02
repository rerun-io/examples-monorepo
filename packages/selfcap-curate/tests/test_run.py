"""``selfcap-run`` against a fake subprocess runner and an in-process catalog: the exact commands, the order, and who survives."""

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Barrier, Lock
from time import sleep

import pytest
import rerun as rr
from dataforge import paths
from dataforge.timing import ConvertRecord, append_record
from rerun.catalog import CatalogClient, DatasetEntry
from rerun.server import Server

from selfcap_curate.apis import run
from selfcap_curate.apis.run import Config

MANIFEST: str = str(Path(__file__).resolve().parents[3] / "pixi.toml")
DATASET: str = "selfcap-new"
real_run_command: Callable[[list[str], dict[str, str], Path], int] = run.run_command
SEGMENT_A: str = "selfcap__2025-12-20__25aeb66a__ep001"
SEGMENT_B: str = "selfcap__2025-12-21__1dbe8c49__ep002"
SESSION_A: str = "2025-12-20__25aeb66a"
SESSION_B: str = "2025-12-21__1dbe8c49"
SEGMENT_A2: str = f"selfcap__{SESSION_A}__ep002"
ONE_PROCESS_TASKS: dict[str, str] = {"soma-xlo": "--segment", "selfcap-preview": "--segment-ids"}
"""Tasks with one process for every segment, and the flag that lists the segments."""


def write_soma_layer(path: Path, segment: str, *, status: str, body_mm: float | None) -> Path:
    """A tiny layer with the catalog property a SOMA layer carries; a missing metric is left out, as ``log_quality`` does."""
    metrics: dict[str, float] = {} if body_mm is None else {"body_median_mm": body_mm}
    with rr.RecordingStream("selfcap-run-test", recording_id=segment) as recording:
        recording.save(path)
        recording.log("/__properties/soma_quality", rr.AnyValues(status=status, **metrics), static=True)
    return path


def pixi(environment: str, task: str, *arguments: str) -> list[str]:
    return ["pixi", "run", "--manifest-path", MANIFEST, "-e", environment, "--frozen", task, *arguments]


@dataclass
class FakePipeline:
    """Stands in for ``run.run_command``: records each call and writes what the real task writes.

    ``failing`` maps a task to its failing segments and the last line of each one's log; ``""`` as the segment fails the whole task.
    """

    catalog_url: str
    raw_segments: tuple[str, ...] = (SEGMENT_A, SEGMENT_B)
    failing: dict[str, dict[str, str]] = field(default_factory=dict)
    quality: dict[str, tuple[str, float | None]] = field(default_factory=dict)
    registered_layers: set[str] = field(default_factory=set)
    """Layers ``dataforge-swap-layer`` really registers, each file rewritten as a small recording."""
    calls: list[tuple[dict[str, str], list[str]]] = field(default_factory=list)

    def __call__(self, command: list[str], env: dict[str, str], log_path: Path) -> int:
        self.calls.append((env, command))
        task: str = command[command.index("--frozen") + 1]
        arguments: list[str] = command[command.index("--frozen") + 2 :]
        for flag in ("--segment-ids", "--segment", "--sequence"):
            segment: str = arguments[arguments.index(flag) + 1] if flag in arguments else ""
            if segment:
                break
        failures: dict[str, str] = self.failing.get(task, {})
        if task not in ONE_PROCESS_TASKS and (segment in failures or "" in failures):
            log_path.write_text(f"some progress\n{failures.get(segment, failures.get('', ''))}\n")
            return 1
        log_path.write_text("some progress\n")
        return WRITERS[task](self, env, arguments, segment, log_path)

    def task_names(self) -> list[str]:
        return [command[command.index("--frozen") + 1] for _, command in self.calls]

    def commands_of(self, task: str) -> list[list[str]]:
        return [command for _, command in self.calls if command[command.index("--frozen") + 1] == task]

    def swapped_files(self, layer: str) -> list[str]:
        commands: list[list[str]] = [command for command in self.commands_of("dataforge-swap-layer") if command[command.index("--layer") + 1] == layer]
        assert len(commands) <= 1
        return [] if not commands else commands[0][commands[0].index("--files") + 1 : commands[0].index("--catalog-url")]


def progress_messages(root: Path) -> list[str]:
    """Each line of progress.log after its ``[HH:MM:SS] <dataset>: `` prefix."""
    lines: list[str] = (root / "progress.log").read_text().splitlines()
    assert all(re.fullmatch(rf"\[\d\d:\d\d:\d\d\] {DATASET}: .+", line) for line in lines)
    return [line.split(": ", 1)[1] for line in lines]


def touch(*paths: Path) -> int:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return 0


def flag_value(arguments: list[str], flag: str) -> Path:
    return Path(arguments[arguments.index(flag) + 1])


def record_conversion(env: dict[str, str], name: str, error: str | None = None) -> None:
    """The timing record ``dataforge-convert`` appends for one sequence."""
    record: ConvertRecord = ConvertRecord(
        dataset="selfcap", recording_id=name, converter_version="test", started_at="now", stage_s={}, total_s=0.0, capture_s=None, layer_bytes={}, host="here", skipped=False, error=error,
    )
    append_record(Path(env["DATAFORGE_OUTPUT_ROOT"]) / "timing/convert.jsonl", record)


def fake_convert(pipeline: FakePipeline, env: dict[str, str], _: list[str], segment: str, __: Path) -> int:
    root: Path = Path(env["DATAFORGE_OUTPUT_ROOT"])
    for name in (segment,) if segment else pipeline.raw_segments:
        touch(root / "base" / f"{name}.rrd", root / "gt" / f"{name}.rrd")
        record_conversion(env, name)
    return 0


def fake_register(pipeline: FakePipeline, _: dict[str, str], arguments: list[str], __: str, ___: Path) -> int:
    CatalogClient(pipeline.catalog_url).create_dataset(arguments[arguments.index("--catalog-name") + 1], exist_ok=True)
    return 0


def fake_align(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, ___: Path) -> int:
    output: Path = flag_value(arguments, "--output-dir")
    for each in arguments[arguments.index("--segment-ids") + 1 :]:
        touch(output / f"{each}.rrd", output / "gt" / f"{each}.rrd", output / "sapiens2" / f"{each}.rrd")
    return 0


def fake_calibrate(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, ___: Path) -> int:
    output: Path = flag_value(arguments, "--output-dir")
    return touch(output / f"{segment}.rrd", output / "egohand" / f"{segment}.rrd")


def fake_one_file(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, ___: Path) -> int:
    return touch(flag_value(arguments, "--output-dir") / f"{segment}.rrd")


def fake_soma(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, ___: Path) -> int:
    return touch(flag_value(arguments, "--output") / arguments[arguments.index("--dataset") + 1] / segment / f"{segment}.rrd")


def fake_swap(pipeline: FakePipeline, _: dict[str, str], arguments: list[str], __: str, log_path: Path) -> int:
    """The soma layer and ``registered_layers`` reach the catalog: nothing else is read from it."""
    layer: str = arguments[arguments.index("--layer") + 1]
    entry: DatasetEntry = CatalogClient(pipeline.catalog_url).get_dataset(arguments[arguments.index("--catalog-name") + 1])
    if layer in pipeline.registered_layers:
        for file in map(Path, arguments[arguments.index("--files") + 1 : arguments.index("--catalog-url")]):
            entry.register([write_soma_layer(file, file.stem, status="pass", body_mm=None).as_uri()], layer_name=layer).wait()
    if layer == "soma":
        for name, (status, body_mm) in pipeline.quality.items():
            entry.register([write_soma_layer(log_path.parent / f"catalog-{name}.rrd", name, status=status, body_mm=body_mm).as_uri()], layer_name="soma").wait()
    return 0


def fake_one_process(task: str) -> Callable[[FakePipeline, dict[str, str], list[str], str, Path], int]:
    """One process for every segment: a segment named in ``failing`` gets no file, and its reason ends the log."""

    def write(pipeline: FakePipeline, _: dict[str, str], arguments: list[str], __: str, log_path: Path) -> int:
        failures: dict[str, str] = pipeline.failing.get(task, {})
        start: int = arguments.index(ONE_PROCESS_TASKS[task]) + 1
        for name in arguments[start : arguments.index("--workers") if "--workers" in arguments else None]:
            if name not in failures:
                touch(flag_value(arguments, "--output-dir") / f"{name}.rrd")
        log_path.write_text("\n".join(["some progress", *failures.values()]) + "\n")
        return 1 if failures else 0

    return write


def fake_nothing(_: FakePipeline, __: dict[str, str], ___: list[str], ____: str, _____: Path) -> int:
    return 0


WRITERS: dict[str, Callable[[FakePipeline, dict[str, str], list[str], str, Path], int]] = {
    "dataforge-convert": fake_convert,
    "dataforge-register": fake_register,
    "selfcap-align": fake_align,
    "dataforge-swap-layer": fake_swap,
    "selfcap-calibrate": fake_calibrate,
    "selfcap-keypoints": fake_one_file,
    "soma-batch": fake_soma,
    "soma-xlo": fake_one_process("soma-xlo"),
    "selfcap-preview": fake_one_process("selfcap-preview"),
    "dataforge-publish-blueprints": fake_nothing,
}


@pytest.fixture
def pipeline(catalog_url: str, monkeypatch: pytest.MonkeyPatch) -> FakePipeline:
    fake: FakePipeline = FakePipeline(catalog_url)
    monkeypatch.setattr(run, "run_command", fake)
    return fake


@pytest.fixture
def catalog_url() -> Iterator[str]:
    with Server(port=0) as server:
        yield server.url()


@pytest.fixture
def config(tmp_path: Path, catalog_url: str) -> Config:
    """A raw_root run; the inputs exist, the run root does not."""
    for directory in ("input/raw/selfcap", "input/egohand", "input/soma-assets"):
        (tmp_path / directory).mkdir(parents=True)
    return Config(
        run_root=tmp_path / "run", catalog_url=catalog_url, dataset=DATASET, oak_egohand_assets=tmp_path / "input/egohand",
        soma_assets=tmp_path / "input/soma-assets", raw_root=tmp_path / "input/raw", donor_catalog_url="rerun+http://donor:1",
    )


def test_both_input_roots_are_refused(config: Config, tmp_path: Path) -> None:
    (tmp_path / "input/base").mkdir()
    with pytest.raises(ValueError, match="exactly one of raw_root and base_dir"):
        run.main(replace(config, base_dir=tmp_path / "input/base"))
    assert not config.run_root.exists()


def test_neither_input_root_is_refused(config: Config) -> None:
    with pytest.raises(ValueError, match="exactly one of raw_root and base_dir"):
        run.main(replace(config, raw_root=None))
    assert not config.run_root.exists()


def test_an_existing_run_root_is_refused(config: Config) -> None:
    config.run_root.mkdir()
    (config.run_root / "kept").write_text("kept")
    with pytest.raises(FileExistsError):
        run.main(config)
    assert sorted(path.name for path in config.run_root.iterdir()) == ["kept"]


def test_an_existing_dataset_is_refused_and_keeps_its_contents(config: Config, catalog_url: str, tmp_path: Path) -> None:
    other: DatasetEntry = CatalogClient(catalog_url).create_dataset(DATASET)
    source: Path = write_soma_layer(tmp_path / "other.rrd", SEGMENT_A, status="pass", body_mm=10.0)
    other.register([source.as_uri()], layer_name="soma").wait()
    with pytest.raises(ValueError, match=f"dataset {DATASET} already exists"):
        run.main(config)
    assert not config.run_root.exists()
    assert other.segment_ids() == [SEGMENT_A]


@pytest.mark.parametrize(("field", "value"), [("jobs", 0), ("soma_jobs", 0), ("preview_workers", 0)])
def test_a_config_that_cannot_run_is_refused(config: Config, field: str, value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run.main(replace(config, **{field: value}))
    assert not config.run_root.exists()


@pytest.mark.parametrize("field", ["raw_root", "oak_egohand_assets", "soma_assets"])
def test_a_missing_input_directory_stops_before_any_step(config: Config, tmp_path: Path, field: str) -> None:
    with pytest.raises(FileNotFoundError, match="absent"):
        run.main(replace(config, **{field: tmp_path / "absent"}))
    assert not config.run_root.exists()


def test_a_run_root_on_the_nas_is_refused(config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "NAS_ROOT", tmp_path)
    with pytest.raises(ValueError, match="refusing to write"):
        run.main(config)
    assert not config.run_root.exists()


def test_a_malformed_segment_id_is_refused_before_any_step(config: Config, pipeline: FakePipeline) -> None:
    with pytest.raises(ValueError, match="invalid session episode key"):
        run.main(replace(config, segment_ids=(SEGMENT_A, "selfcap__not-an-episode")))
    assert not config.run_root.exists()
    assert pipeline.calls == []


def test_segment_ids_reach_the_tasks_as_recording_ids_once_each(config: Config, pipeline: FakePipeline) -> None:
    run.main(replace(config, segment_ids=(SEGMENT_A.removeprefix("selfcap__"), SEGMENT_A)))
    assert [command[-2:] for command in pipeline.commands_of("dataforge-convert")] == [[SEGMENT_A, "selfcap"]]


def test_a_raw_root_without_selfcap_captures_is_refused(config: Config, tmp_path: Path) -> None:
    (tmp_path / "other-raw").mkdir()
    with pytest.raises(FileNotFoundError, match="selfcap"):
        run.main(replace(config, raw_root=tmp_path / "other-raw"))
    assert not config.run_root.exists()


def test_a_raw_root_run_gives_every_step_its_exact_command_in_order(config: Config, pipeline: FakePipeline, catalog_url: str, tmp_path: Path) -> None:
    run.main(replace(config, jobs=1, soma_jobs=1))

    root: Path = tmp_path / "run"
    raw: str = str(tmp_path / "input/raw")
    assets: str = str(tmp_path / "input/soma-assets")
    donor: str = "rerun+http://donor:1"
    a, b = SEGMENT_A, SEGMENT_B
    catalog: list[str] = ["--catalog-url", catalog_url, "--catalog-name", DATASET]
    source: list[str] = ["--catalog-url", catalog_url, "--dataset", DATASET]

    def align(segment: str, session: str) -> list[str]:
        return pixi("selfcap-curate", "selfcap-align", "--base-dir", f"{root}/rrd/base", "--output-dir", f"{root}/align/{session}", *source, "--segment-ids", segment)

    def calibrate(segment: str) -> list[str]:
        return pixi(
            "selfcap-curate", "selfcap-calibrate", "--output-dir", f"{root}/calib/{segment}", "--segment-ids", segment, *source,
            "--exo-method", "head", "--oak-egohand-assets", str(tmp_path / "input/egohand"), "--donor-catalog-url", donor,
        )

    def keypoints(segment: str) -> list[str]:
        return pixi("selfcap-curate", "selfcap-keypoints", "--calib-dir", f"{root}/calib/{segment}", "--output-dir", f"{root}/keypoints/{segment}", "--segment-ids", segment, *source)

    def soma(segment: str) -> list[str]:
        return pixi("selfcap-soma", "soma-batch", "--dataset", DATASET, "--segment", segment, "--catalog-url", catalog_url, "--output", f"{root}/soma", "--assets", assets)

    def publish() -> list[str]:
        return pixi("dataforge", "dataforge-publish-blueprints", *catalog, "--output-root", f"{root}/catalog", "selfcap")

    def swap(layer: str, *files: str) -> list[str]:
        return pixi("dataforge", "dataforge-swap-layer", "--layer", layer, "--files", *files, *catalog)

    assert pipeline.calls == [
        ({"DATAFORGE_RAW_ROOT": raw, "DATAFORGE_OUTPUT_ROOT": f"{root}/rrd"}, pixi("dataforge", "dataforge-convert", "selfcap")),
        ({"DATAFORGE_OUTPUT_ROOT": f"{root}/rrd"}, pixi("dataforge", "dataforge-register", *catalog, "selfcap")),
        ({}, align(a, SESSION_A)),
        ({}, align(b, SESSION_B)),
        ({}, swap("base", f"{root}/align/{SESSION_A}/{a}.rrd", f"{root}/align/{SESSION_B}/{b}.rrd")),
        ({}, swap("gt", f"{root}/align/{SESSION_A}/gt/{a}.rrd", f"{root}/align/{SESSION_B}/gt/{b}.rrd")),
        ({}, swap("sapiens2", f"{root}/align/{SESSION_A}/sapiens2/{a}.rrd", f"{root}/align/{SESSION_B}/sapiens2/{b}.rrd")),
        ({}, calibrate(a)),
        ({}, calibrate(b)),
        ({}, swap("egohand", f"{root}/calib/{a}/egohand/{a}.rrd", f"{root}/calib/{b}/egohand/{b}.rrd")),
        ({}, keypoints(a)),
        ({}, keypoints(b)),
        ({}, swap("calib", f"{root}/keypoints/{a}/{a}.rrd", f"{root}/keypoints/{b}/{b}.rrd")),
        ({}, publish()),
        ({}, soma(a)),
        ({}, soma(b)),
        ({}, swap("soma", f"{root}/soma/{DATASET}/{a}/{a}.rrd", f"{root}/soma/{DATASET}/{b}/{b}.rrd")),
        ({}, pixi("selfcap-soma", "soma-xlo", "--soma-dir", f"{root}/soma", "--dataset", DATASET, "--assets", assets, "--output-dir", f"{root}/soma_xlo", "--segment", a, b)),
        ({}, swap("soma_xlo", f"{root}/soma_xlo/{a}.rrd", f"{root}/soma_xlo/{b}.rrd")),
        ({}, pixi("selfcap-curate", "selfcap-preview", *source, "--output-dir", f"{root}/preview", "--segment-ids", a, b, "--workers", "3")),
        ({}, swap("preview", f"{root}/preview/{a}.rrd", f"{root}/preview/{b}.rrd")),
        ({}, publish()),
    ]


def as_cut_bases(tmp_path: Path, *segments: str) -> Path:
    directory: Path = tmp_path / "input/as-cut"
    directory.mkdir()
    for segment in segments:
        (directory / f"{segment}.rrd").touch()
    (directory / "not-a-segment.rrd").touch()
    return directory


def test_a_base_dir_run_skips_ingest_and_registers_links_to_the_bases(config: Config, pipeline: FakePipeline, catalog_url: str, tmp_path: Path) -> None:
    bases: Path = as_cut_bases(tmp_path, SEGMENT_A, SEGMENT_B)
    run.main(replace(config, raw_root=None, base_dir=bases))

    root: Path = tmp_path / "run"
    assert "dataforge-convert" not in pipeline.task_names()
    assert pipeline.calls[0] == (
        {"DATAFORGE_OUTPUT_ROOT": f"{root}/rrd"},
        pixi("dataforge", "dataforge-register", "--catalog-url", catalog_url, "--catalog-name", DATASET, "selfcap"),
    )
    assert sorted(path.name for path in (root / "rrd/base").iterdir()) == [f"{SEGMENT_A}.rrd", f"{SEGMENT_B}.rrd"]
    assert (root / "rrd/base" / f"{SEGMENT_A}.rrd").resolve() == bases / f"{SEGMENT_A}.rrd"
    assert pipeline.commands_of("selfcap-align")[0][pipeline.commands_of("selfcap-align")[0].index("--base-dir") + 1] == f"{root}/rrd/base"


def test_segment_ids_select_bases_and_a_missing_one_counts_as_failed(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    bases: Path = as_cut_bases(tmp_path, SEGMENT_A, SEGMENT_B)
    absent: str = "selfcap__2025-12-22__00000000__ep001"
    with pytest.raises(SystemExit):
        run.main(replace(config, raw_root=None, base_dir=bases, segment_ids=(SEGMENT_A, absent)))

    root: Path = tmp_path / "run"
    assert [path.name for path in (root / "rrd/base").iterdir()] == [f"{SEGMENT_A}.rrd"]
    assert [command[command.index("--segment-ids") + 1] for command in pipeline.commands_of("selfcap-align")] == [SEGMENT_A]
    assert f"base FAIL {absent} (0 s): no file {bases}/{absent}.rrd" in progress_messages(root)


def test_a_raw_root_run_with_segment_ids_ingests_only_those_one_at_a_time(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.failing["dataforge-convert"] = {SEGMENT_B: "ValueError: No sequence matches"}
    with pytest.raises(SystemExit):
        run.main(replace(config, segment_ids=(SEGMENT_A, SEGMENT_B)))

    root: Path = tmp_path / "run"
    env: dict[str, str] = {"DATAFORGE_RAW_ROOT": str(tmp_path / "input/raw"), "DATAFORGE_OUTPUT_ROOT": f"{root}/rrd"}
    assert pipeline.calls[:2] == [
        (env, pixi("dataforge", "dataforge-convert", "--sequence", SEGMENT_A, "selfcap")),
        (env, pixi("dataforge", "dataforge-convert", "--sequence", SEGMENT_B, "selfcap")),
    ]
    messages: list[str] = progress_messages(root)
    assert f"ingest ok   {SEGMENT_A} (0 s)" in messages
    assert f"ingest FAIL {SEGMENT_B} (0 s): ValueError: No sequence matches" in messages


def test_a_segment_that_fails_calibrate_is_logged_gets_no_further_step_and_is_absent_from_later_swaps(
    config: Config, pipeline: FakePipeline, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline.failing["selfcap-calibrate"] = {SEGMENT_B: "ValueError: no head track"}
    with pytest.raises(SystemExit) as stopped:
        run.main(replace(config, jobs=1))
    assert stopped.value.code not in (0, None)

    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    assert f"calib ok   {SEGMENT_A} (0 s)" in messages
    assert f"calib FAIL {SEGMENT_B} (0 s): ValueError: no head track" in messages
    assert (root / "logs" / f"calib-{SEGMENT_B}.log").read_text() == "some progress\nValueError: no head track\n"
    for later in ("selfcap-keypoints", "soma-batch", "soma-xlo"):
        assert [SEGMENT_B in " ".join(command) for command in pipeline.commands_of(later)] == [False]
    assert SEGMENT_B not in " ".join(pipeline.commands_of("selfcap-preview")[0])
    assert pipeline.swapped_files("base") == [f"{root}/align/{SESSION_A}/{SEGMENT_A}.rrd", f"{root}/align/{SESSION_B}/{SEGMENT_B}.rrd"]
    assert pipeline.swapped_files("egohand") == [f"{root}/calib/{SEGMENT_A}/egohand/{SEGMENT_A}.rrd"]
    assert pipeline.swapped_files("calib") == [f"{root}/keypoints/{SEGMENT_A}/{SEGMENT_A}.rrd"]
    assert pipeline.swapped_files("soma") == [f"{root}/soma/{DATASET}/{SEGMENT_A}/{SEGMENT_A}.rrd"]
    assert pipeline.swapped_files("soma_xlo") == [f"{root}/soma_xlo/{SEGMENT_A}.rrd"]
    assert pipeline.swapped_files("preview") == [f"{root}/preview/{SEGMENT_A}.rrd"]
    assert "SUMMARY calib: ok 1, failed 1, wall 0 s" in capsys.readouterr().out


def test_the_reason_skips_the_blank_lines_and_box_borders_after_the_failure(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.failing["selfcap-calibrate"] = {SEGMENT_B: "RuntimeError: head fit diverged\n\n╰────────╯\n   \n"}
    with pytest.raises(SystemExit):
        run.main(config)
    assert f"calib FAIL {SEGMENT_B} (0 s): RuntimeError: head fit diverged" in progress_messages(tmp_path / "run")


def test_a_clean_exit_without_the_output_file_fails_the_segment(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(WRITERS, "selfcap-keypoints", fake_nothing)
    with pytest.raises(SystemExit, match="no segments left after keypoints"):
        run.main(config)
    root: Path = tmp_path / "run"
    assert f"keypoints FAIL {SEGMENT_A} (0 s): exit 0 but no {root}/keypoints/{SEGMENT_A}/{SEGMENT_A}.rrd" in progress_messages(root)


def test_zero_survivors_stop_the_run_nonzero_with_the_counts(config: Config, pipeline: FakePipeline, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pipeline.failing["selfcap-calibrate"] = {SEGMENT_A: "ValueError: one", SEGMENT_B: "ValueError: two"}
    with pytest.raises(SystemExit, match="no segments left after calib") as stopped:
        run.main(config)
    assert stopped.value.code not in (0, None)
    assert pipeline.task_names().count("selfcap-keypoints") == 0
    assert pipeline.swapped_files("egohand") == []
    assert pipeline.task_names()[-1] == "selfcap-calibrate"
    assert "SUMMARY calib: ok 0, failed 2, wall 0 s" in capsys.readouterr().out


def test_no_ingested_segment_stops_before_register(config: Config, pipeline: FakePipeline) -> None:
    pipeline.raw_segments = ()
    with pytest.raises(SystemExit, match="no segments left after ingest"):
        run.main(config)
    assert pipeline.task_names() == ["dataforge-convert"]


def test_no_donor_uses_the_catalog(config: Config, pipeline: FakePipeline, catalog_url: str) -> None:
    run.main(replace(config, donor_catalog_url=None))
    for command in pipeline.commands_of("selfcap-calibrate"):
        assert command[command.index("--donor-catalog-url") + 1] == catalog_url


def test_a_preview_file_proves_the_segment_and_only_its_survivors_are_swapped(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.failing["selfcap-preview"] = {SEGMENT_B: "preview resolution mismatch"}
    with pytest.raises(SystemExit):
        run.main(config)
    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    assert f"preview ok   {SEGMENT_A}" in messages
    assert f"preview FAIL {SEGMENT_B}: preview resolution mismatch; see {root}/logs/preview.log" in messages
    assert pipeline.swapped_files("preview") == [f"{root}/preview/{SEGMENT_A}.rrd"]
    assert pipeline.task_names()[-1] == "dataforge-publish-blueprints"


def test_a_crash_of_the_one_soma_xlo_process_fails_only_the_segments_with_no_file(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(_: FakePipeline, __: dict[str, str], arguments: list[str], ___: str, log_path: Path) -> int:
        touch(flag_value(arguments, "--output-dir") / f"{SEGMENT_A}.rrd")
        log_path.write_text("Segmentation fault\n")
        return 139

    monkeypatch.setitem(WRITERS, "soma-xlo", crash)
    with pytest.raises(SystemExit):
        run.main(config)
    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    assert f"soma_xlo ok   {SEGMENT_A}" in messages
    assert f"soma_xlo FAIL {SEGMENT_B}: Segmentation fault; see {root}/logs/soma_xlo.log" in messages
    assert pipeline.swapped_files("soma_xlo") == [f"{root}/soma_xlo/{SEGMENT_A}.rrd"]
    assert SEGMENT_B not in " ".join(pipeline.commands_of("selfcap-preview")[0])
    assert "SUMMARY soma_xlo: ok 1, failed 1, wall 0 s" in messages


def test_no_preview_survivor_stops_before_the_last_blueprints(config: Config, pipeline: FakePipeline) -> None:
    pipeline.failing["selfcap-preview"] = {SEGMENT_A: "bad", SEGMENT_B: "worse"}
    with pytest.raises(SystemExit, match="no segments left after preview"):
        run.main(config)
    assert pipeline.task_names().count("dataforge-publish-blueprints") == 1


def test_a_failed_swap_stops_the_run_because_later_steps_would_read_the_old_layer(config: Config, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def refuse(_: FakePipeline, __: dict[str, str], ___: list[str], ____: str, log_path: Path) -> int:
        log_path.write_text("TimeoutError: idle gate timed out after 600s for layer base\n")
        return 1

    monkeypatch.setitem(WRITERS, "dataforge-swap-layer", refuse)
    with pytest.raises(SystemExit, match="swap base FAILED: TimeoutError: idle gate timed out after 600s for layer base"):
        run.main(config)
    assert pipeline.task_names()[-1] == "dataforge-swap-layer"
    assert "selfcap-calibrate" not in pipeline.task_names()
    assert "SUMMARY align: ok 2, failed 0, wall 0 s" in progress_messages(tmp_path / "run")


@pytest.mark.parametrize(("task", "survives"), [("dataforge-register", False), ("dataforge-publish-blueprints", True)])
def test_a_failed_catalog_step_stops_the_run_nonzero(config: Config, pipeline: FakePipeline, tmp_path: Path, task: str, survives: bool) -> None:
    pipeline.failing[task] = {"": "ConnectionError: catalog went away"}
    with pytest.raises(SystemExit, match="ConnectionError: catalog went away"):
        run.main(config)
    assert ("selfcap-align" in pipeline.task_names()) == survives
    assert pipeline.task_names()[-1] == task


def test_a_layer_no_segment_wrote_is_logged_and_not_swapped(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def calibrate_without_egohand(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, ___: Path) -> int:
        return touch(flag_value(arguments, "--output-dir") / f"{segment}.rrd")

    monkeypatch.setitem(WRITERS, "selfcap-calibrate", calibrate_without_egohand)
    run.main(config)
    assert pipeline.swapped_files("egohand") == []
    assert "swap egohand: no files" in progress_messages(tmp_path / "run")


def test_a_failed_ingest_of_one_sequence_is_counted_and_skipped_by_its_timing_record(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record decides, not the log's wording or a file the failed conversion left."""

    def convert_one_badly(_: FakePipeline, env: dict[str, str], __: list[str], ___: str, log_path: Path) -> int:
        touch(Path(env["DATAFORGE_OUTPUT_ROOT"]) / "base" / f"{SEGMENT_A}.rrd")
        touch(Path(env["DATAFORGE_OUTPUT_ROOT"]) / "base" / f"{SEGMENT_B}.rrd")
        record_conversion(env, SEGMENT_A)
        record_conversion(env, SEGMENT_B, "ValueError: bad video")
        log_path.write_text("converting 2 sequence(s)\n1 of 2 sequence(s) failed\n")
        return 1

    monkeypatch.setitem(WRITERS, "dataforge-convert", convert_one_badly)
    with pytest.raises(SystemExit):
        run.main(config)
    messages: list[str] = progress_messages(tmp_path / "run")
    assert f"ingest ok   {SEGMENT_A}" in messages
    assert f"ingest FAIL {SEGMENT_B}: ValueError: bad video" in messages
    assert [command[command.index("--segment-ids") + 1] for command in pipeline.commands_of("selfcap-align")] == [SEGMENT_A]
    assert "SUMMARY ingest: ok 1, failed 1, wall 0 s" in messages


def test_an_ingest_crash_with_no_sequence_named_still_ends_nonzero(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def convert_then_crash(_: FakePipeline, env: dict[str, str], __: list[str], ___: str, log_path: Path) -> int:
        touch(Path(env["DATAFORGE_OUTPUT_ROOT"]) / "base" / f"{SEGMENT_A}.rrd", Path(env["DATAFORGE_OUTPUT_ROOT"]) / "base" / f"{SEGMENT_B}.rrd")
        record_conversion(env, SEGMENT_A)
        log_path.write_text("converting 2 sequence(s)\nKilled\n")
        return 137

    monkeypatch.setitem(WRITERS, "dataforge-convert", convert_then_crash)
    with pytest.raises(SystemExit):
        run.main(config)
    messages: list[str] = progress_messages(tmp_path / "run")
    assert "ingest returned an error: Killed" in messages
    assert "SUMMARY ingest: ok 1, failed 1, wall 0 s" in messages
    assert [command[command.index("--segment-ids") + 1] for command in pipeline.commands_of("selfcap-align")] == [SEGMENT_A]


def test_an_ingest_record_without_its_base_file_fails_the_segment(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def convert_one_without_a_file(_: FakePipeline, env: dict[str, str], __: list[str], ___: str, ____: Path) -> int:
        touch(Path(env["DATAFORGE_OUTPUT_ROOT"]) / "base" / f"{SEGMENT_A}.rrd")
        record_conversion(env, SEGMENT_A)
        record_conversion(env, SEGMENT_B)
        return 0

    monkeypatch.setitem(WRITERS, "dataforge-convert", convert_one_without_a_file)
    with pytest.raises(SystemExit):
        run.main(config)
    root: Path = tmp_path / "run"
    assert f"ingest FAIL {SEGMENT_B}: exit 0 but no {root}/rrd/base/{SEGMENT_B}.rrd" in progress_messages(root)


def test_the_episodes_of_a_session_align_in_one_process_with_their_files_side_by_side(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    """Align pools evidence across a session's episodes; head calibration stays per episode."""
    run.main(replace(config, segment_ids=(SEGMENT_A, SEGMENT_A2, SEGMENT_B)))
    root: Path = tmp_path / "run"
    aligns: list[list[str]] = pipeline.commands_of("selfcap-align")
    assert [command[command.index("--output-dir") + 1 :] for command in aligns] == [
        [f"{root}/align/{SESSION_A}", "--catalog-url", pipeline.catalog_url, "--dataset", DATASET, "--segment-ids", SEGMENT_A, SEGMENT_A2],
        [f"{root}/align/{SESSION_B}", "--catalog-url", pipeline.catalog_url, "--dataset", DATASET, "--segment-ids", SEGMENT_B],
    ]
    assert pipeline.swapped_files("base") == [f"{root}/align/{SESSION_A}/{SEGMENT_A}.rrd", f"{root}/align/{SESSION_A}/{SEGMENT_A2}.rrd", f"{root}/align/{SESSION_B}/{SEGMENT_B}.rrd"]
    assert [command[command.index("--segment-ids") + 1 :][:1] for command in pipeline.commands_of("selfcap-calibrate")] == [[SEGMENT_A], [SEGMENT_A2], [SEGMENT_B]]
    assert "SUMMARY align: ok 3, failed 0, wall 0 s" in progress_messages(root)


def test_a_failed_session_fails_each_of_its_episodes_and_the_others_go_on(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.failing["selfcap-align"] = {SEGMENT_A: "ValueError: the session has no phone clock"}
    with pytest.raises(SystemExit, match="2 segment failures"):
        run.main(replace(config, segment_ids=(SEGMENT_A, SEGMENT_A2, SEGMENT_B)))
    messages: list[str] = progress_messages(tmp_path / "run")
    for segment in (SEGMENT_A, SEGMENT_A2):
        assert f"align FAIL {segment} (0 s): ValueError: the session has no phone clock" in messages
    assert "SUMMARY align: ok 1, failed 2, wall 0 s" in messages
    assert [command[command.index("--segment-ids") + 1] for command in pipeline.commands_of("selfcap-calibrate")] == [SEGMENT_B]


def test_superseded_files_go_once_the_catalog_shows_the_new_ones(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.registered_layers = {"base", "gt", "calib"}
    run.main(config)
    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    for segment in (SEGMENT_A, SEGMENT_B):
        for superseded in (root / "rrd/base" / f"{segment}.rrd", root / "rrd/gt" / f"{segment}.rrd", root / "calib" / segment / f"{segment}.rrd"):
            assert not superseded.exists()
            assert f"removed superseded {superseded}" in messages
        for kept in (root / "align" / segment.split("__", 1)[1].rsplit("__", 1)[0] / f"{segment}.rrd", root / "keypoints" / segment / f"{segment}.rrd", root / "calib" / segment / "egohand" / f"{segment}.rrd"):
            assert kept.is_file()


def test_a_superseded_file_stays_while_the_catalog_does_not_show_the_new_one(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    run.main(config)
    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    assert (root / "rrd/base" / f"{SEGMENT_A}.rrd").is_file()
    assert (root / "calib" / SEGMENT_A / f"{SEGMENT_A}.rrd").is_file()
    assert f"kept {root}/calib/{SEGMENT_A}/{SEGMENT_A}.rrd: the catalog does not show {root}/keypoints/{SEGMENT_A}/{SEGMENT_A}.rrd as calib" in messages


def test_a_failed_segment_keeps_its_superseded_file(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.registered_layers = {"calib"}
    pipeline.failing["selfcap-keypoints"] = {SEGMENT_B: "ValueError: no keypoints"}
    with pytest.raises(SystemExit):
        run.main(config)
    root: Path = tmp_path / "run"
    assert not (root / "calib" / SEGMENT_A / f"{SEGMENT_A}.rrd").exists()
    assert (root / "calib" / SEGMENT_B / f"{SEGMENT_B}.rrd").is_file()


def test_a_base_dir_run_removes_its_link_and_never_the_as_cut_file(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    bases: Path = as_cut_bases(tmp_path, SEGMENT_A, SEGMENT_B)
    pipeline.registered_layers = {"base"}
    run.main(replace(config, raw_root=None, base_dir=bases))
    assert not (tmp_path / "run/rrd/base" / f"{SEGMENT_A}.rrd").is_symlink()
    assert (bases / f"{SEGMENT_A}.rrd").is_file()


def test_the_summary_reads_soma_pass_flag_and_medians_from_the_segment_table(config: Config, pipeline: FakePipeline, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pipeline.quality = {SEGMENT_A: ("pass", 20.0), SEGMENT_B: ("flag", 50.0), "selfcap__2025-12-22__3a5ae0c4__ep001": ("pass", 30.0)}
    run.main(config)
    root: Path = tmp_path / "run"
    messages: list[str] = progress_messages(root)
    assert "SOMA_QUALITY pass 2, flag 1, median mm: body 30.0" in messages
    assert not list(root.rglob("*.json")), "no JSON"
    assert "SOMA_QUALITY pass 2, flag 1, median mm: body 30.0" in capsys.readouterr().out


def test_the_summary_prints_every_step_and_the_wall_lines_number_the_steps_in_order(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    run.main(config)
    messages: list[str] = progress_messages(tmp_path / "run")
    assert [message.split(" ")[0] for message in messages if "_WALL " in message] == [
        "STEP1_INGEST_WALL", "STEP2_REGISTER_WALL", "STEP3_ALIGN_WALL", "STEP4_CALIB_WALL", "STEP5_KEYPOINTS_WALL",
        "STEP6_PUBLISH_WALL", "STEP7_SOMA_WALL", "STEP8_SOMA_XLO_WALL", "STEP9_PREVIEW_WALL", "STEP10_BLUEPRINTS_WALL",
    ]
    assert [message for message in messages if message.startswith("SUMMARY ")] == [
        "SUMMARY ingest: ok 2, failed 0, wall 0 s",
        "SUMMARY register: ok 2, failed 0, wall 0 s",
        "SUMMARY align: ok 2, failed 0, wall 0 s",
        "SUMMARY calib: ok 2, failed 0, wall 0 s",
        "SUMMARY keypoints: ok 2, failed 0, wall 0 s",
        "SUMMARY publish: ok 2, failed 0, wall 0 s",
        "SUMMARY soma: ok 2, failed 0, wall 0 s",
        "SUMMARY soma_xlo: ok 2, failed 0, wall 0 s",
        "SUMMARY preview: ok 2, failed 0, wall 0 s",
        "SUMMARY blueprints: ok 2, failed 0, wall 0 s",
    ]
    assert messages[-1] == "PIPELINE_DONE total 0 s"


def test_ok_lines_carry_what_the_stage_reported(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def keypoints_with_markers(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, log_path: Path) -> int:
        log_path.write_text(f"KEYPOINTS_RATE {segment} 24.997\nnoise\nKEYPOINTS_TIME {segment} 1624 199326 {{'detect': 4.0}}\n")
        return touch(flag_value(arguments, "--output-dir") / f"{segment}.rrd")

    def calibrate_with_marker(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, log_path: Path) -> int:
        log_path.write_text(f"HEAD_CALIBRATION {segment} {{'rig_00': ('head', 3.5)}}\n")
        return touch(flag_value(arguments, "--output-dir") / f"{segment}.rrd")

    monkeypatch.setitem(WRITERS, "selfcap-keypoints", keypoints_with_markers)
    def soma_with_marker(_: FakePipeline, __: dict[str, str], arguments: list[str], segment: str, log_path: Path) -> int:
        log_path.write_text(f"SOMA_TIME {segment} fit_s=3.0 total_s=9.0\n")
        return touch(flag_value(arguments, "--output") / DATASET / segment / f"{segment}.rrd")

    monkeypatch.setitem(WRITERS, "selfcap-calibrate", calibrate_with_marker)
    monkeypatch.setitem(WRITERS, "soma-batch", soma_with_marker)
    run.main(config)
    messages: list[str] = progress_messages(tmp_path / "run")
    assert f"keypoints ok   {SEGMENT_A} (0 s) 24.997 1624 199326 {{'detect': 4.0}}" in messages
    assert f"calib ok   {SEGMENT_A} (0 s) {{'rig_00': ('head', 3.5)}}" in messages
    assert f"soma ok   {SEGMENT_A} (0 s) fit_s=3.0 total_s=9.0" in messages


def test_a_missing_metric_is_left_out_of_its_median_and_a_zero_is_kept(config: Config, pipeline: FakePipeline, tmp_path: Path) -> None:
    pipeline.quality = {SEGMENT_A: ("pass", 0.0), SEGMENT_B: ("flag", None)}
    run.main(config)
    assert "SOMA_QUALITY pass 1, flag 1, median mm: body 0.0" in progress_messages(tmp_path / "run")


def test_align_calibrate_and_soma_run_their_jobs_at_a_time_and_keypoints_one_at_a_time(config: Config, pipeline: FakePipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    """The barrier breaks unless two align, calibrate or SOMA processes are in flight together; the peak of every other task must stay one."""
    both_running: Barrier = Barrier(2, timeout=10.0)
    running: dict[str, int] = {}
    peak: dict[str, int] = {}
    lock: Lock = Lock()

    def traced(command: list[str], env: dict[str, str], log_path: Path) -> int:
        name: str = command[command.index("--frozen") + 1]
        with lock:
            running[name] = running.get(name, 0) + 1
            peak[name] = max(peak.get(name, 0), running[name])
        if name in ("selfcap-align", "selfcap-calibrate", "soma-batch"):
            both_running.wait()
        code: int = pipeline(command, env, log_path)
        sleep(0.01)
        with lock:
            running[name] -= 1
        return code

    monkeypatch.setattr(run, "run_command", traced)
    run.main(replace(config, jobs=2, soma_jobs=2))
    assert peak == {
        "dataforge-convert": 1, "dataforge-register": 1, "selfcap-align": 2, "dataforge-swap-layer": 1, "selfcap-calibrate": 2, "selfcap-keypoints": 1,
        "soma-batch": 2, "soma-xlo": 1, "selfcap-preview": 1, "dataforge-publish-blueprints": 1,
    }


def test_without_asset_flags_the_tasks_download_their_own_models(config: Config, pipeline: FakePipeline) -> None:
    run.main(replace(config, oak_egohand_assets=None, soma_assets=None))
    commands: list[list[str]] = [command for task in ("selfcap-calibrate", "soma-batch", "soma-xlo") for command in pipeline.commands_of(task)]
    assert commands
    assert not [command for command in commands if "--oak-egohand-assets" in command or "--assets" in command]


def test_relative_paths_reach_the_tasks_as_absolute_ones(config: Config, pipeline: FakePipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every task has its own working directory, so a relative path would land elsewhere."""
    monkeypatch.chdir(tmp_path)
    run.main(replace(config, run_root=Path("run"), raw_root=Path("input/raw"), oak_egohand_assets=Path("input/egohand"), soma_assets=Path("input/soma-assets")))
    assert pipeline.calls[0][0] == {"DATAFORGE_RAW_ROOT": str(tmp_path / "input/raw"), "DATAFORGE_OUTPUT_ROOT": str(tmp_path / "run/rrd")}
    calibrate: list[str] = pipeline.commands_of("selfcap-calibrate")[0]
    assert calibrate[calibrate.index("--oak-egohand-assets") + 1] == str(tmp_path / "input/egohand")
    assert calibrate[calibrate.index("--output-dir") + 1] == str(tmp_path / f"run/calib/{SEGMENT_A}")


def env_of_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """The environment a real child process gets from ``run_command`` under this parent."""
    for name, value in parent.items():
        monkeypatch.setenv(name, value)
    log_path: Path = tmp_path / "child-env.log"
    assert real_run_command(["/usr/bin/env"], extra, log_path) == 0
    return dict(line.split("=", 1) for line in log_path.read_text().splitlines() if "=" in line)


def test_a_task_does_not_inherit_the_activation_of_the_env_that_runs_selfcap_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested ``pixi run`` keeps what it does not set: curate's libraries would load into the torch of selfcap-soma."""
    child: dict[str, str] = env_of_child(
        tmp_path, monkeypatch,
        {
            "CONDA_PREFIX": "/x/.pixi/envs/selfcap-curate",
            "LD_LIBRARY_PATH": "/x/.pixi/envs/selfcap-curate/lib:/opt/mine/lib:/x/.pixi/envs/selfcap-curate/lib/python3.12/site-packages/tensorrt_libs",
            "PATH": "/x/.pixi/envs/selfcap-curate/bin:/usr/bin:/bin",
            "CFLAGS": "-O2 -isystem /x/.pixi/envs/selfcap-curate/include",
            "PIXI_DEV_MODE": "1",
            "CUDA_VISIBLE_DEVICES": "1",
        },
        {"DATAFORGE_OUTPUT_ROOT": "/r/rrd"},
    )
    assert child["LD_LIBRARY_PATH"] == "/opt/mine/lib"
    assert child["PATH"] == "/usr/bin:/bin"
    assert child["CUDA_VISIBLE_DEVICES"] == "1"
    assert child["DATAFORGE_OUTPUT_ROOT"] == "/r/rrd"
    for name in ("CONDA_PREFIX", "CFLAGS", "PIXI_DEV_MODE"):
        assert name not in child


def test_a_task_outside_any_pixi_activation_keeps_the_whole_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    child: dict[str, str] = env_of_child(tmp_path, monkeypatch, {"LD_LIBRARY_PATH": "/opt/mine/lib", "PATH": "/usr/bin:/bin"}, {})
    assert child["LD_LIBRARY_PATH"] == "/opt/mine/lib"
    assert child["PATH"] == "/usr/bin:/bin"
