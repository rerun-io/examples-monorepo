"""Exercise registration against a disposable local Rerun catalog."""

import re
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest
from datafusion import DataFrame
from rerun.catalog import CatalogClient

from agent_traces.apis import register
from agent_traces.blueprint import catalog_blueprint
from agent_traces.events import Prompt, Session, TimedRecord, ToolCall, ToolResult, TurnBoundary, Usage, UsageSample
from agent_traces.manifest import CONVERSION_REVISION, Manifest, ManifestEntry, save_manifest
from agent_traces.rerun_log import write_session_rrd


@pytest.fixture
def catalog_port() -> int:
    """Reserve catalog checks to the disposable-server port range."""
    for port in range(52000, 52100):
        with socket.socket() as reservation:
            try:
                reservation.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    pytest.fail("No free disposable catalog port in 52000-52099")


@pytest.mark.integration
def test_catalog_register_skip_replace_and_missing_file(tmp_path: Path, rerun_binary: Path, catalog_port: int) -> None:
    """Known and null totals merge; duplicate policies and CLI failures reach the real server."""
    out: Path = tmp_path / "out"
    profile: Path = out / "synthetic"
    profile.mkdir(parents=True)
    manifest: Manifest = Manifest()
    for identity, total in (("known", 17), ("unknown", None)):
        session: Session = Session(identity, "synthetic", tmp_path / f"{identity}.jsonl",
            [TimedRecord(TurnBoundary("start"), 1_767_225_600_000_000_000, 0, turn_id="turn"),
             TimedRecord(Prompt("catalog fixture"), 1_767_225_600_000_000_000, 0, turn_id="turn"),
             TimedRecord(UsageSample(Usage(output_tokens=10)), 1_767_225_601_000_000_000, 1, model="main-model"),
             TimedRecord(UsageSample(Usage(output_tokens=5)), 1_767_312_001_000_000_000, 2, model="main-model"),
             TimedRecord(ToolCall("Read", "read", "x" * 10000, "file_read"), 1_767_225_602_000_000_000, 3),
             TimedRecord(ToolResult("Read", "read", "error", "{}", "file_read", 12.0, True), 1_767_225_603_000_000_000, 4),
             TimedRecord(ToolResult("Read", "read", "more output", "{}", "file_read", 12.0, True), 1_767_225_604_000_000_000, 5),
             TimedRecord(ToolCall("Read", "read2", "{}", "file_read"), 1_767_225_605_000_000_000, 6),
             TimedRecord(ToolResult("Read", "read2", "ok", "{}", "file_read", 8.0), 1_767_225_606_000_000_000, 7),
             TimedRecord(ToolCall("custom", "pending", "{}", "other"), 1_767_225_607_000_000_000, 8)],
            {"child": [TimedRecord(UsageSample(Usage(output_tokens=3)), 1_767_225_601_000_000_000, 0, model="child-model"),
                       TimedRecord(UsageSample(Usage(output_tokens=2)), 1_767_312_001_000_000_000, 1, model="child-model"),
                       TimedRecord(ToolCall("Read", "read", "{}", "file_read"), 1_767_225_602_000_000_000, 2),
                       TimedRecord(ToolResult("Read", "read", "ok", "{}", "file_read", 4.0), 1_767_225_603_000_000_000, 3)]},
            Counter({"unknown": 1}), properties={"total_input_tokens": total, "total_output_tokens": total,
                "models": "main-model,child-model", "total_cost_usd": 0.25 if total is not None else None})
        written = write_session_rrd(session, profile / f"{identity}.rrd", host="synthetic")
        manifest.sessions[identity] = ManifestEntry(str(session.source_path), "fixture", written.path.name,
            "2026-01-01T00:00:00Z", "synthetic", CONVERSION_REVISION, sum(written.entity_rows.values()))
    save_manifest(manifest, profile / "manifest.json")
    second_out: Path = tmp_path / "second-out"
    second_profile: Path = second_out / profile.name
    shutil.copytree(profile, second_profile)
    extra_profile: Path = second_out / "extra"
    shutil.copytree(profile, extra_profile)
    manifest.sessions.pop("unknown")
    save_manifest(manifest, profile / "manifest.json")
    save_manifest(manifest, extra_profile / "manifest.json")
    with (tmp_path / "server.log").open("w+") as log:
        process: subprocess.Popen[str] = subprocess.Popen([str(rerun_binary), "server", "--host", "127.0.0.1", "--port", str(catalog_port)],
                                                          stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            deadline: float = time.monotonic() + 15.0
            while True:
                assert process.poll() is None, "Disposable catalog exited before accepting connections"
                try:
                    with socket.create_connection(("127.0.0.1", catalog_port), timeout=0.2):
                        break
                except OSError:
                    assert time.monotonic() < deadline, "Disposable catalog did not start within 15 seconds"
                    time.sleep(0.1)
            config: register.Config = register.Config(catalog_url=f"rerun+http://127.0.0.1:{catalog_port}", out=(out, second_out))
            assert register.main(config) == register.Summary(3, 1, 0, 0)
            client = CatalogClient(config.catalog_url)
            assert set(client.dataset_names()) == {"agent-traces-synthetic", "agent-traces-extra"}
            assert set(client.get_dataset("agent-traces-extra").segment_ids()) == {"known"}
            assert len(list(profile.glob("*.rbl"))) == 2
            assert not list(second_profile.glob("*.rbl"))
            assert len(list(extra_profile.glob("*.rbl"))) == 2
            assert_readme_queries(config.catalog_url, "agent-traces-synthetic")
            initial_entries = {str(entry.id) for entry in client.entries(include_hidden=True)}
            entry = client.get_dataset("agent-traces-synthetic")
            external = tmp_path / "external.rbl"
            catalog_blueprint().save("external", external)
            entry.register_blueprint(external.as_uri(), set_default=False)
            old_blueprints = set(entry.blueprints())
            assert entry.default_blueprint() is not None
            assert entry.default_segment_table_blueprint() is not None
            assert register.main(replace(config, out=(second_out, out))) == register.Summary(0, 4, 0, 0)
            assert {str(entry.id) for entry in client.entries(include_hidden=True)} == initial_entries
            entry = client.get_dataset("agent-traces-synthetic")
            assert len(entry.blueprints()) == 2
            assert not list(profile.glob("*.rbl"))
            assert len(list(second_profile.glob("*.rbl"))) == 2
            assert external.is_file()
            assert not old_blueprints & set(entry.blueprints())
            assert entry.default_blueprint() in entry.blueprints()
            assert entry.default_segment_table_blueprint() in entry.blueprints()
            assert register.main(replace(config, replace=True)) == register.Summary(3, 1, 0, 0)
            assert len(list(profile.glob("*.rbl"))) == 2
            assert not list(second_profile.glob("*.rbl"))
            entry = CatalogClient(config.catalog_url).get_dataset("agent-traces-synthetic")
            assert set(entry.segment_ids()) == {"known", "unknown"}
            assert len(list(extra_profile.glob("*.rbl"))) == 2
            (profile / "known.rrd").unlink()
            cli = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "tools/apps/register.py"),
                                  "--catalog-url", config.catalog_url, "--out", str(out), str(second_out)], capture_output=True, text=True, timeout=30)
            assert cli.returncode == 1, cli.stdout + cli.stderr
            assert "missing=1" in cli.stdout
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            with socket.socket() as released:
                released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                released.bind(("127.0.0.1", catalog_port))


def assert_readme_queries(catalog_url: str, dataset_name: str) -> None:
    """Execute the documented source, checking cross-session, child and multi-part results."""
    readme: str = (Path(__file__).parents[1] / "README.md").read_text()
    assert "## Query the catalog" in readme
    section: str = readme.split("## Query the catalog", 1)[1].split("\n## ", 1)[0]
    blocks: list[str] = re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)
    assert len(blocks) == 4
    namespace: dict[str, object] = {"catalog_url": catalog_url, "dataset_name": dataset_name}
    for index, block in enumerate(blocks):
        exec(compile(block, f"README recipe {index + 1}", "exec"), namespace)
    sessions = cast(pa.Table, namespace["sessions"]).to_pylist()
    assert len(sessions) == 2
    for row in sessions:
        assert row["turns"] == 1
        assert row["tool_calls"] == 4
        assert row["subagents"] == 1
        assert row["models"] == "main-model,child-model"
    assert {row["cost_usd"] for row in sessions} == {0.25, None}
    tokens = {(row["rerun_segment_id"], row["model"], row["day"].date().isoformat()): row["output_tokens"]
              for row in cast(pa.Table, namespace["output_per_model_day"]).to_pylist()}
    shares = {row["rerun_segment_id"]: row["share"] for row in cast(pa.Table, namespace["subagent_share"]).to_pylist()}
    summaries = {(row["rerun_segment_id"], row["tool"]): {key: row[key] for key in ("calls", "error_rate", "median_latency_ms", "p90_latency_ms")}
                 for row in cast(pa.Table, namespace["tool_summary"]).to_pylist()}
    for row in sessions:
        segment = row["rerun_segment_id"]
        assert {key[1:]: value for key, value in tokens.items() if key[0] == segment} == {
            ("main-model", "2026-01-01"): 10.0, ("child-model", "2026-01-01"): 3.0,
            ("main-model", "2026-01-02"): 5.0, ("child-model", "2026-01-02"): 2.0}
        assert shares[segment] == 0.25
        assert summaries[segment, "Read"] == {"calls": 3, "error_rate": 1 / 3, "median_latency_ms": 8.0, "p90_latency_ms": 12.0}
        assert summaries[segment, "custom"] == {"calls": 1, "error_rate": None, "median_latency_ms": None, "p90_latency_ms": None}
    assert not {"metadata_json", "input_json", "tool_use_result_json", "text"} & set(cast(DataFrame, namespace["tool_rows"]).schema().names)
