"""The gate replay must use frozen labels and original S1 observations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate_redecision import cached_command, decision_flipped, flips_finished


def test_cached_command_replaces_live_inputs_and_pins_gate_v3():
    template = [
        "python",
        "-m",
        "selfcap_curate.apis.exo_label",
        "--segment",
        "{segment}",
        "--rtmw-rrd",
        "{s1_dir}/rtmw_2d.rrd",
        "--thresholds.sapiens-max-regression-px",
        "2.0",
    ]
    result = cached_command(template, {"segment": "session", "s1_dir": "/old"}, Path("/frozen/labels.json"), Path("/frozen/raw.rrd"))
    assert result[result.index("--cached-labels") + 1] == "/frozen/labels.json"
    assert result[result.index("--rtmw-rrd") + 1] == "/frozen/raw.rrd"
    assert result[result.index("--thresholds.sapiens-max-regression-px") + 1] == "5.0"
    assert result[result.index("--thresholds.max-nonconverged-cv-folds") + 1] == "1"
    assert result.count("--rtmw-rrd") == 1


def test_only_refined_decision_changes_require_s2_and_soma():
    assert decision_flipped({"refined": False, "decision": "rejected"}, "accepted")
    assert decision_flipped({"refined": True, "decision": "accepted"}, "rejected")
    assert not decision_flipped({"refined": True, "decision": "accepted"}, "accepted")
    assert not decision_flipped({"refined": False, "decision": "failed"}, "rejected")


def test_refresh_waits_for_flip_publication_and_cleanup():
    rows = {"flip": {"state": "soma", "info": "{}"}}
    assert not flips_finished(rows, ["flip"])
    rows["flip"] = {"state": "done", "info": '{"cleanup_pending":"readback failed"}'}
    assert not flips_finished(rows, ["flip"])
    rows["flip"] = {"state": "failed", "info": "{}"}
    assert not flips_finished(rows, ["flip"])
    rows["flip"] = {"state": "done", "info": "{}"}
    assert flips_finished(rows, ["flip"])


def test_original_source_downloads_hash_verified_remote_when_local_path_is_null(tmp_path, monkeypatch):
    import hashlib
    import json

    import gate_redecision as replay

    monkeypatch.setattr(replay, "ROOT", tmp_path)
    payload = b"original S1 bytes"
    digest = hashlib.sha256(payload).hexdigest()
    metadata = {"sessions": {"test-session": {"local_path": None, "remote_path": "/remote/original.rrd", "sha256": digest}}}
    (tmp_path / "source-overrides.json").write_text(json.dumps(metadata))
    calls = []

    def download(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(payload)

    monkeypatch.setattr(replay.subprocess, "run", download)
    local, remote = replay.original_source("test-session", {"calib_path": str(tmp_path / "missing/calib.json")})
    assert local.read_bytes() == payload
    assert remote == "/remote/original.rrd"
    assert calls[0][-2] == "spark:/remote/original.rrd"
