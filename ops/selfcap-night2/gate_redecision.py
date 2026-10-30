"""Replay cached labels after the main run, then enqueue only calibration flips."""

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import runner

ROOT = Path("/tmp/selfcap-night2/gate-v3")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def cached_command(template: list[str], fields: dict, labels: Path, raw: Path) -> list[str]:
    command = [part.format(**fields) for part in template]
    for flag, value in [
        ("--cached-labels", str(labels)),
        ("--rtmw-rrd", str(raw)),
        ("--thresholds.sapiens-max-regression-px", "5.0"),
        ("--thresholds.max-nonconverged-cv-folds", "1"),
    ]:
        if flag in command:
            command[command.index(flag) + 1] = value
        else:
            command.extend([flag, value])
    return command


def decision_flipped(previous: dict, decision: str) -> bool:
    return bool(previous.get("refined", previous.get("decision") == "accepted")) != (decision == "accepted")


def flips_finished(rows: dict, flips: list[str]) -> bool:
    """A refresh cannot overtake a flip's final publication and cleanup."""
    return all(rows[seg]["state"] in ("done", "soma_input_limited") and not json.loads(rows[seg]["info"]).get("cleanup_pending") for seg in flips)


def original_source(segment: str, previous: dict) -> tuple[Path, str]:
    """Recover immutable S1 data by its recorded hash, never current S2 raw points."""
    candidates = [Path(previous["calib_path"]).parent / "result.json", runner.ROOT / "runs" / segment / "label/result.json"]
    digest = None
    for path in candidates:
        if path.exists():
            digest = json.loads(path.read_text()).get("rtmw_sha256")
            if digest:
                break
    recovered = Path("/tmp/exo-batch-redecision/sources.json")
    entry = json.loads(recovered.read_text()).get("sessions", {}).get(segment, {}) if recovered.exists() else {}
    supplemental = ROOT / "source-overrides.json"
    if supplemental.exists():
        entry = json.loads(supplemental.read_text()).get("sessions", {}).get(segment, entry)
    if not digest:
        digest = entry.get("sha256")
    if not digest or len(digest) != 64:
        raise ValueError(f"{segment}: original S1 RTMW hash is missing")
    local = ROOT / "sources" / f"{segment}.rrd"
    local.parent.mkdir(parents=True, exist_ok=True)
    remote = f"/home/pablo/dataforge-dev/selfcap-post-ep1-v2/rtmw_2d/{segment}-{digest[:16]}.rrd"
    if entry.get("sha256") == digest:
        remote = entry["remote_path"].removeprefix("file://")
    if not local.exists():
        known = Path(entry.get("local_path") or "/nonexistent")
        if known.is_file() and entry.get("sha256") == digest:
            import shutil

            shutil.copyfile(known, local)
        else:
            temporary = local.with_suffix(".partial")
            subprocess.run(["scp", "-q", "spark:" + remote, str(temporary)], check=True, timeout=180)
            temporary.replace(local)
    with local.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"{segment}: original RTMW hash mismatch")
    return local, remote


def replay(segment: str, saved: dict) -> dict:
    previous = saved["decision"]
    work = runner.ROOT / "runs" / segment
    labels = Path(previous.get("landmarks_path", str(work / "label/landmarks.json")))
    initial = next((work / "s1/calib").glob("*.json"))
    raw, remote = original_source(segment, previous)
    output = ROOT / "decisions" / segment
    output.mkdir(parents=True, exist_ok=True)
    fields = {
        "segment": segment,
        "calib_record": str(initial),
        "output_dir": str(output),
        "s1_dir": str(work / "s1"),
        "work": str(work),
        "s2_dir": str(work / "s2"),
        "dataset": "selfcap-post-ep1-v2",
        "catalog_url": "rerun+http://dgx-spark:51235",
    }
    command = cached_command(runner.config()["label_command"], fields, labels, raw)
    with (output / "command.log").open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=runner.config()["pipeline_worktree"], timeout=400, check=True)
    result = json.loads((output / "result.json").read_text())
    if result.get("gate_version") != "gate-v3" or result["decision"] not in ("accepted", "rejected", "skipped"):
        raise ValueError(f"{segment}: invalid cached result {result.get('decision')}")
    return {
        "before": previous.get("decision"),
        "after": result["decision"],
        "flipped": decision_flipped(previous, result["decision"]),
        "output_dir": str(output),
        "raw_path": str(raw),
        "raw_remote": remote,
        "raw_sha256": result["rtmw_sha256"],
    }


def run() -> None:
    """One durable post-main pass; failures leave the refresh barrier closed."""
    cfg = runner.config()
    if cfg.get("run590_phase") != "gate_redecision":
        raise ValueError("cached replay is allowed only after the main 590 completes")
    snapshot = ROOT / "before.json"
    if not snapshot.exists():
        with runner.connect() as db:
            rows = db.execute("select * from jobs order by ordinal").fetchall()
        selected = {}
        excluded = {}
        for row in rows:
            info = json.loads(row["info"])
            previous = info.get("decision", {})
            labels = Path(previous.get("landmarks_path", str(runner.ROOT / "runs" / row["segment"] / "label/landmarks.json")))
            if previous and labels.is_file():
                selected[row["segment"]] = {"decision": previous, "state": row["state"]}
            else:
                excluded[row["segment"]] = "no cached labels: " + row["state"]
        write_json(snapshot, {"sessions": selected, "excluded": excluded, "created_at": time.time()})
    before = json.loads(snapshot.read_text())
    selected = before["sessions"]
    report_path = ROOT / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"sessions": {}, "flips": [], "started_at": time.time()}
    pending = [(segment, saved) for segment, saved in selected.items() if segment not in report["sessions"]]
    # Three sessions = the same concurrency as the production label pool; no Sol calls.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {segment: pool.submit(replay, segment, saved) for segment, saved in pending}
        for segment, future in futures.items():
            report["sessions"][segment] = future.result()
            write_json(report_path, report)
    from restore_inputs import restore

    for segment, row in report["sessions"].items():
        if row.get("applied"):
            continue
        output = Path(row["output_dir"])
        result = json.loads((output / "result.json").read_text())
        with runner.connect() as db:
            current = db.execute("select * from jobs where segment=?", (segment,)).fetchone()
        if current["owner"] is not None:
            raise ValueError(f"{segment}: job acquired during post-main replay")
        info = json.loads(current["info"])
        info["gate_v3_previous"] = selected[segment]["decision"]
        info["decision"] = {
            "refined": result["decision"] == "accepted",
            "decision": result["decision"],
            "metrics": result.get("gate"),
            "reason": result.get("reason"),
            "calib_path": str(output / "calib.json"),
            "landmarks_path": str(output / "landmarks.json"),
            "gate_version": "gate-v3",
        }
        info["gate_v3_result"] = str(output / "result.json")
        state = current["state"]
        if row["flipped"]:
            restore(segment, Path(row["raw_path"]))
            for key in ("failed_stage", "retries", "waiting_reason", "infrastructure_reason"):
                info.pop(key, None)
            info["gate_v3_flip"] = True
            state = "labelled"
            if segment not in report["flips"]:
                report["flips"].append(segment)
        with runner.connect() as db:
            db.execute(
                "update jobs set state=?,info=?,error=null,updated=? where segment=? and owner is null",
                (state, json.dumps(info), time.time(), segment),
            )
        row["applied"] = True
        # Restored Parquet owns the flip inputs; the immutable remote original remains available.
        Path(row["raw_path"]).unlink(missing_ok=True)
        write_json(report_path, report)
    report["before_accepted"] = sum(v["decision"].get("refined", False) for v in selected.values())
    report["after_accepted"] = sum(v["after"] == "accepted" for v in report["sessions"].values())
    report["excluded"] = before["excluded"]
    report["completed_at"] = time.time()
    write_json(report_path, report)
    write_json(ROOT / "complete.json", report)
    runner.notify(
        f"Gate-v3 cached redecision complete: {len(selected)} sessions; accepted "
        f"{report['before_accepted']}->{report['after_accepted']}; {len(report['flips'])} flips awaiting S2/SOMA; no Sol calls."
    )


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        write_json(ROOT / "error.json", {"error": str(error), "at": time.time()})
        runner.notify(f"Gate-v3 replay paused: {error}; batch100 SOMA refresh remains blocked")
        raise
