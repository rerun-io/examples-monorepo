"""Release orchestration from the same immutable checkout as every stage."""

import argparse
import collections
import datetime as dt
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import runner
from gate_redecision import flips_finished
from soma_timing import timing_window

ROOT = Path("/tmp/selfcap-night2")

BASE_NOTIFY = runner.notify


def notify(message):
    BASE_NOTIFY("RUN590 " + message.replace("d31 real batch example DONE:", "first full session DONE:"))


runner.notify = notify


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + "-", delete=False) as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
        p = Path(f.name)
    p.replace(path)


def terminal(row):
    info = json.loads(row["info"])
    return row["state"] in ("done", "calib_input_limited", "soma_input_limited") or (row["state"] == "failed" and info.get("retries", 0) >= 2)


def launch(role, count):
    cfg = runner.config()
    for index in range(count):
        unit = f"selfcap-run590-{role}-{index}"
        if subprocess.run(["systemctl", "--user", "is-active", "--quiet", unit]).returncode == 0:
            continue
        args = [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--collect",
            "--property=Restart=on-failure",
            "--property=RestartSec=10",
            "--property=WorkingDirectory=" + cfg["pipeline_worktree"],
            "--setenv=CUDA_MPS_PIPE_DIRECTORY=/tmp/selfcap-mps/pipe",
            "--setenv=OMP_NUM_THREADS=1",
            "--setenv=PYTHONUNBUFFERED=1",
            "--setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "--setenv=PYTHONPATH=" + cfg["pipeline_pythonpath"],
            "--setenv=TRITON_PTXAS_BLACKWELL_PATH=" + cfg["pipeline_worktree"] + "/.pixi/envs/selfcap-curate/bin/ptxas",
        ]
        if role == "s2":
            args += ["--setenv=CUDA_VISIBLE_DEVICES="]
        args += [cfg["pipeline_python"], str(Path(__file__).resolve()), "worker", role, str(index)]
        subprocess.run(args, check=True)


def stop(role, count):
    for index in range(count):
        subprocess.run(["systemctl", "--user", "stop", f"selfcap-run590-{role}-{index}"], check=False)


def snapshot():
    cfg = runner.config()
    with runner.connect() as db:
        rows = db.execute("select * from jobs where ordinal<687 order by ordinal").fetchall()
        health = {
            r["stage"]: {"paused": bool(r["paused"]), "consecutive_failures": r["failures"], "reason": r["reason"]}
            for r in db.execute("select * from stage_health")
        }
        events = dict(
            db.execute("select status,count(*) from events where ts>=? and status!='ok' group by status", (cfg["run590_started_at"],)).fetchall()
        )
    main = [r for r in rows if r["ordinal"] < 590]
    refresh = [r for r in rows if r["ordinal"] >= 590]
    decisions = collections.Counter(json.loads(r["info"]).get("decision", {}).get("decision", "pending") for r in main)
    quality = collections.Counter()
    reasons = collections.Counter()
    failures = collections.Counter()
    means = {}
    for row in rows:
        info = json.loads(row["info"])
        if row["state"] == "done":
            path = Path(cfg["soma_output_root"]) / "selfcap-post-ep1-v2" / row["segment"] / "quality.json"
            if path.exists():
                q = json.loads(path.read_text())
                quality[q["status"]] += 1
                reasons.update(q.get("reasons", []))
            else:
                quality["missing"] += 1
        if row["state"] in ("failed", "infrastructure_waiting", "s1_waiting_fix", "s2_waiting_fix", "soma_waiting_fix"):
            message = (
                info.get("infrastructure_reason")
                or row["error"]
                or info.get("waiting_reason")
                or info.get("input_limitation")
                or info.get("s1", {}).get("rtmw_skipped")
                or row["state"]
            )
            last = message.splitlines()[-1]
            error_match = re.match(r"([\w.]+):", last)
            kind = error_match.group(1) if error_match else row["state"]
            failures[kind] += 1
    for stage, default in [("s1", 28.8), ("labelling", 76.6), ("s2", 15.2), ("soma", 75.9)]:
        values = [json.loads(r["info"]).get(stage + "_wall_s") for r in rows if r["ordinal"] < 590 or stage == "soma"]
        values = [v for v in values if isinstance(v, (int, float))]
        means[stage] = sum(values) / len(values) if values else default
    remaining_s1 = sum(r["state"] in ("queued", "s1") for r in main)
    remaining_labels = sum(r["state"] in ("queued", "s1", "s1_done", "labelling") for r in main)
    remaining_soma = sum(not terminal(r) for r in rows)
    main_soma_remaining = sum(not terminal(r) for r in main)
    refresh_remaining = sum(not terminal(r) for r in refresh)
    eta = (
        (
            max(remaining_s1 * means["s1"], remaining_labels * means["labelling"] / 3, main_soma_remaining * means["soma"] / 6)
            + refresh_remaining * means["soma"] / 6
        )
        / 60
        if cfg.get("run590_phase") == "main"
        else remaining_soma * means["soma"] / 6 / 60
    )
    with runner.connect() as db:
        soma_events = [
            (r["ts"], r["seconds"], r["segment"]) for r in db.execute("select e.ts,e.seconds,e.segment from events e join jobs j on j.segment=e.segment where e.stage='soma' and e.status='ok' and not coalesce(json_extract(j.info,'$.soma_cuda_launch_blocking'),0)")
        ]
    timing = timing_window(soma_events, cfg.get("soma_normal_timing_start", 0.0), cfg.get("soma_normal_timing_jobs", 20))
    atomic_json(ROOT / "soma-normal-timing.json", timing)
    return {
        "soma_normal_timing": timing,
        "phase": cfg.get("run590_phase", "main"),
        "main_terminal": sum(terminal(r) for r in main),
        "main_total": 590,
        "refresh_terminal": sum(terminal(r) for r in refresh),
        "refresh_total": 97,
        "terminal": sum(terminal(r) for r in rows),
        "total": 687,
        "states": dict(collections.Counter(r["state"] for r in rows)),
        "main_states": dict(collections.Counter(r["state"] for r in main)),
        "refresh_states": dict(collections.Counter(r["state"] for r in refresh)),
        "refined": decisions["accepted"],
        "rejected": decisions["rejected"],
        "label_timeout": decisions["timeout"],
        "label_failed": decisions["failed"],
        "soma_quality": dict(quality),
        "quality_flag_reasons": dict(reasons),
        "failures_by_class": dict(failures),
        "input_limitations": dict(collections.Counter(r["state"] for r in rows if r["state"] in ("calib_input_limited", "soma_input_limited"))),
        "events": events,
        "stage_health": health,
        "mean_stage_seconds": means,
        "eta_minutes": round(eta, 1),
        "elapsed_minutes": round((time.time() - cfg["run590_started_at"]) / 60, 1),
    }


def recovery():
    cfg = runner.config()
    with runner.connect() as db:
        db.execute("begin immediate")
        stale = [r["name"] for r in db.execute("select * from workers where heartbeat<?", (time.time() - 120,))]
        for owner in stale:
            for row in db.execute("select * from jobs where owner=?", (owner,)).fetchall():
                state = {"s1": "queued", "s2": "labelled", "labelling": "s1_done"}.get(row["state"], row["state"])
                db.execute(
                    "update jobs set state=?,owner=null,updated=? where segment=? and state=? and owner=? and updated=?",
                    (state, time.time(), row["segment"], row["state"], owner, row["updated"]),
                )
        for row in db.execute("select * from jobs where ordinal<? and state='failed' and updated<?", (cfg["limit"], time.time() - 10)).fetchall():
            info = json.loads(row["info"])
            attempt = info.get("retries", 0)
            if attempt >= 2:
                continue
            stage = info.get("failed_stage", "s1")
            info["retries"] = attempt + 1
            state = {"s1": "queued", "s2": "labelled", "labelling": "s1_done", "soma": "soma"}[stage]
            db.execute(
                "update jobs set state=?,owner=null,updated=?,info=? where segment=? and state=? and owner is ? and updated=?",
                (state, time.time(), json.dumps(info), row["segment"], row["state"], row["owner"], row["updated"]),
            )


def controller():
    marker = ROOT / "run590-milestones.json"
    sent = json.loads(marker.read_text()) if marker.exists() else []
    last_pauses = {}
    gpu_stopped = False
    while True:
        recovery()
        cfg = runner.config()
        s = snapshot()
        atomic_json(ROOT / "run590-status.json", s)
        for stage, health in s["stage_health"].items():
            signature = health["reason"] if health["paused"] else None
            if signature and last_pauses.get(stage) != signature:
                notify("STAGE PAUSED " + stage + ": " + json.dumps(s))
            last_pauses[stage] = signature
        for n in range(100, 701, 100):
            if s["terminal"] >= n and n not in sent:
                notify(f"{n} terminal: " + json.dumps(s))
                sent.append(n)
                atomic_json(marker, sent)
        if s["main_terminal"] == 590 and cfg.get("run590_phase") == "main":
            with runner.connect() as db:
                pending_cleanup = any(
                    json.loads(r["info"]).get("cleanup_pending") for r in db.execute("select info from jobs where ordinal<590 and state='done'")
                )
            if not pending_cleanup:
                for role, count in [("gpu", 1), ("label", 3), ("s2", 3), ("soma", 6)]:
                    stop(role, count)
                cfg = runner.config()
                cfg["run590_phase"] = "gate_redecision"
                cfg["corpus_paused"] = True
                runner.write_config(cfg)
                command = [
                    "systemd-run",
                    "--user",
                    "--unit=selfcap-run590-redecision",
                    "--collect",
                    "--property=WorkingDirectory=" + cfg["pipeline_worktree"],
                    "--setenv=PYTHONPATH=" + cfg["pipeline_pythonpath"],
                    "--setenv=OMP_NUM_THREADS=1",
                    "--setenv=PYTHONUNBUFFERED=1",
                    cfg["pipeline_python"],
                    str(Path(cfg["pipeline_worktree"]) / "ops/selfcap-night2/gate_redecision.py"),
                ]
                subprocess.run(command, check=True)
                notify("Main590 terminal; replaying ALL cached labels with gate-v3 before flips and batch100 refresh.")
        if cfg.get("run590_phase") == "gate_redecision":
            replay_root = ROOT / "gate-v3"
            if (replay_root / "complete.json").exists():
                cfg = runner.config()
                cfg["run590_phase"] = "gate_flips"
                cfg["corpus_paused"] = False
                cfg["limit"] = 687
                runner.write_config(cfg)
                launch("s2", 3)
                launch("soma", 6)
            elif (replay_root / "error.json").exists():
                cfg = runner.config()
                cfg["run590_phase"] = "gate_redecision_waiting_fix"
                runner.write_config(cfg)
        if cfg.get("run590_phase") == "gate_flips":
            replay = json.loads((ROOT / "gate-v3/complete.json").read_text())
            with runner.connect() as db:
                by_segment = {r["segment"]: r for r in db.execute("select * from jobs")}
            finished = flips_finished(by_segment, replay["flips"])
            if finished:
                stop("s2", 3)
                with runner.connect() as db:
                    db.execute(
                        "update jobs set state='soma',owner=null,updated=? where ordinal>=590 and ordinal<687 and state='soma_refresh_queued'",
                        (time.time(),),
                    )
                cfg = runner.config()
                cfg["run590_phase"] = "soma_refresh"
                runner.write_config(cfg)
                launch("soma", 6)
                notify("All gate-v3 flips published/read back; starting remaining batch100 SOMA-only refreshes. " + json.dumps(snapshot()))
        if not gpu_stopped and s["main_states"].get("queued", 0) == 0 and s["main_states"].get("s1", 0) == 0:
            stop("gpu", 1)
            gpu_stopped = True
        if s["terminal"] == 687 and cfg.get("run590_phase") == "soma_refresh":
            with runner.connect() as db:
                pending = [
                    r
                    for r in db.execute("select segment,info from jobs where ordinal<687 and state='done'")
                    if json.loads(r["info"]).get("cleanup_pending")
                ]
            if not pending:
                for role, count in [("gpu", 1), ("label", 3), ("s2", 3), ("soma", 6)]:
                    stop(role, count)
                cfg = runner.config()
                cfg["full_go"] = False
                cfg["corpus_paused"] = True
                cfg["run590_phase"] = "complete"
                runner.write_config(cfg)
                manifest = json.loads((ROOT / "run590-manifest.json").read_text())
                manifest["finished_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
                manifest["final"] = s
                atomic_json(ROOT / "run590-manifest.json", manifest)
                notify("COMPLETE687 terminal (590new +97SOMArefresh): " + json.dumps(s))
                return
        with runner.connect() as db:
            completed = db.execute("select segment,info from jobs where ordinal<? and state='done'", (cfg["limit"],)).fetchall()
        for row in completed:
            info = json.loads(row["info"])
            if info.get("cleanup_pending") and time.time() - info.get("cleanup_attempted_at", 0) > 60:
                runner.cleanup_finished(row["segment"], info)
        time.sleep(10)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["controller", "worker", "status", "launch"])
    p.add_argument("role", nargs="?")
    p.add_argument("index", nargs="?", type=int, default=0)
    a = p.parse_args()
    if a.command == "controller":
        controller()
    elif a.command == "worker":
        runner.work(a.role, a.index)
    elif a.command == "status":
        print(json.dumps(snapshot(), indent=2))
    else:
        launch(a.role, a.index)
