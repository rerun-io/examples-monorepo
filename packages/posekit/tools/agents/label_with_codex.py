"""Drive the hand labeler with a Codex model as the reviewer (Astra overnight).

The script is a client of the labeler's HTTP API, the same API the browser page uses. For every unreviewed
1 fps keyframe of a segment it fetches the 6-view contact sheet (numbered proposal boxes) and its metadata,
asks a Codex model to judge each numbered box and to box any hand the proposals missed, posts the decisions
as FrameLabels (reviewer = --reviewer, source "agent" for boxes the model drew or re-sided), range-fills the
5 fps frames between consecutive reviewed keyframes, and commits the segment's ``hand_boxes`` layer.

Why stdlib-only and outside pixi: it runs on the host that holds the Codex login (pablo-dl-server, where no pixi
environment may run), talks HTTP to the backend on the DGX Spark, and shells out to ``codex exec``.

Usage:
    python3 label_with_codex.py --server http://dgx-spark:9898 --segment <sid> [--segment ...] --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPT = """You are reviewing hand-detector proposals for a hand-labeling dataset.

The image is a contact sheet of 6 synchronized fisheye cameras on a head-worn rig (RoboCap), all at the same instant.
Cells: top row cam_00 left_front, cam_01 right_front, cam_02 left_eye; bottom row cam_03 right_eye, cam_04 left,
cam_05 right. The eye cameras look down at the wearer's hands; front cameras see the hands at the bottom edge; side
cameras rarely see them. Each proposal box is drawn and numbered #k with its side (L/R) and score. "left"/"right"
means the WEARER's anatomical hand, not the image side. Only the wearer's own hands count: other people's hands,
gloves lying on a table, and pictures of hands are NOT hands to label.

Proposals (native pixel boxes, for reference):
{proposals}

Tasks:
1. For EVERY numbered proposal give a verdict:
   - "correct": the box tightly covers one of the wearer's hands (wrist to fingertips visible part) with the right side.
   - "wrong_side": it covers a wearer hand but the side is wrong.
   - "loose": it is on a wearer hand but the box is clearly too big/small/offset; give a corrected box.
   - "not_hand": it is not on a wearer hand (background, object, another person's hand).
   - "duplicate": another proposal already covers the same hand in the same camera better.
2. For each camera, list any VISIBLE wearer hand that no correct/loose proposal covers ("missed"), with its side and
   a box in SHEET pixel coordinates (the whole image; x right, y down). Partially visible hands at the image edge count
   if at least a quarter of the hand is visible.
3. List cameras that cannot be judged at all (black frame, extreme blur/exposure) as "unjudgeable".
Box coordinates you give are SHEET pixels of this image. Be precise and conservative: when unsure whether
something is a hand, say not_hand.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdicts", "missed", "unjudgeable"],
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["k", "verdict", "box"],
                "properties": {
                    "k": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["correct", "wrong_side", "loose", "not_hand", "duplicate"]},
                    "box": {
                        "description": "corrected box in SHEET px for 'loose', else null",
                        "anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}],
                    },
                },
            },
        },
        "missed": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cam", "side", "box"],
                "properties": {
                    "cam": {"type": "integer"},
                    "side": {"type": "string", "enum": ["left", "right"]},
                    "box": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
            },
        },
        "unjudgeable": {"type": "array", "items": {"type": "integer"}},
    },
}


def http_json(url: str, body: dict | None = None, timeout: float = 120.0) -> dict | list:
    """GET (body None) or POST JSON; raise with the server's message on HTTP errors."""
    data: bytes | None = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"{url}: HTTP {err.code}: {err.read().decode(errors='replace')[:500]}") from err


def http_bytes(url: str, timeout: float = 120.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def sheet_to_native(cells: list[dict], box: list[float]) -> tuple[int, list[float]] | None:
    """Map a sheet-pixel box to (cam, native box) using the cell that holds its centre; None outside every cell."""
    cx: float = (box[0] + box[2]) / 2
    cy: float = (box[1] + box[3]) / 2
    for cell in cells:
        if cell["x"] <= cx < cell["x"] + cell["w"] and cell["y"] <= cy < cell["y"] + cell["h"]:
            s: float = cell["scale"]
            x0 = (min(box[0], box[2]) - cell["x"]) / s
            y0 = (min(box[1], box[3]) - cell["y"]) / s
            x1 = (max(box[0], box[2]) - cell["x"]) / s
            y1 = (max(box[1], box[3]) - cell["y"]) / s
            return cell["cam"], [x0, y0, x1, y1]
    return None


def decisions_to_labels(info: dict, answer: dict, reviewer: str) -> list[dict]:
    """Turn the model's answer into one ViewLabel dict per (cam, side) (pure; unit-tested)."""
    cells: list[dict] = info["cells"]
    cams: list[int] = [c["cam"] for c in cells]
    by_k: dict[int, dict] = {p["k"]: p for p in info["proposals"]}
    unjudgeable: set[int] = {c for c in answer.get("unjudgeable", []) if c in cams}
    chosen: dict[tuple[int, str], tuple[float, dict]] = {}  # (cam, side) -> (rank, label)

    def offer(cam: int, side: str, rank: float, label: dict) -> None:
        if (cam, side) not in chosen or rank > chosen[(cam, side)][0]:
            chosen[(cam, side)] = (rank, label)

    for verdict in answer.get("verdicts", []):
        prop = by_k.get(verdict["k"])
        if prop is None or prop["cam"] in unjudgeable:
            continue
        cam, side, box = prop["cam"], prop["side"], prop["box"]
        v: str = verdict["verdict"]
        if v == "correct":
            offer(cam, side, prop["score"], {"status": "accepted", "box": box, "source": prop["model"]})
        elif v == "wrong_side":
            other = "left" if side == "right" else "right"
            offer(cam, other, prop["score"], {"status": "corrected", "box": box, "source": "agent"})
        elif v == "loose" and verdict.get("box"):
            mapped = sheet_to_native(cells, verdict["box"])
            if mapped is not None and mapped[0] == cam:
                x0, y0, x1, y1 = mapped[1]
                if x1 > x0 and y1 > y0:
                    offer(cam, side, prop["score"], {"status": "corrected", "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "source": "agent"})
    for miss in answer.get("missed", []):
        mapped = sheet_to_native(cells, miss["box"])
        if mapped is None or mapped[0] in unjudgeable:
            continue
        x0, y0, x1, y1 = mapped[1]
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        # a model-drawn box ranks below any accepted proposal for the same (cam, side)
        offer(mapped[0], miss["side"], -1.0, {"status": "corrected", "box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "source": "agent"})

    labels: list[dict] = []
    for cam in cams:
        for side in ("left", "right"):
            if cam in unjudgeable:
                label = {"status": "rejected", "box": None, "source": "agent"}
            elif (cam, side) in chosen:
                label = chosen[(cam, side)][1]
            else:
                label = {"status": "no_hand", "box": None, "source": "agent"}
            labels.append({"cam": cam, "side": side, "reviewer": reviewer, **label})
    return labels


def ask_codex(sheet_jpg: bytes, info: dict, model: str, effort: str, codex_home: str, workdir: Path) -> dict:
    """One ``codex exec`` call with the sheet image; returns the parsed structured answer."""
    with tempfile.TemporaryDirectory(dir=workdir) as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "sheet.jpg").write_bytes(sheet_jpg)
        (tmp / "schema.json").write_text(json.dumps(SCHEMA))
        lines = [f"#{p['k']} cam_{p['cam']:02d} {p['side']} score {p['score']:.2f} ({p['model']}) box {[round(p['box'][c]) for c in ('x0', 'y0', 'x1', 'y1')]}" for p in info["proposals"]]
        (tmp / "prompt.txt").write_text(PROMPT.format(proposals="\n".join(lines) or "(none)"))
        cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only", "--ephemeral", "--skip-git-repo-check",
               "-C", str(tmp), "--output-schema", str(tmp / "schema.json"), "-o", str(tmp / "out.json"), "-i", str(tmp / "sheet.jpg"), "-"]
        with (tmp / "prompt.txt").open("rb") as stdin:
            proc = subprocess.run(cmd, stdin=stdin, capture_output=True, env={**os.environ, "CODEX_HOME": codex_home}, timeout=600)
        if proc.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"codex exec rc={proc.returncode}: {proc.stderr.decode(errors='replace')[-800:]}")
        return json.loads((tmp / "out.json").read_text())


def review_keyframe(server: str, sid: str, t_ns: int, args: argparse.Namespace, workdir: Path) -> dict:
    info = http_json(f"{server}/api/sheet/{sid}/{t_ns}.json")
    jpg = http_bytes(f"{server}/api/sheet/{sid}/{t_ns}.jpg")
    answer: dict = {}
    for attempt in range(3):
        try:
            answer = ask_codex(jpg, info, args.model, args.effort, args.codex_home, workdir)
            break
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as err:
            print(f"[{sid} {t_ns}] codex attempt {attempt + 1} failed: {err}", file=sys.stderr, flush=True)
            if "usage limit" in str(err).lower():
                raise
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"codex failed 3 times on {sid} {t_ns}")
    labels = decisions_to_labels(info, answer, args.reviewer)
    http_json(f"{server}/api/labels", {"segment_id": sid, "t_ns": t_ns, "labels": labels})
    log = {"segment_id": sid, "t_ns": t_ns, "answer": answer, "n_labels": len(labels)}
    with (workdir / "answers.jsonl").open("a") as fh:
        fh.write(json.dumps(log) + "\n")
    return log


def label_segment(server: str, sid: str, args: argparse.Namespace, workdir: Path) -> None:
    info = http_json(f"{server}/api/segment/{sid}")
    keyframes: list[int] = info["keyframes_ns"]
    todo: list[int] = [t for t in (q["t_ns"] for q in http_json(f"{server}/api/queue?sid={sid}&limit=100000")) if t in set(keyframes)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{sid}] {len(todo)} of {len(keyframes)} keyframes to review", flush=True)
    t_start = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(review_keyframe, server, sid, t, args, workdir): t for t in todo}
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(time.time() - t_start, 1e-6)
                print(f"[{sid}] {done}/{len(todo)} keyframes, {rate * 60:.1f}/min", flush=True)
    if args.limit:
        return
    filled = 0
    for t0, t1 in zip(keyframes, keyframes[1:], strict=False):
        result = http_json(f"{server}/api/range", {"segment_id": sid, "t0_ns": t0, "t1_ns": t1, "reviewer": args.reviewer})
        filled += result["filled"]
    commit = http_json(f"{server}/api/commit/{sid}", {})
    print(f"[{sid}] range-filled {filled}; committed {commit}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://dgx-spark:9898")
    parser.add_argument("--segment", action="append", default=[], help="segment id; repeat; default = every segment")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--reviewer", default="astra")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="review only the first N keyframes per segment (no range fill / commit)")
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex-work"))
    parser.add_argument("--workdir", default="/mnt/nas/ssd-8tb/scratch/hand-labeler/astra")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    segments: list[str] = args.segment or [s["segment_id"] for s in http_json(f"{args.server}/api/segments")]
    for sid in segments:
        label_segment(args.server, sid, args, workdir)


if __name__ == "__main__":
    main()
