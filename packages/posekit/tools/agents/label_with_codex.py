"""Drive the hand labeler with a Codex model in the reviewer's seat (Astra overnight).

The script is a client of the labeler's HTTP API: it only makes the calls a person's clicks in the browser make.

The driver uses MEgATrack's semi-automatic loop (section 4.2): the machine propagates boxes
across frames and views, the reviewer watches 2-second playback windows of all six cameras with the machine boxes
drawn, accepts every window that looks right, and acts only at the first failure it sees: it accepts the frames
before it, fixes the failing hand at that frame (pick a proposal, draw a box, set the side, "no hand", reject), and
resumes watching from there. The server logs every action with wall-clock time, so the effort is measurable.


Why stdlib-only and outside pixi: it runs on the host that holds the Codex login (pablo-dl-server, where no pixi
environment may run), talks HTTP to the backend on the DGX Spark, and shells out to ``codex exec``.

Usage:
    python3 label_with_codex.py --server https://dgx-spark.<tailnet>.ts.net:9898 [--segment <sid> ...] --workers 10
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


def http_json(url: str, body: dict | None = None, timeout: float = 120.0, attempts: int = 8) -> object:
    """GET (body None) or POST JSON; retry 5xx and connection errors (a restarting server); raise on 4xx."""
    data: bytes | None = None if body is None else json.dumps(body).encode()
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == attempts - 1:
                raise RuntimeError(f"{url}: HTTP {err.code}: {err.read().decode(errors='replace')[:500]}") from err
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def json_object(url: str, body: dict | None = None) -> dict:
    """An endpoint that answers a JSON object."""
    value = http_json(url, body)
    if not isinstance(value, dict):
        raise RuntimeError(f"{url}: expected a JSON object, got {type(value).__name__}")
    return value


def json_list(url: str) -> list:
    """An endpoint that answers a JSON array."""
    value = http_json(url)
    if not isinstance(value, list):
        raise RuntimeError(f"{url}: expected a JSON array, got {type(value).__name__}")
    return value


def http_bytes(url: str, timeout: float = 120.0, attempts: int = 8) -> bytes:
    """GET raw bytes with the same retry rule as ``http_json``."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code < 500 or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(min(60, 5 * 2**attempt))
    raise AssertionError("unreachable")


def sheet_to_native(cells: list[dict], cam: int, box: list[float]) -> list[float] | None:
    """Map a sheet-pixel box inside camera ``cam``'s cell to native pixels, clipped to that image.

    None when the box's centre is not in that camera's cell (the answer contradicts itself), or when the
    clipped box is smaller than 4 px on a side.
    """
    cell = next((c for c in cells if c["cam"] == cam), None)
    if cell is None or not all(isinstance(v, int | float) and v == v and abs(v) < 1e6 for v in box):
        return None
    cx: float = (box[0] + box[2]) / 2
    cy: float = (box[1] + box[3]) / 2
    if not (cell["x"] <= cx < cell["x"] + cell["w"] and cell["y"] <= cy < cell["y"] + cell["h"]):
        return None
    s: float = cell["scale"]
    width: float = cell["w"] / s
    height: float = cell["h"] / s
    x0 = min(max((min(box[0], box[2]) - cell["x"]) / s, 0.0), width)
    y0 = min(max((min(box[1], box[3]) - cell["y"]) / s, 0.0), height)
    x1 = min(max((max(box[0], box[2]) - cell["x"]) / s, 0.0), width)
    y1 = min(max((max(box[1], box[3]) - cell["y"]) / s, 0.0), height)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return [x0, y0, x1, y1]


def run_codex(image: bytes, prompt: str, schema: dict, args: argparse.Namespace) -> dict:
    """One ``codex exec`` call with one image and a structured answer; returns the parsed answer."""
    with tempfile.TemporaryDirectory(prefix="label-codex-") as tmp_str:
        tmp = Path(tmp_str)
        (tmp / "image.jpg").write_bytes(image)
        (tmp / "schema.json").write_text(json.dumps(schema))
        (tmp / "prompt.txt").write_text(prompt)
        cmd = ["codex", "exec", "-m", args.model, "-c", f'model_reasoning_effort="{args.effort}"', "-s", "read-only", "--ephemeral",
               "--skip-git-repo-check", "-C", str(tmp), "--output-schema", str(tmp / "schema.json"), "-o", str(tmp / "out.json"),
               "-i", str(tmp / "image.jpg"), "-"]
        with (tmp / "prompt.txt").open("rb") as stdin:
            proc = subprocess.run(cmd, stdin=stdin, capture_output=True, env={**os.environ, "CODEX_HOME": args.codex_home}, timeout=600)
        if proc.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"codex exec rc={proc.returncode}: {proc.stderr.decode(errors='replace')[-800:]}")
        return json.loads((tmp / "out.json").read_text())


def ask_with_retries(image: bytes, prompt: str, schema: dict, args: argparse.Namespace, check, where: str) -> dict:
    """``run_codex`` with up to 3 attempts; ``check(answer)`` raises ValueError on an incomplete answer."""
    for attempt in range(3):
        try:
            answer = run_codex(image, prompt, schema, args)
            check(answer)
            return answer
        except (RuntimeError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, json.JSONDecodeError) as err:
            print(f"[{where}] codex attempt {attempt + 1} failed: {err}", file=sys.stderr, flush=True)
            if "usage limit" in str(err).lower():
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"codex failed 3 times on {where}")


def proposal_lines(info: dict) -> str:
    return "\n".join(
        f"#{p['k']} cam_{p['cam']:02d} {p['side']} score {p['score']:.2f} ({p['model']}) box {[round(p['box'][c]) for c in ('x0', 'y0', 'x1', 'y1')]}"
        for p in info["proposals"]
    ) or "(none)"


WATCH_PROMPT = """You are a hand-labeling reviewer watching a playback of a head-worn camera rig (RoboCap), like a person
watching a tracker's output and stopping only when it fails (MEgATrack's semi-automatic box labelling).

The image is a filmstrip: {rows} frames, {gap} s apart, one per row (row label r0, r1, ... at the left), and the
6 cameras of the rig as columns (names in the header): cam_00 left_front, cam_01 right_front, cam_02 left_eye,
cam_03 right_eye, cam_04 left, cam_05 right. The eye cameras look down at the wearer's hands; the front cameras see them
at the bottom edge; the side cameras rarely see them. The boxes are the MACHINE's current hand boxes: blue = the
wearer's LEFT hand, orange = the wearer's RIGHT hand (anatomical sides, not image sides). A thick box marked * was set by
a reviewer; a dashed box is held over one missed detection. Only the wearer's own hands count.

Decide whether the machine is right in EVERY frame and EVERY camera:
- a visible wearer hand (at least about a quarter of it in the image) must have one box of the right colour, roughly
  tight (it may be loose by ~25 %);
- no box may sit on something that is not a wearer hand;
- no hand may have the wrong colour.
Small jitter is fine. Judge like a careful person at normal playback speed: report a failure only when you are sure.
If everything is right, answer ok=true and failure=null. Otherwise answer ok=false with the FIRST failure in time
(lowest row; if several in that row, the most serious): its row, camera index (0-5), the hand side that is wrong or
missing (for a box on a non-hand, the side of that box), and the kind:
  "missed_hand" (visible hand without a box), "off_hand" (box not on a wearer hand, or badly misplaced),
  "wrong_side" (box on a hand but wrong colour), "bad_frame" (the camera image is unusable: black, frozen, smeared).
"""

WATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok", "failure"],
    "properties": {
        "ok": {"type": "boolean"},
        "failure": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["row", "cam", "side", "kind"],
                    "properties": {
                        "row": {"type": "integer"},
                        "cam": {"type": "integer"},
                        "side": {"type": "string", "enum": ["left", "right"]},
                        "kind": {"type": "string", "enum": ["missed_hand", "off_hand", "wrong_side", "bad_frame"]},
                    },
                },
            ]
        },
    },
}

FIX_PROMPT = """You stopped the playback at a tracker failure and now fix it, like a person with a box tool.

The image is one frame of all 6 cameras (cells labelled cam_0X name; whole image in SHEET pixels, x right, y down).
Dashed boxes labelled "T L"/"T R" are the machine's current boxes; numbered boxes #k are the hand detector's
proposals (blue = left, orange = right; "left"/"right" = the WEARER's anatomical hand).
The failure you saw: camera cam_{cam:02d}, {side} hand, kind "{kind}".

Proposals (native pixel boxes, for reference):
{proposals}

Give the smallest set of fixes that makes this frame right, for the failure above and any other failure you see in
THIS frame. One fix per (camera, side) that needs a change; leave everything that is already right alone. Actions:
  "pick"    : the right box is proposal #k (use it for this camera and side; this also fixes a wrong side);
  "draw"    : draw the box yourself, in SHEET pixels [x0, y0, x1, y1], tight around the visible hand;
  "no_hand" : this camera does not see that hand (removes a wrong box);
  "reject"  : this camera's image is unusable at this frame.
If, looking closely, nothing is actually wrong, return an empty list.
"""

FIX_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["fixes"],
    "properties": {
        "fixes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cam", "side", "action", "k", "box"],
                "properties": {
                    "cam": {"type": "integer"},
                    "side": {"type": "string", "enum": ["left", "right"]},
                    "action": {"type": "string", "enum": ["pick", "draw", "no_hand", "reject"]},
                    "k": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
                    "box": {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}]},
                },
            },
        }
    },
}


def check_watch(n_rows: int, answer: dict) -> None:
    """A failure must point at a real row and camera; ok and failure must agree."""
    failure = answer.get("failure")
    if answer["ok"] != (failure is None):
        raise ValueError(f"ok={answer['ok']} but failure={failure}")
    if failure is not None and not (0 <= failure["row"] < n_rows and 0 <= failure["cam"] <= 5):
        raise ValueError(f"failure outside the filmstrip: {failure}")


def fixes_to_labels(info: dict, answer: dict, reviewer: str) -> list[dict]:
    """Turn fix actions into ViewLabel dicts for only the (cam, side) the reviewer changed (pure)."""
    by_k: dict[int, dict] = {p["k"]: p for p in info["proposals"]}
    labels: dict[tuple[int, str], dict] = {}
    for fix in answer["fixes"]:
        cam, side, action = fix["cam"], fix["side"], fix["action"]
        if cam not in {c["cam"] for c in info["cells"]}:
            continue
        if action == "pick":
            prop = by_k.get(fix["k"]) if fix["k"] is not None else None
            if prop is None or prop["cam"] != cam:
                continue
            label = {"status": "accepted" if prop["side"] == side else "corrected", "box": prop["box"],
                     "source": prop["model"] if prop["side"] == side else "agent"}
        elif action == "draw":
            drawn = sheet_to_native(info["cells"], cam, fix["box"]) if fix["box"] else None
            if drawn is None:
                continue
            label = {"status": "corrected", "box": dict(zip(("x0", "y0", "x1", "y1"), drawn, strict=True)), "source": "agent"}
        elif action == "no_hand":
            label = {"status": "no_hand", "box": None, "source": "agent"}
        else:
            label = {"status": "rejected", "box": None, "source": "agent"}
        labels[(cam, side)] = {"cam": cam, "side": side, "reviewer": reviewer, **label}
    return list(labels.values())


def megatrack_segment(server: str, sid: str, args: argparse.Namespace, workdir: Path) -> None:
    """Watch the whole segment in windows; accept what is right; fix only the first failure of each window."""
    info = json_object(f"{server}/api/segment/{sid}")
    times: list[int] = info["times_ns"]
    if args.clip_s is not None:
        # label one clip only: [start, start + length) seconds after the segment's first grid time
        start_s, length_s = args.clip_s
        t_lo: int = times[0] + round(start_s * 1e9)
        times = [t for t in times if t_lo <= t < t_lo + round(length_s * 1e9)]
        print(f"[{sid}] clip {start_s:.1f}-{start_s + length_s:.1f} s: {len(times)} grid frames", flush=True)
    window: int = args.window
    log_path = workdir / f"{sid}.jsonl"
    fails_at: dict[int, int] = {}
    # Resume where the reviewer left off: after the last accepted run, or at the last fixed frame if that is later.
    actions: list = json_list(f"{server}/api/actions/{sid}")
    accepted_to: int = max((a["t1_ns"] for a in actions if a["kind"] == "accept_range" and a["t1_ns"] is not None), default=-1)
    last_fix: int = max((a["t_ns"] for a in actions if a["kind"] not in ("accept_range", "accept_frame")), default=-1)
    i = next((k for k, t in enumerate(times) if t > accepted_to), len(times))
    if last_fix > accepted_to and last_fix in times:
        i = times.index(last_fix)
    if i:
        print(f"[{sid}] resuming at frame {i}/{len(times)} ({len(actions)} earlier actions)", flush=True)
    t_start = time.time()
    n_watch = n_fix = 0
    stride = 1

    def log(event: dict) -> None:
        with log_path.open("a") as fh:
            fh.write(json.dumps({"wall": time.time(), **event}) + "\n")

    def accept(a: int, b: int) -> None:  # inclusive grid indices
        result = json_object(f"{server}/api/accept_range", {"segment_id": sid, "t0_ns": times[a], "t1_ns": times[b], "reviewer": args.reviewer})
        log({"action": "accept_range", "t0_ns": times[a], "t1_ns": times[b], "result": result})

    def step(i: int) -> int:
        """Watch one window from grid index i; accept it or fix its first failure. Returns the next index."""
        nonlocal n_watch, n_fix, stride
        s = stride
        n = min(window, (len(times) - i - 1) // s + 1)
        jpg = http_bytes(f"{server}/api/playback/{sid}/{times[i]}.jpg?n={n}&stride={s}")
        prompt = WATCH_PROMPT.format(rows=n, gap=f"{0.2 * s:.1f}")
        watch = ask_with_retries(jpg, prompt, WATCH_SCHEMA, args, lambda a, n=n: check_watch(n, a), f"{sid} watch {times[i]}")
        n_watch += 1
        log({"action": "watch", "t_ns": times[i], "rows": n, "stride": s, "answer": watch})
        failure = watch["failure"]
        if failure is None:
            last = i + (n - 1) * s
            accept(i, last)
            stride = min(stride + 1, args.max_stride)  # the track holds: look at a longer stretch next time
            return last + 1
        stride = 1
        r: int = failure["row"]
        if r > 0:
            accept(i, i + (r - 1) * s)
            if s > 1:
                return i + (r - 1) * s + 1  # re-watch the unseen frames before the failure at full rate
        t_f: int = times[i + r * s]
        fails_at[t_f] = fails_at.get(t_f, 0) + 1
        if fails_at[t_f] > 2:
            # the reviewer keeps stopping on the same frame after fixing it twice: reject that view there and go on
            json_object(f"{server}/api/labels", {"segment_id": sid, "t_ns": t_f, "labels": [
                {"cam": failure["cam"], "side": failure["side"], "status": "rejected", "box": None, "source": "agent", "reviewer": args.reviewer}]})
            log({"action": "reject_after_repeats", "t_ns": t_f, "failure": failure})
            accept(i + r * s, i + r * s)
            return i + r * s + 1
        sheet = json_object(f"{server}/api/sheet/{sid}/{t_f}.json")
        sheet_jpg = http_bytes(f"{server}/api/sheet/{sid}/{t_f}.jpg")
        prompt = FIX_PROMPT.format(cam=failure["cam"], side=failure["side"], kind=failure["kind"], proposals=proposal_lines(sheet))
        fix = ask_with_retries(sheet_jpg, prompt, FIX_SCHEMA, args, lambda a: None, f"{sid} fix {t_f}")
        n_fix += 1
        labels = fixes_to_labels(sheet, fix, args.reviewer)
        log({"action": "fix", "t_ns": t_f, "failure": failure, "answer": fix, "n_labels": len(labels)})
        if labels:
            json_object(f"{server}/api/labels", {"segment_id": sid, "t_ns": t_f, "labels": labels})
            return i + r * s  # watch again from the fixed frame; propagation now starts from the fix
        accept(i + r * s, i + r * s)  # on a closer look nothing was wrong
        return i + r * s + 1

    errors_in_a_row = 0
    while i < len(times):
        try:
            i = step(i)
        except (RuntimeError, OSError) as err:
            # a server hiccup or a model failure: wait and watch the same window again; give up only if it keeps failing
            if "usage limit" in str(err).lower():
                raise
            errors_in_a_row += 1
            print(f"[{sid}] window at {times[i]} failed ({errors_in_a_row}): {err}", file=sys.stderr, flush=True)
            if errors_in_a_row >= 6:
                raise
            time.sleep(30 * errors_in_a_row)
            continue
        errors_in_a_row = 0
        if n_watch % 20 == 0:
            done_s = (times[min(i, len(times) - 1)] - times[0]) / 1e9
            print(f"[{sid}] {i}/{len(times)} frames, {n_watch} windows, {n_fix} fixes, {done_s / max(time.time() - t_start, 1e-6):.2f}x real time", flush=True)
    commit = json_object(f"{server}/api/commit/{sid}", {})
    effort = json_object(f"{server}/api/effort/{sid}")
    print(f"[{sid}] done: {n_watch} windows, {n_fix} fixes; committed {commit}; effort {effort}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default="http://dgx-spark:9898")
    parser.add_argument("--segment", action="append", default=[], help="segment id; repeat; default = every segment")
    parser.add_argument("--window", type=int, default=10, help="megatrack: filmstrip rows per playback window")
    parser.add_argument("--clip-s", type=float, nargs=2, metavar=("START", "LENGTH"), default=None,
                        help="megatrack: label only this clip, seconds after the segment's first grid time")
    parser.add_argument("--max-stride", type=int, default=3, help="megatrack: grid steps between rows once windows keep passing (1 = every 5 fps frame)")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--reviewer", default="astra")
    parser.add_argument("--workers", type=int, default=4, help="parallel segments")
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex-work"))
    parser.add_argument("--workdir", default="/mnt/nas/ssd-8tb/scratch/hand-labeler/astra")
    args = parser.parse_args()
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    segments: list[str] = args.segment or [s["segment_id"] for s in json_list(f"{args.server}/api/segments")]
    # One reviewer per segment, segments in parallel: inside a segment the loop is sequential (a fix changes what follows).
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(megatrack_segment, args.server, sid, args, workdir): sid for sid in segments}
        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except (RuntimeError, OSError) as err:
                print(f"[{futures[fut]}] stopped: {err}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
