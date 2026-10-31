"""Restore published S1 inputs without inference; preserve original selected detections."""

import hashlib
import json
import subprocess
from pathlib import Path

import cache2d
import numpy as np
import pyarrow as pa
from rerun.chunk import RrdReader
from rerun.server import Server
from selfcap_curate.calib_layer import CalibSession, write_calib_layer
from selfcap_curate.catalog_io import DATASET_NAME, open_segment, read_time_sync
from selfcap_curate.keypoints import Keypoints2D, ViewDetections
from selfcap_curate.quest_fit import read_session_episode
from serde.json import from_json

ROOT = Path("/tmp/selfcap-night2")


def fetch(uri: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(["scp", "-q", "spark:" + uri.removeprefix("file://"), str(target)], check=True, timeout=180)


def restore(segment: str, original_rtmw: Path) -> dict:
    work = ROOT / "runs" / segment
    s1 = work / "s1"
    receipts = json.loads((work / "publish-receipts.json").read_text())
    for layer, target in [
        ("base", s1 / "align" / f"{segment}.rrd"),
        ("sapiens2", s1 / "align/sapiens2" / f"{segment}.rrd"),
        ("egohand_calib", s1 / "calib/egohand" / f"{segment}.rrd"),
    ]:
        if layer in receipts:
            fetch(receipts[layer]["uri"], target)
    calib = s1 / "calib" / f"{segment}.rrd"
    if not calib.exists():
        state = from_json(CalibSession, next((s1 / "calib").glob("*.json")).read_text())
        with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: [s1 / "align" / f"{segment}.rrd"]}) as server:
            dataset = server.client().get_dataset(DATASET_NAME)
            record = open_segment(dataset, segment)
            write_calib_layer(calib, state, read_session_episode(record, read_time_sync(dataset, segment)))
    cache = s1 / "rtmw-cache"
    if list(cache.glob("*.parquet")):
        return {"cache": "existing"}
    # Layout from the base is authoritative; lexical phone order is not the rig order.
    with Server(host="127.0.0.1", port=0, datasets={DATASET_NAME: [s1 / "align" / f"{segment}.rrd"]}) as server:
        dataset = server.client().get_dataset(DATASET_NAME)
        phones = open_segment(dataset, segment).layout.phones
        sync = read_time_sync(dataset, segment)
        expected = tuple(phone for phone in phones if (sync.offsets_ms or {}).get("exo/" + phone) is not None)
    batches = {}
    for chunk in RrdReader(original_rtmw).stream():
        if chunk.is_static:
            continue
        path = str(chunk.entity_path)
        if not path.endswith("/pinhole/pred/rtmw/coco133_uv"):
            continue
        batch = chunk.to_record_batch()
        if "Points2D:positions" not in batch.schema.names:
            continue
        rig = int(path.split("/")[2].removeprefix("rig_"))
        batches.setdefault(rig, []).append(batch)
    views = {}
    times = None
    for rig, items in batches.items():
        t = pa.concat_tables([pa.Table.from_batches([b]) for b in items], promote_options="default")
        positions = t["Points2D:positions"]
        valid = positions.is_valid()
        t = t.filter(valid)
        stamps = np.asarray(t["video_time"].cast(pa.int64()))
        order = np.argsort(stamps)
        stamps = stamps[order]
        xy = np.asarray(t["Points2D:positions"].to_pylist(), np.float32)[order]
        score = np.asarray(t["simplecv.KeypointConfidence2D:confidences"].to_pylist(), np.float32)[order].reshape(-1, 133)
        if times is None:
            times = stamps.astype(np.float64) / 1e9
        else:
            np.testing.assert_array_equal(np.rint(times * 1e9).astype(np.int64), stamps)
        finite = np.isfinite(xy).all(-1)
        lo = np.min(np.where(finite[..., None], xy, np.inf), axis=1)
        hi = np.max(np.where(finite[..., None], xy, -np.inf), axis=1)
        boxes = np.concatenate([lo, hi], axis=1).astype(np.float32)
        boxes[~np.isfinite(boxes)] = 0
        views[phones[rig]] = ViewDetections(boxes[:, None], np.max(score, axis=1)[:, None], xy[:, None], score[:, None])
    if times is None:
        raise ValueError("original S1 recording has no RTMW point rows")
    # The raw S1 layer omits uncalibrated views. Keep their empty slots for the
    # detector-cache phone contract; they cannot contribute to triangulation.
    for phone in expected:
        if phone not in views:
            views[phone] = ViewDetections(
                np.zeros((len(times), 1, 4), np.float32),
                np.zeros((len(times), 1), np.float32),
                np.full((len(times), 1, 133, 2), np.nan, np.float32),
                np.zeros((len(times), 1, 133), np.float32),
            )
    rate = json.loads((s1 / "complete.json").read_text())["rate"]
    cache2d.save(cache, Keypoints2D(times, views, rate))
    restored = cache2d.load(cache, expected_phones=expected, expected_rate=rate)
    for phone, view in views.items():
        np.testing.assert_array_equal(restored.views[phone].xy, view.xy)
        np.testing.assert_array_equal(restored.views[phone].scores, view.scores)
    with original_rtmw.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"cache": "original_s1_selected", "source": str(original_rtmw), "sha256": digest, "frames": len(times)}
    (cache / "source.json").write_text(json.dumps(result, indent=2))
    return result
