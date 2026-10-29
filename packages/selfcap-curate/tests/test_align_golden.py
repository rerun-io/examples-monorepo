"""Raw PR2 conversion → temporary catalog → actual Tyro alignment command."""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import torch
from dataforge import schema
from dataforge.datasets.selfcap import SelfcapConfig, build_episode_plan, episode_is_readable
from dataforge.identity import SequenceIdentity
from dataforge.schema import CAPTURE_TIMELINE
from rerun.chunk import RrdReader

from selfcap_curate.catalog_io import DATASET_NAME
from selfcap_curate.records import read_frozen_sync, read_sync

pytestmark = pytest.mark.golden
V2 = Path("/home/pablo/exoego-data/dataforge-selfcap-aligned/v2")
RAW = Path("/mnt/nas/datasets/exoego-self-collected/main/cut")


def test_align_reproduces_session_from_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_server) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for raw-to-aligned golden chain")
    keys = [f"2025-12-30__65d00748__ep{number:03d}" for number in (1, 2)]
    references = {}
    for key in keys:
        source = V2 / "sync" / f"{key}.json"
        if not source.is_file():
            pytest.skip(f"golden sync record absent: {source}")
        references[key] = read_frozen_sync(source)
    bases, plans = [], {}
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "converted"))
    for key, record in references.items():
        raw = RAW / record.date / record.session / "episodes" / f"episode-{record.episode[2:]}"
        if not episode_is_readable(raw) or not (raw / "quest/left_hand_poses.csv").is_file():
            pytest.skip(f"golden raw episode absent or unreadable: {raw}")
        plans[key] = build_episode_plan(raw)
        start = time.monotonic()
        bases.append(
            SelfcapConfig()
            .setup()
            .convert(SequenceIdentity(dataset="selfcap", parts=(record.date, record.session[:8], record.episode)), raw, force=False)
        )
        print("RAW_ALIGN_TIME", key, "convert", time.monotonic() - start, flush=True)
    output = tmp_path / "aligned"
    # Read-only warm model assets may be copied into the test cache. The final
    # session directory must remain absent until publication. No frozen detections, links or sync inputs are read.
    for source, destination, patterns in [
        (
            Path("/home/pablo/.cache/posekit/onnx"),
            tmp_path / "cache/onnx",
            ("yolox_m_8xb8-300e_humanart-c2c7a14a*.onnx", "sapiens2_0.4b_pose_b8_bf16_v2.onnx*"),
        ),
        (Path("/tmp/selfcap-c2-cache/trt"), tmp_path / "cache/trt", ("*.engine*",)),
    ]:
        destination.mkdir(parents=True, exist_ok=True)
        for pattern in patterns:
            for asset in source.glob(pattern):
                shutil.copyfile(asset, destination / asset.name)
    monkeypatch.setenv("POSEKIT_ONNX_CACHE", str(tmp_path / "cache/onnx"))
    monkeypatch.setenv("TRTKIT_TRT_CACHE", str(tmp_path / "cache/trt"))
    with private_server({DATASET_NAME: bases}) as server:
        assert ":51235" not in server.url()
        key_file = tmp_path / "keys.txt"
        key_file.write_text("\n".join(keys) + "\n")
        command = [
            sys.executable,
            str(Path(__file__).parents[1] / "tools/apps/align.py"),
            "--base-dir",
            str(bases[0].parent),
            "--output-dir",
            str(output),
            "--key-file",
            str(key_file),
            "--catalog-url",
            server.url(),
        ]
        start = time.monotonic()
        completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
        print(completed.stdout, flush=True)
        (tmp_path / "align.log").write_text(completed.stdout)
        print("RAW_ALIGN_TIME", "session", "align", time.monotonic() - start, flush=True)
        assert completed.returncode == 0, completed.stdout[-6000:]

    failures = []
    for key, reference in references.items():
        record = read_sync(output / f"selfcap__{key}.json")
        assert record.session == reference.session
        assert record.oak_rgb_model.t_start_session_s == reference.oak_rgb_model.t_start_session_s
        assert record.offsets_ms is not None and reference.offsets_ms is not None
        for stream, expected in reference.offsets_ms.items():
            actual = record.offsets_ms[stream]
            delta = None if actual is None or expected is None else abs(actual - expected)
            print("RAW_ALIGN_OFFSET", key, stream, expected, actual, delta, flush=True)
            if actual is not None if expected is None else actual is None or delta > 2.0:
                failures.append(f"{key} {stream}: frozen={expected}, port={actual}")
        plan = plans[key]
        streams = {
            schema.video_path(rig.rig, camera.cam): (
                "exo/" + rig.name if rig.kind == "exo" else "quest_video" if rig.kind == "quest" else "oak_" + camera.name
            )
            for rig in plan.rigs
            for camera in rig.cameras
        }
        clock_errors = {}
        for chunk in RrdReader(output / f"selfcap__{key}.rrd").stream():
            if chunk.is_static or str(chunk.entity_path) not in streams:
                continue
            batch = chunk.to_record_batch()
            capture = np.asarray(batch.column(CAPTURE_TIMELINE).cast(pa.int64()))
            video = np.asarray(batch.column("video_time").cast(pa.int64()))
            stream = streams[str(chunk.entity_path)]
            if stream == "oak_rgb":
                model = record.oak_rgb_model
                offset = np.rint((model.a_ms + model.b_ms_per_s * (model.t_start_session_s + capture / 1e9)) * 1e6).astype(np.int64)
            else:
                offset = round((record.offsets_ms[stream] or 0.0) * 1e6)
            error = int(np.max(np.abs(video - capture - offset)))
            entity = str(chunk.entity_path)
            clock_errors[entity] = max(clock_errors.get(entity, 0), error)
            assert error == 0
        assert len(clock_errors) == sum(len(rig.cameras) for rig in plan.rigs)
        for stream, error in clock_errors.items():
            print("RAW_ALIGN_CLOCK", key, stream, error, flush=True)
    assert not failures, "\n".join(failures)
