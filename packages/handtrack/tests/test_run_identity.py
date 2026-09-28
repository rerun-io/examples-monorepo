"""Run reuse is tied to immutable settings and complete segment artifacts."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from serde.json import to_json
from test_results import _track

from handtrack.apis import run_pipeline
from handtrack.apis.evaluate import _complete
from handtrack.apis.run_pipeline import RunConfig, file_sha256, metrics_path
from handtrack.eval.segment import DetNetAloneMetrics
from handtrack.fit.pose_fit import FitConfig
from handtrack.results import save_track
from handtrack.tracker import TrackerConfig


def test_identity_first_writer_wins_and_rejects_changed_settings(tmp_path: Path) -> None:
    config = RunConfig(output_root=tmp_path, detector="oracle", keypoints="oracle", max_frames=30)
    with ThreadPoolExecutor(max_workers=4) as workers:
        identities = list(workers.map(lambda _: run_pipeline.ensure_run_identity(config), range(4)))
    assert len(set(identities)) == 1
    path = tmp_path / config.name / "identity.json"
    original = path.read_bytes()
    for changed, field in [
        (replace(config, max_frames=None), "max_frames"),
        (replace(config, seed=1), "seed"),
        (replace(config, tracker=TrackerConfig(fit=FitConfig(max_iterations=7))), "tracker"),
        (replace(config, hand_modes=("unknown",)), "hand_modes"),
        (replace(config, domain="synthetic"), "domain"),
    ]:
        with pytest.raises(ValueError, match=field):
            run_pipeline.ensure_run_identity(changed)
        assert path.read_bytes() == original
    assert run_pipeline.ensure_run_identity(replace(config, shard=3, shards=4)) == identities[0]
    assert json.loads(original)["seed"] == 0
    assert json.loads(original)["split"] == "test"


def test_completion_requires_sidecar_metrics_hash_and_identity(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "detnet.weights.pt").write_bytes(b"weights")
    config = RunConfig(output_root=tmp_path, checkpoints=weights, hand_modes=(), keypoints="oracle")
    identity = run_pipeline.ensure_run_identity(config)
    track = _track(2)
    track = replace(track, meta=replace(track.meta, kind="detnet_alone", hand_mode="known", run_identity_sha256=identity))
    directory = tmp_path / config.name / "detnet"
    path = save_track(track, directory)
    metrics = DetNetAloneMetrics(track.meta.segment, "real", "hand_hand", 2, "a" * 64, [], [], file_sha256(path), run_identity_sha256=identity)
    metrics_path(directory, track.meta.segment).write_text(to_json(metrics))
    assert _complete(config, track.meta.segment)
    with pytest.raises(ValueError, match="max_frames"):
        _complete(replace(config, max_frames=30), track.meta.segment)
    metrics_file = metrics_path(directory, track.meta.segment)
    metrics_text = metrics_file.read_text()
    for content in ("{}", metrics_text.replace(identity, "wrong")):
        metrics_file.write_text(content)
        assert not _complete(config, track.meta.segment)
    metrics_file.unlink()
    assert not _complete(config, track.meta.segment)
    metrics_file.write_text(metrics_text)
    sidecar = path.with_suffix(".json")
    original = sidecar.read_text()
    sidecar.unlink()
    assert not _complete(config, track.meta.segment)
    sidecar.write_text(original.replace(identity, "wrong"))
    assert not _complete(config, track.meta.segment)
    sidecar.write_text(original)
    path.write_bytes(b"corrupt")
    assert not _complete(config, track.meta.segment)
    (weights / "detnet.weights.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="detnet_sha256"):
        run_pipeline.ensure_run_identity(config)
