from __future__ import annotations

import json
from pathlib import Path

from mv_api.api.batch_calibration import ManifestEntry, SequenceManifest, discover_episodes_in_sequence


def test_discover_episodes_in_sequence(tmp_path: Path) -> None:
    sequence_path = tmp_path / "seq_001"
    episode_dir = sequence_path / "episodes" / "episode_001"
    episode_dir.mkdir(parents=True)
    (episode_dir / "episode_001.rrd").write_bytes(b"rrd")

    episodes = discover_episodes_in_sequence(sequence_path)

    assert len(episodes) == 1
    assert episodes[0].rrd_path == episode_dir / "episode_001.rrd"
    assert episodes[0].calibrated_path == episode_dir / "episode_001-calibrated.rrd"
    assert episodes[0].sequence_id == "seq_001"


def test_sequence_manifest_roundtrip(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = SequenceManifest.load_or_create(manifest_path, "seq_001")
    manifest.episodes["episode_001"] = ManifestEntry(
        episode_path="episode.rrd",
        calibrated_path="episode-calibrated.rrd",
        status="success",
        calibrated_at="2026-04-07T00:00:00",
    )

    manifest.save(manifest_path)
    loaded = SequenceManifest.load_or_create(manifest_path, "seq_001")

    payload = json.loads(manifest_path.read_text())
    assert payload["sequence_id"] == "seq_001"
    assert loaded.episodes["episode_001"].status == "success"
