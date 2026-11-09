"""GSI Egocentric Six-View against GSI's live API: download and convert the edition's smallest episode."""

import os
from pathlib import Path

import pytest
import rerun.chunk as rrc
from conftest import read_back, read_chunks, recording_properties

from dataforge import schema
from dataforge.datasets.gsi_six_view import GsiSixViewConfig

SMALLEST: str = "ep_v1_6a03abdb6245d608f61851028dc83477"
"""The edition's smallest episode, 33 MB of hands.rrd."""


@pytest.mark.integration
def test_the_live_api_serves_an_episode_that_converts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.environ.get("GSI_PARTNER_PASSWORD"):
        pytest.skip("needs GSI's partner sign-in: GSI_PARTNER_USERNAME and GSI_PARTNER_PASSWORD (source ~/.secrets.env)")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = GsiSixViewConfig(root=tmp_path / "raw", sequences=(SMALLEST,)).setup()
    dataset.download()
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=False)

    targets: dict[str, Path] = dataset.targets(identity)
    assert all(target.is_file() for target in targets.values())
    base: rrc.ChunkStore = read_back(targets["base"])
    assert recording_properties(base, "capture")["category"] == "Food preparation"
    assert {schema.video_path(0, cam) for cam in range(6)} <= {str(chunk.entity_path) for chunk in read_chunks(targets["base"])}
    assert not source.exists()


