"""What the PC11 dump producer refuses before it opens a segment.

``tests/tools/dump_clip.py`` is a test-only producer — it writes the ``.npz``
bundle :mod:`slam_rs.apis.bench_track` replays and the PGM directory the Rust
lanes and the C++ oracle read — and a whole segment is gigabytes, so what is
worth a test here is the selection it will not spend a feed on. It is reached as
a namespace module under ``tests``, the same path pytest and pyrefly are given.
"""

from pathlib import Path

import pytest
from fixture_types import never
from tools.dump_clip import Config, main

from slam_rs.reference import SMOKE_SEGMENTS
from tools import dump_clip


def test_an_npz_dump_of_no_framesets_is_refused_before_the_feed_is_opened(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``--max-framesets 0 --npz`` wrote a partial directory and then raised out of NumPy.

    Zero was accepted, the loop broke on its first frameset, and the non-NPZ
    side files (``calib.json``, ``imu.csv``, ``timestamps.txt``, ``imu.json``,
    ``frames.sha256``, ``clip.json``) were all written before ``np.stack`` on the
    empty image list raised ``ValueError: need at least one array to stack`` — so
    the tool left a clip directory that reads as a dump and holds no frameset
    (S25 review). The refusal comes before the manifest, the catalog and the
    output directory.
    """

    monkeypatch.setattr(dump_clip, "load_manifest", never("the manifest was read for a dump of no framesets"))
    output: Path = tmp_path / "clip"
    with pytest.raises(ValueError, match="--max-framesets 0.*--npz"):
        main(Config(segment=SMOKE_SEGMENTS[1], output=output, max_framesets=0, npz=True))
    assert not output.exists()
