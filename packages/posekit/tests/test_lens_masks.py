"""The lens mask step's input boundary: checks that must hold before any model loads."""

from pathlib import Path

import pytest


def test_an_output_that_is_an_input_recording_is_refused_and_the_input_kept(tmp_path: Path) -> None:
    """``--output-dir`` beside the base names ``<segment>.rrd``: the atomic write would replace the base with the masks."""
    import rerun as rr

    from posekit.apis.lens_masks import LensMasksConfig, main

    base: Path = tmp_path / "seg.rrd"
    recording = rr.RecordingStream("dataforge", recording_id="seg")
    recording.save(base)
    recording.log("/point", rr.Points3D([[0.0, 0.0, 0.0]]), static=True)
    recording.flush()
    recording.disconnect()
    before: bytes = base.read_bytes()
    with pytest.raises(ValueError, match="input recording"):
        main(LensMasksConfig(output_dir=tmp_path, rrd=base, people_tracks=False))
    assert base.read_bytes() == before
