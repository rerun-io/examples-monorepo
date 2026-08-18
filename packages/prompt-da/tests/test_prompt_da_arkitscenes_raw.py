"""Behavioral tests for raw-disk ARKitScenes PromptDA inference.

Helpers shared with the catalog-dataloader tool are covered by
``test_prompt_da_arkitscenes.py``; timeline helpers live in
``arkitscenes-download``'s ``test_ingest_clock.py``.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("pyarrow", reason="ARKitScenes catalog deps live in the prompt-da-stream envs")
pytest.importorskip("torchcodec", reason="raw video decode deps live in the prompt-da-stream envs")
pytest.importorskip("arkitscenes_download", reason="ARKitScenes ingest deps live in the prompt-da-stream envs")

from rerun_prompt_da.apis.arkitscenes_shared import segments_to_process  # noqa: E402
from rerun_prompt_da.apis.prompt_da_arkitscenes_raw import (  # noqa: E402
    nearest_timestamped_path,
    raw_sequence_dir,
    rotate_depth_for_catalog,
)


def test_nearest_timestamped_path_enforces_pairing_tolerance() -> None:
    """Pair the closest sensor sample and reject one beyond two milliseconds."""
    paths = [Path("frame_1.000.png"), Path("frame_1.010.png"), Path("frame_1.020.png")]
    assert nearest_timestamped_path(paths, 1.0119, tolerance_s=0.002) == paths[1]
    assert nearest_timestamped_path(paths, 1.0121, tolerance_s=0.002) is None
    assert nearest_timestamped_path(paths, 1.016, tolerance_s=None) == paths[2]
    assert nearest_timestamped_path([], 1.0, tolerance_s=None) is None


def test_process_all_skips_existing_layer_and_missing_raw_data(tmp_path: Path) -> None:
    """Select only unfinished catalog segments that also exist on local disk."""
    (tmp_path / "Training" / "ready").mkdir(parents=True)
    rows = [
        {"rerun_segment_id": "done", "rerun_layer_names": ["base", "promptda_raw"]},
        {"rerun_segment_id": "ready", "rerun_layer_names": ["base"]},
        {"rerun_segment_id": "missing", "rerun_layer_names": ["base"]},
    ]

    def raw_data_available(segment_id: str) -> bool:
        return raw_sequence_dir(tmp_path, segment_id) is not None

    assert segments_to_process(rows, None, True, "promptda_raw", raw_data_available) == ["ready"]


def test_catalog_depth_rotation_reverses_unbaking() -> None:
    """Rotate landscape prediction back to the catalog's baked orientation."""
    baked = np.arange(6, dtype=np.uint16).reshape(3, 2)
    for quarter_turns in range(4):
        landscape = np.ascontiguousarray(np.rot90(baked, -quarter_turns))
        assert_array_equal(rotate_depth_for_catalog(landscape, quarter_turns), baked)
