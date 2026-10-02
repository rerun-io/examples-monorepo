"""A short hands layer from a real catalog segment through the ONNX nets (not registered).

Needs a catalog server holding the robocap dataset with base and slam_rs layers, and the ONNX models: set
``ROBOCAP_LIVE_CATALOG_URL``, ``ROBOCAP_LIVE_MODELS_DIR`` (detnet_full.onnx, keynet.onnx) and ``ROBOCAP_LIVE_SEGMENT``.
"""

import os
from pathlib import Path

import pytest

from robocap_live.apis.hands_layer import Config, LayerRun, main

pytestmark = pytest.mark.integration


def test_a_catalog_segment_becomes_a_hands_layer(tmp_path: Path) -> None:
    catalog: str | None = os.environ.get("ROBOCAP_LIVE_CATALOG_URL")
    models: str | None = os.environ.get("ROBOCAP_LIVE_MODELS_DIR")
    segment: str | None = os.environ.get("ROBOCAP_LIVE_SEGMENT")
    if not (catalog and models and segment):
        pytest.skip("set ROBOCAP_LIVE_CATALOG_URL, ROBOCAP_LIVE_MODELS_DIR and ROBOCAP_LIVE_SEGMENT (catalog server + ONNX models)")
    if not (Path(models) / "detnet_full.onnx").is_file():
        pytest.skip(f"ONNX models absent from {models}")
    try:
        run: LayerRun = main(Config(segment=segment, output_dir=tmp_path, catalog=catalog, models_dir=Path(models), max_framesets=30))
    except ConnectionError as error:
        pytest.skip(f"catalog {catalog} unreachable: {error}")
    assert run.framesets == 30 and Path(run.output).is_file() and run.register_s == 0.0
    assert run.with_pose + run.held_pose > 0
