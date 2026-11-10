"""A short hands layer from a real catalog segment through the ONNX nets (not registered).

Needs a catalog server holding the robocap dataset with base and slam_rs layers, and the ONNX models: set
``ROBOCAP_LIVE_CATALOG_URL``, ``ROBOCAP_LIVE_MODELS_DIR`` (keynet.onnx and detnet/<version>/detnet_full.onnx) and
``ROBOCAP_LIVE_SEGMENT``. Runs once per DetNet version, each with its own camera turn.
"""

import os
from pathlib import Path
from typing import Literal

import pytest

from robocap_live.apis.hands_layer import Config, LayerRun, main

VERSIONS: list[Literal["mix1-upright", "0929"]] = ["mix1-upright", "0929"]

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("version", VERSIONS)
def test_a_catalog_segment_becomes_a_hands_layer(tmp_path: Path, version: Literal["mix1-upright", "0929"]) -> None:
    catalog: str | None = os.environ.get("ROBOCAP_LIVE_CATALOG_URL")
    models: str | None = os.environ.get("ROBOCAP_LIVE_MODELS_DIR")
    segment: str | None = os.environ.get("ROBOCAP_LIVE_SEGMENT")
    if not (catalog and models and segment):
        pytest.skip("set ROBOCAP_LIVE_CATALOG_URL, ROBOCAP_LIVE_MODELS_DIR and ROBOCAP_LIVE_SEGMENT (catalog server + ONNX models)")
    if not (Path(models) / "detnet" / version / "detnet_full.onnx").is_file():
        pytest.skip(f"DetNet {version} absent from {models}/detnet/{version}")
    try:
        run: LayerRun = main(Config(segment=segment, output_dir=tmp_path, catalog=catalog, models_dir=Path(models), max_framesets=30,
                              detnet_version=version))
    except ConnectionError as error:
        pytest.skip(f"catalog {catalog} unreachable: {error}")
    assert run.framesets == 30 and Path(run.output).is_file() and run.register_s == 0.0
    assert run.with_pose + run.held_pose > 0
