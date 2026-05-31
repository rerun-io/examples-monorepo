from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_benchmark_module() -> ModuleType:
    module_path: Path = Path(__file__).parent.parent / "tools" / "benchmark_torchcodec_cuda_multiview.py"
    spec = importlib.util.spec_from_file_location("benchmark_torchcodec_cuda_multiview", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_epfl_exo_rgb_paths_returns_configured_camera_order(tmp_path: Path) -> None:
    """EPFL benchmark inputs follow the dataset's exo camera order."""
    module: ModuleType = _load_benchmark_module()
    video_dir: Path = tmp_path / "Public_release_videos" / "train" / "YH2002" / "2023_12_04_10_15_23" / "videos"
    video_dir.mkdir(parents=True)
    for camera_name in module.EPFL_EXO_CAMERA_NAMES:
        (video_dir / f"{camera_name}.mp4").touch()

    video_paths: list[Path] = module.epfl_exo_rgb_paths(tmp_path, "train", "YH2002", "2023_12_04_10_15_23")

    assert [path.name for path in video_paths] == [f"{camera_name}.mp4" for camera_name in module.EPFL_EXO_CAMERA_NAMES]


def test_benchmark_parser_defaults_to_chunked_approximate_decode() -> None:
    """The benchmark defaults to the chosen simple chunked CUDA path."""
    module: ModuleType = _load_benchmark_module()

    args = module.build_parser().parse_args([])

    assert args.chunk_size == 32
    assert args.seek_mode == "approximate"
    assert not hasattr(args, "temporal_shards")


def test_benchmark_parser_rejects_materialized_decode_mode() -> None:
    """The benchmark should only expose positive chunked decode."""
    module: ModuleType = _load_benchmark_module()

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--chunk-size", "0"])
