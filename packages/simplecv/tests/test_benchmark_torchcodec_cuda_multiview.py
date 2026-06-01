from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_benchmark_module() -> ModuleType:
    module_path: Path = Path(__file__).parent.parent / "simplecv" / "apis" / "benchmark_torchcodec_cuda_multiview.py"
    spec = importlib.util.spec_from_file_location("simplecv.apis.benchmark_torchcodec_cuda_multiview", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_video_paths_searches_directory_in_sorted_order(tmp_path: Path) -> None:
    """Benchmark inputs are discovered as sorted mp4s from a directory."""
    module: ModuleType = _load_benchmark_module()
    video_dir: Path = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "cam_b.mp4").touch()
    (video_dir / "notes.txt").touch()
    (video_dir / "cam_a.mp4").touch()

    video_paths: list[Path] = module.find_video_paths(video_dir)

    assert [path.name for path in video_paths] == ["cam_a.mp4", "cam_b.mp4"]


def test_find_video_paths_prefers_rgb_low_videos(tmp_path: Path) -> None:
    """Assembly101-style RGB exo videos are selected when mixed mp4s exist."""
    module: ModuleType = _load_benchmark_module()
    video_dir: Path = tmp_path / "videos"
    video_dir.mkdir()
    (video_dir / "C10095_rgb_low.mp4").touch()
    (video_dir / "C10115_rgb_low.mp4").touch()
    (video_dir / "HMC_84346135_mono10bit_low.mp4").touch()

    video_paths: list[Path] = module.find_video_paths(video_dir)

    assert [path.name for path in video_paths] == ["C10095_rgb_low.mp4", "C10115_rgb_low.mp4"]


def test_benchmark_config_defaults_to_chunked_approximate_decode(tmp_path: Path) -> None:
    """The benchmark defaults to the chosen simple chunked CUDA path."""
    module: ModuleType = _load_benchmark_module()

    config = module.BenchmarkConfig(video_dir=tmp_path)

    assert config.chunk_size == 32
    assert config.seek_mode == "approximate"


def test_tool_is_minimal_tyro_wrapper() -> None:
    """The CLI tool should delegate parsing to tyro and implementation to the API."""
    tool_path: Path = Path(__file__).parent.parent / "tools" / "benchmark_torchcodec_cuda_multiview.py"
    tool_text: str = tool_path.read_text()

    assert "tyro.cli(BenchmarkConfig" in tool_text
    assert "from simplecv.apis.benchmark_torchcodec_cuda_multiview import BenchmarkConfig, main" in tool_text
