"""Processes that find no engine build it once: the rest wait for the first."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import pytest

from trtkit import trt_builder
from trtkit.trt_builder import TrtBuildConfig, ensure_engine


def test_concurrent_first_uses_build_one_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[Path] = []

    def build(_: Path, engine: Path, __: TrtBuildConfig, *, onnx_sha256: str) -> None:
        builds.append(engine)
        sleep(0.2)
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_bytes(b"engine")

    onnx: Path = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")
    monkeypatch.setattr(trt_builder, "build_engine", build)
    monkeypatch.setattr(trt_builder, "onnx_content_hash", lambda *_, **__: "hash")
    monkeypatch.setattr(trt_builder, "cached_engine_path", lambda *_, cache_dir, **__: cache_dir / "model.engine")
    with ThreadPoolExecutor(3) as pool:
        engines: list[Path] = list(pool.map(lambda _: ensure_engine(onnx, TrtBuildConfig(), cache_dir=tmp_path / "cache"), range(3)))
    assert engines == [tmp_path / "cache/model.engine"] * 3
    assert len(builds) == 1
