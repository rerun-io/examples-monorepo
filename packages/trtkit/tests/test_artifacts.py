"""Concurrent callers share one build of a missing artifact."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from trtkit.artifacts import build_once


def test_concurrent_callers_build_a_missing_artifact_once(tmp_path: Path) -> None:
    artifact = tmp_path / "cache" / "model.onnx"
    builds: list[int] = []
    barrier = Barrier(4)

    def build() -> None:
        builds.append(1)
        artifact.write_bytes(b"model")

    def call(_: int) -> Path:
        barrier.wait(timeout=10)
        return build_once(artifact, build)

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(call, range(4)))
    assert results == [artifact] * 4
    assert artifact.read_bytes() == b"model"
    assert len(builds) == 1
    assert build_once(artifact, lambda: builds.append(2)) == artifact and len(builds) == 1
