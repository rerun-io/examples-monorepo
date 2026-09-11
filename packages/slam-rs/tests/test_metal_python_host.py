"""Exercise Metal detection through the Python host, with an external hang guard."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixture_types import RigFactory, TextureFactory
from jaxtyping import UInt8
from numpy import ndarray

from slam_rs import _core


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS Metal")
def test_python_host_metal_detects_stereo_features(rig: RigFactory, texture: TextureFactory, tmp_path: Path) -> None:
    """Construction alone misses shader compilation in the first detection pass."""
    assert _core.gpu_backend == "wgpu", "the Metal gate requires a freshly built wgpu core"
    image_path: Path = tmp_path / "texture.npy"
    np.save(image_path, texture(0, 0))
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), rig(2).to_json(), str(image_path)],
        capture_output=True,
        text=True,
        timeout=120.0,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip("wgpu found no Metal adapter on this host")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "detected stereo features" in result.stdout


def run_probe(calibration_json: str, image_path: Path) -> None:
    """Feed inertial samples and the fixture's textured stereo scene to Vio."""
    try:
        vio: _core.Vio = _core.Vio(_core.Calibration.from_json(calibration_json), _core.VioConfig(), gpu=True)
    except ValueError as error:
        if "wgpu found no metal adapter on this host" in str(error):
            sys.exit(77)
        raise
    assert vio.gpu
    image: UInt8[ndarray, "h w"] = np.load(image_path, allow_pickle=False)
    for step in range(12):
        t_ns: int = step * 50_000_000
        for sample_ns in range(t_ns, t_ns + 50_000_000, 1_000_000):
            vio.push_imu(sample_ns, [0.0, 0.0, 0.0], [0.0, 0.0, 9.81])
        left: UInt8[ndarray, "h w"] = np.ascontiguousarray(np.roll(image, step, axis=1))
        right: UInt8[ndarray, "h w"] = np.ascontiguousarray(np.roll(image, step + 1, axis=1))
        vio.track(t_ns, [left, right])
        frame: _core.FlowFrame | None = vio.flow_frame()
        if frame is not None and len(frame.ids(0)) > 0 and len(frame.ids(1)) > 0:
            print("detected stereo features")
            return
    raise AssertionError("no stereo frontend output after twelve textured framesets")


if __name__ == "__main__":
    run_probe(sys.argv[1], Path(sys.argv[2]))
