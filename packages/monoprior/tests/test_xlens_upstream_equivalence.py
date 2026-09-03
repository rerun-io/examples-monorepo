"""Keep the vendored X-Lens inference code aligned with pristine upstream fixtures."""

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
from jaxtyping import Float32, UInt8

from monopriors.third_party.xlens.inference import preprocess as owned_preprocess

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "xlens"


def _load_upstream_preprocess() -> ModuleType:
    """Load the pristine preprocessing fixture under a synthetic package name."""
    path: Path = REFERENCE_DIR / "upstream_preprocess.py"
    spec: importlib.machinery.ModuleSpec | None = importlib.util.spec_from_file_location("xlens_upstream.inference.preprocess", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream fixture {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_preprocessing_matches_upstream() -> None:
    """The as-is vendor produces bit-identical image, ray, and pose preprocessing."""
    upstream: ModuleType = _load_upstream_preprocess()
    image0: UInt8[np.ndarray, "28 42 3"] = np.arange(28 * 42 * 3, dtype=np.uint8).reshape(28, 42, 3)
    image1: UInt8[np.ndarray, "28 42 3"] = np.flip(image0, axis=1).copy()
    K_33: Float32[np.ndarray, "3 3"] = np.array([[40.0, 0.0, 21.0], [0.0, 41.0, 14.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rays0: Float32[np.ndarray, "28 42 3"] = upstream.pinhole_d_cam(K_33, 28, 42)
    rays1: Float32[np.ndarray, "28 42 3"] = owned_preprocess.pinhole_d_cam(K_33, 28, 42)
    np.testing.assert_array_equal(rays0, rays1)
    cam_T_ref: Float32[np.ndarray, "2 4 4"] = np.stack((np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)))
    cam_T_ref[1, 0, 3] = 0.1

    upstream_batch: dict = upstream.assemble_batch([image0, image1], [rays0, rays0], [1, 1], c2w=cam_T_ref, device=torch.device("cpu"))
    owned_batch: dict = owned_preprocess.assemble_batch([image0, image1], [rays1, rays1], [1, 1], c2w=cam_T_ref, device=torch.device("cpu"))
    assert upstream_batch.keys() == owned_batch.keys()
    for key in upstream_batch:
        assert torch.equal(upstream_batch[key], owned_batch[key])
