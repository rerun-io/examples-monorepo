"""SOMA-X hand fits of MANO hands: the XLO hand lands on the MANO surface within the trust bar."""

import os
from pathlib import Path

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

pytest.importorskip("soma", reason="SOMA-X is installed only in the dataforge-soma-x environments")
torch = pytest.importorskip("torch")

from simplecv.ops.mano.mano_np import MANOLayerNP  # noqa: E402
from simplecv.ops.smplx.smplx_torch import SmplxLayerTorch  # noqa: E402

from dataforge import soma_x  # noqa: E402
from dataforge.soma_x_fit import BODY_MAX_MEDIAN_ERROR_MM, HAND_MAX_MEDIAN_ERROR_MM, BodyFit, BodyFitter, HandFit, HandFitter  # noqa: E402

pytestmark = pytest.mark.integration

FRAMES: int = 24


def mano_hands(side: str) -> tuple[Float32[ndarray, "c"], Float32[ndarray, "n 778 3"], Float32[ndarray, "n 21 3"]]:
    """A MANO hand closing its fingers while it moves and turns, 0.4 m in front of the origin."""
    rng: np.random.Generator = np.random.default_rng(7)
    betas: Float32[ndarray, "c"] = rng.normal(0.0, 0.8, 10).astype(np.float32)
    ramp: Float32[ndarray, "n 1"] = np.linspace(0.0, 1.0, FRAMES, dtype=np.float32)[:, None]
    pose: Float32[ndarray, "n 48"] = np.zeros((FRAMES, 48), dtype=np.float32)
    pose[:, :3] = ramp * np.array([0.3, -0.6, 0.2], dtype=np.float32)
    pose[:, 3:] = ramp * rng.normal(0.0, 0.25, 45).astype(np.float32)
    translation: Float32[ndarray, "n 3"] = (
        np.array([0.05, 0.0, 0.4], dtype=np.float32) + ramp * np.array([0.1, 0.05, 0.0], dtype=np.float32)
    ).astype(np.float32)
    layer: MANOLayerNP = MANOLayerNP(side=side, betas=betas, use_pca=False)  # type: ignore[arg-type]
    result: tuple[Float32[ndarray, "n 778 3"], Float32[ndarray, "n 21 3"]] = layer(pose, translation)
    return betas, result[0], result[1]


@pytest.fixture(scope="module", params=["left", "right"])
def fitted(request: pytest.FixtureRequest) -> tuple[HandFitter, HandFit, Float32[ndarray, "n 778 3"]]:
    if not torch.cuda.is_available():
        pytest.skip("SOMA-X fitting needs a CUDA GPU")
    side: str = request.param
    betas, vertices, joints = mano_hands(side)
    fitter: HandFitter = HandFitter(side)  # type: ignore[arg-type]
    return fitter, fitter.fit(betas, vertices, joints[:, 0]), vertices


def test_the_fit_meets_the_hand_bar_on_every_frame(fitted: tuple[HandFitter, HandFit, Float32[ndarray, "n 778 3"]]) -> None:
    fit: HandFit = fitted[1]
    assert fit.error_mm.shape == (FRAMES,)
    assert float(fit.error_mm.max()) < HAND_MAX_MEDIAN_ERROR_MM


def test_the_xlo_hand_sits_on_the_mano_surface_in_world_coordinates(fitted: tuple[HandFitter, HandFit, Float32[ndarray, "n 778 3"]]) -> None:
    fitter, fit, mano = fitted
    assert fit.vertices.shape == (FRAMES, 134, 3)
    assert fit.joints.shape == (FRAMES, 25, 3)
    nearest_mm: Float32[ndarray, "n v"] = np.sqrt(((fit.vertices[:, :, None] - mano[:, None]) ** 2).sum(-1)).min(-1) * 1000.0
    assert float(np.median(nearest_mm)) < 4.0
    assert len(fitter.topology.links) == 24
    assert fitter.topology.faces.max() < 134


def test_the_logged_parameters_regenerate_the_logged_mesh(fitted: tuple[HandFitter, HandFit, Float32[ndarray, "n 778 3"]]) -> None:
    fitter, fit, _ = fitted
    again: Float32[ndarray, "n 134 3"] = fitter.pose(fit.identity_coeffs, fit.rotations, fit.root_translation)
    np.testing.assert_allclose(again, fit.vertices, atol=1e-4)
    assert fitter.model().identity_model_type == "mano"
    assert soma_x.LOD == "xlo"


def smpl_model_root() -> Path:
    """Body-model root holding ``smpl/SMPL_NEUTRAL.pkl`` (``DATAFORGE_BODY_MODEL_ROOT``)."""
    root: Path = Path(os.environ.get("DATAFORGE_BODY_MODEL_ROOT", Path.home() / "exoego-data/harmony4d/body_models"))
    if not (root / "smpl/SMPL_NEUTRAL.pkl").is_file():
        pytest.skip(f"needs the official neutral SMPL at {root}/smpl/SMPL_NEUTRAL.pkl (set DATAFORGE_BODY_MODEL_ROOT)")
    return root


@pytest.fixture(scope="module")
def body_fitted() -> tuple[BodyFitter, BodyFit, Float32[ndarray, "n 6890 3"]]:
    if not torch.cuda.is_available():
        pytest.skip("SOMA-X fitting needs a CUDA GPU")
    root: Path = smpl_model_root()
    rng: np.random.Generator = np.random.default_rng(3)
    betas: Float32[ndarray, "c"] = rng.normal(0.0, 1.0, 10).astype(np.float32)
    ramp: Float32[ndarray, "n 1"] = np.linspace(0.0, 1.0, FRAMES, dtype=np.float32)[:, None]
    pose: Float32[ndarray, "n 72"] = np.zeros((FRAMES, 72), dtype=np.float32)
    pose[:, 3:] = ramp * rng.normal(0.0, 0.3, 69).astype(np.float32)
    pose[:, :3] = ramp * np.array([0.0, 1.2, 0.0], dtype=np.float32)
    translation: Float32[ndarray, "n 3"] = (np.array([1.0, 0.2, 3.0], dtype=np.float32) + ramp * np.array([0.5, 0.0, 0.0], dtype=np.float32)).astype(
        np.float32
    )
    layer: SmplxLayerTorch = SmplxLayerTorch(betas=betas, model_type="smpl", gender="neutral", model_root_dir=root)
    vertices: Float32[ndarray, "n 6890 3"] = layer.forward_batched(pose, translation).vertices
    fitter: BodyFitter = BodyFitter("smpl", model_path=root / "smpl/SMPL_NEUTRAL.pkl")
    return fitter, fitter.fit(betas, vertices), vertices


def test_a_smpl_body_fit_meets_the_body_bar(body_fitted: tuple[BodyFitter, BodyFit, Float32[ndarray, "n 6890 3"]]) -> None:
    fitter, fit, smpl = body_fitted
    assert float(fit.error_mm.max()) < BODY_MAX_MEDIAN_ERROR_MM
    assert fit.vertices.shape == (FRAMES, 612, 3)
    assert fit.joints.shape[1] == len(fitter.topology.parents)
    nearest_mm: Float32[ndarray, "n v"] = np.sqrt(((fit.vertices[:, :, None] - smpl[:, None]) ** 2).sum(-1)).min(-1) * 1000.0
    assert float(np.median(nearest_mm)) < 15.0
    np.testing.assert_allclose(fitter.pose(fit.identity_coeffs, fit.rotations, fit.root_translation), fit.vertices, atol=1e-4)
