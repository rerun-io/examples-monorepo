"""The fast SOMA-X identity setup builds the same skeleton-transfer state, refit cache and fits as upstream."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from jaxtyping import Float32
from numpy import ndarray

pytest.importorskip("soma", reason="SOMA-X is installed only in the dataforge-soma-x environments")
torch = pytest.importorskip("torch")

from simplecv.ops.mano.mano_np import MANOLayerNP  # noqa: E402
from soma.fitting import pose_inversion  # noqa: E402
from soma.geometry.skeleton_transfer import SkeletonTransfer  # noqa: E402

from dataforge import soma_x_identity  # noqa: E402
from dataforge.datasets.egoexo4d_body import SMPLH_FILE  # noqa: E402
from dataforge.soma_x_fit import BodyFitter, HandFitter, ModelFit, rotvecs_to_matrices  # noqa: E402

pytestmark = pytest.mark.integration

IDENTITIES: int = 3
FRAMES: int = 16
ATOL: float = 1e-6
"""Metres (the rigs are metric): float32 round-off of a ~1 m body is ~1e-7."""
JOINT_ATOL: float = 5e-6
"""Metres: bar for joints regressed through the RBF weights (float32 solves of ill-conditioned systems, see below)."""


def _cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("SOMA-X fitting needs a CUDA GPU")


@pytest.fixture(scope="module")
def body() -> BodyFitter:
    _cuda()
    path: Path = Path.home() / "exoego-data/egoexo4d" / SMPLH_FILE
    if not path.is_file():
        pytest.skip(f"needs the SMPL-H model at {path}")
    return BodyFitter("smpl", model_path=path, gender="male", num_betas=16)


@pytest.fixture(scope="module")
def hand() -> HandFitter:
    _cuda()
    return HandFitter("right")


@pytest.fixture
def installed() -> Iterator[None]:
    soma_x_identity.install_fast_identity_setup()
    yield
    soma_x_identity.uninstall_fast_identity_setup()


def assert_same(fast: Any, upstream: Any, where: str) -> None:
    """Recursive equality: integer/bool tensors and plain values exactly, float tensors to float32 round-off."""
    if isinstance(upstream, tuple) and len(upstream) == 2 and isinstance(upstream[1], float):  # (tensor, own tolerance)
        torch.testing.assert_close(fast[0], upstream[0], atol=upstream[1], rtol=0.0, msg=lambda m: f"{where}: {m}")
    elif isinstance(upstream, torch.Tensor):
        assert isinstance(fast, torch.Tensor), where
        assert fast.layout == upstream.layout, where
        if upstream.layout == torch.sparse_csr:
            assert torch.equal(fast.crow_indices(), upstream.crow_indices()), where
            assert torch.equal(fast.col_indices(), upstream.col_indices()), where
            assert_same(fast.values(), upstream.values(), where + ".values")
            return
        assert fast.shape == upstream.shape and fast.dtype == upstream.dtype and fast.device == upstream.device, where
        if upstream.is_floating_point():
            torch.testing.assert_close(fast, upstream, atol=ATOL, rtol=1e-5, msg=lambda m: f"{where}: {m}")
        else:
            assert torch.equal(fast, upstream), where
    elif isinstance(upstream, dict):
        assert isinstance(fast, dict) and fast.keys() == upstream.keys(), where
        for key in upstream:
            assert_same(fast[key], upstream[key], f"{where}[{key!r}]")
    elif isinstance(upstream, list | tuple):
        assert type(fast) is type(upstream) and len(fast) == len(upstream), where
        for index, (a, b) in enumerate(zip(fast, upstream, strict=True)):
            assert_same(a, b, f"{where}[{index}]")
    else:
        assert fast == upstream, where


def transfer_state(transfer: SkeletonTransfer) -> dict[str, Any]:
    """Every tensor the skeleton transfer holds after ``update_bind``; the RBF weights as the joints they regress.

    The per-joint RBF systems (linear kernel, up to ~1,000 control points) are ill-conditioned: upstream's float32
    weights differ from a float64 solve by ~3e-4, as much as the batched solve's do. What they are for is the
    regressed joint position, and that agrees to float32 round-off (~1 um on the rest shape).
    """
    state: dict[str, Any] = {name: value for name, value in transfer._buffers.items() if value is not None}
    sparse: Any = transfer.sparse_rbf_matrix
    state["sparse_structure"] = (sparse.crow_indices(), sparse.col_indices())
    state["regressed_joints"] = (sparse @ transfer.bind_shape, JOINT_ATOL)
    regressors: list[Any] = transfer.joint_pos_regressors or []
    state["regressors"] = [
        None
        if rbf is None
        else (rbf.source_control_points, (rbf.interpolate(rbf.source_control_points, transfer.bind_world_transforms[i : i + 1, :3, 3]), JOINT_ATOL))
        for i, rbf in enumerate(regressors)
    ]
    return state


def identities(count: int, size: int, seed: int) -> list[Float32[ndarray, "1 c"]]:
    rng: np.random.Generator = np.random.default_rng(seed)
    return [rng.normal(0.0, 1.0, (1, size)).astype(np.float32) for _ in range(count)]


def check_identity_setup(fitter: BodyFitter | HandFitter, betas: int) -> None:
    inversion: pose_inversion.PoseInversion = fitter.inversion
    inversion.prepare_identity(torch.zeros(1, betas, device=fitter.device))  # builds the structural caches
    assert inversion._skel_transfer is not None
    for identity in identities(IDENTITIES, betas, seed=betas):
        inversion.soma.prepare_identity(torch.as_tensor(identity, device=fitter.device))
        view: Any = inversion.soma.public_rig_view() if hasattr(inversion.soma, "public_rig_view") else None
        bind_world: Any = (view.bind_transforms_world if view is not None else inversion.soma._cached_bind_transforms_world)[0]
        bind_shape: Any = inversion.soma._cached_rest_shape[0]
        parents: Any = view.joint_parent_ids if view is not None else inversion.soma.joint_parent_ids
        weights: Any = view.skinning_weights if view is not None else inversion.soma.skinning_weights
        t_pose: Any = view.t_pose_world if view is not None else inversion.soma.t_pose_world
        args: tuple[Any, ...] = (inversion.joint_names, parents, bind_world, bind_shape, weights, t_pose)

        soma_x_identity.UPSTREAM_UPDATE_BIND(inversion._skel_transfer, bind_world, bind_shape)
        upstream_transfer: dict[str, Any] = transfer_state(inversion._skel_transfer)
        upstream_cache: dict[str, Any] = soma_x_identity.UPSTREAM_REFIT_CACHE(*args, root_idx=inversion._root_joint_idx)
        for _ in range(2):  # the second pass runs on the captured structure
            soma_x_identity.fast_update_bind(inversion._skel_transfer, bind_world, bind_shape)
            assert_same(transfer_state(inversion._skel_transfer), upstream_transfer, "skeleton_transfer")
            fast_cache: dict[str, Any] = soma_x_identity.fast_precompute_refit_cache(*args, root_idx=inversion._root_joint_idx)
            assert_same(fast_cache, upstream_cache, "refit_cache")


def test_body_identity_setup_matches_upstream(body: BodyFitter) -> None:
    check_identity_setup(body, 16)


def test_hand_identity_setup_matches_upstream(hand: HandFitter) -> None:
    check_identity_setup(hand, 10)


def posed_targets(fitter: BodyFitter | HandFitter, identity: Float32[ndarray, "1 c"]) -> tuple[Float32[ndarray, "n v 3"], Float32[ndarray, "n 3"]]:
    """Source meshes of ``identity`` in small random poses and their centres: the fitting layer's own low-LOD body
    (centred on the origin), or a posed MANO hand (centred on its wrist)."""
    rng: np.random.Generator = np.random.default_rng(11)
    if fitter.kind == "hand":
        pose: Float32[ndarray, "n 48"] = rng.normal(0.0, 0.2, (FRAMES, 48)).astype(np.float32)
        translation: Float32[ndarray, "n 3"] = np.tile(np.array([0.05, 0.0, 0.4], dtype=np.float32), (FRAMES, 1))
        mano: MANOLayerNP = MANOLayerNP(side="right", betas=identity[0], use_pca=False)
        meshes: tuple[Float32[ndarray, "n 778 3"], Float32[ndarray, "n 21 3"]] = mano(pose, translation)
        return meshes[0], meshes[1][:, 0]
    rotations: Float32[ndarray, "n j 3"] = rng.normal(0.0, 0.15, (FRAMES, len(fitter.inversion.joint_names) - 1, 3)).astype(np.float32)
    layer: Any = fitter.fit_layer
    with torch.no_grad():
        layer.prepare_identity(torch.as_tensor(identity, device=fitter.device).expand(FRAMES, -1))
        posed: Any = layer.pose(
            torch.as_tensor(rotvecs_to_matrices(rotations), device=fitter.device),
            transl=torch.zeros(FRAMES, 3, device=fitter.device),
            pose2rot=False,
            absolute_pose=True,
            apply_correctives=False,
        )
    return posed["vertices"].float().cpu().numpy(), np.zeros((FRAMES, 3), dtype=np.float32)


def fit_runs(fitter: BodyFitter | HandFitter, betas: int) -> list[list[ModelFit]]:
    """Fits of two identities (one batch each): twice with upstream's identity setup, then with the fast one."""
    ids: list[Float32[ndarray, "1 c"]] = identities(2, betas, seed=100 + betas)
    targets: list[tuple[Float32[ndarray, "n v 3"], Float32[ndarray, "n 3"]]] = [posed_targets(fitter, identity) for identity in ids]
    settings: dict[str, int] = {"body_iters": 2 if fitter.kind == "body" else 0}
    runs: list[list[ModelFit]] = []
    for install in (False, False, True):
        if install:
            soma_x_identity.install_fast_identity_setup()
        fitter._identity_key = b""
        runs.append([fitter._fit(identity, target[0], target[1], **settings) for identity, target in zip(ids, targets, strict=True)])
    soma_x_identity.uninstall_fast_identity_setup()
    return runs


def largest_difference(a: list[ModelFit], b: list[ModelFit], field: str) -> float:
    return max(float(np.abs(getattr(x, field) - getattr(y, field)).max()) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("kind", ["body", "hand"])
def test_fits_match_upstream(kind: str, request: pytest.FixtureRequest) -> None:
    """The fit itself is not bit-reproducible on the GPU (upstream twice differs by ~0.1 mm on these hard random poses):
    the fast setup stays within twice upstream's own run-to-run spread, plus float32 slack."""
    fitter: BodyFitter | HandFitter = request.getfixturevalue(kind)
    upstream, again, fast = fit_runs(fitter, 16 if kind == "body" else 10)
    for field, slack in (("error_mm", 1e-3), ("vertices", 1e-6), ("root_translation", 1e-6)):
        spread: float = largest_difference(upstream, again, field)
        assert largest_difference(fast, upstream, field) <= 2.0 * spread + slack, field
    for x, y in zip(fast, upstream, strict=True):
        assert abs(float(np.median(x.error_mm)) - float(np.median(y.error_mm))) < 0.05  # mm


def test_install_is_idempotent_and_reversible(installed: None) -> None:
    soma_x_identity.install_fast_identity_setup()
    assert SkeletonTransfer.update_bind is soma_x_identity.fast_update_bind
    assert pose_inversion._precompute_refit_cache is soma_x_identity.fast_precompute_refit_cache
    soma_x_identity.uninstall_fast_identity_setup()
    assert SkeletonTransfer.update_bind is soma_x_identity.UPSTREAM_UPDATE_BIND
    assert pose_inversion._precompute_refit_cache is soma_x_identity.UPSTREAM_REFIT_CACHE
