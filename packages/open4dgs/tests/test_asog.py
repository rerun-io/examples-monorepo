"""Animated SOG round trip on a tiny synthetic clip, and the TRiGS <-> .asog parameter mapping."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("asog", reason="asog-src is in the open4dgs environments only")
pytest.importorskip("trigs", reason="open4dgs-trigs-src is in the open4dgs environments only")

FPS: float = 30.0
NUM_FRAMES: int = 30


def synthetic_params(n: int = 512, seed: int = 0) -> dict[str, torch.Tensor]:
    """Splats on a 8 x 8 x 8 grid (10 cm apart, so canonical centres identify them after the encoder's reordering)."""
    rng = np.random.default_rng(seed)
    side: int = round(n ** (1 / 3))
    grid = np.stack(np.meshgrid(*[np.arange(side)] * 3, indexing="ij"), axis=-1).reshape(-1, 3) * 0.1 - 0.35
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    p: dict[str, np.ndarray] = {
        "means": grid,
        "quats": q,
        "scales": rng.uniform(-5, -3, (n, 3)),
        "opacities": rng.normal(2, 1, n),
        "sh0": rng.normal(0, 0.5, (n, 3)),
        "shN": rng.normal(0, 0.05, (n, 15, 3)),
        "times": rng.uniform(0.2, 0.8, (n, 1)),
        "log_durations": np.log(rng.uniform(0.2, 0.5, (n, 1))),
        "anchors": grid + rng.normal(0, 0.05, (n, 3)),
        "is_static": (np.arange(n) % 16 == 0)[:, None].astype(np.float64),
        "zeta_base_r": rng.normal(0, 1.0, (n, 3)),
        "zeta_base_t": rng.normal(0, 0.2, (n, 3)),
    }
    for k in range(3):
        p[f"zeta_c{k}_r"] = rng.normal(0, 0.3, (n, 3))
        p[f"zeta_c{k}_t"] = rng.normal(0, 0.1, (n, 3))
    return {name: torch.from_numpy(value.astype(np.float32)) for name, value in p.items()}


def deform(params: dict[str, torch.Tensor], t: float, threshold: float = 0.01) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TRiGS motion at ``t``; ``threshold`` is the smooth-cull edge (TRiGS default 0.01, ``.asog`` v1 fixes 0.05)."""
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

    config = TrigsTrainingConfig(fps=FPS, temporal_active_threshold=threshold)
    means, quats, gamma = TrigsTrainer(device="cpu")._deform(params, t, config)  # pyrefly: ignore[bad-argument-type]
    return means, torch.nn.functional.normalize(quats, dim=-1), gamma.reshape(-1)


def test_nu_control_points_reproduce_the_quartic() -> None:
    from asog.motion import bezier2, bezier4

    from open4dgs.asog import nu_control_points

    rng = np.random.default_rng(1)
    omega = rng.normal(size=(3, 64, 3))
    nu_true = rng.normal(size=(3, 64, 3))
    tau = np.linspace(0, 1, 7)
    # the quartic g = nu . omega in Bernstein form, from samples (5 coefficients <- 7 samples, exact)
    basis = np.stack([[1, 4, 6, 4, 1][k] * (1 - tau) ** (4 - k) * tau**k for k in range(5)], axis=1)
    g_samples = np.stack([(bezier2(np.full(64, s), nu_true) * bezier2(np.full(64, s), omega)).sum(1) for s in tau])
    g = np.linalg.lstsq(basis, g_samples, rcond=None)[0]
    nu = nu_control_points(omega, g)
    for s in tau:
        got = (bezier2(np.full(64, s), nu) * bezier2(np.full(64, s), omega)).sum(1)
        np.testing.assert_allclose(got, bezier4(np.full(64, s), g), atol=1e-9)


def test_round_trip_positions_rotations_and_weights(tmp_path: Path) -> None:
    from asog import read as read_asog

    from open4dgs.asog import AsogSettings, decode_asog, encode_params

    src = synthetic_params()
    path = tmp_path / "clip.asog"
    stats = encode_params(src, FPS, NUM_FRAMES, path, AsogSettings(prune=0.0))
    assert stats.count == 512 and path.stat().st_size == stats.bytes
    decoded, _ = decode_asog(path, torch.device("cpu"))
    # the encoder reorders the splats: match them by canonical centre (grid spacing 10 cm >> quantisation)
    order = torch.cdist(src["means"].double(), decoded["means"].double()).argmin(dim=1)
    assert len(set(order.tolist())) == 512
    reference = read_asog(path)
    for t in (0.3, 0.5, 0.7):
        m0, q0, w0 = deform(src, t)
        m1, q1, w1 = deform(decoded, t)
        m1, q1, w1 = m1[order], q1[order], w1[order]
        visible = (w0 > 0.05) & (w1 > 0.05)
        assert int(visible.sum()) > 200
        # positions within a few quantisation steps (qp 2 mm over the life, 14-bit log positions, 8 mm anchors)
        assert float((m1 - m0)[visible].norm(dim=-1).median()) < 2e-3
        assert float((m1 - m0)[visible].norm(dim=-1).max()) < 2e-2
        # rotations: 8-bit smallest-three, so a few degrees at most
        angle = 2 * torch.acos((q0 * q1).sum(-1).abs().clamp(max=1.0))
        assert float(angle[visible].max()) < np.radians(4.0)
        # temporal weight: centre time 2^-11 s, 8-bit log sigma
        assert float((w1 - w0)[visible].abs().max()) < 0.05
        # the mapping back to TRiGS gives exactly the reference decoder's motion (whose weight cut is 0.05)
        frame = reference.evaluate(t)
        md, qd, wd = deform(decoded, t, threshold=0.05)
        shown = torch.from_numpy(frame.weights) > 0
        np.testing.assert_allclose(md[shown].double().numpy(), frame.positions[shown.numpy()], atol=1e-5)
        np.testing.assert_allclose((qd[shown].double() * torch.from_numpy(frame.rotations[shown.numpy()])).sum(-1).abs().numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(wd.double().numpy(), frame.weights, atol=1e-5)


def test_trigs_runtime_folds_the_base_twist_and_zeroes_statics() -> None:
    from open4dgs.asog import trigs_runtime

    src = synthetic_params(n=64)
    rt = trigs_runtime(src, FPS, NUM_FRAMES)
    static = src["is_static"].reshape(-1) > 0.5
    np.testing.assert_allclose(rt.omega[1][~static.numpy()], (src["zeta_base_r"] + src["zeta_c1_r"])[~static].double().numpy(), rtol=1e-6)
    assert np.all(rt.omega[:, static.numpy()] == 0) and np.all(rt.nu[:, static.numpy()] == 0)
    np.testing.assert_allclose(rt.opacity, torch.sigmoid(src["opacities"]).double().numpy(), rtol=1e-6)
    assert rt.manifest["timeline"] == {"frameRate": FPS, "totalFrames": NUM_FRAMES, "durationSeconds": NUM_FRAMES / FPS}
