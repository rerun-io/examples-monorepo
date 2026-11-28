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

    from open4dgs.asog import AsogSettings, decode_asog, encode_asog

    src = synthetic_params()
    path = tmp_path / "clip.asog"
    data, stats = encode_asog(src, FPS, NUM_FRAMES, AsogSettings(preset="upstream", prune=0.0))
    path.write_bytes(data)
    assert stats.count == 512 and path.stat().st_size == stats.bytes
    decoded, _ = decode_asog(path.read_bytes(), torch.device("cpu"))
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
        assert float((m1 - m0)[visible].norm(dim=-1).max()) < 2.5e-2  # axis-anchored rows drop a rotation of < 8 qp steps
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


def test_presets_band_safe_default_and_upstream() -> None:
    from open4dgs.asog import AsogOptions, AsogSettings

    assert AsogSettings().options() == AsogOptions(qp=0.5, qa=2.0, prune=0.0, mbits=16, tbits=11, lsbits=8)
    assert AsogSettings(preset="upstream").options() == AsogOptions(qp=2.0, qa=8.0, prune=0.3, mbits=14, tbits=11, lsbits=8)
    # ``small`` is the name for upstream's preset (web streaming); ``upstream`` stays as an alias
    assert AsogSettings(preset="small").options() == AsogSettings(preset="upstream").options()
    # an override changes only its own option
    assert AsogSettings(preset="upstream", prune=0.0).options() == AsogOptions(qp=2.0, qa=8.0, prune=0.0, mbits=14, tbits=11, lsbits=8)


def test_upstream_preset_matches_the_encoder_defaults() -> None:
    import dataclasses
    import inspect

    from asog.encode import encode_runtime

    from open4dgs.asog import AsogSettings

    defaults = {name: p.default for name, p in inspect.signature(encode_runtime).parameters.items() if p.default is not inspect.Parameter.empty}
    options = dataclasses.asdict(AsogSettings(preset="upstream").options())
    assert {name: defaults[name] for name in options} == options


def test_decode_keeps_trigs_timing_and_a_stock_player_cuts_at_0_05(tmp_path: Path) -> None:
    """K2 (a): ``decode_asog`` hands back mu / sigma, not a cut; the cut is the renderer's. The trainer path renders the
    decoded clip with TRiGS's 0.01, a stock ``.asog`` player with the format's fixed 0.05: never more opaque, at most
    0.0404 less (at gamma = 0.05), and blank once gamma < 0.05 (|dt| > 2.45 sigma instead of 3.03 sigma)."""
    from asog import read as read_asog

    from open4dgs.asog import AsogSettings, decode_asog, encode_asog

    src = synthetic_params()
    path = tmp_path / "clip.asog"
    path.write_bytes(encode_asog(src, FPS, NUM_FRAMES, AsogSettings())[0])
    decoded, _ = decode_asog(path.read_bytes(), torch.device("cpu"))
    reference = read_asog(path)
    np.testing.assert_allclose(decoded["log_durations"].reshape(-1).double().numpy(), np.log(reference.sigma), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(decoded["times"].reshape(-1).double().numpy(), reference.mu, atol=1e-6)
    dynamic = ~torch.from_numpy(reference.static)
    blanked = 0
    for t in np.linspace(0.0, (NUM_FRAMES - 1) / FPS, 15):
        trainer_w = deform(decoded, float(t), threshold=0.01)[2].double()
        player_w = torch.from_numpy(reference.evaluate(float(t)).weights)
        diff = (trainer_w - player_w)[dynamic]
        assert float(diff.min()) > -1e-6 and float(diff.max()) < 0.0405
        blanked += int(((player_w == 0) & (trainer_w > 0))[dynamic].sum())
    assert blanked > 0


def test_translation_needs_a_rotation_axis_and_the_axis_anchor_keeps_it(tmp_path: Path) -> None:
    """K2 (b): ``.asog`` carries translation only as ``g = nu . omega``, along the quantised rotation axis.

    A pure-translation row (omega = 0; TRiGS's gauge fix then moves it by the full nu) comes back without its
    translation. A row with a slow rotation (|omega| = 1e-3 rad/s, above TRiGS's 1e-8 threshold on |omega|^2) whose
    omega would round to zero steps keeps it, because ``axis_anchors`` moves its anchor onto its axis; with the
    original anchor the encoder zeroes its omega and drops the translation too.
    """
    from asog import read as read_asog
    from asog.encode import encode_runtime

    from open4dgs.asog import AsogSettings, decode_asog, encode_asog, trigs_runtime

    src = synthetic_params()
    pure, slow = torch.arange(1, 512, 16), torch.arange(2, 512, 16)  # dynamic rows (row % 16 == 0 is static)
    direction = torch.nn.functional.normalize(torch.randn(len(slow), 3, generator=torch.Generator().manual_seed(3)), dim=-1)
    for k in ("base", "c0", "c1", "c2"):
        src[f"zeta_{k}_r"][pure] = 0.0
        src[f"zeta_{k}_t"][pure] = 0.0 if k != "base" else torch.tensor([0.3, 0.0, 0.0])
        src[f"zeta_{k}_r"][slow] = 0.0 if k != "base" else 1e-3 * direction
        src[f"zeta_{k}_t"][slow] = 0.0 if k != "base" else 0.3 * direction  # 0.3 m/s along the axis
    options = AsogSettings().options()
    raw = tmp_path / "original-anchor.asog"
    encode_runtime(trigs_runtime(src, FPS, NUM_FRAMES), raw, qp=options.qp, qa=options.qa, prune=options.prune, mbits=options.mbits)
    path = tmp_path / "clip.asog"
    path.write_bytes(encode_asog(src, FPS, NUM_FRAMES, AsogSettings())[0])
    decoded, _ = decode_asog(path.read_bytes(), torch.device("cpu"))
    order = torch.cdist(src["means"].double(), decoded["means"].double()).argmin(dim=1)
    assert np.all(np.abs(read_asog(raw).omega[:, order[slow].numpy()]).sum(axis=(0, 2)) == 0)  # original anchor: omega -> 0
    assert np.all(np.abs(read_asog(path).omega[:, order[pure].numpy()]).sum(axis=(0, 2)) == 0)
    for rows, kept in ((pure, False), (slow, True)):
        t = float(src["times"][rows].mean()) + 0.1
        m0 = deform(src, t)[0][rows]
        m1 = deform(decoded, t)[0][order[rows]]
        dt = t - src["times"][rows].reshape(-1)
        np.testing.assert_allclose((m0 - src["means"][rows]).norm(dim=-1).numpy(), (0.3 * dt.abs()).numpy(), rtol=1e-3)  # TRiGS: full 0.3 m/s
        moved = (m1 - decoded["means"][order[rows]]).norm(dim=-1)
        if kept:
            assert float((m1 - m0).norm(dim=-1).max()) < 1e-2  # direction within about 1 / AXIS_ANCHOR_TARGET rad
        else:
            assert float(moved.max()) < 1e-6  # not at all


def test_threaded_encode_writes_the_serial_file(tmp_path: Path) -> None:
    from asog.encode import encode_runtime

    from open4dgs.asog import AsogSettings, axis_anchors, encode_asog, trigs_runtime

    src = synthetic_params(4096, seed=3)
    del src["shN"]  # the SH palette k-means on CUDA is not bit-reproducible between runs
    options = AsogSettings().options()
    runtime = trigs_runtime(src, FPS, NUM_FRAMES)
    runtime.anchors = axis_anchors(runtime, options.qp)
    serial = tmp_path / "serial.asog"
    encode_runtime(
        runtime, serial, qp=options.qp, qa=options.qa, prune=options.prune, mbits=options.mbits, tbits=options.tbits, lsbits=options.lsbits
    )
    data, stats = encode_asog(src, FPS, NUM_FRAMES, AsogSettings())
    assert data == serial.read_bytes()
    assert stats.bytes == len(data)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sorted_codebook_matches_upstream(seed: int) -> None:
    from asog.quantize import codebook1d

    from open4dgs.asog import sorted_codebook1d

    rng = np.random.default_rng(seed)
    # repeated values and a few far outliers: ties at the cell edges and empty cells
    x = np.concatenate([rng.normal(-4, 0.5, (3000, 3)).round(2).ravel(), rng.normal(0, 3, 50), np.full(200, 0.25)])
    want_cb, want_idx = codebook1d(x)
    got_cb, got_idx = sorted_codebook1d(x)
    np.testing.assert_array_equal(got_cb, want_cb)
    np.testing.assert_array_equal(got_idx, want_idx)
