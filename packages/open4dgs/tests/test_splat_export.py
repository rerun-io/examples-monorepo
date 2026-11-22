"""Splat export on a synthetic checkpoint (CPU, no catalog): quaternion order, scale activation, colours, culling."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("trigs", reason="the TRiGS trainer lives in the open4dgs envs")


def _params(n: int, *, static: bool) -> dict[str, torch.Tensor]:
    zeros3 = torch.zeros(n, 3)
    quat_wxyz = torch.tensor([[math.cos(0.3), math.sin(0.3), 0.0, 0.0]]).repeat(n, 1)  # rotation about x
    return {
        "means": torch.arange(n * 3, dtype=torch.float32).reshape(n, 3),
        "quats": quat_wxyz,
        "scales": torch.log(torch.tensor([[0.01, 0.02, 0.03]])).repeat(n, 1),
        "opacities": torch.linspace(-8.0, 4.0, n),
        "sh0": torch.zeros(n, 3),
        "shN": torch.full((n, 15, 3), 0.25),
        "times": torch.zeros(n, 1),
        "log_durations": torch.zeros(n, 1),
        "anchors": zeros3.clone(),
        "is_static": torch.full((n, 1), 1.0 if static else 0.0),
        **{f"zeta_{k}": zeros3.clone() for k in ("base_t", "base_r", "c0_t", "c0_r", "c1_t", "c1_r", "c2_t", "c2_r")},
    }


def _eval(params: dict[str, torch.Tensor], t: float, with_sh: bool = False):
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

    from open4dgs.splat_export import splats_at

    return splats_at(params, t, TrigsTrainer(device="cpu"), TrigsTrainingConfig(fps=25.0), with_sh=with_sh)


def test_wxyz_to_xyzw() -> None:
    from open4dgs.splat_export import wxyz_to_xyzw

    assert wxyz_to_xyzw(torch.tensor([[1.0, 2.0, 3.0, 4.0]])).tolist() == [[2.0, 3.0, 4.0, 1.0]]


def test_static_rows_keep_pose_scale_and_order() -> None:
    from open4dgs.splat_export import MIN_OPACITY, SH_C0

    params = _params(8, static=True)
    splats = _eval(params, 1.0, with_sh=True)
    opacity = torch.sigmoid(params["opacities"])
    keep = (opacity > MIN_OPACITY).numpy()
    assert 0 < keep.sum() < 8  # the lowest logits are culled
    np.testing.assert_allclose(splats.centers, params["means"].numpy()[keep])
    np.testing.assert_allclose(splats.scales, np.tile([[0.01, 0.02, 0.03]], (int(keep.sum()), 1)), rtol=1e-6)
    np.testing.assert_allclose(splats.quaternions, np.tile([[math.sin(0.3), 0.0, 0.0, math.cos(0.3)]], (int(keep.sum()), 1)), atol=1e-6)
    assert splats.colors[:, :3].tolist() == [[128, 128, 128]] * int(keep.sum())  # sh0 = 0 -> 0.5 grey
    np.testing.assert_array_equal(splats.colors[:, 3], (opacity.numpy()[keep] * 255.0 + 0.5).astype(np.uint8))
    assert splats.sh_rest is not None and splats.sh_rest.shape == (int(keep.sum()), 15, 3) and splats.sh_rest.dtype == np.float16
    assert SH_C0 > 0.28


def test_temporal_gamma_fades_dynamic_rows() -> None:
    params = _params(4, static=False)
    params["opacities"] = torch.full((4,), 6.0)
    params["times"] = torch.tensor([[0.0], [0.0], [2.0], [2.0]])
    params["log_durations"] = torch.full((4, 1), math.log(0.1))
    at_zero = _eval(params, 0.0)
    assert len(at_zero.centers) == 2  # rows centred at t = 2 s are 20 sigma away
    np.testing.assert_allclose(at_zero.centers, params["means"].numpy()[:2], atol=1e-6)
    assert len(_eval(params, 2.0).centers) == 2


def test_translation_motion_moves_centres() -> None:
    params = _params(2, static=False)
    params["opacities"] = torch.full((2,), 6.0)
    params["log_durations"] = torch.full((2, 1), math.log(10.0))
    params["zeta_base_t"] = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)  # constant velocity 1 m/s along x, no rotation
    splats = _eval(params, 0.5)
    np.testing.assert_allclose(splats.centers - params["means"].numpy(), [[0.5, 0.0, 0.0]] * 2, atol=1e-5)


def test_up_view_coordinates() -> None:
    import rerun as rr

    from open4dgs.splat_export import up_view_coordinates

    # OpenCV camera looking along +x with image y = -z_world: up = +z.
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    assert up_view_coordinates(pose[None]) == rr.ViewCoordinates.RIGHT_HAND_Z_UP
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)  # image y = -y_world: up = +y
    assert up_view_coordinates(pose[None]) == rr.ViewCoordinates.RIGHT_HAND_Y_UP


def test_log_splat_frames_partitions_per_frame(tmp_path) -> None:
    import rerun as rr
    from rerun.chunk import RrdReader

    from open4dgs.splat_export import log_splat_frames

    params = _params(6, static=True)
    recording = rr.RecordingStream("test-splats", recording_id="t")
    recording.save(tmp_path / "s.rrd")
    counts = log_splat_frames(recording, iter([(10, _eval(params, 0.0)), (11, _eval(params, 0.04))]), batch_frames=1)
    recording.flush()
    recording.disconnect()
    chunks = RrdReader(tmp_path / "s.rrd").stream().filter(content="/world/person/splats").to_chunks()
    frames = sorted(f for c in chunks for f in c.to_record_batch().column("frame_index").to_pylist())
    assert frames == [10, 11] and counts[0] == counts[1] > 0
