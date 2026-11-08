"""Tracking's pose solve on the Mojo kernels (`mojo_backend.track_pose_rays`) on synthetic matches."""

import pytest

torch = pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")
pytest.importorskip("mast3r_slam.mojo_backend._mast3r_mojo", reason="build the Mojo kernels first (pixi run -e mast3r-slam-dev _mast3r-slam-build-mojo-kernels)")

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def _problem(n: int) -> tuple:
    """Keyframe points, the frame's view of them under a known relative Sim3, and that Sim3."""
    import lietorch_mojo as lietorch

    generator = torch.Generator(device="cpu").manual_seed(0)
    keyframe_points = torch.rand(n, 3, generator=generator) * torch.tensor([2.0, 2.0, 3.0]) + torch.tensor([-1.0, -1.0, 1.0])
    tangent = torch.tensor([[0.05, -0.02, 0.03, 0.02, -0.04, 0.01, 0.03]])
    camk_sim3_camf = lietorch.Sim3.exp(tangent.cuda())
    frame_points = camk_sim3_camf.inv().act(keyframe_points.cuda())
    return keyframe_points.cuda(), frame_points, camk_sim3_camf


def test_track_pose_rays_recovers_the_relative_pose() -> None:
    import lietorch_mojo as lietorch

    from mast3r_slam.mojo_backend import track_pose_rays

    n = 4096
    keyframe_points, frame_points, camk_sim3_camf = _problem(n)
    twc = torch.cat([lietorch.Sim3.Identity(1, device="cuda").data, lietorch.Sim3.Identity(1, device="cuda").data])
    iterations = track_pose_rays(
        twc,
        torch.stack([keyframe_points, frame_points]),
        torch.ones(1, n, 1, dtype=torch.bool, device="cuda"),
        torch.full((1, n, 1), 2.0, device="cuda"),
        0.003,
        10.0,
        50,
        1e-6,
    )
    assert iterations < 50
    assert torch.equal(twc[0], lietorch.Sim3.Identity(1, device="cuda").data[0]), "the keyframe pose is fixed"
    error = (lietorch.Sim3(twc[1:2]).inv() * camk_sim3_camf).log().abs().max().item()
    assert error < 1e-4


def test_track_pose_rays_raises_without_valid_matches() -> None:
    import lietorch_mojo as lietorch

    from mast3r_slam.mojo_backend import track_pose_rays

    n = 1024
    keyframe_points, frame_points, _ = _problem(n)
    twc = torch.cat([lietorch.Sim3.Identity(1, device="cuda").data, lietorch.Sim3.Identity(1, device="cuda").data])
    with pytest.raises(torch.linalg.LinAlgError):
        track_pose_rays(
            twc,
            torch.stack([keyframe_points, frame_points]),
            torch.zeros(1, n, 1, dtype=torch.bool, device="cuda"),
            torch.full((1, n, 1), 2.0, device="cuda"),
            0.003,
            10.0,
            50,
            1e-6,
        )
