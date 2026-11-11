"""CUDA capture must include BA, preserve state, and avoid CPU tensor reads."""

import pytest
import torch

pytest.importorskip("vipe_mojo")
from vipe_mojo.slam_ext import _PoseSystem


@pytest.mark.integration
def test_device_pose_solve_capture() -> None:
    """A positive definite reduced system and a failed factorization keep CPU semantics."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    ii = torch.tensor([0, 1, 1, 2])
    jj = torch.tensor([1, 0, 2, 1])
    system = _PoseSystem(2, False, ii, jj, 1, None)
    system.prepare_device(torch.device("cuda:0"))
    reduced = torch.zeros(system.host_size)
    # Positive diagonals in each edge's lower-triangle block; arbitrary right-hand sides.
    for edge in range(4):
        for row in range(12):
            reduced[edge * 105 + row * (row + 1) // 2 + row] = 2.0
        reduced[edge * 105 + 78:edge * 105 + 90] = torch.arange(12).float() / 13.0
    reference, _ = system.solve(reduced, 1e-3, 0.1, 1e-6, 1e-6)
    static = reduced.cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        system.solve_device(static, 1e-3, 0.1, 1e-6, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, _ = system.solve_device(static, 1e-3, 0.1, 1e-6, 1e-6)
    graph.replay()
    torch.testing.assert_close(result.cpu(), reference, rtol=1e-6, atol=1e-7)
    static.fill_(float("nan"))
    graph.replay()
    assert torch.count_nonzero(result).item() == 0


@pytest.mark.integration
def test_whole_step_matches_eager_and_updates_inputs() -> None:
    """Capture/replay must consume changed sensor depths and preserve every state tensor."""
    import copy
    from types import SimpleNamespace
    from typing import cast

    from omegaconf import OmegaConf
    from vipe.slam.components.buffer import GraphBuffer
    from vipe.slam.components.factor_graph import FactorGraph
    from vipe.slam.components.sparse_tracks import SparseTracks
    from vipe.slam.networks.droid_net import DroidNet, UpdateModule
    from vipe.utils.cameras import CameraType

    from metrictrail.calibration_graphs import CalibrationGraph, calibration_cuda_graphs

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(1234)
    device = torch.device("cuda:0")
    buffer = GraphBuffer(128, 192, 1, 8, 1.0, None,
                         OmegaConf.create({"fused": True, "dense_disp_alpha": 0.05}),
                         cast(SparseTracks, SimpleNamespace(enabled=False)), CameraType.PINHOLE, device)
    buffer.n_frames = 4
    buffer.intrinsics[0] = torch.tensor([150.0, 150.0, 96.0, 64.0], device=device)
    buffer.poses[:4, 0] = torch.arange(4, device=device) * 0.01
    buffer.fmaps.normal_(0.0, 0.1)
    buffer.nets.normal_(0.0, 0.1)
    buffer.inps.normal_(0.0, 0.1)
    net = SimpleNamespace(update=UpdateModule().cuda().eval())
    reference_buffer = copy.deepcopy(buffer)
    with torch.no_grad():
        reference = FactorGraph(cast(DroidNet, net), reference_buffer, device, 48, True, False)
        accelerated = CalibrationGraph(net, buffer, device, 48, True, False)
        reference.add_neighborhood_factors(0, 4, r=2)
        with calibration_cuda_graphs():
            accelerated.add_neighborhood_factors(0, 4, r=2)
        for iteration in range(4):
            reference_buffer.disps_sens[:4] = 0.5 + 0.1 * iteration
            buffer.disps_sens[:4] = 0.5 + 0.1 * iteration
            reference.update(t0=1, t1=4)
            with calibration_cuda_graphs():
                accelerated.update(t0=1, t1=4)
            for name in ("poses", "disps"):
                torch.testing.assert_close(getattr(buffer, name), getattr(reference_buffer, name), rtol=3e-4, atol=3e-5)
            for name in ("f_net", "target", "weight", "damping"):
                torch.testing.assert_close(getattr(accelerated, name), getattr(reference, name), rtol=3e-3, atol=3e-4)
        assert accelerated.capture_count == 1
        assert accelerated.replay_count == 3
        reference.rm_factors(reference.ii == 0, store=True)
        accelerated.rm_factors(accelerated.sources == 0, store=True)
        for name in ("ii", "jj", "ii_inac", "jj_inac"):
            torch.testing.assert_close(getattr(accelerated, name), getattr(reference, name), rtol=0.0, atol=0.0)
        reference.update(t0=1, t1=4, use_inactive=True)
        with calibration_cuda_graphs():
            accelerated.update(t0=1, t1=4, use_inactive=True)
        torch.testing.assert_close(buffer.poses, reference_buffer.poses, rtol=3e-4, atol=3e-5)
