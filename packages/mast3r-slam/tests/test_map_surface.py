"""fuse_keyframes: keyframes at their Sim3 poses become one front-faced TSDF mesh, fused on a latest-wins thread."""

import threading

import lietorch_mojo as lietorch
import numpy as np
import pytest
import rerun as rr
from jaxtyping import Float32
from numpy import ndarray

pytest.importorskip("tsdf_rs", reason="the map surface needs the tsdf-rs extension from the mast3r-slam envs")

from mast3r_slam import map_surface  # noqa: E402
from mast3r_slam.map_surface import (  # noqa: E402
    KeyframeDepth,
    LatestWinsWorker,
    LiveSurface,
    SurfaceRequest,
    SurfaceSettings,
    integrate_keyframe,
    new_volume,
)

H: int = 48
W: int = 64


def _plane_keyframe(depth: float) -> KeyframeDepth:
    """A fronto-parallel wall at ``depth`` (before scale) filling a pinhole with f=W."""
    k_33: Float32[ndarray, "3 3"] = np.array([[W, 0.0, W / 2], [0.0, W, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    return KeyframeDepth(
        depth_hw=np.full((H, W), depth, dtype=np.float32),
        k_33=k_33,
        rgb_hw3=np.full((H, W, 3), 200, dtype=np.uint8),
    )


def _sim3_data(scale: float) -> Float32[ndarray, "8"]:
    data: Float32[ndarray, "8"] = lietorch.Sim3.Identity(1, device="cpu").data.squeeze().numpy().astype(np.float32).copy()
    data[7] = scale
    return data


def test_mesh_lands_at_the_scaled_depth() -> None:
    """A Sim3 scale of 2 puts a wall seen at Z=1 in its camera frame at Z=2 in the world."""
    volume = new_volume(SurfaceSettings(voxel_size=0.02))
    for _ in range(2):
        integrate_keyframe(volume, _plane_keyframe(1.0), _sim3_data(2.0))
    vertices: Float32[ndarray, "v 3"] = volume.extract_mesh()[0]
    np.testing.assert_allclose(np.median(vertices[:, 2]), 2.0, atol=0.03)


def _live_logs(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    logged: list[tuple[str, str]] = []
    monkeypatch.setattr(map_surface.rr, "log", lambda path, entity, *_a, **_k: logged.append((str(path), type(entity).__name__)))
    monkeypatch.setattr(map_surface.rr, "set_time", lambda *_a, **_k: None)
    return logged


def test_live_surface_logs_changed_blocks_then_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """New keyframes log only changed blocks; the re-fuse clears the subtree and logs every block, front-faced."""
    logged: list[tuple[str, str]] = _live_logs(monkeypatch)
    meshes: list[rr.Mesh3D] = []
    real_front_faced = map_surface.front_faced_mesh

    def _capture_mesh(*args: object) -> rr.Mesh3D:
        meshes.append(real_front_faced(*args))  # pyrefly: ignore
        return meshes[-1]

    monkeypatch.setattr(map_surface, "front_faced_mesh", _capture_mesh)
    live: LiveSurface = LiveSurface("world/mesh", None, ("frame", "video_time"), refuse_every=3, settings=SurfaceSettings(voxel_size=0.02))
    wall: KeyframeDepth = _plane_keyframe(1.0)
    far_wall: KeyframeDepth = _plane_keyframe(2.0)

    live.handle(SurfaceRequest(keyframes=[(0, wall, _sim3_data(1.0))], frame_idx=0, timestamp_ns=None))
    first: list[tuple[str, str]] = list(logged)
    assert first and all(path.startswith("world/mesh/") and kind == "Mesh3D" for path, kind in first)

    logged.clear()
    live.handle(SurfaceRequest(keyframes=[(0, wall, _sim3_data(1.0)), (1, wall, _sim3_data(1.0))], frame_idx=1, timestamp_ns=None))
    assert logged and set(logged) <= set(first), "the same wall again only touches blocks it already had"

    logged.clear()
    keyframes = [(0, wall, _sim3_data(1.0)), (1, wall, _sim3_data(1.0)), (2, far_wall, _sim3_data(1.0))]
    live.handle(SurfaceRequest(keyframes=keyframes, frame_idx=2, timestamp_ns=None))
    assert logged[0] == ("world/mesh", "Clear"), "a re-fuse clears the live blocks first"
    assert {path for path, _ in logged[1:]} > {path for path, _ in first}, "then logs every block, the new wall included"
    for mesh in meshes:
        assert mesh.face_rendering is not None
        assert mesh.face_rendering.pa_array.to_pylist() == [rr.components.MeshFaceRendering.Front.value]


def test_final_request_always_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run's last request re-fuses at the final poses whatever the cadence."""
    logged: list[tuple[str, str]] = _live_logs(monkeypatch)
    live: LiveSurface = LiveSurface("world/mesh", None, ("frame", "video_time"), refuse_every=100)
    live.handle(SurfaceRequest(keyframes=[(0, _plane_keyframe(1.0), _sim3_data(1.0))], frame_idx=0, timestamp_ns=None, final=True))
    assert logged[0] == ("world/mesh", "Clear")


def test_latest_wins_worker_coalesces_pending_requests_and_runs_the_last_on_close() -> None:
    """Requests that arrive while the worker is busy collapse to the newest; close() still runs a pending one."""
    handled: list[int] = []
    started: threading.Event = threading.Event()
    release: threading.Event = threading.Event()

    def _handle(request: int) -> None:
        if request == 1:
            started.set()
            assert release.wait(timeout=5.0)
        handled.append(request)

    worker: LatestWinsWorker[int] = LatestWinsWorker(_handle, name="test-latest-wins")
    worker.submit(1)
    assert started.wait(timeout=5.0)
    for request in (2, 3, 4):
        worker.submit(request)
    release.set()
    worker.submit(5)  # pending while 1 (or 4) runs: the final request must still run before close returns
    worker.close(timeout=5.0)
    assert handled[0] == 1
    assert handled[-1] == 5
    assert 2 not in handled and 3 not in handled


def test_latest_wins_worker_re_raises_a_crash_on_close() -> None:
    """A request that raises ends the worker; close() raises it on the caller's thread instead of dropping it."""

    def _handle(_request: int) -> None:
        raise ValueError("surface failed")

    worker: LatestWinsWorker[int] = LatestWinsWorker(_handle, name="test-latest-wins-crash")
    worker.submit(1)
    with pytest.raises(ValueError, match="surface failed"):
        worker.close(timeout=5.0)
