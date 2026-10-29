import math
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("open3d", reason="surface fusion needs an importable open3d (hexels env)", exc_type=ImportError)

from hexels_bench.apis.surface_run import frame_parity, surface_out_dir, total_parity  # noqa: E402
from hexels_bench.cameras import ContractCamera  # noqa: E402
from hexels_bench.layout import Roots  # noqa: E402
from hexels_bench.surface import (  # noqa: E402
    FusionView,
    TsdfSettings,
    alpha_mask,
    fuse_tsdf,
    intrinsic,
    normals_from_depth,
    post_process_mesh,
    scene_cameras,
    surface_depth,
    tsdf_settings,
)
from hexels_bench.surface_frames import SurfaceFrame  # noqa: E402
from hexels_bench.surface_records import DepthParity  # noqa: E402


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


VIEWS: list[tuple[str, tuple[float, float, float, float], tuple[float, float, float]]] = [
    ("b.png", (0.9, 0.1, -0.3, 0.2), (0.5, -1.0, 2.0)),
    ("a.png", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
]


def write_colmap_text(sparse: Path) -> None:
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("# id model w h params\n1 PINHOLE 101 60 80.0 70.0 50.5 30.0\n")
    lines: list[str] = ["# images"]
    for index, (name, q, t) in enumerate(VIEWS):
        norm: float = math.sqrt(sum(v * v for v in q))
        lines.append(f"{index + 1} {' '.join(str(v / norm) for v in q)} {' '.join(map(str, t))} 1 {name}")
        lines.append("")
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")


def write_colmap_binary(sparse: Path) -> None:
    sparse.mkdir(parents=True)
    (sparse / "cameras.bin").write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, 101, 60) + struct.pack("<4d", 80.0, 70.0, 50.5, 30.0))
    body: bytes = struct.pack("<Q", len(VIEWS))
    for index, (name, q, t) in enumerate(VIEWS):
        norm: float = math.sqrt(sum(v * v for v in q))
        body += struct.pack("<i7di", index + 1, *(v / norm for v in q), *t, 1) + name.encode() + b"\x00"
        body += struct.pack("<Q", 2) + struct.pack("<ddq", 1.0, 2.0, -1) * 2
    (sparse / "images.bin").write_bytes(body)


@pytest.mark.parametrize("writer", [write_colmap_text, write_colmap_binary])
def test_scene_cameras_follow_the_2dgs_loader(tmp_path: Path, writer) -> None:
    writer(tmp_path / "sparse" / "0")
    (tmp_path / "images").mkdir()
    for name, _, _ in VIEWS:
        Image.new("RGB", (101, 60)).save(tmp_path / "images" / name)
    cameras: list[ContractCamera] = scene_cameras(tmp_path, "images", 2)
    assert [camera.name for camera in cameras] == ["a", "b"]
    assert (cameras[0].width, cameras[0].height) == (round(101 / 2), 30)  # Python rounds 50.5 to 50
    assert cameras[0].fov_x == pytest.approx(2 * math.atan(101 / 160))
    assert cameras[0].fov_y == pytest.approx(2 * math.atan(60 / 140))
    q, t = VIEWS[0][1], np.array(VIEWS[0][2])
    rotation: np.ndarray = rotation_from_quaternion(np.asarray(q) / np.linalg.norm(q))
    world_view: np.ndarray = np.asarray(cameras[1].world_view_transform)
    np.testing.assert_allclose(world_view[:3, :3], rotation.T, atol=1e-6)  # row-vector form: [x 1] @ M
    np.testing.assert_allclose(world_view[3, :3], t, atol=1e-6)
    np.testing.assert_allclose(cameras[1].camera_center, -rotation.T @ t, atol=1e-5)


def test_scene_cameras_reject_resolutions_outside_the_fixed_factors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution"):
        scene_cameras(tmp_path, "images", 3)


def test_alpha_mask_keeps_half_opaque_pixels_and_ignores_rgb_images(tmp_path: Path) -> None:
    rgba: np.ndarray = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = [[0, 127, 128], [255, 255, 0]]
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "a.png")
    Image.new("RGB", (3, 2)).save(tmp_path / "b.png")
    np.testing.assert_array_equal(alpha_mask(tmp_path / "a.png", 3, 2), [[False, False, True], [True, True, False]])
    assert alpha_mask(tmp_path / "b.png", 3, 2) is None


def test_surface_depth_applies_mode_alpha_cut_and_mask() -> None:
    median: np.ndarray = np.full((1, 3), 2.0, dtype=np.float32)
    expected: np.ndarray = np.full((1, 3), 3.0, dtype=np.float32)
    alpha: np.ndarray = np.array([[0.2, 0.6, 0.9]], dtype=np.float32)
    mask: np.ndarray = np.array([[True, True, False]])
    np.testing.assert_array_equal(surface_depth(median, expected, alpha, "median", 0.0, None), [[2.0, 2.0, 2.0]])
    np.testing.assert_array_equal(surface_depth(median, expected, alpha, "expected", 0.5, mask), [[0.0, 3.0, 0.0]])


def test_tsdf_settings_follow_the_two_upstream_groups() -> None:
    assert tsdf_settings("dtu", "scan24") == TsdfSettings(voxel_size=0.004, sdf_trunc=0.016, depth_trunc=3.0)
    assert tsdf_settings("tnt", "Truck") == tsdf_settings("dtu", "scan24")
    assert tsdf_settings("tnt", "Courthouse") == TsdfSettings(voxel_size=0.006, sdf_trunc=0.024, depth_trunc=4.5)
    assert tsdf_settings("mipnerf360", "garden").depth_trunc == 6.0


def frontal_camera(name: str, width: int = 64, height: int = 48) -> ContractCamera:
    return ContractCamera(
        name=name, width=width, height=height, fov_x=1.0, fov_y=2 * math.atan(math.tan(0.5) * height / width),
        world_view_transform=np.eye(4), camera_center=np.zeros(3),
    )  # fmt: skip


def test_intrinsic_puts_the_principal_point_at_the_pixel_grid_centre() -> None:
    matrix: np.ndarray = np.asarray(intrinsic(frontal_camera("a")).intrinsic_matrix)
    assert matrix[0, 2] == pytest.approx(31.5) and matrix[1, 2] == pytest.approx(23.5)
    assert matrix[0, 0] == pytest.approx(32 / math.tan(0.5))


def test_fusing_a_frontal_plane_gives_a_flat_mesh_at_its_depth() -> None:
    camera: ContractCamera = frontal_camera("a")
    depth: np.ndarray = np.full((48, 64), 1.0, dtype=np.float32)
    rgb: np.ndarray = np.full((48, 64, 3), 200, dtype=np.uint8)
    settings = TsdfSettings(voxel_size=0.01, sdf_trunc=0.04, depth_trunc=3.0)
    result = fuse_tsdf([FusionView(camera=camera, rgb=rgb, depth=depth)], settings)
    assert result.views == 1 and result.valid_depth_fraction == 1.0
    vertices: np.ndarray = np.asarray(post_process_mesh(result.mesh).vertices)
    assert len(vertices) > 100
    assert np.abs(vertices[:, 2] - 1.0).max() < settings.voxel_size


def test_post_process_keeps_only_the_largest_cluster() -> None:
    import open3d as o3d

    big = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    small = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=10).translate((5.0, 0.0, 0.0))
    mesh = post_process_mesh(big + small)
    assert len(mesh.triangles) == len(big.triangles)
    assert np.asarray(mesh.vertices)[:, 0].max() < 1.5


def test_normals_from_depth_face_the_camera() -> None:
    camera: ContractCamera = frontal_camera("a", 16, 12)
    normal: np.ndarray = normals_from_depth(np.full((12, 16), 2.0, dtype=np.float32), camera)
    np.testing.assert_allclose(normal[1:-1, 1:-1], np.broadcast_to([0.0, 0.0, -1.0], (10, 14, 3)), atol=1e-6)
    assert not normal[0].any() and not normal[:, -1].any()


def frame(name: str, seed: int) -> SurfaceFrame:
    rng: np.random.Generator = np.random.default_rng(seed)
    return SurfaceFrame(
        name=name,
        rgb=rng.integers(0, 256, size=(3, 4, 3), dtype=np.uint8),
        median_depth=rng.random((3, 4), dtype=np.float32),
        expected_depth=rng.random((3, 4), dtype=np.float32),
        alpha=rng.random((3, 4), dtype=np.float32),
    )


def test_parity_totals_weight_views_by_their_pixels() -> None:
    a: SurfaceFrame = frame("a", 0)
    b = SurfaceFrame(name="a", rgb=a.rgb, median_depth=a.median_depth + np.float32(0.01), expected_depth=a.expected_depth, alpha=a.alpha)
    single: DepthParity = frame_parity(b, a)
    assert single.both_hit == 12 and single.hit_mismatch == 0
    assert single.median_abs_mean == pytest.approx(0.01, abs=1e-6) and single.median_over_tolerance == 12
    total: DepthParity = total_parity([single, frame_parity(a, a)])
    assert total.views == 2 and total.pixels == 24 and total.median_abs_mean == pytest.approx(0.005, abs=1e-6)


def test_surface_outputs_go_to_a_mode_directory_only_for_an_override() -> None:
    roots = Roots(data=Path("/d"), runs=Path("/r"))
    assert surface_out_dir(roots, "dtu", "scan24", "hexels-v3-7k", None) == Path("/r/dtu/scan24/hexels-v3-7k/surface")
    assert surface_out_dir(roots, "dtu", "scan24", "hexels-v3-7k", "median") == Path("/r/dtu/scan24/hexels-v3-7k/surface")
    assert surface_out_dir(roots, "dtu", "scan24", "brush-7k", "median") == Path("/r/dtu/scan24/brush-7k/surface-median")


def test_rrd_frames_read_the_cli_layout_in_view_order(tmp_path: Path) -> None:
    rr = pytest.importorskip("rerun")
    from hexels_bench.surface_frames import rrd_frames

    cameras: list[ContractCamera] = [frontal_camera("v0", 4, 3), frontal_camera("v1", 4, 3)]
    sent: list[SurfaceFrame] = [frame("v0", 0), frame("v1", 1)]
    recording = rr.RecordingStream("hexels-surface")
    recording.save(tmp_path / "surface.rrd")
    for view in (1, 0):  # out of order: the reader must still yield view 0 first
        recording.set_time("view", sequence=view)
        recording.log("world/camera/m/rgb", rr.Image(sent[view].rgb))
        recording.log("world/camera/m/median_depth", rr.DepthImage(sent[view].median_depth, meter=1.0))
        recording.log("world/camera/m/expected_depth", rr.DepthImage(sent[view].expected_depth, meter=1.0))
        recording.log("world/camera/m/alpha", rr.Image(sent[view].alpha, color_model="L"))
    recording.disconnect()
    received: list[SurfaceFrame] = list(rrd_frames(tmp_path / "surface.rrd", "m", cameras))
    assert [item.name for item in received] == ["v0", "v1"]
    for a, b in zip(sent, received, strict=True):
        for field in ("rgb", "median_depth", "expected_depth", "alpha"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    with pytest.raises(ValueError, match="complete views of"):
        list(rrd_frames(tmp_path / "surface.rrd", "m", [*cameras, frontal_camera("v2", 4, 3)]))


@pytest.mark.integration
@pytest.mark.parametrize(("dataset", "scene"), [("dtu", "scan24"), ("tnt", "Barn"), ("mipnerf360", "garden")])
def test_scene_cameras_equal_upstream_camera_contracts(dataset, scene) -> None:
    """All registered views of a real scene equal the HF camera contract's training + test records."""
    from hexels_bench.cameras import CameraContract
    from hexels_bench.layout import read_json

    roots = Roots()
    contract_path: Path = roots.contract(dataset, scene)
    if not contract_path.is_file() or not roots.scene_dir(dataset, scene).is_dir():
        pytest.skip(f"needs {contract_path} and {roots.scene_dir(dataset, scene)}")
    contract = read_json(CameraContract, contract_path)
    reference: list[ContractCamera] = sorted(contract.geometry.training + contract.geometry.test, key=lambda camera: camera.name)
    cameras: list[ContractCamera] = scene_cameras(roots.scene_dir(dataset, scene), contract.settings.images, contract.settings.resolution)
    assert [(c.name, c.width, c.height, c.fov_x, c.fov_y) for c in cameras] == [(c.name, c.width, c.height, c.fov_x, c.fov_y) for c in reference]
    for ours, theirs in zip(cameras, reference, strict=True):
        np.testing.assert_allclose(ours.world_view_transform, theirs.world_view_transform, atol=2e-6)
        np.testing.assert_allclose(ours.camera_center, theirs.camera_center, atol=2e-6)


def test_post_process_keeps_everything_when_asked_for_more_clusters_than_exist() -> None:
    import open3d as o3d

    big = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    small = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10).translate((5.0, 0.0, 0.0))
    total: int = len(big.triangles) + len(small.triangles)
    assert len(post_process_mesh(big + small, clusters_to_keep=50).triangles) == total


def test_every_surface_method_has_one_table_row() -> None:
    from typing import get_args

    from hexels_bench.apis.surface_run import SURFACE_METHODS, SurfaceMethod, surface_model_dir

    roots = Roots(data=Path("/d"), runs=Path("/r"))
    assert list(SURFACE_METHODS) == list(get_args(SurfaceMethod))
    assert all(SURFACE_METHODS[spec.ply_from].depth_source == "wgsl" for spec in SURFACE_METHODS.values())
    assert surface_model_dir(roots, "dtu", "scan24", "hexels-v3-surf-30k-pub-cuda") == roots.published_surface_model("dtu", "scan24")
    assert surface_model_dir(roots, "dtu", "scan24", "hexels-v3-surf-30k") == Path("/r/dtu/scan24/hexels-v3-surf-30k/upstream")
    assert surface_model_dir(roots, "dtu", "scan24", "brush-7k") is None


def test_tnt_evaluator_runs_through_pixi_from_the_workspace_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The T&T evaluator needs the tnt-eval env: `pixi run --frozen` installs it from the lock or says why it cannot."""
    from hexels_bench.apis import surface_eval
    from hexels_bench.layout import WORKSPACE

    calls: list[tuple[list[str], Path | None]] = []

    def fake_run(command: list[str], **kwargs) -> None:
        calls.append((command, kwargs.get("cwd")))
        (tmp_path / "tnt_eval" / "Barn.prf_tau_plotstr.txt").write_text("0.5 0.4 0.45 0.01\n")

    monkeypatch.setattr(surface_eval.subprocess, "run", fake_run)
    (tmp_path / "tnt_eval").mkdir()
    config = surface_eval.Config(dataset="tnt", scenes=["Barn"], roots=Roots(data=tmp_path / "data", runs=tmp_path / "runs"))
    metrics = surface_eval.evaluate_tnt(config, "Barn", "m", tmp_path / "fuse_post.ply", tmp_path / "tnt_eval", tmp_path / "log.txt")
    command, cwd = calls[0]
    assert command[:6] == ["pixi", "run", "-e", "tnt-eval", "--frozen", "python"] and command[6].endswith("scripts/eval_tnt/run.py")
    assert cwd == WORKSPACE
    assert (metrics.precision, metrics.recall, metrics.f1, metrics.tau) == (50.0, 40.0, 45.0, 0.01)
