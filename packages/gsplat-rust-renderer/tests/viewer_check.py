"""Subprocess harness for the native viewer integration and golden pixel tests."""
from __future__ import annotations

import math
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro
from jaxtyping import Float32, Float64, UInt8
from PIL import Image
from rerun.encodings import EntityPathLike
from rerun.experimental import ViewerClient
from serde.json import to_json

from gsplat_rust_renderer.gaussians3d import compute_visualizer, log_ply
from gsplat_rust_renderer.scene_io import NerfFrame, NerfTransforms


@dataclass(frozen=True)
class Config:
    binaries: Path
    ply: Path
    out: Path
    mode: Literal["relog", "pair", "views", "portable", "pair-white", "depth"] = "relog"


def capture(viewer: ViewerClient, view: rrb.Spatial3DView, path: Path, allow_blank: bool = False, ready: Callable[[UInt8[np.ndarray, "h w 3"]], bool] | None = None) -> UInt8[np.ndarray, "h w 3"]:
    """Poll fresh screenshots until visible pixels settle, with a bounded deadline."""
    recording = rr.get_global_data_recording()
    if recording is not None:
        recording.flush(timeout_sec=30.0)
    deadline = time.monotonic() + 60.0
    previous = None
    while time.monotonic() < deadline:
        if not any(v.view_id.replace("-", "") == str(view.id).replace("-", "") for v in viewer.viewer_state().views):
            time.sleep(0.1)
            continue
        path.unlink(missing_ok=True)
        try:
            viewer.save_screenshot(str(path), view_id=view.id)
        except ConnectionError as error:
            if "not found" not in str(error):
                raise
            time.sleep(0.1)
            continue
        pixels = None
        screenshot_deadline = min(deadline, time.monotonic() + 2.0)
        while time.monotonic() < screenshot_deadline:
            try:
                with Image.open(path) as screenshot:
                    pixels = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
                break
            except (FileNotFoundError, OSError):
                time.sleep(0.1)
        if pixels is not None:
            visible = ready(pixels) if ready is not None else allow_blank or int(pixels.max()) - int(pixels.min()) > 20
            if visible and previous is not None and np.array_equal(previous, pixels):
                return pixels
            previous = pixels
        time.sleep(0.1)
    raise RuntimeError(f"Viewer pixels did not settle: {path}")


def yellow_fraction(pixels: UInt8[np.ndarray, "h w 3"]) -> float:
    """Return the Lego-yellow fraction of UInt8[ndarray, "h w 3"] pixels."""
    return float(np.mean((pixels[:, :, 0] > 100) & (pixels[:, :, 1] > 60) & (pixels[:, :, 2] < 80)))


def has_lego_color(pixels: UInt8[np.ndarray, "h w 3"]) -> bool:
    """Exclude stable loading indicators from a successful Lego capture."""
    return yellow_fraction(pixels) > 0.01


def splat_view(background: tuple[int, int, int], eye: rrb.EyeControls3D | None, overrides: dict[EntityPathLike, rrb.Visualizer] | None = None, contents: list[str] | str = "$origin/**", name: str | None = None) -> rrb.Spatial3DView:
    """Build a solid-background splat view, retaining automatic eye and content defaults."""
    return rrb.Spatial3DView(origin="/", name=name, contents=contents, overrides=overrides,
        background=rrb.Background(color=background, kind="SolidColor"), line_grid=False, eye_controls=eye)


def spawn_viewer(binary: str | None) -> ViewerClient:
    """Start an owned headless viewer on a free port; the client is a context manager."""
    return ViewerClient.spawn(headless=True, port=free_port(), executable_path=binary, hide_welcome_screen=True)


def free_port() -> int:
    """Reserve a free loopback port for an owned viewer process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def check_portable_recording(config: Config) -> None:
    """Prove automatic, explicit native, and explicit compute choices in both viewers."""
    assert rr.__version__ == "0.38.1"
    for choice in (None, "GaussianSplats3D", "ComputeGaussianSplats3D"):
        label = {None: "portable", "GaussianSplats3D": "explicit-native", "ComputeGaussianSplats3D": "explicit-compute"}[choice]
        view = splat_view((0, 0, 0),
            rrb.EyeControls3D(position=(2.0, -2.0, 1.2), look_target=(0.0, 0.0, 0.25), eye_up=(0.0, 0.0, 1.0)),
            {"world/splats": rrb.Visualizer(choice)} if choice is not None else {}, ["world/splats"], name=label)
        recording_path = config.out / f"{label}.rrd"
        rr.init(label, strict=True)
        rr.save(recording_path)
        rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log_ply(config.ply)
        recording = rr.get_global_data_recording()
        assert recording is not None
        recording.flush(timeout_sec=30.0)
        rr.disconnect()
        for name, executable in (("compute", str(config.binaries / "gsplat-rust-renderer")), ("stock", None)):
            print(f"PORTABLE_VIEWER {label} {name}", flush=True)
            with spawn_viewer(executable) as viewer:
                viewer.open_url(str(recording_path.resolve()))
                unsupported = choice == "ComputeGaussianSplats3D" and name == "stock"
                pixels = capture(viewer, view, config.out / f"{label}-{name}.png", allow_blank=unsupported, ready=None if unsupported else has_lego_color)
                yellow: float = yellow_fraction(pixels)
                reports = [report.summary for item in viewer.viewer_state().views for report in item.reports]
                print(f"PORTABLE_PIXELS {label} {name} yellow={yellow:.6f} reports={reports}", flush=True)
                if unsupported:
                    assert yellow < 0.001, "Stock must not silently reinterpret the explicit compute choice"
                    # Stock 0.38.1 silently skips unknown instruction types.
                    print("UNSUPPORTED_STOCK_VISUALIZER: ComputeGaussianSplats3D requires the custom viewer", flush=True)
                else:
                    assert yellow > 0.01, f"{label} {name} did not draw splats"


def check_depth_and_framing(config: Config) -> None:
    """Exercise a white scene with opaque meshes, image planes, and frustums."""
    with spawn_viewer(str(config.binaries / "gsplat-rust-renderer")) as viewer:
        rr.init("depth-and-framing", strict=True)
        rr.connect_grpc(viewer.url)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        centers = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        rr.log("world/splats", rr.GaussianSplats3D(centers=centers, scales=[1.0, 1.0, 1.0], colors=[255, 0, 0, 255]), static=True)
        framed = splat_view((255, 255, 255), None, contents=["world/splats"])
        rr.send_blueprint(rrb.Blueprint(framed, collapse_panels=True))
        pixels = capture(viewer, framed, config.out / "framing.png")
        red = (pixels[:, :, 0] > 150) & (pixels[:, :, 1] < 100)
        assert float(red.mean()) > 0.0005 and red[pixels.shape[0] // 2, pixels.shape[1] // 2], "Default framing ignored the distant splat AABB"
        rr.log("world/splats", rr.GaussianSplats3D(centers=[[0, 0, 0]], scales=[0.7, 0.7, 0.05], colors=[255, 0, 0, 255]), static=True)
        # Large green plane behind, small blue plane in front.
        for name, x0, x1, z, color in [("rear", -2.0, 2.0, -1.0, [0, 255, 0]), ("front", -0.7, -0.2, 1.0, [0, 0, 255])]:
            rr.log(f"world/{name}", rr.Mesh3D(vertex_positions=[[x0, -0.3, z], [x1, -0.3, z], [x1, 0.3, z], [x0, 0.3, z]],
                                             triangle_indices=[[0, 1, 2], [0, 2, 3]], albedo_factor=color), static=True)
        # Same Pinhole + image-plane primitives as log-scene, placed behind the splat.
        rr.log("world/camera", rr.Transform3D(translation=[0, 0, -2]), static=True)
        rr.log("world/camera/pinhole", rr.Pinhole(focal_length=80, width=64, height=64, image_plane_distance=0.5), static=True)
        rr.log("world/camera/pinhole/image", rr.Image(np.full((64, 64, 3), [255, 0, 255], dtype=np.uint8)), static=True)
        view = splat_view((255, 255, 255),
            rrb.EyeControls3D(position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)))
        rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
        pixels = capture(viewer, view, config.out / "depth.png")
        h, w = pixels.shape[:2]
        center = pixels[h // 2, w // 2]
        assert center[0] > 240 and center[1] < 30 and center[2] < 30, f"Rear content punched through opaque splats: {center}"
        blue = (pixels[:, :, 2] > 200) & (pixels[:, :, 0] < 30) & (pixels[:, :, 1] < 30)
        assert float(blue.mean()) > 0.002, "Front mesh did not occlude splats"
        green = (pixels[:, :, 1] > 200) & (pixels[:, :, 0] < 100)
        assert green.any(), "Rear mesh should remain visible outside the splat footprint"
        # One invalid entity must not blank the valid cloud.
        rr.log("world/bad", rr.GaussianSplats3D(centers=[[0, 0, 0]], scales=[1, 1, 1]), rr.Transform3D(scale=0), static=True)
        pixels = capture(viewer, view, config.out / "bad-entity.png")
        assert pixels[h // 2, w // 2, 0] > 240
        rr.disconnect()


def check_views(viewer: ViewerClient, config: Config) -> None:
    """Check shared uploads and independent blueprint modes."""
    rr.log("world/splats", rr.GaussianSplats3D(centers=[[0.0, 0.0, 0.0]], scales=[0.0001] * 3, colors=[255, 0, 0, 255], spherical_harmonics_degree=0), static=True)
    views: list[rrb.Spatial3DView] = [splat_view((0, 0, 0),
        rrb.EyeControls3D(position=(0.0, 0.0, 2.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)),
        {"world/splats": compute_visualizer(mode)}, ["world/splats"]) for mode in ("default", "mip")]
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True))
    peaks: list[int] = []
    for index, mode_view in enumerate(views):
        pixels: UInt8[np.ndarray, "h w 3"] = capture(viewer, mode_view, config.out / f"mode-{index}.png", allow_blank=index == 1)
        h, w = pixels.shape[:2]
        peaks.append(int(pixels[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20, 0].max()))
    print(f"default/mip central red peaks: {peaks}", flush=True)
    # One u8 alpha step becomes ~13 display levels when blended in linear light.
    assert peaks[0] > 80 and peaks[1] < 16, "Blueprint mode override did not reach the shared core"


def check_relog(viewer: ViewerClient, config: Config) -> None:
    """Check replacement after optional-field and count updates."""
    view = splat_view((0, 0, 0),
        rrb.EyeControls3D(position=(0.0, 0.0, 5.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, 1.0, 0.0)), contents=["world/splats"])
    rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
    for channel, side in enumerate([8, 8, 16]):
        axis: Float32[np.ndarray, "side"] = np.linspace(-1.2, 1.2, side, dtype=np.float32)
        gx, gy = np.meshgrid(axis, axis)
        centers: Float32[np.ndarray, "n 3"] = np.stack([gx.ravel(), gy.ravel(), np.zeros(side * side, dtype=np.float32)], axis=1)
        rgba: list[int] = [13, 13, 13, 242]
        rgba[channel] = 255
        if channel == 1:
            # Optional attributes must invalidate uploads even with unchanged center row IDs.
            rr.log("world/splats", rr.GaussianSplats3D.from_fields(colors=rgba), static=True)
        else:
            rr.log("world/splats", rr.GaussianSplats3D(centers=centers, scales=[0.18] * 3, colors=rgba, spherical_harmonics_degree=0), static=True)
        image = capture(viewer, view, config.out / f"relog-{'rgb'[channel]}-{side * side}.png")
        h, w = image.shape[:2]
        mean: Float64[np.ndarray, "3"] = image[h // 4:3 * h // 4, w // 4:3 * w // 4].mean(axis=(0, 1)) / 255.0
        print(f"relog {side * side}: {mean}", flush=True)
        assert int(np.argmax(mean)) == channel and mean[channel] > 0.2, "Stale or blank relog"
        assert np.delete(mean, channel).max() < 0.1, "Previous cloud color remains"


def look_at_c2w(
    position: Float64[np.ndarray, "3"],
    target: Float64[np.ndarray, "3"],
    world_up: Float64[np.ndarray, "3"],
) -> Float64[np.ndarray, "4 4"]:
    """Build an OpenGL camera-to-world matrix from Float64[ndarray, "3"] eye, target, and up vectors."""
    forward: Float64[np.ndarray, "3"] = target - position
    forward = forward / np.linalg.norm(forward)
    right: Float64[np.ndarray, "3"] = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    true_up: Float64[np.ndarray, "3"] = np.cross(right, forward)
    c2w: Float64[np.ndarray, "4 4"] = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = position
    return c2w


def check_pair(config: Config, image: UInt8[np.ndarray, "h w 3"], position: Float64[np.ndarray, "3"], target: Float64[np.ndarray, "3"], up: Float64[np.ndarray, "3"]) -> None:
    """Compare a captured viewport with the matching standalone camera."""
    # At 1 px/point, Rerun 0.38.1 crops 2 pixels at top/left and 3 at bottom/right.
    h, w = image.shape[:2]
    h, w = h + 5, w + 5
    fov_y: float = float(np.float32(55.0) * np.float32(math.tau) / np.float32(360.0))
    camera: NerfTransforms = NerfTransforms(2.0 * math.atan(math.tan(fov_y / 2.0) * w / h), [NerfFrame("fixed", look_at_c2w(position, target, up))])
    camera_path: Path = config.out / "camera.json"
    camera_path.write_text(to_json(camera))
    reference_path: Path = config.out / "standalone.png"
    subprocess.run(
        [str(config.binaries / "gsplat"), "render", "--ply", str(config.ply), "--camera", str(camera_path), "--width", str(w), "--height", str(h), "--background", "1,1,1" if config.mode == "pair-white" else "0,0,0", "--output", str(reference_path)],
        check=True, timeout=120,
    )
    reference: UInt8[np.ndarray, "h w 3"] = np.asarray(Image.open(reference_path).convert("RGB"), dtype=np.uint8)[2:-3, 2:-3]
    error: Float64[np.ndarray, "h w 3"] = (image.astype(np.float64) - reference.astype(np.float64)) / 255.0
    psnr: float = -10.0 * math.log10(float(np.mean(error**2)))
    print(f"viewer/standalone: {w}x{h}, {psnr:.6f} dB", flush=True)
    # Linear-light viewer blending differs from the standalone display-space background.
    floor = 40.0 if config.mode == "pair-white" else 39.0
    assert psnr >= floor, f"Viewer/standalone PSNR below {floor} dB: {psnr}"


def main(config: Config) -> None:
    """Assert pixels through a real viewer; the calling test sets a process timeout."""
    config.out.mkdir(parents=True, exist_ok=True)
    if check := {"depth": check_depth_and_framing, "portable": check_portable_recording}.get(config.mode):
        check(config)
        return
    position: Float64[np.ndarray, "3"] = np.array([2.0, -2.0, 1.2])
    target: Float64[np.ndarray, "3"] = np.array([0.0, 0.0, 0.25])
    up: Float64[np.ndarray, "3"] = np.array([0.0, 0.0, 1.0])
    with spawn_viewer(str(config.binaries / "gsplat-rust-renderer")) as viewer:
        rr.init("native-viewer-check", strict=True)
        rr.connect_grpc(viewer.url)
        try:
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            log_ply(config.ply)
            view: rrb.Spatial3DView = splat_view((255, 255, 255) if config.mode == "pair-white" else (0, 0, 0),
                rrb.EyeControls3D(position=position, look_target=target, eye_up=up), contents=["world/splats"])
            rr.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
            image: UInt8[np.ndarray, "h w 3"] = capture(viewer, view, config.out / "lego.png", ready=has_lego_color)
            assert float(np.mean(np.max(image, axis=2) > 20)) > 0.05, "Lego render is blank"
            if config.mode == "views":
                check_views(viewer, config)
            elif config.mode == "relog":
                check_relog(viewer, config)
            else:
                check_pair(config, image, position, target, up)
        finally:
            rr.disconnect()


if __name__ == "__main__":
    main(tyro.cli(Config))
