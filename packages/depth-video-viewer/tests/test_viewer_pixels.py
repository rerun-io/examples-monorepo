"""rerun-canonical draws a `DepthVideoStream` exactly as the PNG `EncodedDepthImage` of the same 10-bit codes.

Two recordings hold one depth sequence under one pinhole, with the same meter, range, colormap and layout: one as AV1
video (this package's extension), one as 16-bit PNGs (stock Rerun). Each view of the video recording must equal the
same view of the PNG recording pixel for pixel, in 2D and as a 3D cloud, at a frame in each GOP. One 3D view adds
gsplat splats, so both extensions draw together in one view.
"""

import time
from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from canonical_viewer.pixels import capture, spawn_viewer
from jaxtyping import Float32, Int64, UInt8, UInt16

from depth_video_viewer.apis.comparison import png_u16
from depth_video_viewer.encode import encode_av1, to_codes
from depth_video_viewer.log import DepthVideoProperties, log_depth_video

pytestmark = pytest.mark.golden

PINHOLE = "/world/cam/pinhole"
DEPTH = f"{PINHOLE}/depth"
BOUNDS = f"{PINHOLE}/image_bounds"
SPLATS = "/world/splats"
TIMELINE = "video_time"
WIDTH, HEIGHT, FRAMES, GOP = 64, 48, 14, 10
FRAME_NS = 100_000_000
SHOWN_FRAMES = (3, 12)
"""One frame in each GOP: the second needs a seek to the next keyframe."""
EYE = rrb.EyeControls3D(position=(0.0, -0.6, -0.9), look_target=(0.0, 0.0, 1.0), eye_up=(0.0, -1.0, 0.0))


def depth_sequence() -> UInt16[np.ndarray, "n h w"]:
    """A tilted plane (600-1500 mm) with a disc that moves right, and a hole of no depth (0) in one corner."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    frames: list[UInt16[np.ndarray, "h w"]] = []
    for frame in range(FRAMES):
        depth: Float32[np.ndarray, "h w"] = (600.0 + 800.0 * x / (WIDTH - 1) + 100.0 * y / (HEIGHT - 1)).astype(np.float32)
        depth[(x - 8 - 3 * frame) ** 2 + (y - HEIGHT // 2) ** 2 < 8**2] = 500.0
        depth[:8, :8] = 0.0
        frames.append(depth.astype(np.uint16))
    return np.stack(frames)


def write_scene(path: Path, as_video: bool) -> dict[str, rrb.View]:
    """Save the depth sequence as video or as PNGs, with splats and the same layout either way; return its views by name."""
    codes, unit_mm = to_codes(depth_sequence())
    times_ns: Int64[np.ndarray, "n"] = np.arange(FRAMES, dtype=np.int64) * FRAME_NS
    meter: float = 1000.0 / unit_mm
    code_range: tuple[float, float] = (0.0, 1500.0 / unit_mm)
    views: dict[str, rrb.View] = {
        "2d": rrb.Spatial2DView(origin=PINHOLE, visual_bounds=rrb.VisualBounds2D(x_range=[0.0, float(WIDTH)], y_range=[0.0, float(HEIGHT)])),
        "cloud": rrb.Spatial3DView(origin="/world", contents=["/world/**", f"- {SPLATS}", f"- {BOUNDS}"], eye_controls=EYE, background=[20, 20, 24]),
        "cloud+splats": rrb.Spatial3DView(origin="/world", contents=["/world/**", f"- {BOUNDS}"], eye_controls=EYE, background=[20, 20, 24]),
    }
    recording = rr.RecordingStream("depth-video-canonical-pixels", recording_id=f"depth-video-canonical-pixels-{as_video}")
    recording.save(path)
    recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views.values()), collapse_panels=True))
    # Explicit frustum and axis sizes: Rerun's defaults follow the scene bounds, which the out-of-tree visualizer cannot
    # extend (the PNG cloud's stock visualizer does), so the defaults would differ between the two recordings.
    recording.log(PINHOLE, rr.Pinhole(focal_length=60.0, width=WIDTH, height=HEIGHT, image_plane_distance=0.3), rr.TransformAxes3D(0.1), static=True)
    if as_video:
        props = DepthVideoProperties(meter, rr.components.Colormap.Turbo, code_range, point_fill_ratio=1.0)
        log_depth_video(DEPTH, encode_av1(codes, gop=GOP), times_ns, props, timeline=TIMELINE, recording=recording)
    else:
        recording.log(DEPTH, rr.EncodedDepthImage.from_fields(media_type="image/png", meter=meter, depth_range=code_range, colormap="turbo",
                                                              point_fill_ratio=1.0), static=True)  # fmt: skip
        recording.send_columns(DEPTH, indexes=[rr.TimeColumn(TIMELINE, duration=times_ns.astype("timedelta64[ns]"))],
                               columns=rr.EncodedDepthImage.columns(blob=[png_u16(frame) for frame in codes]))  # fmt: skip
    # The out-of-tree visualizer gives a 2D view no bounds; a transparent box over the image does (both recordings have it).
    recording.log(BOUNDS, rr.Boxes2D(mins=[0, 0], sizes=[WIDTH, HEIGHT], colors=[[0, 0, 0, 0]]), static=True)
    recording.log(SPLATS, rr.GaussianSplats3D(centers=[[-0.25, -0.15, 0.7], [0.25, 0.15, 0.8]], scales=[[0.08, 0.05, 0.02], [0.05, 0.08, 0.02]],
                                              quaternions=[[0, 0, 0, 1], [0, 0, 0, 1]], colors=[[255, 60, 200, 255], [60, 255, 120, 255]]),
                  static=True)  # fmt: skip
    recording.flush()
    recording.disconnect()
    return views


def test_depth_video_draws_the_png_codes_alone_and_beside_splats(canonical_viewer: Path, tmp_path: Path) -> None:
    shown: dict[tuple[bool, str, int], UInt8[np.ndarray, "h w 3"]] = {}
    messages: list[str] = []
    problems: list[str] = []
    # PNG first: its pixels are the reference that each video view must reach once its frame has decoded.
    for as_video in (False, True):
        recording_path: Path = tmp_path / f"depth-{'video' if as_video else 'png'}.rrd"
        views: dict[str, rrb.View] = write_scene(recording_path, as_video)
        with spawn_viewer(canonical_viewer) as viewer:
            viewer.open(str(recording_path))
            for frame in SHOWN_FRAMES:
                viewer.set_time(TIMELINE, duration=np.timedelta64(frame * FRAME_NS, "ns"), play_state="paused")
                for name, view in views.items():
                    path: Path = tmp_path / f"{'video' if as_video else 'png'}-{name}-{frame}.png"
                    deadline: float = time.monotonic() + 60.0
                    pixels = capture(viewer, view, path)
                    while as_video and not np.array_equal(pixels, shown[False, name, frame]) and (remaining := deadline - time.monotonic()) > 0:
                        pixels = capture(viewer, view, path, timeout_s=remaining)
                    shown[as_video, name, frame] = pixels
            logs = viewer.get_logs()
        messages += [entry.message for entry in logs]
        problems += [f"{entry.level} {entry.target}: {entry.message}" for entry in logs if entry.level in ("ERROR", "WARN")]

    for name in views:
        for frame in SHOWN_FRAMES:
            video, reference = shown[True, name, frame], shown[False, name, frame]
            assert video.shape == reference.shape, f"{name}, frame {frame}"
            differ: float = float((video != reference).mean())
            assert differ == 0.0, f"{name}, frame {frame}: the video differs from the PNG codes in {differ:.4%} of values"
        first, second = (shown[True, name, frame] for frame in SHOWN_FRAMES)
        assert not np.array_equal(first, second), f"{name}: both frames draw the same pixels"
    # The magenta splat: turbo never has red and blue high together, so only the gsplat extension draws these pixels.
    for frame in SHOWN_FRAMES:
        magenta: dict[str, float] = {}
        for name in ("cloud", "cloud+splats"):
            rgb = shown[True, name, frame].astype(np.int16)
            magenta[name] = float(((rgb[..., 0] > 180) & (rgb[..., 2] > 140) & (rgb[..., 1] < 120)).mean())
        assert magenta["cloud"] == 0.0 and magenta["cloud+splats"] > 0.002, f"frame {frame}: magenta splat pixels {magenta}"
    for extension in ("gsplat", "depth-video"):
        assert f"canonical viewer: extension {extension} registered" in messages
    assert not problems, problems
