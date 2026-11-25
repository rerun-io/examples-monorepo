"""Viewer contracts of `SogLodSplats3D` through `gsplat probe` (release binaries, a GPU): an LOD recording draws through
the compute path, mode off draws the same pixels as the plain `SogSplats3D` path of level 0, and a still camera
encodes no new splat frame. Level parameters reach the viewer through `GSPLAT_LOD_*` environment variables."""

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from test_sog_lod import write_lod_set  # pyrefly: ignore[missing-import]  (pytest rootdir import)

from gsplat_rust_renderer.sog3d import log_sog, log_sog_lod
from gsplat_rust_renderer.sog_decode import read_sog
from gsplat_rust_renderer.sog_lod import read_sog_lod

ROOT: Path = Path(__file__).resolve().parents[1]
ENTITY: str = "/world/splats"


@pytest.fixture(scope="module")
def gsplat() -> Path:
    """The release `gsplat` with the probe (one crate build; `GSPLAT_BIN` skips it)."""
    if "GSPLAT_BIN" in os.environ:
        return Path(os.environ["GSPLAT_BIN"])
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "-p", "gsplat-cli", "--features", "gsplat-cli/probe"],
        cwd=ROOT,
        env=os.environ,
        check=True,
        timeout=3600,
    )
    return ROOT / "target/release/gsplat"


def write_recording(out: Path, log: Callable[[rr.RecordingStream], None], spin: float) -> Path:
    """A fixed eye on the synthetic set (`spin` rad/s around +y), solid black background."""
    recording = rr.RecordingStream("lod-viewer-contract")
    recording.save(out)
    view = rrb.Spatial3DView(
        origin="/",
        contents=["/world/**"],
        line_grid=False,
        background=rrb.Background(color=(0, 0, 0), kind="SolidColor"),
        eye_controls=rrb.EyeControls3D(position=(0.0, -2.0, -12.0), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, -1.0, 0.0), spin_speed=spin),
    )
    recording.send_blueprint(rrb.Blueprint(view, collapse_panels=True))
    recording.log("/", rr.ViewCoordinates.RDF, static=True)
    log(recording)
    recording.flush()
    recording.disconnect()
    return out


def probe(gsplat: Path, rrd: Path, out: Path, env: dict[str, str], *extra: str) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    result: subprocess.CompletedProcess[str] = subprocess.run(
        [str(gsplat), "probe", str(rrd), "--out", str(out), "--window-size", "640x480", *extra],
        env={**os.environ, "RUST_LOG": "info", **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    print(result.stdout, result.stderr, sep="\n")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Gaussian compute fallback" not in result.stderr
    return json.loads(out.read_text()), result


def pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


@pytest.mark.integration
def test_lod_recording_draws_and_mode_off_equals_plain_sog(gsplat: Path, tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set")
    lod_rrd: Path = write_recording(tmp_path / "lod.rrd", lambda r: log_sog_lod(ENTITY, read_sog_lod(root), 64, recording=r), spin=0.0)
    plain_rrd: Path = write_recording(tmp_path / "plain.rrd", lambda r: log_sog(ENTITY, read_sog(root / "0_0"), 64, recording=r), spin=0.0)

    lod, lod_run = probe(gsplat, lod_rrd, tmp_path / "lod.json", {"GSPLAT_LOD_MODE": "lod", "GSPLAT_LOD_PX": "2"})
    assert "Streamed SOG decode: 380 rows in 2 levels" in lod_run.stderr
    [view] = lod["lod"]
    assert view["mode"] == "lod" and view["tau_px"] == 2.0 and view["scene_rows"] == 380 and view["level_rows"] == [300, 80]
    assert view["rows"] == sum(view["rows_per_level"]) > 0
    assert pixels(tmp_path / "lod.png").max() > 0, "the LOD recording drew nothing"

    off, _ = probe(gsplat, lod_rrd, tmp_path / "off.json", {"GSPLAT_LOD_MODE": "off"})
    [view] = off["lod"]
    assert view["mode"] == "off" and view["rows_per_level"][0] == view["rows"] == 300
    plain, plain_run = probe(gsplat, plain_rrd, tmp_path / "plain.json", {})
    assert plain["lod"] == [] and "SOG decode: 300 splats" in plain_run.stderr
    difference: np.ndarray = np.abs(pixels(tmp_path / "off.png") - pixels(tmp_path / "plain.png"))
    assert difference.max() == 0, f"mode off differs from the plain SOG path: max {difference.max()}, {np.count_nonzero(difference)} values"


@pytest.mark.integration
def test_still_camera_encodes_no_new_frame_and_selects_once(gsplat: Path, tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set")
    rrd: Path = write_recording(tmp_path / "still.rrd", lambda r: log_sog_lod(ENTITY, read_sog_lod(root), 64, recording=r), spin=0.0)
    report, _ = probe(gsplat, rrd, tmp_path / "still.json", {"GSPLAT_LOD_MODE": "lod"}, "--frames-in-flight", "2")
    assert report["fresh_frames"] == 0, "a still camera encoded splat frames during the measurement"
    [view] = report["lod"]
    assert view["selections"] == 1, "a still camera selected more than once"
    moving: Path = write_recording(tmp_path / "moving.rrd", lambda r: log_sog_lod(ENTITY, read_sog_lod(root), 64, recording=r), spin=1.2)
    report, _ = probe(gsplat, moving, tmp_path / "moving.json", {"GSPLAT_LOD_MODE": "lod"}, "--frames-in-flight", "2", "--require-motion")
    [view] = report["lod"]
    assert report["fresh_frames"] > 0 and view["selections"] > 1


@pytest.mark.integration
def test_two_views_select_for_their_own_eye_and_share_one_scene(gsplat: Path, tmp_path: Path) -> None:
    root: Path = write_lod_set(tmp_path / "set")
    out: Path = tmp_path / "two.rrd"
    recording = rr.RecordingStream("lod-viewer-two-views")
    recording.save(out)
    views: list[rrb.Spatial3DView] = [
        rrb.Spatial3DView(
            origin="/",
            name=name,
            contents=["/world/**"],
            line_grid=False,
            background=rrb.Background(color=(0, 0, 0), kind="SolidColor"),
            eye_controls=rrb.EyeControls3D(position=(0.0, 0.0, -distance), look_target=(0.0, 0.0, 0.0), eye_up=(0.0, -1.0, 0.0), spin_speed=0.0),
        )
        for name, distance in (("near", 6.0), ("far", 2000.0))
    ]
    recording.send_blueprint(rrb.Blueprint(rrb.Horizontal(*views), collapse_panels=True))
    recording.log("/", rr.ViewCoordinates.RDF, static=True)
    log_sog_lod(ENTITY, read_sog_lod(root), 64, recording=recording)
    recording.flush()
    recording.disconnect()
    report, run = probe(gsplat, out, tmp_path / "two.json", {"GSPLAT_LOD_MODE": "lod", "GSPLAT_LOD_PX": "2"}, "--multi-view")
    assert report["views"] == 2
    assert run.stderr.count("Streamed SOG decode:") == 1, "two views must share one upload"
    near, far = sorted(report["lod"], key=lambda v: -v["rows_per_level"][0])
    assert near["rows_per_level"][0] > 0 and far["rows_per_level"][0] == 0 and far["rows"] > 0, (near, far)
