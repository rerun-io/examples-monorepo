from pathlib import Path

import pytest

pytest.importorskip("rerun.catalog", reason="mvs catalog imports need the Rerun catalog dependencies")
pytest.importorskip("rerun.experimental.dataloader", reason="mvs catalog imports need the Rerun dataloader")
pytest.importorskip("torchcodec", reason="mvs catalog imports need TorchCodec")

import depth_png  # noqa: E402
import numpy as np  # noqa: E402
import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402
import rerun.chunk as rrc  # noqa: E402

from mvs.catalog import FRAME_TIMELINE, PINHOLE_LOWRES_PATH, PINHOLE_PATH, TIMELINE, blueprint, log_depth  # noqa: E402


def test_log_depth_writes_the_lowres_pinhole_and_a_millimetre_png(tmp_path: Path) -> None:
    """The depth lands under its own 256x192 pinhole as a 16-bit millimetre PNG on both timelines."""
    depth_hw = np.full((192, 256), 2.5, dtype=np.float32)
    depth_hw[:96] = 0.75  # near top half, far bottom half
    K_depth_33 = np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 96.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    target: Path = tmp_path / "depth.rrd"
    recording = rr.RecordingStream("mvs-test")
    recording.save(target)
    with recording:
        rr.set_time(TIMELINE, duration=np.timedelta64(1_000_000_000, "ns"))
        rr.set_time(FRAME_TIMELINE, sequence=60)
        log_depth(depth_hw, K_depth_33)
    recording.flush()
    recording.disconnect()

    chunks: list[rrc.Chunk] = list(rrc.RrdReader(target).stream())
    columns: dict[str, dict[str, list]] = {}
    for chunk in chunks:
        batch = chunk.to_record_batch()
        columns.setdefault(str(chunk.entity_path), {}).update({name: batch.column(name).to_pylist() for name in batch.schema.names})
    pinhole = columns[f"/{PINHOLE_LOWRES_PATH}"]
    depth = columns[f"/{PINHOLE_LOWRES_PATH}/depth"]
    # One row per column, one instance per row.
    assert pinhole["Pinhole:resolution"] == [[[256.0, 192.0]]]
    assert np.allclose(np.array(pinhole["Pinhole:image_from_camera"][0][0]).reshape(3, 3).T, K_depth_33)
    assert depth["EncodedDepthImage:meter"] == [[1000.0]]
    assert depth[FRAME_TIMELINE] == [60]
    assert TIMELINE in depth
    decoded = depth_png.decode(bytes(depth["EncodedDepthImage:blob"][0][0]))
    assert decoded.dtype == np.uint16
    assert decoded[0, 0] == 750 and decoded[-1, -1] == 2500


def test_blueprint_shows_the_scene_the_video_and_the_depth() -> None:
    """One screen: the 3D scene (mesh + camera) beside the camera video above the predicted depth."""
    views: dict[str, rrb.View] = {}

    def collect(container: rrb.Container | rrb.View) -> None:
        if isinstance(container, rrb.View):
            views[str(container.name)] = container
        else:
            for child in container.contents:
                collect(child)

    layout = blueprint()
    assert layout.collapse_panels
    collect(layout.root_container)
    assert set(views) == {"Scene", "Video", "Predicted depth"}
    assert isinstance(views["Scene"], rrb.Spatial3DView) and views["Scene"].origin == "world"
    assert isinstance(views["Video"], rrb.Spatial2DView) and views["Video"].origin == PINHOLE_PATH
    assert isinstance(views["Predicted depth"], rrb.Spatial2DView) and views["Predicted depth"].origin == PINHOLE_LOWRES_PATH
