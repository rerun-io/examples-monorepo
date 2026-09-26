"""EPFL source contracts: native clocks, fit gates, and annotation intervals."""

from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets.epfl_source import CAMERA_NAMES, read_timestamps


def test_camera_order_and_device_clock(tmp_path: Path) -> None:
    assert CAMERA_NAMES == ("output0", "Aoutput0", "Aoutput1", "Aoutput2", "Aoutput3", "Boutput0", "Boutput1", "Boutput2", "Boutput3", "hololens")
    path = tmp_path / "timestamps.txt"
    path.write_text("29300000\n29333333\n29400000\n")
    np.testing.assert_array_equal(read_timestamps(path), [29300000000, 29333333000, 29400000000])


def test_fit_gate_and_raw_confidence() -> None:
    from dataforge.datasets.epfl_source import parse_pose_row

    mano, smpl = pose_cells()
    row = parse_pose_row(mano, smpl)
    assert row.left.parameters.l2_dist == 0.012
    assert row.confidence[91] == pytest.approx(1.04)
    assert row.confidence[92] == 0.0
    assert np.isfinite(row.positions[92]).all()
    mano["l2_dist_left"] = ""
    mano["l2_dist_right"] = "0.06"
    smpl["l2_dist"] = "287.6"
    row = parse_pose_row(mano, smpl)
    assert np.isnan(row.positions).all()
    assert not row.confidence.any()
    smpl["rgb_frameid"] = "871"
    with pytest.raises(ValueError, match="rgb_frameid"):
        parse_pose_row(mano, smpl)


def pose_cells() -> tuple[dict[str, str], dict[str, str]]:
    import json

    mano = {"rgb_frameid": "870", "kp3ds": json.dumps([[1, 2, 3]] * 42), "kp3ds_conf": json.dumps([1.04, 0.0] + [0.7] * 40)}
    for side in ("left", "right"):
        mano.update(
            {
                f"{side}_{key}": json.dumps(value)
                for key, value in {"poses": [0.0] * 48, "Rh": [0.0] * 3, "Th": [0.0] * 3, "shapes": [0.0] * 10}.items()
            }
        )
        mano[f"l2_dist_{side}"] = "0.012"
    smpl = {"rgb_frameid": "870", "kp3ds": json.dumps([[1, 2, 3]] * 17), "kp3ds_conf": json.dumps([0.8] * 17), "l2_dist": "0.025"}
    smpl.update({key: json.dumps(value) for key, value in {"poses": [0.0] * 72, "Rh": [0.0] * 3, "Th": [0.0] * 3, "shapes": [0.0] * 10}.items()})
    return mano, smpl


def test_actions_overlap_and_xlsx(tmp_path: Path) -> None:
    from zipfile import ZipFile

    from dataforge.datasets.epfl_actions import Segment, action_rows, read_xlsx

    path = tmp_path / "actions.xlsx"
    with ZipFile(path, "w") as z:
        z.writestr(
            "xl/worksheets/sheet1.xml",
            """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
        <row><c r="A1" t="inlineStr"><is><t>Start</t></is></c><c r="B1" t="inlineStr"><is><t>End</t></is></c><c r="C1" t="inlineStr"><is><t>Verbs</t></is></c><c r="D1" t="inlineStr"><is><t>Nouns</t></is></c><c r="E1" t="inlineStr"><is><t>Confusion</t></is></c></row>
        <row><c r="A2" t="n"><v>0.1</v></c><c r="B2" t="n"><v>0.3</v></c><c r="C2" t="inlineStr"><is><t>Grab</t></is></c><c r="E2" t="inlineStr"><is><t>0</t></is></c></row>
        </sheetData></worksheet>""",
        )
    rows = read_xlsx(path)
    assert (rows[0].Start, rows[0].End, rows[0].Verbs, rows[0].Nouns, rows[0].Confusion) == (0.1, 0.3, "Grab", "", "0")
    times = np.array([100, 200, 300, 400, 500], dtype=np.int64)
    frames, texts = action_rows([Segment(1, 3, "Grab"), Segment(2, 5, "Hold")], times)
    assert texts == ["", "Grab", "Grab\nHold", "Hold"]
    np.testing.assert_array_equal(times[frames], [100, 200, 300, 400])


def test_hololens_dropout_and_frame_sequence(tmp_path: Path) -> None:
    import csv
    import json

    from dataforge.datasets.epfl_source import holo_batches, pose_batches

    path = tmp_path / "holo.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["world2holo"])
        transform = np.eye(4)
        transform[0, 3] = 2
        writer.writerow([json.dumps(transform.tolist())])
        writer.writerow(["[]"])
    transforms = next(holo_batches(path, 2, total=None))
    assert transforms[0, 0, 3] == -2
    assert np.isnan(transforms[1]).all()
    mano, smpl = pose_cells()
    for model, row in [("mano", mano), ("smpl", smpl)]:
        with (tmp_path / f"pose3d_{model}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            writer.writerow(row)
    with pytest.raises(ValueError, match="consecutive"):
        list(pose_batches(tmp_path, 2, total=None))


def test_full_rational_projection_and_visibility() -> None:
    from dataforge.datasets.epfl_layers import project_keypoints
    from dataforge.datasets.epfl_source import ExoCamera

    camera = ExoCamera(
        K=np.array([[100.0, 0.0, 640.0], [0.0, 100.0, 360.0], [0.0, 0.0, 1.0]]),
        dist=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        word2cam=np.eye(4),
    )
    points = np.full((1, 133, 3), np.nan, dtype=np.float32)
    points[0, :3] = [[1, 0, 1], [0, 0, -1], [0, 0, 1]]
    pixels = project_keypoints(camera, points)
    # k4=1 divides x=1 by two: ignoring the rational denominator gives 740.
    np.testing.assert_allclose(pixels[0, 0], [690, 360])
    assert np.isnan(pixels[0, 1]).all()
    np.testing.assert_allclose(pixels[0, 2], [640, 360])
    assert np.isnan(pixels[0, 3:]).all()


def test_pose_rrd_preserves_residuals_and_layer_ownership(tmp_path: Path) -> None:
    from conftest import read_chunks

    from dataforge import writing
    from dataforge.datasets.epfl_layers import start_parameters, write_hand_pose, write_pose
    from dataforge.datasets.epfl_source import FITS, parse_pose_row

    mano, smpl = pose_cells()
    row = parse_pose_row(mano, smpl)
    times = np.array([29300000000], dtype=np.int64)
    frames = np.array([0], dtype=np.int64)
    for body in (False, True):
        target = tmp_path / f"{body}.rrd"
        with writing.atomic_recording(target, recording_id="epfl-test", send_properties=False) as recording:
            specs = FITS[2:] if body else FITS[:2]
            start_parameters(recording, specs)
            (write_pose if body else write_hand_pose)(recording, [row], times, frames, specs=specs)
        chunks = read_chunks(target)
        assert any(c.entity_path == "/world/gt/coco133_xyz" for c in chunks) == (not body)
        residuals = [c.to_record_batch().column("Scalars:scalars").to_pylist()[0][0] for c in chunks if str(c.entity_path).endswith("/l2_dist")]
        np.testing.assert_allclose(residuals, [0.025] if body else [0.012, 0.012])
        for spec in specs:
            metadata = [c.to_record_batch() for c in chunks if c.is_static and str(c.entity_path) == spec.parameters_path]
            assert len(metadata) == 1
            expected = {"use_pca": False, "root": "Rh", "translation_pivot": "origin", "source": "pose3d_smpl.csv" if body else "pose3d_mano.csv"}
            if body:
                expected["gender"] = "neutral (assumed)"
            for name, value in expected.items():
                assert metadata[0].column(name).to_pylist() == [[value]]
            assert ("gender" in metadata[0].schema.names) == body


def test_blueprint_and_discovery_missing_assets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from conftest import blueprint_views

    from dataforge.datasets.epfl import EpflConfig

    config = EpflConfig(root=tmp_path)
    dataset = config.setup()
    assert len(blueprint_views(dataset.default_blueprint())) == 11
    assert len(blueprint_views(dataset.table_blueprint())) == 1
    session = tmp_path / "Public_release_pose/train/person/session"
    session.mkdir(parents=True)
    assert dataset.discover() == []
    assert "pose3d_mano.csv" in capsys.readouterr().out
    with pytest.raises(ValueError, match="frame_limit"):
        EpflConfig(frame_limit=0)


def test_actions_and_projection_rrds(tmp_path: Path) -> None:
    from conftest import read_chunks

    from dataforge import writing
    from dataforge.datasets.epfl_actions import Segment
    from dataforge.datasets.epfl_layers import write_actions, write_projections
    from dataforge.datasets.epfl_source import EXO_CAMERAS, ExoCamera, parse_pose_row

    mano, smpl = pose_cells()
    row = parse_pose_row(mano, smpl)
    times = np.array([29300000000], dtype=np.int64)
    frames = np.array([0], dtype=np.int64)
    camera = ExoCamera(np.eye(3), np.zeros(8), np.eye(4))
    target = tmp_path / "derived.rrd"
    with writing.atomic_recording(target, recording_id="derived", send_properties=False) as recording:
        write_actions(recording, {"fine": [Segment(0, 1, "Grab")]}, times)
        write_projections(recording, {name: camera for name in EXO_CAMERAS}, [row], times, frames)
    chunks = read_chunks(target)
    action = next(c.to_record_batch() for c in chunks if str(c.entity_path) == "/task/actions/fine")
    assert action.column("TextDocument:text").to_pylist() == [["Grab"]]
    assert len({str(c.entity_path) for c in chunks if str(c.entity_path).endswith("/coco133_uv_projected")}) == 9


def test_scene_centre_is_the_exo_camera_centroid() -> None:
    from dataforge.datasets.epfl import scene_centre
    from dataforge.datasets.epfl_source import ExoCamera

    def camera(centre: tuple[float, float, float]) -> ExoCamera:
        world2cam = np.eye(4)
        world2cam[:3, 3] = -np.asarray(centre)
        return ExoCamera(K=np.eye(3), dist=np.zeros(8), word2cam=world2cam)

    cameras = {"a": camera((1.0, 0.0, 0.5)), "b": camera((3.0, 2.0, 0.3))}
    x, y = scene_centre(cameras)
    assert (x, y) == pytest.approx((2.0, 1.0))


def test_action_end_at_clock_length_keeps_last_frame_active() -> None:
    from dataforge.datasets.epfl_actions import Segment, action_rows

    frames, texts = action_rows([Segment(2, 5, "Hold")], np.arange(5, dtype=np.int64))
    np.testing.assert_array_equal(frames, [0, 2])
    assert texts == ["", "Hold"]


@pytest.mark.parametrize("raw_kind", ["pose", "video", "symlink"])
def test_work_directory_cannot_overlap_raw_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw_kind: str) -> None:
    from dataforge.datasets.epfl import EpflConfig
    from dataforge.identity import SequenceIdentity

    output = tmp_path / "out"
    work = output / "work"
    raw = tmp_path / "raw"
    if raw_kind == "symlink":
        output.mkdir()
        work.symlink_to("/mnt/nas/epfl-work")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(output))
    config = EpflConfig(root=work if raw_kind == "pose" else raw, video_root=work if raw_kind == "video" else raw)
    with pytest.raises(ValueError, match="outside raw roots"):
        config.setup().convert(SequenceIdentity("epfl", ("train", "subject", "session")), "train/subject/session", force=True)
    assert not list(output.glob("**/*.rrd"))


def test_calibration_separates_static_and_moving_cameras(tmp_path: Path) -> None:
    import json

    from dataforge.datasets.epfl_source import EXO_CAMERAS, EgoCamera, ExoCamera, read_cameras

    raw = {name: {"K": np.eye(3).tolist(), "dist": [0.0] * 8, "word2cam": np.eye(4).tolist()} for name in EXO_CAMERAS}
    raw["hololens"] = {"K": np.eye(3).tolist(), "dist": [0.0] * 8}
    path = tmp_path / "camera_matrix.json"
    path.write_text(json.dumps(raw))
    cameras, ego = read_cameras(path)
    assert tuple(cameras) == EXO_CAMERAS
    assert all(isinstance(camera, ExoCamera) for camera in cameras.values())
    assert isinstance(ego, EgoCamera)
    assert ego.size == (896, 504)
    del raw["output0"]["word2cam"]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="camera_matrix.json.*calibration"):
        read_cameras(path)


def test_readers_check_full_count_but_allow_prefix(tmp_path: Path) -> None:
    import csv

    from dataforge.datasets.epfl_source import holo_batches, pose_batches

    holo = tmp_path / "holo.csv"
    holo.write_text("world2holo\n[]\n[]\n")
    assert sum(len(batch) for batch in holo_batches(holo, 1, total=2)) == 1
    with pytest.raises(ValueError, match="expected count"):
        list(holo_batches(holo, 1))
    with pytest.raises(ValueError, match="expected count"):
        list(holo_batches(holo, 3))
    for model, row in zip(("mano", "smpl"), pose_cells(), strict=True):
        with (tmp_path / f"pose3d_{model}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            writer.writerow({**row, "rgb_frameid": "871"})
    assert sum(len(batch) for batch in pose_batches(tmp_path, 1, total=2)) == 1
    assert sum(len(batch) for batch in pose_batches(tmp_path, 2, total=2)) == 2
    with pytest.raises(ValueError, match="expected count"):
        list(pose_batches(tmp_path, 1))
    with pytest.raises(ValueError, match="expected count"):
        list(pose_batches(tmp_path, 3))


@pytest.mark.parametrize("unsupported", ["shared_strings", "extra_sheet", "formula", "boolean"])
def test_xlsx_rejects_other_layouts(tmp_path: Path, unsupported: str) -> None:
    from zipfile import ZipFile

    from dataforge.datasets.epfl_actions import read_xlsx

    path = tmp_path / "other.xlsx"
    with ZipFile(path, "w") as archive:
        if unsupported == "shared_strings":
            archive.writestr("xl/sharedStrings.xml", "<sst/>")
        elif unsupported == "extra_sheet":
            archive.writestr("xl/worksheets/sheet2.xml", "<worksheet/>")
        cell = '<c r="A1"><f>1+1</f><v>2</v></c>' if unsupported == "formula" else '<c r="A1" t="b"><v>1</v></c>'
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row>{cell}</row></sheetData></worksheet>',
        )
    with pytest.raises(ValueError, match="other.xlsx"):
        read_xlsx(path)
