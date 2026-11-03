"""SelfCap layouts: the default layout at every stage, the table card, and the table fields."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rerun.blueprint as rrb
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer
from test_selfcap import rig_layout_base

from dataforge import schema
from dataforge.datasets.selfcap import SelfcapConfig, SelfcapDataset
from dataforge.datasets.selfcap_blueprint import CameraPane, build_blueprint
from dataforge.writing import blueprint_views

# The 3D scene starts about 3.2 m from the world origin, where the wearer stands in most captures, and orbits slowly.
SCENE_EYE = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=[2.0, 2.9, 2.0], look_target=[0.0, 1.6, 0.0], eye_up=[0.0, 1.0, 0.0], spin_speed=0.1)


def test_selfcap_layouts_open_on_the_aligned_clock() -> None:
    # video_time equals capture_time at ingest and is the aligned clock after selfcap-align, so it is the right
    # timeline at every stage; left unset, the viewer picked capture_time and showed the as-cut streams.
    panes = (CameraPane("Dragon", 0, 0, "exo"), CameraPane("oak rgb", 4, 0, "ego"), CameraPane("quest left", 5, 0, "quest"))
    for blueprint in (build_blueprint(panes, ego_rig=4), SelfcapDataset(SelfcapConfig()).default_blueprint()):
        assert blueprint.time_panel is not None
        assert str(blueprint.time_panel.timeline) == schema.TIMELINE


HEAD_CAMERAS: dict[str, tuple[int, int]] = {"OAK rgb": (4, 0), "OAK left": (4, 1), "OAK right": (4, 2), "Quest left": (5, 0), "Quest right": (5, 1)}
"""The default layout's head panes and their (rig, camera) in a four-phone session."""
PHONES: dict[str, int] = {"Phone 1": 0, "Phone 2": 1, "Phone 3": 2, "Phone 4": 3}
"""The default layout's phone panes and their rigs: index labels, as device names vary per session."""


def test_default_blueprint_shows_the_raw_capture_before_calibration() -> None:
    # One layout for every stage. Before calibrate the phones have no pose: a phone left in 3D would sit at the
    # world origin, and the Quest hands drawn through its pinhole would land in the wrong place.
    blueprint: rrb.Blueprint = SelfcapDataset(SelfcapConfig()).default_blueprint(posed_phones=False)
    views: dict[str, rrb.View] = {str(view.name): view for view in blueprint_views(blueprint)}
    sounds: list[str] = [f"{name} · sound" for name in PHONES]
    loudness: str = "Loudness per phone (dB over its own floor)"
    assert sorted(views) == sorted(["Scene", *HEAD_CAMERAS, *PHONES, *sounds, loudness, "Gyroscope", "Accelerometer"])

    scene = views["Scene"]
    assert str(scene.origin) == "/world"
    scene_contents: list[str] = cast(list[str], scene.contents)
    assert "/world/**" in scene_contents
    for rig in PHONES.values():
        assert f"- /world/rig_{rig:02d}/**" in scene_contents
    for rig, cam in HEAD_CAMERAS.values():  # head frustums show; their image planes would hide the hands
        assert f"+ /world/rig_{rig:02d}/cam_{cam:02d}/pinhole" in scene_contents
        assert f"- /world/rig_{rig:02d}/cam_{cam:02d}/pinhole/**" in scene_contents
    assert "/world/gt" not in scene.visualizer_overrides  # the Quest hands and their joint axes show
    assert str(scene.properties["Background"]) == str(rrb.Background(kind=rrb.BackgroundKind.GradientDark))
    assert str(scene.properties["EyeControls3D"]) == str(SCENE_EYE), "a close, slow orbit, like the catalog cards"
    assert scene.visualizer_overrides["/world/pred"] == rrb.EntityBehavior(visible=False)

    for name, (rig, cam) in HEAD_CAMERAS.items():
        pinhole: str = f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole"
        assert str(views[name].origin) == pinhole
        assert f"{pinhole}/**" in cast(list[str], views[name].contents)
        assert "/world/gt/**" in cast(list[str], views[name].contents)
        assert "/world/gt" not in views[name].visualizer_overrides
        assert views[name].visualizer_overrides[f"{pinhole}/sapiens2"] == rrb.EntityBehavior(visible=False)
    for name, rig in PHONES.items():
        assert cast(list[str], views[name].contents) == [f"/world/rig_{rig:02d}/cam_00/pinhole/**"]
        # Sapiens2 shows only its points at or above 0.7; the full set stays in the data, hidden.
        overrides = views[name].visualizer_overrides
        assert overrides[f"/world/rig_{rig:02d}/cam_00/pinhole/pred/sapiens2/coco133_uv"] == rrb.EntityBehavior(visible=False)
        assert overrides[f"/world/rig_{rig:02d}/cam_00/pinhole/pred/rtmw/coco133_uv"] == rrb.EntityBehavior(visible=False)
        assert overrides[f"/world/rig_{rig:02d}/cam_00/pinhole/sapiens2/coco133_uv"] == rrb.EntityBehavior(visible=False)
        assert f"/world/rig_{rig:02d}/cam_00/pinhole/pred/sapiens2/coco133_uv_min70" not in overrides
        sound = views[f"{name} · sound"]
        assert isinstance(sound, rrb.AudioView)
        assert str(sound.origin) == f"/world/rig_{rig:02d}/audio_00"
        assert str(sound.properties["AudioPlayback"]) == str(rrb.AudioPlayback(volume=0.0))  # muted until asked
    assert cast(list[str], views[loudness].contents) == [f"/world/rig_{rig:02d}/audio_00/loudness" for rig in range(4)]
    assert str(views["Gyroscope"].origin) == "/world/rig_04/imu_00"
    assert cast(str, views["Accelerometer"].contents) == "/world/rig_04/imu_00/accel"


def test_default_blueprint_draws_the_hands_and_body_through_posed_phones() -> None:
    # After calibrate the phones have poses: their cameras show in 3D and every overlay projects through them.
    views = {str(view.name): view for view in blueprint_views(SelfcapDataset(SelfcapConfig()).default_blueprint())}
    scene_contents: list[str] = cast(list[str], views["Scene"].contents)
    assert "- /world/rig_00/**" not in scene_contents
    for rig, cam in HEAD_CAMERAS.values():
        assert f"+ /world/rig_{rig:02d}/cam_{cam:02d}/pinhole" in scene_contents
    for name, rig in PHONES.items():
        pinhole: str = f"/world/rig_{rig:02d}/cam_00/pinhole"
        assert not [rule for rule in scene_contents if rule.startswith(f"- {pinhole}")], "each posed phone shows its video in 3D"
        contents: list[str] = cast(list[str], views[name].contents)
        assert contents[:4] == [f"{pinhole}/**", "/world/gt/**", "/world/pred/**", "/world/skeleton/**"]
        # The other rigs' camera frames and names project into each phone (cam nodes only: no frustum or video).
        others = [r for r in range(6) if r != rig]
        assert {f"/world/rig_{r:02d}/cam_00/rig_label" for r in others if r < 4} <= set(contents)
        assert {"/world/rig_04/cam_01", "/world/rig_04/cam_02", "/world/rig_05/cam_01"} <= set(contents)
        assert not [c for c in contents[4:] if c.startswith(f"/world/rig_{rig:02d}/")], "not its own frame"
        hidden = views[name].visualizer_overrides
        for overlay in ("pred", "sapiens2", "coco133_uv_projected", "quest_coco133_uv_projected"):
            assert hidden[f"{pinhole}/{overlay}"] == rrb.EntityBehavior(visible=False)
        assert hidden["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=True)
    assert views["Scene"].visualizer_overrides["/world/pred/soma_x/mesh"] == rrb.EntityBehavior(visible=True)
    assert views["Scene"].visualizer_overrides["/world/rig_00/cam_00/pinhole/skeleton"] == rrb.EntityBehavior(visible=False)


def test_each_oak_pane_draws_the_skeleton_written_beneath_its_own_pinhole() -> None:
    views = {str(view.name): view for view in blueprint_views(SelfcapDataset(SelfcapConfig()).default_blueprint())}
    for name in ("OAK rgb", "OAK left", "OAK right"):
        rig, cam = HEAD_CAMERAS[name]
        skeleton: str = f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole/skeleton"
        assert cast(list[str], views[name].contents)[0] == f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole/**"
        assert "/world/skeleton/**" not in cast(list[str], views[name].contents), "the OAK lens model is in its own 2D skeleton, not the viewer's pinhole"
        assert skeleton not in views[name].visualizer_overrides
        assert views["Scene"].visualizer_overrides[skeleton] == rrb.EntityBehavior(visible=False)


@pytest.mark.integration
def test_register_publishes_the_posed_layouts_only_when_every_phone_has_a_pose(tmp_path: Path) -> None:
    # Calibrate writes each phone's pose on its cam_00. The catalog has one default layout for all its segments, so it
    # draws the phones in 3D only when every phone of every segment has a pose: an unposed phone would sit at the origin.
    import rerun as rr

    from dataforge.writing import recording_to

    def calib(target: Path, recording_id: str, rigs: range) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with recording_to(target, recording_id=recording_id, send_properties=False) as recording:
            for rig in rigs:
                rr.log(schema.cam_path(rig, 0), rr.Transform3D(translation=[0.0, 0.0, 1.0], mat3x3=np.eye(3)), static=True, recording=recording)

    def phones_in_scene(entry: DatasetEntry) -> bool:
        segment, table = SelfcapDataset(SelfcapConfig()).catalog_blueprints(entry)
        scene = next(view for view in blueprint_views(segment) if str(view.name) == "Scene")
        posed: bool = "- /world/rig_00/**" not in cast(list[str], scene.contents)
        assert isinstance(table, rrb.Blueprint)
        assert (str(blueprint_views(table)[1].origin) == "/world/rig_00/cam_00/pinhole") == posed
        return posed

    four, three = "selfcap__2025-12-20__25aeb66a__ep001", "selfcap__2025-12-21__1dbe8c49__ep002"
    rig_layout_base(tmp_path / "base" / f"{four}.rrd", four, phones=4)
    rig_layout_base(tmp_path / "base" / f"{three}.rrd", three, phones=3)
    calib(tmp_path / "calib" / f"{four}.rrd", four, range(4))
    calib(tmp_path / "calib" / f"{three}.rrd", three, range(2))  # calibrate posed only two of its three phones
    with rr.server.Server(port=0) as server:
        entry = server.client().create_dataset("selfcap-test")
        entry.register([(tmp_path / "base" / f"{segment}.rrd").as_uri() for segment in (four, three)], layer_name="base")
        assert not phones_in_scene(entry)  # raw captures
        entry.register([(tmp_path / "calib" / f"{four}.rrd").as_uri()], layer_name="calib")
        assert not phones_in_scene(entry)  # one segment calibrated, the other still raw
        entry.register([(tmp_path / "calib" / f"{three}.rrd").as_uri()], layer_name="calib")
        assert not phones_in_scene(entry)  # every segment calibrated, but one phone has no pose
        calib(tmp_path / "calib2" / f"{three}.rrd", three, range(3))
        entry.register([(tmp_path / "calib2" / f"{three}.rrd").as_uri()], layer_name="calib", on_duplicate=OnDuplicateSegmentLayer.REPLACE)
        assert phones_in_scene(entry)


def test_table_card_decodes_one_video_and_follows_the_phone_poses() -> None:
    # A card decodes exactly one stream. Before calibrate it is the OAK with the Quest hands; after, phone 1 with
    # the body drawn through its pose.
    for posed_phones, pane in ((False, (4, 0)), (True, (0, 0))):
        views: list[rrb.View] = blueprint_views(SelfcapDataset(SelfcapConfig()).table_blueprint(posed_phones=posed_phones))
        assert len(views) == 2
        scene, camera = views
        assert isinstance(scene, rrb.Spatial3DView) and isinstance(camera, rrb.Spatial2DView)
        assert str(scene.properties["Background"]) == str(rrb.Background(kind=rrb.BackgroundKind.GradientDark))
        assert str(scene.properties["EyeControls3D"]) == str(SCENE_EYE)
        assert str(scene.origin) == "/world"
        assert str(camera.origin) == f"/world/rig_{pane[0]:02d}/cam_{pane[1]:02d}/pinhole"
        scene_contents: list[str] = cast(list[str], scene.contents)
        for phone_count in (3, 4):
            slots = [(rig, 0) for rig in range(phone_count)] + [(phone_count, cam) for cam in range(3)] + [(phone_count + 1, cam) for cam in range(2)]
            for rig, cam in slots:
                assert f"- /world/rig_{rig:02d}/cam_{cam:02d}/pinhole/video/**" in scene_contents
        assert "/world/gt/**" in cast(list[str], camera.contents)
        assert ("- /world/rig_00/**" in scene_contents) == (not posed_phones)
        assert ("/world/skeleton/**" in cast(list[str], camera.contents)) == posed_phones


def test_table_fields_show_only_the_session_phones_and_soma_quality_sorted_by_session() -> None:
    # The rank and refined columns no longer exist; the collector's name never shows, even before a layer provides it.
    fields = SelfcapConfig().setup().table_fields()
    expected = [
        ("property:calib:session", "Capture / session"),
        ("property:calib:exo_cameras", "Phones"),
        ("property:soma_quality:status", "SOMA quality"),
    ]
    assert [(field.column, field.name) for field in fields.cards] == expected
    assert [(field.column, field.name) for field in fields.table] == expected
    assert fields.hidden == ("property:capture:collector",)
    assert fields.sort_by == "property:calib:session"
