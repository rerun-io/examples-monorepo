"""``cube_layer.read_cube_faces``: the faces are the camera nodes the layer's ``cube`` property lists, nothing else."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rerun.catalog", reason="rerun.catalog is required by this test module")

import rerun as rr  # noqa: E402

SIZE: int = 64
FACES: tuple[str, ...] = ("cube6_front0", "cube6_left")


def _face_layer(path: Path, faces: tuple[str, ...] | None) -> None:
    """Two face nodes and one lens node, as dataforge writes them; ``faces`` is the ``cube`` property's list (None: no property)."""
    with rr.RecordingStream("test", recording_id="segment") as rec:
        rec.save(path)
        if faces is not None:
            rec.send_property("cube", rr.AnyValues(faces=list(faces)))
        for cam, name in ((0, "lens0"), (2, FACES[0]), (3, FACES[1])):
            node: str = f"/world/rig_00/cam_0{cam}"
            rec.log(node, rr.AnyValues(name=name), rr.Transform3D(translation=[0.0, 0.0, 0.0], mat3x3=np.eye(3)), static=True)
            rec.log(f"{node}/pinhole", rr.Pinhole(focal_length=SIZE / 2, principal_point=[(SIZE - 1) / 2, (SIZE - 1) / 2], width=SIZE, height=SIZE), static=True)


def test_faces_are_the_cameras_the_cube_property_lists(tmp_path: Path) -> None:
    from monopriors.cube_layer import open_rrd, read_cube_faces

    path: Path = tmp_path / "faces.rrd"
    _face_layer(path, FACES)
    with open_rrd(path, "faces") as layer:
        faces = read_cube_faces(layer)
    assert [(face.entity, face.name) for face in faces] == [("/world/rig_00/cam_02", FACES[0]), ("/world/rig_00/cam_03", FACES[1])]


def test_a_cube_layer_without_its_face_list_is_refused(tmp_path: Path) -> None:
    from monopriors.cube_layer import open_rrd, read_cube_faces

    path: Path = tmp_path / "faces.rrd"
    _face_layer(path, None)
    with open_rrd(path, "faces") as layer, pytest.raises(ValueError, match="property:cube:faces"):
        read_cube_faces(layer)
