"""The lens-mask to face-mask tool on synthetic layers: stamps, geometry read back from the cube layer, and a missing mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Float64, UInt8
from numpy import ndarray

LENS: int = 384
FACE: int = 160
FOCAL: float = 1835.37 * LENS / 1920
STAMPS: list[int] = [1_000_000, 334_000_000]


def lens_from_rig(lens: int) -> Float64[ndarray, "4 4"]:
    """Lens 1 looks the other way, half a turn about y; 16.6 mm either side of the rig origin."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.eye(3) if lens == 0 else np.diag([-1.0, 1.0, -1.0])
    pose[:3, 3] = [0.0, 0.0, -0.0166]
    return pose


def write_base(path: Path) -> None:
    from simplecv.rerun_custom_types import CameraDistortion, PinholeWithDistortion

    centre: float = (LENS - 1) / 2
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for lens in range(2):
            pose = lens_from_rig(lens)
            recording.log(
                f"/world/rig_00/cam_0{lens}",
                rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], relation=rr.TransformRelation.ChildFromParent),
                static=True,
            )
            pinhole = rr.Pinhole(image_from_camera=[[FOCAL, 0, centre], [0, FOCAL, centre], [0, 0, 1]], width=LENS, height=LENS)
            distortion = CameraDistortion("mei_v6", np.array([2.315, np.deg2rad(100.0)] + [0.0] * 13))
            recording.log(f"/world/rig_00/cam_0{lens}/pinhole", PinholeWithDistortion(pinhole, distortion), static=True)


def write_faces(path: Path) -> None:
    """A cube layer as ``write_cube`` lays it out, with placeholder video samples."""
    from dataforge.datasets import insta360_cube as cube

    lenses = [lens_from_rig(0), lens_from_rig(1)]
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for k, face in enumerate(cube.cube_faces(FACE)):
            node: str = f"/world/rig_00/cam_{2 + k:02d}"
            pose = cube.face_from_rig(face, lenses)
            recording.log(node, rr.AnyValues(name=face.name), static=True)
            recording.log(
                node, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], relation=rr.TransformRelation.ChildFromParent), static=True
            )
            recording.log(f"{node}/pinhole", rr.Pinhole(image_from_camera=face.intrinsics, width=FACE, height=FACE), static=True)
            recording.log(f"{node}/pinhole/video", rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
            for stamp in STAMPS:
                recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
                recording.log(f"{node}/pinhole/video", rr.VideoStream.from_fields(sample=b"\x00"))


def write_masks(path: Path, stamps: list[int]) -> None:
    """Lens 0: a disc on its optical axis. Lens 1: a disc 80 deg toward its +X (the cube's -X: ``cube6_left``)."""
    import insta360_io

    centre: float = (LENS - 1) / 2
    camera = insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [2.315, np.deg2rad(100.0)] + [0.0] * 13)
    side, _ = camera.project(np.array([[np.sin(np.deg2rad(80.0)), 0.0, np.cos(np.deg2rad(80.0))]]))
    u, v = np.meshgrid(np.arange(LENS), np.arange(LENS))
    discs: list[UInt8[ndarray, "h w"]] = [
        (np.hypot(u - centre, v - centre) <= 20).astype(np.uint8),
        (np.hypot(u - side[0, 0], v - side[0, 1]) <= 8).astype(np.uint8),
    ]
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for stamp in stamps:
            recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            for lens in range(2):
                recording.log(f"/world/rig_00/cam_0{lens}/pinhole/people_mask", rr.SegmentationImage(discs[lens]))


def read_masks(path: Path) -> dict[str, list[tuple[int, UInt8[ndarray, "h w"]]]]:
    """Every face mask of a written layer: ``{entity: [(stamp, mask)]}``."""
    found: dict[str, list[tuple[int, UInt8[ndarray, "h w"]]]] = {}
    for chunk in rrc.RrdReader(path).stream():
        batch = chunk.to_record_batch()
        if "SegmentationImage:buffer" in batch.schema.names:
            for stamp, buffer, fmt in zip(
                batch.column("video_time").cast("int64").to_pylist(),
                batch.column("SegmentationImage:buffer").to_pylist(),
                batch.column("SegmentationImage:format").to_pylist(),
                strict=True,
            ):
                found.setdefault(chunk.entity_path, []).append(
                    (stamp, np.asarray(buffer[0], dtype=np.uint8).reshape(fmt[0]["height"], fmt[0]["width"]))
                )
    return found


def test_each_lens_disc_lands_on_its_face_at_every_face_stamp(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import Config, main

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, [*STAMPS, 999_000_000])
    main(Config(base=base, masks=masks, faces=faces, output=output))
    found = read_masks(output)
    assert sorted(found) == [f"/world/rig_00/cam_{cam:02d}/pinhole/people_mask" for cam in range(2, 8)]
    hit: dict[str, bool] = {"cam_02": True, "cam_03": False, "cam_04": True, "cam_05": False, "cam_06": False, "cam_07": False}
    for entity, rows in found.items():
        assert sorted(stamp for stamp, _ in rows) == STAMPS, entity
        assert all((mask.sum() > 0) == hit[entity.split("/")[3]] for _, mask in rows), entity
        assert all(mask.shape == (FACE, FACE) and set(np.unique(mask)) <= {0, 1} for _, mask in rows)
    # Lens 1's disc sits on the lens-1 half of cube6_left: the image's left half (image right points to +Z0).
    left: UInt8[ndarray, "n n"] = found["/world/rig_00/cam_04/pinhole/people_mask"][0][1]
    assert left[:, : FACE // 2].sum() > 0 and left[:, FACE // 2 :].sum() == 0


def test_a_face_stamp_without_a_mask_on_both_lenses_is_an_error(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import Config, main

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS[:1])
    with pytest.raises(ValueError, match="no mask on both lenses"):
        main(Config(base=base, masks=masks, faces=faces, output=output))
    assert not output.exists()
