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


def write_faces(path: Path, base: Path) -> None:
    """A cube layer as ``write_cube`` lays it out, with placeholder video samples."""
    from dataforge.datasets import insta360_cube as cube

    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for lens in range(2):
            for k, face in enumerate(cube.lens_faces(lens, FACE)):
                node: str = f"/world/rig_00/cam_{2 + 5 * lens + k:02d}"
                pose = cube.face_from_rig(face, lens_from_rig(lens))
                recording.log(node, rr.AnyValues(name=face.name, source_lens=lens), static=True)
                recording.log(
                    node, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3], relation=rr.TransformRelation.ChildFromParent), static=True
                )
                recording.log(f"{node}/pinhole", rr.Pinhole(image_from_camera=face.intrinsics, width=face.width, height=face.height), static=True)
                recording.log(f"{node}/pinhole/video", rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
                for stamp in STAMPS:
                    recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
                    recording.log(f"{node}/pinhole/video", rr.VideoStream.from_fields(sample=b"\x00"))


def write_masks(path: Path, stamps: list[int]) -> None:
    """A disc at each lens's centre (on its optical axis)."""
    u, v = np.meshgrid(np.arange(LENS), np.arange(LENS))
    centre: float = (LENS - 1) / 2
    disc: UInt8[ndarray, "h w"] = (np.hypot(u - centre, v - centre) <= 20).astype(np.uint8)
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for stamp in stamps:
            recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            for lens in range(2):
                recording.log(f"/world/rig_00/cam_0{lens}/pinhole/people_mask", rr.SegmentationImage(disc))


def test_lens_discs_land_on_the_front_faces_at_every_face_stamp(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import Config, main

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces, base)
    write_masks(masks, [*STAMPS, 999_000_000])
    main(Config(base=base, masks=masks, faces=faces, output=output))
    found: dict[str, list[tuple[int, int]]] = {}
    for chunk in rrc.RrdReader(output).stream():
        batch = chunk.to_record_batch()
        if "SegmentationImage:buffer" in batch.schema.names:
            for stamp, buffer, fmt in zip(
                batch.column("video_time").cast("int64").to_pylist(),
                batch.column("SegmentationImage:buffer").to_pylist(),
                batch.column("SegmentationImage:format").to_pylist(),
                strict=True,
            ):
                mask = np.asarray(buffer[0], dtype=np.uint8).reshape(fmt[0]["height"], fmt[0]["width"])
                found.setdefault(chunk.entity_path, []).append((stamp, int(mask.sum())))
    assert sorted(found) == [f"/world/rig_00/cam_{cam:02d}/pinhole/people_mask" for cam in range(2, 12)]
    for entity, rows in found.items():
        assert sorted(stamp for stamp, _ in rows) == STAMPS, entity
        front: bool = entity.endswith(("cam_02/pinhole/people_mask", "cam_07/pinhole/people_mask"))
        assert all((count > 0) == front for _, count in rows), entity


def test_a_face_stamp_without_a_lens_mask_is_an_error(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import Config, main

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces, base)
    write_masks(masks, STAMPS[:1])
    with pytest.raises(ValueError, match="face stamps have no mask"):
        main(Config(base=base, masks=masks, faces=faces, output=output))
    assert not output.exists()
