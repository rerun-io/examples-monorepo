"""The lens-mask to face-mask tool on synthetic layers: stamps, geometry read back from the cube layer, and a missing mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rerun as rr
import rerun.chunk as rrc
from conftest import read_back, recording_properties
from jaxtyping import Float64, UInt8
from numpy import ndarray
from simplecv.run_record import RunStart, start_run


def _run() -> RunStart:
    return start_run(None, source=Path(__file__))

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


def write_masks(path: Path, stamps: list[int], factor: int = 1, scale: float | None = None) -> None:
    """Lens 0: a disc on its optical axis. Lens 1: a disc 80 deg toward its +X (the cube's -X: ``cube6_left``).

    ``factor`` > 1 writes each disc at 1/factor of the lens size (every factor-th pixel) with a static
    ``Transform3D(scale=[scale, scale, 1])`` (mask contract v2; ``scale`` defaults to ``factor``); ``factor`` = 1 writes it full size
    with no transform.
    """
    import insta360_io

    centre: float = (LENS - 1) / 2
    camera = insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [2.315, np.deg2rad(100.0)] + [0.0] * 13)
    side, _ = camera.project(np.array([[np.sin(np.deg2rad(80.0)), 0.0, np.cos(np.deg2rad(80.0))]]))
    u, v = np.meshgrid(np.arange(LENS), np.arange(LENS))
    discs: list[UInt8[ndarray, "h w"]] = [
        (np.hypot(u - centre, v - centre) <= 20).astype(np.uint8),
        (np.hypot(u - side[0, 0], v - side[0, 1]) <= 8).astype(np.uint8),
    ]
    discs = [disc[factor // 2 :: factor, factor // 2 :: factor] for disc in discs]
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(path)
        for lens in range(2):  # what posekit's mask layers log on each lens mask
            recording.log(f"/world/rig_00/cam_0{lens}/pinhole/people_mask", rr.AnnotationContext([(0, "kept", (255, 255, 255, 0)), (1, "masked", (40, 110, 255, 140))]), static=True)
        if factor > 1:
            for lens in range(2):
                s: float = float(factor) if scale is None else scale
                recording.log(f"/world/rig_00/cam_0{lens}/pinhole/people_mask", rr.Transform3D(scale=[s, s, 1.0]), static=True)
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
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, [*STAMPS, 999_000_000])
    write_face_masks(base, masks, faces, output, _run())
    found = read_masks(output)
    assert sorted(found) == [f"/world/rig_00/cam_{cam:02d}/pinhole/people_mask" for cam in range(2, 8)]
    record: dict[str, object] = recording_properties(read_back(output), "run_masks")
    assert record["layer"] == "masks" and record["segment"] == rrc.RrdReader(base).recordings()[0].recording_id
    hit: dict[str, bool] = {"cam_02": True, "cam_03": False, "cam_04": True, "cam_05": False, "cam_06": False, "cam_07": False}
    for entity, rows in found.items():
        assert sorted(stamp for stamp, _ in rows) == STAMPS, entity
        assert all((mask.sum() > 0) == hit[entity.split("/")[3]] for _, mask in rows), entity
        assert all(mask.shape == (FACE, FACE) and set(np.unique(mask)) <= {0, 1} for _, mask in rows)
    # Lens 1's disc sits on the lens-1 half of cube6_left: the image's left half (image right points to +Z0).
    left: UInt8[ndarray, "n n"] = found["/world/rig_00/cam_04/pinhole/people_mask"][0][1]
    assert left[:, : FACE // 2].sum() > 0 and left[:, FACE // 2 :].sum() == 0


def test_each_face_mask_draws_masked_pixels_as_a_tint_over_a_clear_face(tmp_path: Path) -> None:
    """The face masks carry the lens masks' own annotation context (posekit's): class 0 transparent, class 1 a blue tint."""
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS)
    write_face_masks(base, masks, faces, output, _run())
    contexts: dict[str, list[dict[str, Any]]] = {}
    for chunk in rrc.RrdReader(output).store().stream().filter(is_static=True):
        batch = chunk.to_record_batch()
        if "AnnotationContext:context" in batch.schema.names:
            contexts[str(chunk.entity_path)] = batch.column("AnnotationContext:context").to_pylist()[0][0]
    assert sorted(contexts) == [f"/world/rig_00/cam_{cam:02d}/pinhole/people_mask" for cam in range(2, 8)]
    classes = {entry["class_id"]: entry["class_description"]["info"] for entry in contexts["/world/rig_00/cam_02/pinhole/people_mask"]}
    assert classes[0]["color"] & 255 == 0 and classes[1]["color"] & 255 == 140


def test_the_face_mask_layer_joins_the_base_recording_whatever_the_file_is_called(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("renamed-base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS)
    write_face_masks(base, masks, faces, output, _run())
    assert [entry.recording_id for entry in rrc.RrdReader(output).recordings()] == ["insta360__clip"]


def test_a_face_stamp_without_a_mask_on_both_lenses_is_an_error(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS[:1])
    with pytest.raises(ValueError, match="no mask on both lenses"):
        write_face_masks(base, masks, faces, output, _run())
    assert not output.exists()


def test_a_quarter_size_mask_with_its_scale_joins_like_its_nearest_upsample(tmp_path: Path) -> None:
    """Contract v2: a mask at 1/4 of the lens size with a static scale of 4 gives the faces of the same mask repeated 4x4."""
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, small, full, out_small, out_full = (tmp_path / f"{n}.rrd" for n in ("base", "faces", "small", "full", "o1", "o2"))
    write_base(base)
    write_faces(faces)
    write_masks(small, STAMPS, factor=4)
    # The full-size reference: each quarter-size pixel repeated 4x4 (the contract's lookup for an integer scale).
    with rr.RecordingStream("test", recording_id="insta360__clip") as recording:
        recording.save(full)
        for stamp in STAMPS:
            recording.set_time("video_time", duration=np.timedelta64(stamp, "ns"))
            for entity, rows in read_masks(small).items():
                mask = next(m for t, m in rows if t == stamp)
                recording.log(entity, rr.SegmentationImage(np.repeat(np.repeat(mask, 4, axis=0), 4, axis=1)))
    write_face_masks(base, small, faces, out_small, _run())
    write_face_masks(base, full, faces, out_full, _run())
    got, want = read_masks(out_small), read_masks(out_full)
    assert sorted(got) == sorted(want)
    for entity in want:
        assert all(
            np.array_equal(a, b)
            for (_, a), (_, b) in zip(sorted(got[entity], key=lambda r: r[0]), sorted(want[entity], key=lambda r: r[0]), strict=True)
        )
    assert any(m.sum() > 0 for rows in got.values() for _, m in rows)


def test_a_scale_that_does_not_reach_the_lens_size_is_an_error(tmp_path: Path) -> None:
    from dataforge.apis.insta360_face_masks import write_face_masks

    base, faces, masks, output = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "out"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS, factor=4, scale=3.0)
    with pytest.raises(ValueError, match="scale"):
        write_face_masks(base, masks, faces, output, _run())


@pytest.mark.integration
def test_the_catalog_names_the_input_layers_and_gets_the_face_mask_layer(tmp_path: Path) -> None:
    """From the catalog: the files of base, cube and lens_mask; the masks equal the file route's; the layer is registered."""
    from rerun.catalog import CatalogClient
    from simplecv.catalog_layers import ensure_dataset, register_layer, registered_layer_files

    from dataforge.apis.insta360_face_masks import Config, main, write_face_masks

    base, faces, masks, by_file = (tmp_path / f"{name}.rrd" for name in ("base", "faces", "masks", "by-file"))
    write_base(base)
    write_faces(faces)
    write_masks(masks, STAMPS)
    write_face_masks(base, masks, faces, by_file, _run())
    with rr.server.Server() as server:
        entry = ensure_dataset(server.client(), "insta360")
        register_layer(entry, [base], None)
        register_layer(entry, [faces], "cube")
        register_layer(entry, [masks], "lens_mask")
        main(Config(segment="insta360__clip", catalog_url=server.url(), output_dir=tmp_path / "out"))
        registered = registered_layer_files(CatalogClient(server.url()).get_dataset(name="insta360"), "insta360__clip")
        # An output directory holding another layer's file under the segment's name would replace it: the base (a link
        # to it), or a later step's layer such as slam_rs.
        same: Path = tmp_path / "inputs" / "insta360__clip.rrd"
        same.parent.mkdir()
        same.hardlink_to(base)
        register_layer(entry, [same], None)
        with pytest.raises(ValueError, match="file of layer base"):
            main(Config(segment="insta360__clip", catalog_url=server.url(), output_dir=same.parent))
        slam: Path = tmp_path / "slam_rs" / "insta360__clip.rrd"
        slam.parent.mkdir()
        slam.write_bytes(faces.read_bytes())
        register_layer(entry, [slam], "slam_rs")
        with pytest.raises(ValueError, match="file of layer slam_rs"):
            main(Config(segment="insta360__clip", catalog_url=server.url(), output_dir=slam.parent))
        assert slam.read_bytes() == faces.read_bytes()
    output: Path = tmp_path / "out" / "insta360__clip.rrd"
    assert registered["masks"] == output.resolve()
    by_catalog = read_masks(output)
    assert sorted(by_catalog) == sorted(read_masks(by_file))
    for entity, rows in read_masks(by_file).items():
        assert [(stamp, mask.tobytes()) for stamp, mask in rows] == [(stamp, mask.tobytes()) for stamp, mask in by_catalog[entity]]
