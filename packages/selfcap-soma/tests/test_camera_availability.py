import pyarrow as pa

from selfcap_soma.evaluation_io import missing_camera_calibration


def test_flagged_camera_with_intrinsics_but_no_extrinsics_is_unavailable():
    camera = "/world/rig_02/cam_00"
    table = pa.table(
        {
            camera + "/pinhole:Pinhole:image_from_camera": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]],
            camera + ":Transform3D:translation": pa.array([None], type=pa.list_(pa.float64())),
            camera + ":Transform3D:mat3x3": pa.array([None], type=pa.list_(pa.float64())),
        }
    )
    missing = missing_camera_calibration(table, camera)
    assert set(missing) == {camera + ":Transform3D:translation", camera + ":Transform3D:mat3x3"}


def test_valid_camera_and_rgb_identity_extrinsics_are_distinguished():
    camera = "/world/rig_00/cam_00"
    table = pa.table({camera + "/pinhole:Pinhole:image_from_camera": [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]})
    assert missing_camera_calibration(table, camera, extrinsics=False) == []
    assert len(missing_camera_calibration(table, camera)) == 2
