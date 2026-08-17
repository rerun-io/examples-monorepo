"""Behavior checks for stream-10 camera distortion metadata."""

import unittest
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import RrdReader
from simplecv.camera_parameters import Intrinsics as SimpleCVIntrinsics

from arkitscenes_download.ingest.cli import _log_distortions
from arkitscenes_download.ingest.distortion import (
    APPLE_FORWARD_DISTORTION_COMPONENT,
    APPLE_INVERSE_DISTORTION_COMPONENT,
)
from arkitscenes_download.ingest.metadata import CameraDistortion, decode_camera_distortions
from arkitscenes_download.ingest.paths import PINHOLE_ULTRAWIDE


def test_calibration_rrd_logs_canonical_brown_conrady_and_apple_provenance(tmp_path: Path) -> None:
    """The pinhole gets standard coefficients while both exact Apple arrays remain separate."""
    forward_8: np.ndarray = np.array(
        [-0.08353952, -6.350463, 5.248554, -1.9897834, 0.5831521, -0.10259675, 0.009266047, -0.0003309832],
        dtype=np.float32,
    )
    inverse_8: np.ndarray = np.array(
        [0.06613547, 7.4000044, -8.394783, 5.206168, -1.9468352, 0.3531596, -0.028574973, 0.000752457],
        dtype=np.float32,
    )
    calibration = CameraDistortion(
        camera_name="ultrawide",
        coefficients=forward_8,
        inverse_coefficients=inverse_8,
        center_xy=np.array([1853.150634765625, 1371.03173828125], dtype=np.float64),
        reference_dimensions_wh=np.array([3680.0, 2760.0], dtype=np.float64),
        timestamp=0.0,
        temporal_sample_count=3,
    )
    pinhole_intrinsics = SimpleCVIntrinsics.from_k_matrix(
        camera_conventions="RDF",
        k_matrix=np.array(
            [[245.3910065, 0.0, 321.8739929], [0.0, 245.3910065, 238.0269928], [0.0, 0.0, 1.0]], dtype=np.float32
        ),
        width=640,
        height=480,
    )
    rrd_path: Path = tmp_path / "calibration.rrd"
    with rr.RecordingStream(application_id="distortion_test", recording_id="segment", send_properties=False) as recording:
        recording.save(rrd_path)
        _log_distortions(
            recording,
            [calibration],
            pinhole_intrinsics_by_camera={"ultrawide": pinhole_intrinsics},
            quarter_turns=0,
        )

    reader = RrdReader(rrd_path)
    batches = [chunk.to_record_batch() for chunk in reader.stream() if str(chunk.entity_path) == f"/{PINHOLE_ULTRAWIDE}"]
    values_by_component: dict[str, object] = {}
    for batch in batches:
        for name, column in zip(batch.schema.names, batch.columns, strict=True):
            if "Distortion" in name:
                values_by_component[name] = column.to_pylist()[0][0]

    assert values_by_component["simplecv.components.DistortionModel"] == "brown_conrady"
    standard_14: np.ndarray = np.asarray(values_by_component["simplecv.components.DistortionCoefficients"], dtype=np.float32)
    assert standard_14.shape == (14,)
    np.testing.assert_allclose(
        standard_14[:8],
        [0.14226905, -0.03264365, -1.7744861e-5, -1.9211448e-5, 0.00754465, 0.14195952, -0.04144133, 0.00897725],
        atol=3e-5,
        rtol=0.0,
    )
    np.testing.assert_array_equal(standard_14[8:], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(values_by_component[APPLE_FORWARD_DISTORTION_COMPONENT], forward_8)
    np.testing.assert_array_equal(values_by_component[APPLE_INVERSE_DISTORTION_COMPONENT], inverse_8)
    assert "simplecv.components.InverseDistortionCoefficients" not in values_by_component


class CameraDistortionTest(unittest.TestCase):
    """Validate real ARKit calibration payloads."""

    def test_real_movie_has_sane_radial_lookup_tables(self) -> None:
        """Decoded LUTs are nonempty, bounded, and mostly monotonic."""
        mov_path: Path = Path("data/raw/Training/47332195/47332195.mov")
        if not mov_path.is_file():
            self.skipTest("ARKitScenes sample MOV is unavailable")
        calibrations = decode_camera_distortions(mov_path)

        self.assertGreaterEqual(len(calibrations), 1)
        for calibration in calibrations:
            self.assertGreater(len(calibration.coefficients), 0)
            self.assertTrue(np.all(calibration.center_xy >= 0.0))
            self.assertTrue(np.all(calibration.center_xy <= calibration.reference_dimensions_wh))
            self.assertEqual(len(calibration.coefficients), 8)
            self.assertTrue(np.all(np.isfinite(calibration.coefficients)))
            self.assertGreater(calibration.temporal_sample_count, 1)


if __name__ == "__main__":
    unittest.main()
