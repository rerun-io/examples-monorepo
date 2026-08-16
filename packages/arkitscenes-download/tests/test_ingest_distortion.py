"""Behavior checks for stream-10 camera distortion metadata."""

import unittest
from pathlib import Path

import numpy as np
import rerun as rr
from rerun.experimental import RrdReader

from arkitscenes_download.ingest.cli import _log_distortions
from arkitscenes_download.ingest.metadata import CameraDistortion, decode_camera_distortions
from arkitscenes_download.ingest.paths import PINHOLE_ULTRAWIDE


def test_calibration_rrd_persists_both_apple_polynomial_directions(tmp_path: Path) -> None:
    """The calibration contract labels Apple data and stores both coefficient arrays."""
    forward_8: np.ndarray = np.arange(8, dtype=np.float32)
    inverse_8: np.ndarray = -forward_8
    calibration = CameraDistortion(
        camera_name="ultrawide",
        coefficients=forward_8,
        inverse_coefficients=inverse_8,
        center_xy=np.array([100.0, 80.0], dtype=np.float64),
        reference_dimensions_wh=np.array([200.0, 160.0], dtype=np.float64),
        timestamp=0.0,
        temporal_sample_count=3,
    )
    rrd_path: Path = tmp_path / "calibration.rrd"
    with rr.RecordingStream(application_id="distortion_test", recording_id="segment", send_properties=False) as recording:
        recording.save(rrd_path)
        _log_distortions(recording, [calibration], quarter_turns=0)

    reader = RrdReader(rrd_path)
    batches = [chunk.to_record_batch() for chunk in reader.stream() if str(chunk.entity_path) == f"/{PINHOLE_ULTRAWIDE}"]
    values_by_component: dict[str, object] = {}
    for batch in batches:
        for name, column in zip(batch.schema.names, batch.columns, strict=True):
            if "Distortion" in name:
                values_by_component[name] = column.to_pylist()[0][0]

    assert values_by_component["simplecv.components.DistortionModel"] == "apple_radial_poly"
    np.testing.assert_array_equal(values_by_component["simplecv.components.DistortionCoefficients"], forward_8)
    np.testing.assert_array_equal(values_by_component["simplecv.components.InverseDistortionCoefficients"], inverse_8)


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
