"""Behavior checks for atomic RRD publication."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rerun as rr

from arkitscenes_download.ingest.cli import atomic_recording


class _FakeRecording:
    """Minimal context-managed recording sink used to inject a late failure."""

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def save(self, path: Path, **_kwargs: object) -> None:
        path.write_bytes(b"partial")


class AtomicPublicationTest(unittest.TestCase):
    """A failed logging phase cannot replace an existing valid recording."""

    def test_late_failure_preserves_target_and_removes_partial(self) -> None:
        """An exception after opening the sink leaves no publication debris."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "sequence.rrd"
            target.write_bytes(b"known-good")
            with (
                patch("arkitscenes_download.ingest.cli.rr.RecordingStream", return_value=_FakeRecording()),
                self.assertRaisesRegex(RuntimeError, "late"),
                atomic_recording(target, "sequence", send_properties=False),
            ):
                raise RuntimeError("late logging failure")
            self.assertEqual(target.read_bytes(), b"known-good")
            self.assertEqual(list(Path(directory).glob("*.partial.*")), [])

    def test_non_base_recording_disables_property_chunk(self) -> None:
        """A non-base layer constructs its stream without recording properties."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "calibration.rrd"
            with (
                patch("arkitscenes_download.ingest.cli.rr.RecordingStream", return_value=_FakeRecording()) as recording_stream,
                atomic_recording(target, "sequence", send_properties=False),
            ):
                pass

            recording_stream.assert_called_once_with(application_id="arkitscenes", recording_id="sequence", send_properties=False)

    def test_real_non_base_rrd_has_no_properties_entity(self) -> None:
        """Serialized non-base layers contain no colliding /__properties chunk."""
        with tempfile.TemporaryDirectory() as directory:
            target: Path = Path(directory) / "calibration.rrd"
            with atomic_recording(target, "sequence", send_properties=False) as recording:
                recording.log("world/calibration", rr.AnyValues(source="fixture"), static=True)

            printed: subprocess.CompletedProcess[str] = subprocess.run(
                ["rerun", "rrd", "print", str(target)], check=True, capture_output=True, text=True
            )
            self.assertNotIn("/__properties", printed.stdout)


if __name__ == "__main__":
    unittest.main()
