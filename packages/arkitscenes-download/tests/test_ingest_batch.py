"""Behavior checks for concurrent batch ingestion."""

import tempfile
import unittest
from pathlib import Path

from arkitscenes_download.ingest.batch import SequenceResult, discover_video_ids, has_all_layers, summarize


class IngestBatchTest(unittest.TestCase):
    """Check batch selection and reporting behavior."""

    def test_autodiscovery_is_sorted_and_deduplicated_across_splits(self) -> None:
        """Sequence directories from both splits produce one sorted id list."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir: Path = Path(temporary_directory)
            for relative_path in (
                "raw/Training/200",
                "raw/Training/100",
                "raw/Validation/300",
                "raw/Validation/100",
            ):
                (data_dir / relative_path).mkdir(parents=True)
            (data_dir / "raw" / "Training" / "not-a-directory.txt").write_text("ignored")

            self.assertEqual(discover_video_ids(data_dir), ["100", "200", "300"])

    def test_summary_computes_aggregate_wall_time_and_effective_speedup(self) -> None:
        """Summary totals sequence walls and compares them with elapsed wall time."""
        results: list[SequenceResult] = [
            SequenceResult("100", 10.0, True, "done"),
            SequenceResult("200", 14.0, False, "failed"),
        ]

        summary = summarize(results, total_wall_seconds=8.0)

        self.assertEqual(summary.sum_sequence_wall_seconds, 24.0)
        self.assertEqual(summary.effective_speedup, 3.0)
        self.assertEqual(summary.failed_video_ids, ["200"])

    def test_complete_sequence_requires_all_eight_layers(self) -> None:
        """Skip-existing accepts only a sequence directory with the complete layer set."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence_dir: Path = Path(temporary_directory) / "100"
            sequence_dir.mkdir()
            layer_names: tuple[str, ...] = (
                "base",
                "calibration",
                "video_wide",
                "video_ultrawide",
                "arkit_depth",
                "imu",
                "arkit_mesh",
                "gt_boxes",
            )
            for layer_name in layer_names:
                (sequence_dir / f"{layer_name}.rrd").touch()

            self.assertTrue(has_all_layers(Path(temporary_directory), "100"))
            (sequence_dir / "gt_boxes.rrd").unlink()
            self.assertFalse(has_all_layers(Path(temporary_directory), "100"))


if __name__ == "__main__":
    unittest.main()
