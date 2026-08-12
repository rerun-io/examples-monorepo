"""Behavior check for catalog registration progress."""

import io
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
from rerun.catalog import CatalogClient, DatasetEntry
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from arkitscenes_download.ingest import catalog
from arkitscenes_download.ingest.catalog import Config, register_layer


class CatalogProgressTest(unittest.TestCase):
    """Check progress after a dropped registration call."""

    def test_server_error_is_not_misreported_as_a_dropped_connection(self) -> None:
        """Fatal server errors surface immediately instead of entering the polling loop."""
        dataset_mock: MagicMock = MagicMock(spec=DatasetEntry)
        dataset_mock.register.return_value.wait.side_effect = RuntimeError(
            "/RegisterWithDataset failed (Internal): IO error: Too many open files (os error 24)"
        )
        dataset_mock.segment_table.return_value.df.to_pandas.return_value = pd.DataFrame(
            {"rerun_segment_id": ["100"], "rerun_layer_names": [[]]}
        )
        dataset: DatasetEntry = cast(DatasetEntry, dataset_mock)
        client_mock: MagicMock = MagicMock(spec=CatalogClient)
        client_mock.get_dataset.return_value = dataset

        progress: Progress = Progress()
        with (
            patch.object(catalog, "CatalogClient", return_value=client_mock),
            patch.object(catalog.time, "sleep") as sleep_mock,
            progress,
            self.assertRaisesRegex(RuntimeError, "Too many open files"),
        ):
            task_id: TaskID = progress.add_task("gt_boxes", total=None)
            register_layer(Config(), "gt_boxes", [Path("/tmp/rrd/gt_boxes/100.rrd")], progress, task_id)

        sleep_mock.assert_not_called()

    def test_polling_reports_server_completion_without_a_false_percentage(self) -> None:
        """Opaque server work remains indeterminate while observed counts remain visible."""
        dataset_mock: MagicMock = MagicMock(spec=DatasetEntry)
        dataset_mock.register.return_value.wait.side_effect = ConnectionError("service unavailable")
        tables: list[pd.DataFrame] = [
            pd.DataFrame({"rerun_segment_id": ["100"], "rerun_layer_names": [[]]}),
            pd.DataFrame({"rerun_segment_id": ["100"], "rerun_layer_names": [["base"]]}),
        ]
        dataset_mock.segment_table.return_value.df.to_pandas.side_effect = tables
        dataset: DatasetEntry = cast(DatasetEntry, dataset_mock)
        client_mock: MagicMock = MagicMock(spec=CatalogClient)
        client_mock.get_dataset.return_value = dataset

        output: io.StringIO = io.StringIO()
        console: Console = Console(file=output, force_terminal=True, color_system=None, width=80)
        progress: Progress = Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console)
        with (
            patch.object(catalog, "CatalogClient", return_value=client_mock),
            patch.object(catalog, "CONSOLE", console),
            patch.object(catalog.time, "sleep"),
            patch.object(catalog.time, "perf_counter", side_effect=[0.0, 0.0, 1.0, 200.3]),
            progress,
        ):
            task_id: TaskID = progress.add_task("base · 1 file · submitting", total=None)
            register_layer(Config(), "base", [Path("/tmp/rrd/base/100.rrd")], progress, task_id)

        rendered: str = output.getvalue()
        self.assertIn("base · 1/1 registered", rendered)
        self.assertIn("base: 1 files in 200.3s", rendered)
        self.assertNotIn("%", rendered)

    def test_completeness_requires_ingest_layers_and_reports_optional_gt_coverage(self) -> None:
        """Optional CA-1M layers count as coverage without making uncovered segments incomplete."""
        required_layers: list[str] = [
            "base",
            "calibration",
            "video_wide",
            "video_ultrawide",
            "arkit_depth",
            "imu",
            "arkit_mesh",
            "gt_boxes",
        ]
        dataset_mock: MagicMock = MagicMock(spec=DatasetEntry)
        dataset_mock.segment_table.return_value.df.to_pandas.return_value = pd.DataFrame(
            {
                "rerun_segment_id": ["100", "200"],
                "rerun_layer_names": [required_layers, [*required_layers, "gt_poses", "gt_depth"]],
            }
        )
        dataset: DatasetEntry = cast(DatasetEntry, dataset_mock)
        output: io.StringIO = io.StringIO()
        console: Console = Console(file=output, force_terminal=False, color_system=None, width=120)

        with patch.object(catalog, "register_sequences", return_value=dataset), patch.object(catalog, "CONSOLE", console):
            catalog.main(Config())

        self.assertIn("2 segments, 2 complete, 1 gt-covered", output.getvalue())

    def test_layer_discovery_includes_present_optional_gt_layers(self) -> None:
        """Layer-major discovery returns optional CA-1M files without requiring them."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            rrd_dir: Path = Path(temporary_directory)
            for layer_name in ("base", "gt_depth"):
                layer_dir: Path = rrd_dir / layer_name
                layer_dir.mkdir()
                (layer_dir / "100.rrd").touch()

            files_by_layer: dict[str, list[Path]] = catalog.layer_files(Config(rrd_dir=rrd_dir))

        self.assertEqual(
            tuple(files_by_layer),
            (
                "base",
                "calibration",
                "video_wide",
                "video_ultrawide",
                "arkit_depth",
                "imu",
                "arkit_mesh",
                "gt_boxes",
                "gt_poses",
                "gt_depth",
            ),
        )
        self.assertEqual([path.name for path in files_by_layer["gt_depth"]], ["100.rrd"])
        self.assertEqual(files_by_layer["gt_poses"], [])


if __name__ == "__main__":
    unittest.main()
