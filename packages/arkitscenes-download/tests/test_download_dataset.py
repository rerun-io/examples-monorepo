"""Behavior checks for ARKitScenes download planning."""

import tempfile
import unittest
from pathlib import Path

from arkitscenes_download.download_dataset import (
    ALL_ASSETS,
    VideoMetadata,
    download_video,
    parse_content_length,
    plan_video_downloads,
)
from arkitscenes_download.schema import DEFAULT_ASSETS


class DownloadDatasetTest(unittest.TestCase):
    """Check download planning and HTTP size parsing."""

    def test_plan_video_downloads_returns_only_pending_applicable_assets(self) -> None:
        """Planning skips existing outputs and assets gated by metadata."""
        metadata: VideoMetadata = VideoMetadata("123", "7", "Training", "Up", False, False, True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_dir: Path = Path(temporary_directory)
            video_dir: Path = download_dir / "raw" / "Training" / "123"
            (video_dir / "lowres_depth").mkdir(parents=True)
            (video_dir / "123.mov").touch()

            plans = plan_video_downloads(metadata, ("mov", "lowres_depth", "confidence", "highres_depth", "annotation"), download_dir)

            self.assertEqual([plan.asset for plan in plans], ["confidence", "annotation"])
            self.assertEqual(plans[0].filename, "confidence.zip")
            self.assertEqual(plans[0].dst_path, video_dir / "confidence.zip")
            self.assertTrue(plans[0].is_zip)
            self.assertTrue(plans[0].url.endswith("/raw/Training/123/confidence.zip"))

    def test_default_assets_match_ingest_without_removing_explicit_highres_support(self) -> None:
        """Default downloads contain every ingest input while high-res depth remains explicitly requestable."""
        self.assertEqual(
            DEFAULT_ASSETS,
            (
                "mov",
                "annotation",
                "mesh",
                "lowres_wide.traj",
                "confidence",
                "lowres_depth",
                "lowres_wide_intrinsics",
                "ultrawide_intrinsics",
            ),
        )
        self.assertNotIn("highres_depth", DEFAULT_ASSETS)
        self.assertIn("highres_depth", ALL_ASSETS)

    def test_plan_video_downloads_deduplicates_repeated_assets(self) -> None:
        """A ZIP asset requested twice is planned exactly once."""
        metadata: VideoMetadata = VideoMetadata("123", "7", "Training", "Up", False, False, True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            plans = plan_video_downloads(metadata, ("confidence", "confidence"), Path(temporary_directory))

            self.assertEqual([plan.asset for plan in plans], ["confidence"])

    def test_download_video_skips_plans_already_satisfied_on_disk(self) -> None:
        """An asset extracted after planning is not re-downloaded or re-extracted."""
        metadata: VideoMetadata = VideoMetadata("123", "7", "Training", "Up", False, False, True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            download_dir: Path = Path(temporary_directory)
            plans = plan_video_downloads(metadata, ("confidence",), download_dir)
            # Simulate the extraction landing between planning and transfer.
            (download_dir / "raw" / "Training" / "123" / "confidence").mkdir(parents=True)

            download_video(metadata, plans, download_dir, keep_zip=False, include_point_clouds=False)  # would hit the network if not skipped

    def test_parse_content_length_uses_last_case_insensitive_header(self) -> None:
        """Redirect response headers contribute only their final content length."""
        headers: str = "HTTP/1.1 302 Found\r\nContent-Length: 12\r\n\r\nHTTP/2 200\r\ncontent-length: 345\r\n"

        self.assertEqual(parse_content_length(headers), 345)
        self.assertIsNone(parse_content_length("HTTP/2 200\r\ncontent-type: application/zip\r\n"))


if __name__ == "__main__":
    unittest.main()
