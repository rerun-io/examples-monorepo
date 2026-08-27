"""WildCap discovery and video grouping against synthetic fixtures (no corpus needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataforge.datasets.wildcap import WildcapConfig, WildcapDataset, group_videos
from dataforge.identity import SequenceIdentity

UNREADABLE: int = 0o000
"""The mode a botched sync can leave on a file."""


def make_capture(capture_dir: Path, *, exo: tuple[str, ...] = ("cam-a", "cam-b"), ego: tuple[str, ...] = ("head",)) -> None:
    """Create a minimal WildCap capture: bare mp4s under ``exo/`` and ``ego/``.

    Args:
        capture_dir: Target ``<root>/<capture-name>`` directory.
        exo: File stems written as ``exo/<stem>.mp4``.
        ego: File stems written as ``ego/<stem>.mp4``.
    """
    for group, stems in (("exo", exo), ("ego", ego)):
        if not stems:
            continue
        (capture_dir / group).mkdir(parents=True)
        for stem in stems:
            (capture_dir / group / f"{stem}.mp4").touch()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A fake WildCap root: two convertible captures and two that must be skipped."""
    make_capture(tmp_path / "kitchen-01", exo=("iphone-1", "iphone-2"), ego=("rgb", "left"))
    make_capture(tmp_path / "garage-02", exo=("gopro",), ego=())
    make_capture(tmp_path / "empty-03", exo=(), ego=())  # no videos at all
    make_capture(tmp_path / "locked-04", exo=("iphone-1",), ego=())
    (tmp_path / "locked-04" / "exo" / "iphone-1.mp4").chmod(UNREADABLE)
    (tmp_path / "notes.txt").write_text("not a capture")
    return tmp_path


def test_discover_keeps_only_captures_with_a_readable_video(corpus: Path) -> None:
    dataset: WildcapDataset = WildcapDataset(WildcapConfig(root=corpus))
    discovered: list[tuple[SequenceIdentity, Path]] = dataset.discover()
    assert [identity.sequence_key for identity, _ in discovered] == ["garage-02", "kitchen-01"]
    assert dataset.sequences() == [identity for identity, _ in discovered]


def test_discover_pairs_each_identity_with_its_capture_directory(corpus: Path) -> None:
    dataset: WildcapDataset = WildcapDataset(WildcapConfig(root=corpus))
    pair: tuple[SequenceIdentity, Path] = dataset.discover()[1]
    assert pair[0].recording_id == "wildcap__kitchen-01"
    assert pair[1] == corpus / "kitchen-01"


def test_download_verifies_the_local_corpus(corpus: Path, tmp_path: Path) -> None:
    WildcapConfig(root=corpus).setup().download()
    empty_root: Path = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError, match="missing"):
        WildcapConfig(root=empty_root).setup().download()


def test_group_videos_sorts_each_group_by_stem(corpus: Path) -> None:
    capture_dir: Path = corpus / "kitchen-01"
    assert [video.stem for video in group_videos(capture_dir, "exo")] == ["iphone-1", "iphone-2"]
    assert [video.stem for video in group_videos(capture_dir, "ego")] == ["left", "rgb"]


def test_group_videos_returns_empty_for_a_missing_group(corpus: Path) -> None:
    assert group_videos(corpus / "garage-02", "ego") == []


def test_group_videos_skips_unreadable_videos(corpus: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert group_videos(corpus / "locked-04", "exo") == []
    assert "skipping" in capsys.readouterr().out


def test_group_videos_skips_mov_files_and_accepts_uppercase_mp4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # iPhones ship .MOV, which Rerun does not support; Pixels ship .mp4/.MP4.
    capture_dir: Path = tmp_path / "phones-01"
    (capture_dir / "exo").mkdir(parents=True)
    (capture_dir / "exo" / "IMG_0078.MOV").touch()
    (capture_dir / "exo" / "PXL_1234.MP4").touch()
    assert [video.stem for video in group_videos(capture_dir, "exo")] == ["PXL_1234"]
    assert "IMG_0078.MOV — Rerun does not support .mov" in capsys.readouterr().out


def test_default_blueprint_is_none_because_camera_counts_vary() -> None:
    assert WildcapConfig().setup().default_blueprint() is None
