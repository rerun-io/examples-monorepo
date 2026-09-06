"""The archive readers: a plain zip in-process, and an Info-ZIP volume set through 7-Zip.

The tree written here is deliberately not MSD-shaped — the reader knows nothing
about the dataset it serves — but the frames are noisy on purpose, because a
gradient compresses to a couple of kilobytes and the split fixture has to really
span volumes for the test to mean anything.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from conftest import png_frame

from dataforge.archives import group_archives, open_member_reader

TOP: str = "capture-01"
"""Top-level directory inside every archive below, as a sequence archive has one."""
FRAME_WIDTH: int = 192
"""Frame width; wide enough that a noisy PNG is worth several kilobytes."""
FRAME_HEIGHT: int = 160


def frame_bytes(index: int) -> bytes:
    """One noisy grayscale PNG, so the archive is genuinely incompressible."""
    return png_frame(index, width=FRAME_WIDTH, height=FRAME_HEIGHT, noisy=True)


def member_tree(root: Path, *, num_directories: int, num_frames: int) -> None:
    """Write ``<TOP>/data<N>/index.csv`` plus its frames, for ``num_directories`` groups."""
    for group in range(num_directories):
        data_dir: Path = root / TOP / f"data{group}" / "frames"
        data_dir.mkdir(parents=True, exist_ok=True)
        names: list[str] = []
        for index in range(num_frames):
            name: str = f"{group}-{index:05d}.png"
            (data_dir / name).write_bytes(frame_bytes(index))
            names.append(name)
        (data_dir.parent / "index.csv").write_text("filename\n" + "\n".join(names) + "\n")


def frame_members(root: Path, group: int) -> list[str]:
    """Archive-relative member names of one group's frames, in index order."""
    index_csv: Path = root / TOP / f"data{group}" / "index.csv"
    names: list[str] = index_csv.read_text().splitlines()[1:]
    return [f"{TOP}/data{group}/frames/{name}" for name in names]


def test_group_archives_orders_volumes_parts_first_and_drops_non_archives() -> None:
    """7-Zip wants the closing ``.zip`` named last, and ``.zip`` sorts after ``.z01``."""
    grouped: dict[str, list[tuple[str, int]]] = group_archives(
        [("d/b.zip", 4), ("d/b.z02", 2), ("d/b.z01", 1), ("d/a.zip", 8), ("d/README.md", 5)]
    )

    assert list(grouped) == ["a", "b"]
    assert [path for path, _ in grouped["b"]] == ["d/b.z01", "d/b.z02", "d/b.zip"]


def test_plain_zip_reader_serves_csvs_and_frames_in_order(tmp_path: Path) -> None:
    tree: Path = tmp_path / "tree"
    member_tree(tree, num_directories=2, num_frames=4)
    archive: Path = tmp_path / f"{TOP}.zip"
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=tree)

    with open_member_reader([archive], tmp_path / "work") as reader:
        assert reader.csv_bytes(f"{TOP}/data1/index.csv").startswith(b"filename")
        frames: list[bytes] = list(reader.png_frames(frame_members(tree, 1)))
    assert len(frames) == 4
    assert all(frame.startswith(b"\x89PNG") for frame in frames)
    assert frames == [frame_bytes(index) for index in range(4)]


@pytest.mark.skipif(shutil.which("zip") is None, reason="needs Info-ZIP's zip to build a multi-volume fixture")
def test_split_archive_reader_extracts_one_directory_at_a_time(tmp_path: Path) -> None:
    """Python's zipfile cannot read a spanned archive at all, so 7-Zip does."""
    tree: Path = tmp_path / "tree"
    member_tree(tree, num_directories=2, num_frames=12)
    subprocess.run(["zip", "-q", "-0", "-r", "-s", "64k", str(tmp_path / f"{TOP}.zip"), TOP], cwd=tree, check=True)
    volumes: list[Path] = sorted(tmp_path.glob(f"{TOP}.z*"))
    assert len(volumes) > 1, "the fixture must really be split for this test to mean anything"
    members: list[str] = frame_members(tree, 0)
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(tmp_path / f"{TOP}.zip").read(members[0])

    work: Path = tmp_path / "work"
    with open_member_reader(volumes, work) as reader:
        frames: list[bytes] = list(reader.png_frames(members))
    assert frames == [frame_bytes(index) for index in range(12)]
    # The extracted PNGs are gone again; peak scratch is one directory, not the archive.
    assert not list(work.rglob("*.png"))
