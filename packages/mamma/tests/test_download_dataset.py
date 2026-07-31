from pathlib import Path

import pytest

from mamma.apis.download_dataset import (
    build_collection_phases,
    credentials_from_environment,
    download_url,
    valid_download_size,
)
from mamma.apis.download_manifest import (
    DANCE_SEQUENCES,
    EVAL_DANCE_SEQUENCES,
    EVAL_EXTRA_SEQUENCES,
    EVAL_SINGLES_SEQUENCES,
    IPHONE_INDOOR_SEQUENCES,
    IPHONE_OUTDOOR_SEQUENCES,
    MULTI_PEOPLE_SEQUENCES,
)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (b"", False),
        (b"Error: File not found.", False),
        (b"<!DOCTYPE html><title>Login</title>", False),
        (b"<HTML><body>Access denied</body></HTML>", False),
        (b"\x00\x00\x00 ftypisom" + b"\x00" * 300, True),
        (b"x" * 256 + b"<html>outside inspected prefix</html>", True),
    ],
)
def test_download_file_validity_matches_mpi_script(tmp_path: Path, contents: bytes, expected: bool) -> None:
    download_path: Path = tmp_path / "download"
    download_path.write_bytes(contents)

    size: int | None = valid_download_size(download_path)
    assert (size is not None) is expected
    if expected:
        assert size == len(contents)


def test_manifest_sequence_counts_match_reference_scripts() -> None:
    assert (len(EVAL_SINGLES_SEQUENCES), len(EVAL_EXTRA_SEQUENCES), len(EVAL_DANCE_SEQUENCES)) == (22, 12, 18)
    assert sum(len(sequences) for sequences in DANCE_SEQUENCES.values()) == 123
    assert sum(len(sequences) for sequences in MULTI_PEOPLE_SEQUENCES.values()) == 34
    assert len(IPHONE_INDOOR_SEQUENCES) + len(IPHONE_OUTDOOR_SEQUENCES) == 42


def test_mpi_url_preserves_dataset_relative_path() -> None:
    relative_path: Path = Path("mamma_eval_singles/230929_WhiteRabbit_CatchBall_50048_1/gt/IOI_01.npz")

    assert download_url(relative_path) == (
        "https://download.is.tue.mpg.de/download.php?domain=mamma&resume=1"
        "&sfile=datasets/mamma_eval_singles/230929_WhiteRabbit_CatchBall_50048_1/gt/IOI_01.npz"
    )


def test_collection_phase_paths_match_reference_scripts() -> None:
    eval_files: set[Path] = {path for phase in build_collection_phases("eval") for path in phase.files}
    assert Path(f"{EVAL_SINGLES_SEQUENCES[0]}/gt/global.npz") in eval_files
    assert Path(f"{EVAL_SINGLES_SEQUENCES[0]}/masks/IOI_01_masks.tar") in eval_files
    assert Path(f"{EVAL_EXTRA_SEQUENCES[0]}/markers/vicon_m37.npy") in eval_files
    assert Path(f"{EVAL_EXTRA_SEQUENCES[0]}/gt/IOI_16.npz") in eval_files
    assert Path(f"{EVAL_EXTRA_SEQUENCES[0]}/gt/IOI_17.npz") not in eval_files
    assert Path(f"{EVAL_DANCE_SEQUENCES[0]}/videos_crf16/IOI_32.mp4") in eval_files

    dance_pair: str = "mamma_markerless_dance/140725_Breakdance_Improv_1_03684_03686_1"
    dance_files: set[Path] = {path for phase in build_collection_phases("dance") for path in phase.files}
    assert Path(f"{dance_pair}/pred/params_01.npz") in dance_files

    multi_six: str = MULTI_PEOPLE_SEQUENCES[6][0]
    multi_files: set[Path] = {path for phase in build_collection_phases("multi_people") for path in phase.files}
    assert Path(f"{multi_six}/pred/params_05.npz") in multi_files

    iphone_two: str = IPHONE_INDOOR_SEQUENCES[2][1]
    iphone_files: set[Path] = {path for phase in build_collection_phases("iphone") for path in phase.files}
    assert Path(f"{iphone_two}/pred/params_01.npz") in iphone_files
    assert Path(f"{iphone_two}/videos/D001.mp4") in iphone_files


def test_mpi_credentials_must_be_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAMMA_USERNAME", raising=False)
    monkeypatch.delenv("MAMMA_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="MAMMA_USERNAME and MAMMA_PASSWORD"):
        credentials_from_environment()
