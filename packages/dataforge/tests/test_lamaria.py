"""LaMAria: the manifest, discovery, remote index resolution, blueprints, and convert.

Nothing here touches the network or a VRS. ``download`` runs against a threaded
``http.server`` serving verbatim Apache index pages, and ``convert`` runs against
the ``open_streams`` seam, so the orchestration (temp mp4s, deletion vs
``--keep-raw``, capture properties) is exercised with synthetic frames while the
real encoder and the real writers do their jobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import serde.json

from dataforge import paths
from dataforge.datasets import dataset_defaults
from dataforge.datasets.lamaria import (
    DEFAULT_SEQUENCES,
    LamariaConfig,
    LamariaDataset,
    LamariaManifest,
    LamariaSource,
    SequenceRecord,
)
from dataforge.identity import SequenceIdentity

REFERENCE_DIR: Path = Path(__file__).parent / "reference_data" / "lamaria"
"""Verbatim excerpts of published LaMAria files, shared with ``test_aria.py``."""


# ── config and registration ───────────────────────────────────────────────


def test_lamaria_is_registered_under_its_own_command() -> None:
    assert dataset_defaults["lamaria"].command == "lamaria"
    # The catalog dataset is the command: one Aria layout serves every sequence.
    assert dataset_defaults["lamaria"].name == "lamaria"


def test_the_default_selection_is_the_five_surveyed_training_sequences() -> None:
    config: LamariaConfig = LamariaConfig()
    assert config.sequences == DEFAULT_SEQUENCES
    assert config.sequences == ("R_01_easy", "R_04_medium", "R_11_5cp", "sequence_1_19", "sequence_4_11")
    assert config.root == paths.raw_root() / "lamaria"
    assert config.base_url == "https://cvg-data.inf.ethz.ch/lamaria/"
    assert config.keep_raw is False


# ── the manifest ──────────────────────────────────────────────────────────


def manifest_fixture(base_url: str = "https://cvg-data.inf.ethz.ch/lamaria/") -> LamariaManifest:
    """A two-sequence manifest covering both GT shapes and both splits."""
    return LamariaManifest(
        base_url=base_url,
        sequences=[
            SequenceRecord(
                sequence="R_01_easy",
                split="training",
                vrs_url=f"{base_url}raw_data/training/R_01_easy.vrs",
                vrs_display_bytes=940_572_672,
                has_pseudo_gt=True,
                has_control_points=False,
            ),
            SequenceRecord(
                sequence="R_11_5cp",
                split="training",
                vrs_url=f"{base_url}raw_data/training/R_11_5cp.vrs",
                vrs_display_bytes=2_791_728_742,
                has_pseudo_gt=True,
                has_control_points=True,
            ),
        ],
    )


def test_the_manifest_round_trips_through_json(tmp_path: Path) -> None:
    written: LamariaManifest = manifest_fixture()
    path: Path = tmp_path / "manifest.json"
    path.write_text(serde.json.to_json(written))
    assert serde.json.from_json(LamariaManifest, path.read_text()) == written


# ── discovery ─────────────────────────────────────────────────────────────


def small_files(root: Path, record: SequenceRecord) -> None:
    """Touch the files ``download`` leaves on disk for one sequence."""
    sequence_dir: Path = root / record.split / record.sequence
    (sequence_dir / "aria_calibrations").mkdir(parents=True, exist_ok=True)
    (sequence_dir / "aria_calibrations" / f"{record.sequence}.json").write_text("{}")
    if record.has_pseudo_gt:
        (sequence_dir / "ground_truth" / "pGT").mkdir(parents=True, exist_ok=True)
        (sequence_dir / "ground_truth" / "pGT" / f"{record.sequence}.txt").write_text("")
    if record.has_control_points:
        (sequence_dir / "ground_truth" / "control_points").mkdir(parents=True, exist_ok=True)
        (sequence_dir / "ground_truth" / "control_points" / f"{record.sequence}.json").write_text("{}")


def downloaded_root(tmp_path: Path, *, manifest: LamariaManifest, complete: tuple[str, ...]) -> Path:
    """A raw root holding ``manifest`` plus the small files of the named sequences."""
    root: Path = tmp_path / "raw"
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(serde.json.to_json(manifest))
    for record in manifest.sequences:
        if record.sequence in complete:
            small_files(root, record)
    return root


def test_discover_without_a_manifest_names_the_download_verb(tmp_path: Path) -> None:
    dataset: LamariaDataset = LamariaDataset(LamariaConfig(root=tmp_path / "raw"))
    with pytest.raises(FileNotFoundError, match="dataforge-download lamaria"):
        dataset.discover()


def test_discover_pairs_each_selected_sequence_with_its_local_paths(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy", "R_11_5cp"))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_11_5cp", "R_01_easy"))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    # Sorted by name whatever order --sequences named them in.
    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy", "R_11_5cp"]
    assert discovered[0][0].recording_id == "lamaria__R_01_easy"
    easy: LamariaSource = discovered[0][1]
    assert easy.split == "training"
    assert easy.vrs_path == root / "training" / "R_01_easy" / "raw_data" / "R_01_easy.vrs"
    assert easy.calibration_path == root / "training" / "R_01_easy" / "aria_calibrations" / "R_01_easy.json"
    assert easy.pseudo_gt_path == root / "training" / "R_01_easy" / "ground_truth" / "pGT" / "R_01_easy.txt"
    assert easy.control_points_path is None, "R_01_easy has no surveyed control points"
    assert discovered[1][1].control_points_path == root / "training" / "R_11_5cp" / "ground_truth" / "control_points" / "R_11_5cp.json"


def test_discover_skips_a_sequence_whose_small_files_are_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy",))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_11_5cp"))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy"]
    # Silence would look like the sequence was never selected.
    assert "R_11_5cp" in capsys.readouterr().out


def test_discover_ignores_a_manifest_sequence_the_config_did_not_select(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy", "R_11_5cp"))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy",))

    discovered: list[tuple[SequenceIdentity, LamariaSource]] = LamariaDataset(config).discover()

    assert [identity.sequence_key for identity, _ in discovered] == ["R_01_easy"]


def test_a_selected_sequence_the_manifest_never_saw_is_an_error(tmp_path: Path) -> None:
    manifest: LamariaManifest = manifest_fixture()
    root: Path = downloaded_root(tmp_path, manifest=manifest, complete=("R_01_easy",))
    config: LamariaConfig = LamariaConfig(root=root, sequences=("R_01_easy", "R_99_nonesuch"))

    with pytest.raises(ValueError, match="R_99_nonesuch"):
        LamariaDataset(config).discover()
