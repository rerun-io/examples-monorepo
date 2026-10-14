"""EgoHumans (Khirodkar et al., ICCV 2023): multi-person activities seen by up to four Aria glasses and ~15 static GoPros.

The release is a Google Drive folder with one tar.gz per sequence (``<scene>/<sequence>.tar.gz``, 133 sequences, 598 GB);
``egohumans_download`` fetches it and each sequence's upstream config, which names the glasses and cameras the authors left
out and the cameras they posed by hand. A tar.gz cannot be read in place, so conversion extracts one sequence into the work
directory, converts it with the layers Harmony4D uses (``egohumans_writers``), and deletes the extraction. See ``docs/egohumans.md``.
"""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import ClassVar

import requests
import rerun.blueprint as rrb

from dataforge import paths, writing
from dataforge.archives import remove_tree
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.egohumans_download import (
    CONFIG_REPO_REVISION,
    DriveFile,
    config_path,
    fetch_config,
    fetch_drive_file,
    read_manifest,
)
from dataforge.egohumans_layout import FPS, CaptureConfig, DirectorySequenceFiles, read_sequence
from dataforge.egohumans_writers import EGO_RIG_OFFSET, LAYERS, Clock, capture_blueprint, capture_card, layer_writers
from dataforge.identity import SequenceIdentity
from dataforge.records import decode
from dataforge.smpl import fetch_smpl_neutral
from dataforge.transports import FetchReport

POSE_DIRS: tuple[str, ...] = ("fit_poses3d", "refine_poses3d", "poses3d")
"""COCO-17 keypoints: the release's benchmark is built from ``fit_poses3d`` (tools/create_benchmark), older takes may only ship
the earlier stages."""
EGO_PANES: tuple[int, ...] = tuple(EGO_RIG_OFFSET + number for number in (1, 2, 3, 4))
"""Rigs of aria01..aria04 (``egohumans_writers.ego_rig``)."""
EXO_PANES: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 13, 15)
"""Eight of the ~15 GoPros (MAMMA's eight-pane rule)."""
EXO_SLOTS: int = 20
"""Most GoPros a take can have; the 3D view's rules cover them all."""


@dataclass(frozen=True, slots=True)
class EgohumansSequence:
    """One downloaded sequence archive."""

    archive: Path
    """``<root>/<scene>/<sequence>.tar.gz``."""
    config: Path
    """Its upstream config, ``<root>/configs/<activity>/<sequence>.yaml``."""


def extract_sequence(archive: Path, name: str, destination: Path) -> Path:
    """Extract one release tar.gz into ``destination/<name>`` and return that directory.

    The archives keep the authors' absolute paths (``.../main/01_tagging/004_tagging/exo/...``; upstream strips 8 components),
    so every regular file is placed by the part of its path after the ``<name>`` component; members outside it, links and
    any path with ``..`` are skipped. The archive is untrusted: nothing is extracted by its own path.
    """
    root: Path = destination / name
    remove_tree(root)
    root.mkdir(parents=True)
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            parts: tuple[str, ...] = PurePosixPath(member.name).parts
            if not member.isfile() or name not in parts or ".." in parts:
                continue
            relative: tuple[str, ...] = parts[parts.index(name) + 1 :]
            if not relative:
                continue
            target: Path = root.joinpath(*relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = stream.extractfile(member)
            assert source is not None
            with target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 8 << 20)
    return root


@dataclass
class EgohumansConfig(FrameLimitedConfig):
    """The release as ``dataforge-download egohumans`` lays it out: ``<root>/<scene>/<sequence>.tar.gz`` and ``<root>/configs``."""

    command: ClassVar[str] = "egohumans"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: EgohumansDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "egohumans")
    """Raw root: the sequence archives and the upstream configs."""
    sequences: tuple[str, ...] | None = None
    """Scene keys (``01_tagging``) or sequence keys (``01_tagging/004_tagging``); None selects all."""
    smpl_model_root: Path | None = None
    """Official neutral SMPL root holding ``smpl/SMPL_NEUTRAL.pkl`` (for the mesh triangles); defaults to ``root/body_models``."""

    @property
    def smpl_root(self) -> Path:
        return self.smpl_model_root or self.root / "body_models"

    def selects(self, key: str) -> bool:
        """Whether ``sequences`` names this sequence or its scene."""
        return self.sequences is None or any(key == wanted or key.startswith(wanted + "/") for wanted in self.sequences)


class EgohumansDataset(DataforgeDataset[EgohumansConfig, EgohumansSequence]):
    """Download from Drive, then convert one extracted sequence at a time."""

    layers = LAYERS

    def selected(self) -> list[DriveFile]:
        files: list[DriveFile] = read_manifest().sequences
        known: set[str] = {file.key for file in files} | {file.key.split("/")[0] for file in files}
        unknown: list[str] = [key for key in self.config.sequences or () if key not in known]
        if unknown:
            raise ValueError(f"egohumans: no sequence or scene {unknown} in the release; see `dataforge-download egohumans --list-remote`")
        return [file for file in files if self.config.selects(file.key)]

    def remote_sequences(self) -> list[RemoteSequence]:
        """One entry per sequence archive, from the pinned folder listing (no request)."""
        return [RemoteSequence(file.key, file.size_bytes, (file.path, config_path(file.key))) for file in read_manifest().sequences]

    def download(self) -> None:
        """Fetch the selected archives (size checked, resumable) and their configs, then the SMPL model unless in place.

        A file Drive refuses (quota) is reported and the rest continue; the call fails at the end so a rerun picks it up.
        """
        report: FetchReport = FetchReport()
        refused: list[str] = []
        for file in self.selected():
            fetch_config(file.key, self.config.root)
            try:
                report.count(fetch_drive_file(file, self.config.root / file.path), file.size_bytes)
            except (RuntimeError, ValueError, requests.RequestException) as error:  # quota page, short transfer, dropped connection: all resume on the next run
                refused.append(str(error))
        fetch_smpl_neutral(self.config.root, self.config.smpl_root, explicit=self.config.smpl_model_root is not None)
        print(f"egohumans download: {report.summary()}")
        if refused:
            raise RuntimeError(f"egohumans download: {len(refused)} file(s) not fetched, rerun later:\n" + "\n".join(refused))

    def discover(self) -> list[tuple[SequenceIdentity, EgohumansSequence]]:
        """Every complete sequence archive under the root that has its config."""
        found: list[tuple[SequenceIdentity, EgohumansSequence]] = []
        for archive in sorted(self.config.root.glob("*/*.tar.gz")):
            key: str = f"{archive.parent.name}/{archive.name.removesuffix('.tar.gz')}"
            if not self.config.selects(key):
                continue
            config: Path = self.config.root / config_path(key)
            if not config.is_file():
                print(f"skip {key}: no upstream config at {config} (dataforge-download egohumans fetches it)")
                continue
            found.append((SequenceIdentity(self.config.name, tuple(key.split("/"))), EgohumansSequence(archive, config)))
        return found

    def convert(self, identity: SequenceIdentity, source: EgohumansSequence, *, force: bool) -> Path:
        """Extract the sequence into the work directory, write the four layers from it, delete the extraction."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        scene, sequence_name = identity.parts
        capture: CaptureConfig = decode(CaptureConfig, source.config.read_text(), source=str(source.config), fmt="yaml")
        work: Path = paths.work_root() / "egohumans"
        try:
            with self.timer.stage("extract"):
                directory: Path = extract_sequence(source.archive, sequence_name, work)
            files: DirectorySequenceFiles = DirectorySequenceFiles(directory)
            with self.timer.stage("read"):
                sequence = read_sequence(files, pose_dirs=POSE_DIRS, config=capture)
            clock: Clock = Clock.of(sequence, self.config.frame_limit)
            episode: dict[str, str] = dict(scene=scene, sequence=sequence_name, activity=scene.split("_", 1)[-1])
            revision: str = f"Google Drive folder (listing {read_manifest().listed}), configs rawalkhirodkar/egohumans@{CONFIG_REPO_REVISION}"
            writers = layer_writers(identity, files, sequence, clock, self.timer, self.config.smpl_root, source_revision=revision, episode=episode)
            self.write_layers(identity, targets, pending, writers)
        finally:
            remove_tree(work / sequence_name)
        self.timer.capture_s = len(clock.frames) / FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """Scene on top with the four glasses' rgb cameras beside it; eight GoPros along the bottom."""
        return capture_blueprint(EXO_SLOTS, EGO_PANES, EXO_PANES)

    def table_blueprint(self) -> rrb.Blueprint:
        return capture_card(EXO_SLOTS, EGO_PANES)

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("scene", "activity", "num_people")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
