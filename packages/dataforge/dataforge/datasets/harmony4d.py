"""Harmony4D (Khirodkar, Song et al., NeurIPS 2024): close two-person interactions, ~20 static GoPros, optional Aria glasses.

The release is 22 zips on the Hugging Face Hub (``Jyun-Ting/Harmony4D``, MIT), one per scene and split, each holding several
sequences in the EgoHumans capture layout (``egohumans_layout``). Conversion reads the members straight from the zip; nothing is
extracted. Layers: base (cameras, AV1 video, glasses poses), body_pose (SMPL parameters, SMPL-45 joints, COCO-133 keypoints),
body_mesh (shipped SMPL vertices at 10 Hz) and projections (keypoints through every lens, shipped boxes). See ``docs/harmony4d.md``.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import ClassVar

import rerun.blueprint as rrb

from dataforge import paths, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.egohumans_layout import FPS, ZipSequenceFiles, read_sequence, sequences_in
from dataforge.egohumans_writers import EGO_RIG_OFFSET, LAYERS, Clock, capture_blueprint, capture_card, layer_writers
from dataforge.identity import SequenceIdentity
from dataforge.smpl import fetch_smpl_neutral
from dataforge.transports import FetchReport, HfFileInfo, hf_fetch_verified, hf_list_files

SOURCE_REPO: str = "Jyun-Ting/Harmony4D"
SOURCE_REVISION: str = "3fedb23fd9d1a92541d98ccbce025c695bd752e4"
"""The release as of 2024-11-09: 22 zips, 351.6 GB."""
EGO_PANES: tuple[int, ...] = (EGO_RIG_OFFSET + 1, EGO_RIG_OFFSET + 2)
"""Rigs of aria01 and aria02 (``egohumans_writers.ego_rig``); takes without glasses leave the panes empty."""
EXO_PANES: tuple[int, ...] = (1, 4, 6, 9, 12, 15, 17, 20)
"""Eight GoPros evenly spaced over cam01-cam20, which every take ships (only two 01_hugging takes add cam21/cam22); MAMMA's eight-pane rule."""
EXO_SLOTS: int = 30
"""Most GoPros a take can have; the 3D view's rules cover them all (rules on absent entities do nothing)."""


@dataclass(frozen=True, slots=True)
class Harmony4dSequence:
    """Where one sequence lives: its zip and the member prefix of its directory."""

    archive: Path
    """Local release zip, ``<root>/<split>/<scene>.zip``."""
    prefix: str
    """``<scene>/<sequence>`` inside the zip."""


@dataclass
class Harmony4dConfig(FrameLimitedConfig):
    """The release as ``dataforge-download harmony4d`` lays it out: ``<root>/<split>/<scene>.zip``."""

    command: ClassVar[str] = "harmony4d"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: Harmony4dDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "harmony4d")
    """Raw root: the release zips under ``train/`` and ``test/``."""
    sequences: tuple[str, ...] | None = None
    """Archive keys (``test/01_hugging``) or sequence keys (``test/01_hugging/002_hugging``); None selects all."""
    smpl_model_root: Path | None = None
    """Official neutral SMPL root holding ``smpl/SMPL_NEUTRAL.pkl`` (for the mesh triangles); defaults to ``root/body_models``."""

    @property
    def smpl_root(self) -> Path:
        return self.smpl_model_root or self.root / "body_models"

    def selects(self, key: str) -> bool:
        """Whether ``sequences`` names this archive or sequence key, or one that contains it."""
        return self.sequences is None or any(key == wanted or key.startswith(wanted + "/") or wanted.startswith(key + "/") for wanted in self.sequences)


def archive_key(path: str) -> str:
    """``test/01_hugging.zip`` -> ``test/01_hugging``."""
    return str(PurePosixPath(path).with_suffix(""))


class Harmony4dDataset(DataforgeDataset[Harmony4dConfig, Harmony4dSequence]):
    """Download the release zips, then convert each sequence from its zip in place."""

    layers = LAYERS
    _open: tuple[Path, dict[str, ZipSequenceFiles]] | None = None
    """The zip the last sequence came from, kept open with every sequence of it listed: discover sorts a zip's sequences
    together, and parsing a 40 GB zip's central directory takes seconds."""

    def archives(self) -> list[HfFileInfo]:
        """The release zips at the pinned revision (one listing call)."""
        return [file for file in hf_list_files(SOURCE_REPO, SOURCE_REVISION) if file.path.endswith(".zip")]

    def remote_sequences(self) -> list[RemoteSequence]:
        """One entry per zip: the release packs a scene's sequences into one archive, so a zip is the unit to fetch and prune."""
        return [RemoteSequence(archive_key(file.path), file.size_bytes, (file.path,)) for file in self.archives()]

    def download(self) -> None:
        """Fetch the selected zips (size and LFS sha256 checked) and the SMPL model unless in place; never deletes a raw file."""
        archives: list[HfFileInfo] = self.archives()
        known: set[str] = {archive_key(file.path) for file in archives}
        if self.config.sequences is not None:
            unknown: list[str] = [key for key in self.config.sequences if "/".join(key.split("/")[:2]) not in known]
            if unknown:
                raise ValueError(f"harmony4d: no archive for {unknown} at {SOURCE_REPO}@{SOURCE_REVISION[:8]}; see `dataforge-download harmony4d --list-remote`")
        report: FetchReport = FetchReport()
        for file in archives:
            if self.config.selects(archive_key(file.path)):
                report.count(hf_fetch_verified(SOURCE_REPO, file, local_dir=self.config.root, revision=SOURCE_REVISION), file.size_bytes)
        fetch_smpl_neutral(self.config.root, self.config.smpl_root, explicit=self.config.smpl_model_root is not None)
        print(f"harmony4d download: {report.summary()}")

    def discover(self) -> list[tuple[SequenceIdentity, Harmony4dSequence]]:
        """Every sequence directory (one with an ``exo/`` tree) inside the local zips; zips not yet downloaded are skipped."""
        found: list[tuple[SequenceIdentity, Harmony4dSequence]] = []
        for archive in sorted(self.config.root.glob("*/*.zip")):
            split: str = archive.parent.name
            if not self.config.selects(f"{split}/{archive.stem}"):
                continue
            with zipfile.ZipFile(archive) as opened:
                prefixes: list[str] = list(sequences_in(opened))
            for prefix in prefixes:
                identity: SequenceIdentity = SequenceIdentity(self.config.name, (split, *prefix.split("/")))
                if self.config.selects(identity.sequence_key):
                    found.append((identity, Harmony4dSequence(archive, prefix)))
        if self.config.sequences is not None:
            keys: set[str] = {identity.sequence_key for identity, _ in found}
            missing: list[str] = [key for key in self.config.sequences if not any(k == key or k.startswith(key + "/") for k in keys)]
            if missing:
                raise ValueError(f"harmony4d: {missing} not under {self.config.root}; download the archive first")
        return found

    def sequence_files(self, source: Harmony4dSequence) -> ZipSequenceFiles:
        """``source``'s files, from the zip already open when the last sequence came from the same one."""
        if self._open is None or self._open[0] != source.archive:
            if self._open is not None:
                next(iter(self._open[1].values())).archive.close()
            self._open = (source.archive, sequences_in(zipfile.ZipFile(source.archive)))
        return self._open[1][source.prefix]

    def convert(self, identity: SequenceIdentity, source: Harmony4dSequence, *, force: bool) -> Path:
        """Base, body_pose, body_mesh and projections of one sequence, read from its zip in place."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        split, scene, sequence_name = identity.parts
        with self.timer.stage("read"):
            files: ZipSequenceFiles = self.sequence_files(source)
            sequence = read_sequence(files)
        clock: Clock = Clock.of(sequence, self.config.frame_limit)
        episode: dict[str, str] = dict(split=split, scene=scene, sequence=sequence_name, activity=re.sub(r"_part\d+$", "", scene.split("_", 1)[-1]))
        writers = layer_writers(
            identity, files, sequence, clock, self.timer, self.config.smpl_root, source_revision=f"{SOURCE_REPO}@{SOURCE_REVISION}", episode=episode
        )
        self.write_layers(identity, targets, pending, writers)
        self.timer.capture_s = len(clock.frames) / FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """Scene on top with both glasses' rgb cameras beside it; eight GoPros along the bottom (lens-projected keypoints, boxes)."""
        return capture_blueprint(EXO_SLOTS, EGO_PANES, EXO_PANES)

    def table_blueprint(self) -> rrb.Blueprint:
        return capture_card(EXO_SLOTS, EGO_PANES)

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("split", "scene", "num_people")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
