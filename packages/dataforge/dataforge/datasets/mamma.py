"""MAMMA (MPI's markerless multi-person capture release): one command, one catalog dataset per family.

``--family`` picks the catalog dataset: ``mamma-eval`` (the Singles, Extra and Dance sets, MoSh++ ground truth from
Vicon markers), ``mamma-markerless-dance``, ``mamma-markerless-multiple-people`` and ``mamma-markerless-iphones`` (the
MAMMA model's own predictions). Layers: base (cameras, AV1 video), body_pose, body_mesh, projections, and for eval masks
and markers. The release is read in place (``/mnt/nas/datasets/mamma`` on the dl-server) and never modified; download only
verifies it and proves a fetch on one or two sequences. See ``docs/mamma.md``.
"""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import ClassVar, TypeVar

import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.datasets.mamma_body import PosedPerson, SmplxBody, fetch_models
from dataforge.datasets.mamma_download import fetch_sequence, verify_release
from dataforge.datasets.mamma_layers import (
    Clock,
    write_base,
    write_body_mesh,
    write_body_pose,
    write_eval_visibility,
    write_markers,
    write_masks,
    write_projections,
)
from dataforge.datasets.mamma_source import Family, MammaCamera, MammaSequence, SequenceLabels, discover_sequences, read_camera
from dataforge.datasets.mamma_syn_source import SynFamily
from dataforge.identity import SequenceIdentity

PANES: int = 8
"""Camera panes of the default layout; a rig with more cameras shows every n-th one (Pablo, 2026-10-10)."""
VISIBLE_3D_VIDEOS: int = 4
"""Rigs with more cameras log their videos into the 3D view hidden (toggle them in the blueprint panel): the iPhones show theirs."""
FAMILY_RIGS: dict[Family, int] = {"eval": 32, "markerless-dance": 32, "markerless-multiple-people": 32, "markerless-iphones": 4}
"""Most cameras a rig of the family has (eval Dance 32, Singles and Extra 16): the catalog layouts' scene rules cover them all
(rules on absent entities do nothing)."""
FAMILY_PANES: dict[Family, int] = {**FAMILY_RIGS, "eval": 16}
"""Cameras the catalog default lays panes out for: eval shows the first 16, which every eval rig has."""
SCENE_EYE: rrb.EyeControls3D = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)


def pane_rigs(cameras: int) -> list[int]:
    """The rigs a layout gives a pane: all of a rig of up to 16 cameras, else ``PANES`` evenly spaced ones."""
    return list(range(1, cameras + 1)) if cameras <= 16 else list(range(1, cameras + 1, cameras // PANES))[:PANES]


def scene_view(rigs: int, *, card: bool = False) -> rrb.Spatial3DView:
    """The scene (z up) of a rig of up to ``rigs`` cameras: projections, masks and visibility landmarks stay in the panes, and a
    card drops every video while a layout hides them on rigs of more than ``VISIBLE_3D_VIDEOS`` cameras (a hidden video is not
    decoded)."""
    numbers: range = range(1, rigs + 1)
    contents: list[str] = [
        "+ /world/**",
        *(f"- {schema.coco133_uv_projected_path(rig, 0)}/**" for rig in numbers),
        *(f"- {schema.pinhole_path(rig, 0)}/masks/**" for rig in numbers),
        *(f"- {schema.pinhole_path(rig, 0)}/visibility/**" for rig in numbers),
        *(blueprints.video_exclusions((rig, 0) for rig in numbers) if card else []),
    ]
    return rrb.Spatial3DView(
        name="Scene",
        origin="/world",
        contents=contents,
        overrides=None if card or rigs <= VISIBLE_3D_VIDEOS else {schema.video_path(rig, 0): rrb.EntityBehavior(visible=False) for rig in numbers},
        eye_controls=SCENE_EYE,
    )


def layout(rigs: int, names: list[str] | None = None, *, panes: int | None = None) -> rrb.Blueprint:
    """The scene above a strip of camera panes, for ``panes`` of the cameras (all ``rigs`` by default)."""
    views: list[rrb.Spatial2DView] = [
        blueprints.camera_view(names[rig - 1] if names else f"Camera {rig}", rig, 0) for rig in pane_rigs(panes or rigs)
    ]
    return blueprints.exoego_blueprint(scene_view(rigs), ego_panes=[], exo_panes=views)


@dataclass
class MammaConfig(FrameLimitedConfig):
    """The MAMMA release as the MPI downloader lays it out (``mamma_eval_*``, ``mamma_markerless_*`` beside each other)."""

    command: ClassVar[str] = "mamma"
    """Registry key and CLI subcommand; ``--family`` picks the catalog dataset."""
    _target: type = field(default_factory=lambda: MammaDataset)
    family: Family | SynFamily = "eval"
    """Catalog family to work on; also the catalog dataset (``mamma-<family>``). ``syn`` and ``syn-bedlam`` are MammaSyn."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "mamma")
    """Release root, read only."""
    model_root: Path = field(default_factory=lambda: paths.raw_root() / "mamma-body-models")
    """Where the SMPL-X model files go (``dataforge-download mamma`` fetches them)."""
    sequences: tuple[str, ...] | None = None
    """Sequence keys (``singles/230929_WhiteRabbit_CatchBall_50048_1``, ``indoors/crossing_arms``); None selects all."""
    index_root: Path = field(default_factory=lambda: paths.raw_root() / "mamma-syn-index")
    """MammaSyn: where each sequence directory's shard index is cached (local disk; rebuilt when a shard changes)."""
    fetch_into: Path | None = None
    """Download: also fetch the ``sequences`` from the MPI server into this directory and compare every file with the release
    copy, as a proof that the download path works (needs MAMMA_USERNAME and MAMMA_PASSWORD). Never the whole release."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per family."""
        return f"{self.command}-{self.family}"

    def setup(self) -> DataforgeDataset:
        """The MPI families' dataset, or MammaSyn's for ``syn`` and ``syn-bedlam``."""
        from dataforge.datasets.mamma_syn import MammaBedlamDataset, MammaSynDataset  # imports this module

        if self.family == "syn":
            return MammaSynDataset(self)
        if self.family == "syn-bedlam":
            return MammaBedlamDataset(self)
        return MammaDataset(self)


SourceT = TypeVar("SourceT")


def select_sequences(config: MammaConfig, pairs: list[tuple[SequenceIdentity, SourceT]], noun: str) -> list[tuple[SequenceIdentity, SourceT]]:  # noqa: UP047 — beartype requires legacy generics
    """``pairs`` narrowed to ``config.sequences`` when given; raises when one of them is not there (``noun``: sequence, clip)."""
    if config.sequences is None:
        return pairs
    missing: set[str] = set(config.sequences) - {identity.sequence_key for identity, _ in pairs}
    if missing:
        raise ValueError(f"{config.name}: no {noun} {sorted(missing)} under {config.root}")
    return [(identity, source) for identity, source in pairs if identity.sequence_key in config.sequences]


class MammaDataset(DataforgeDataset[MammaConfig, MammaSequence]):
    """Convert one sequence at a time from the release in place."""

    @property
    def family(self) -> Family:
        """The MPI family (``MammaConfig.setup`` gives the MammaSyn families their own dataset)."""
        family: Family | SynFamily = self.config.family
        if family == "syn" or family == "syn-bedlam":
            raise ValueError(f"{family} is a MammaSyn family: MammaConfig.setup builds MammaSynDataset for it")
        return family

    layers = (
        paths.BASE_LAYER,
        paths.BODY_POSE_LAYER,
        paths.BODY_MESH_LAYER,
        paths.PROJECTIONS_LAYER,
        paths.MASKS_LAYER,
        paths.MARKERS_LAYER,
        paths.VISIBILITY_LAYER,
    )
    """masks and visibility only where the release ships them (eval), markers only in eval Extra; register skips a layer no
    segment has."""

    @cached_property
    def body(self) -> SmplxBody:
        return SmplxBody(self.config.model_root)

    def download(self) -> None:
        """Fetch the SMPL-X models and verify the release in place; with ``fetch_into``, prove the MPI download on ``sequences``."""
        fetched: int = fetch_models(self.config.model_root)
        sequences, problems = verify_release(self.config.root, self.family)
        print(f"{self.config.name} download: {len(sequences)} sequences verified in place under {self.config.root}; {fetched} SMPL-X file(s) fetched")
        for problem in problems:
            print(f"  problem: {problem}")
        if self.config.fetch_into is not None:
            if self.config.sequences is None:
                raise ValueError("--fetch-into proves the MPI download on chosen sequences: name them with --sequences (the release is ~221 GB)")
            paths.require_outside([self.config.fetch_into], roots=[paths.NAS_ROOT, self.config.root])
            for sequence in sequences:
                key: str = SequenceIdentity(self.config.name, sequence.parts).sequence_key
                if key in self.config.sequences:
                    mismatches: list[str] = fetch_sequence(self.config.root, sequence, self.config.fetch_into)
                    problems += mismatches
                    print(f"  fetched {key} from the MPI server: {mismatches or 'identical to the release copy'}")
        if problems:
            raise ValueError(f"{self.config.name}: {len(problems)} problem(s) in the release, listed above")

    def discover(self) -> list[tuple[SequenceIdentity, MammaSequence]]:
        found, skipped = discover_sequences(self.config.root, self.family)
        for directory, reason in skipped:
            print(f"  mamma: skipped {directory}: {reason}")
        return select_sequences(self.config, [(SequenceIdentity(self.config.name, sequence.parts), sequence) for sequence in found], "sequence")

    def convert(self, identity: SequenceIdentity, source: MammaSequence, *, force: bool) -> Path:
        """Base, then the body, projection, mask, marker and visibility layers, all from the release in place."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        # By collection, not by file presence: a missing mask tar or marker file fails its layer instead of skipping it.
        ships: dict[str, bool] = {paths.MASKS_LAYER: source.is_eval, paths.MARKERS_LAYER: source.has_markers, paths.VISIBILITY_LAYER: source.is_eval}
        pending = [layer for layer in pending if ships.get(layer, True)]
        if not pending:
            return targets[paths.BASE_LAYER]
        with self.timer.stage("read"):
            labels: SequenceLabels = source.labels()
            cameras: list[MammaCamera] = [read_camera(source.calibration(camera)) for camera in source.cameras]
        clock: Clock = Clock.of(labels, self.config.frame_limit)
        if paths.BASE_LAYER in pending:
            # The base layer embeds this sequence's own layout (16 or 32 panes differ within eval), not the family default.
            with (
                self.timer.stage(f"write:{paths.BASE_LAYER}"),
                writing.atomic_recording(
                    targets[paths.BASE_LAYER],
                    recording_id=identity.recording_id,
                    default_blueprint=layout(len(cameras), [c.name for c in cameras]),
                    send_properties=True,
                ) as recording,
            ):
                write_base(recording, identity, source, labels, cameras, clock, self.timer)
        posed: list[PosedPerson] = []
        if {paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER, paths.VISIBILITY_LAYER} & set(pending):
            with self.timer.stage("smplx"):
                posed = [self.body.pose(person, clock.rows) for person in labels.people]

        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, labels.people, labels.method, posed, clock),
                paths.BODY_MESH_LAYER: lambda recording: write_body_mesh(recording, posed, clock),
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(
                    recording, cameras, posed, clock, camera_model="vicon_radial_2 (IOI cameras), pinhole (iPhones)", calibration_source="per-camera release NPZ"
                ),
                paths.MASKS_LAYER: lambda recording: write_masks(recording, source, cameras, len(labels.people), clock),
                paths.MARKERS_LAYER: lambda recording: write_markers(recording, source, clock),
                paths.VISIBILITY_LAYER: lambda recording: write_eval_visibility(recording, source, cameras, labels, posed, self.body.landmark_ids, clock),
            },
        )
        self.timer.capture_s = len(clock.frames) / labels.fps
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """The family's layout, registered on its catalog dataset; each base rrd embeds its own sequence's."""
        return layout(FAMILY_RIGS[self.family], panes=FAMILY_PANES[self.family])

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the scene without video beside the first camera."""
        return blueprints.exoego_table_blueprint(scene_view(FAMILY_RIGS[self.family], card=True), blueprints.camera_view("Camera 1", 1, 0))

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("collection", "num_people", "body_source")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
