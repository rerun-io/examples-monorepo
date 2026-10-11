"""MAMMA (MPI's markerless multi-person capture release): one command, one catalog dataset per family.

``--family`` picks the catalog dataset: ``mamma-eval`` (the Singles, Extra and Dance sets, MoSh++ ground truth from
Vicon markers), ``mamma-markerless-dance``, ``mamma-markerless-multiple-people`` and ``mamma-markerless-iphones`` (the
MAMMA model's own predictions). Layers: base (cameras, AV1 video), body_pose, body_mesh, projections, and for eval masks
and markers. The release is read in place (``/mnt/nas/datasets/mamma`` on the dl-server) and never modified; download only
verifies it and proves a fetch on one or two sequences. See ``docs/mamma.md``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import rerun as rr
import rerun.blueprint as rrb

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.datasets.mamma_body import PosedPerson, SmplxBody, fetch_models
from dataforge.datasets.mamma_download import fetch_sequence, verify_release
from dataforge.datasets.mamma_layers import Clock, write_base, write_body_mesh, write_body_pose, write_markers, write_masks, write_projections
from dataforge.datasets.mamma_source import Family, MammaCamera, MammaSequence, SequenceLabels, discover_sequences, read_camera
from dataforge.identity import SequenceIdentity

PANES: int = 8
"""Camera panes of the default layout; a rig with more cameras shows every n-th one (Pablo, 2026-10-10)."""
VISIBLE_3D_VIDEOS: int = 4
"""Rigs with more cameras log their videos into the 3D view hidden (toggle them in the blueprint panel): the iPhones show theirs."""
FAMILY_RIGS: dict[Family, int] = {"eval": 16, "markerless-dance": 32, "markerless-multiple-people": 32, "markerless-iphones": 4}
"""Cameras the catalog dataset's default layout is built for: every eval rig has the first 16 (Dance has 32)."""
SCENE_EYE: rrb.EyeControls3D = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)


def pane_rigs(cameras: int) -> list[int]:
    """The rigs a layout gives a pane: all of a rig of up to 16 cameras, else ``PANES`` evenly spaced ones."""
    return list(range(1, cameras + 1)) if cameras <= 16 else list(range(1, cameras + 1, cameras // PANES))[:PANES]


def layout(cameras: int, names: list[str] | None = None) -> rrb.Blueprint:
    """The scene (z up, projections and masks left to the panes) above a strip of camera panes."""
    rigs: range = range(1, cameras + 1)
    scene = rrb.Spatial3DView(
        name="Scene",
        origin="/world",
        contents=[
            "+ /world/**",
            *(f"- {schema.coco133_uv_projected_path(rig, 0)}/**" for rig in rigs),
            *(f"- {schema.pinhole_path(rig, 0)}/masks/**" for rig in rigs),
        ],
        overrides={schema.video_path(rig, 0): rrb.EntityBehavior(visible=False) for rig in rigs} if cameras > VISIBLE_3D_VIDEOS else None,
        eye_controls=SCENE_EYE,
    )
    panes: list[rrb.Spatial2DView] = [blueprints.camera_view(names[rig - 1] if names else f"Camera {rig}", rig, 0) for rig in pane_rigs(cameras)]
    return blueprints.exoego_blueprint(scene, ego_panes=[], exo_panes=panes)


@dataclass
class MammaConfig(FrameLimitedConfig):
    """The MAMMA release as the MPI downloader lays it out (``mamma_eval_*``, ``mamma_markerless_*`` beside each other)."""

    command: ClassVar[str] = "mamma"
    """Registry key and CLI subcommand; ``--family`` picks the catalog dataset."""
    _target: type = field(default_factory=lambda: MammaDataset)
    family: Family = "eval"
    """Catalog family to work on; also the catalog dataset (``mamma-<family>``)."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "mamma")
    """Release root, read only."""
    model_root: Path = field(default_factory=lambda: paths.raw_root() / "mamma-body-models")
    """Where the SMPL-X model files go (``dataforge-download mamma`` fetches them)."""
    sequences: tuple[str, ...] | None = None
    """Sequence keys (``singles/230929_WhiteRabbit_CatchBall_50048_1``, ``indoors/crossing_arms``); None selects all."""
    fetch_into: Path | None = None
    """Download: also fetch the ``sequences`` from the MPI server into this directory and compare every file with the release
    copy, as a proof that the download path works (needs MAMMA_USERNAME and MAMMA_PASSWORD). Never the whole release."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per family."""
        return f"{self.command}-{self.family}"


class MammaDataset(DataforgeDataset[MammaConfig, MammaSequence]):
    """Convert one sequence at a time from the release in place."""

    def __init__(self, config: MammaConfig) -> None:
        super().__init__(config)
        self.body: SmplxBody | None = None

    @property
    def layers(self) -> tuple[str, ...]:  # type: ignore[override]
        derived: tuple[str, ...] = (paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER)
        return (paths.BASE_LAYER, *derived, *((paths.MASKS_LAYER, paths.MARKERS_LAYER) if self.config.family == "eval" else ()))

    def download(self) -> None:
        """Fetch the SMPL-X models and verify the release in place; with ``fetch_into``, prove the MPI download on ``sequences``."""
        fetched: int = fetch_models(self.config.model_root)
        sequences, problems = verify_release(self.config.root, self.config.family)
        print(f"{self.config.name} download: {len(sequences)} sequences verified in place under {self.config.root}; {fetched} SMPL-X file(s) fetched")
        for problem in problems:
            print(f"  problem: {problem}")
        if self.config.fetch_into is not None:
            if self.config.sequences is None:
                raise ValueError("--fetch-into proves the MPI download on chosen sequences: name them with --sequences (the release is ~221 GB)")
            paths.require_outside([self.config.fetch_into], roots=[paths.NAS_ROOT, self.config.root])
            for identity, sequence in self.discover():
                mismatches: list[str] = fetch_sequence(self.config.root, sequence, self.config.fetch_into)
                problems += mismatches
                print(f"  fetched {identity.sequence_key} from the MPI server: {mismatches or 'identical to the release copy'}")
        if problems:
            raise ValueError(f"{self.config.name}: {len(problems)} problem(s) in the release, listed above")

    def discover(self) -> list[tuple[SequenceIdentity, MammaSequence]]:
        found, skipped = discover_sequences(self.config.root, self.config.family)
        for reason in skipped:
            print(f"  mamma: skipped {reason}")
        pairs: list[tuple[SequenceIdentity, MammaSequence]] = [(SequenceIdentity(self.config.name, sequence.parts), sequence) for sequence in found]
        if self.config.sequences is not None:
            known: set[str] = {identity.sequence_key for identity, _ in pairs}
            missing: set[str] = set(self.config.sequences) - known
            if missing:
                raise ValueError(f"mamma-{self.config.family}: no sequence {sorted(missing)} under {self.config.root}")
            pairs = [(identity, sequence) for identity, sequence in pairs if identity.sequence_key in self.config.sequences]
        return pairs

    def convert(self, identity: SequenceIdentity, source: MammaSequence, *, force: bool) -> Path:
        """Base, then the body, projection, mask and marker layers, all from the release in place."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        shipped: dict[str, bool] = {
            paths.MASKS_LAYER: all(source.mask_tar(camera).is_file() for camera in source.cameras),
            paths.MARKERS_LAYER: source.markers_dir.is_dir(),
        }
        pending = [layer for layer in pending if shipped.get(layer, True)]
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
        if {paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER} & set(pending):
            if self.body is None:
                self.body = SmplxBody(self.config.model_root)
            body: SmplxBody = self.body
            with self.timer.stage("smplx"):
                posed = [body.pose(person, clock.frames) for person in labels.people]

        def body_mesh(recording: rr.RecordingStream) -> None:
            assert self.body is not None and self.body.faces is not None
            write_body_mesh(recording, posed, self.body.faces, clock)

        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, labels, posed, clock),
                paths.BODY_MESH_LAYER: body_mesh,
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(recording, cameras, posed, clock),
                paths.MASKS_LAYER: lambda recording: write_masks(recording, source, cameras, len(labels.people), clock),
                paths.MARKERS_LAYER: lambda recording: write_markers(recording, source, clock),
            },
        )
        self.timer.capture_s = len(clock.frames) / labels.fps
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """The family's layout, registered on its catalog dataset; each base rrd embeds its own sequence's."""
        return layout(FAMILY_RIGS[self.config.family])

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the scene without video beside the first camera."""
        rigs: int = FAMILY_RIGS[self.config.family]
        scene = rrb.Spatial3DView(
            name="Scene",
            origin="/world",
            contents=[
                "+ /world/**",
                *blueprints.video_exclusions((rig, 0) for rig in range(1, rigs + 1)),
                *(f"- {schema.coco133_uv_projected_path(rig, 0)}/**" for rig in range(1, rigs + 1)),
                *(f"- {schema.pinhole_path(rig, 0)}/masks/**" for rig in range(1, rigs + 1)),
            ],
            eye_controls=SCENE_EYE,
        )
        return blueprints.exoego_table_blueprint(scene, blueprints.camera_view("Camera 1", 1, 0))

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("collection", "num_people", "body_source")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
