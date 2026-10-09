"""Ego-Exo4D takes with the Ego-Exo4D-HM body fits: four layers per take on the Aria device clock.

``base`` is the raw take (localized GoPros at 1080p, the Aria's RGB, SLAM and eye cameras, the Aria
trajectory); ``body_pose``, ``body_mesh`` and ``projections`` come from the take's HM fit. Only takes the HM
release covers (2,649) are converted. Take files are fetched on demand from the Ego-Exo4D release (licence keys
as an AWS profile) and deleted once base is written, unless ``keep_raw``; the fits, the body models, the
release metadata and base's small sidecars stay. See ``docs/egoexo4d.md``.
"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int64
from numpy import ndarray

from dataforge import aria, blueprints, paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.egoexo4d_body import HM_REPO, HM_REVISION, HmFit, SmplhModel, read_fit, write_body_mesh, write_body_pose
from dataforge.datasets.egoexo4d_download import HM_DIR, ManifestPath, Release, fetch_models, fit_path, hm_fits
from dataforge.datasets.egoexo4d_layers import (
    ARIA_STREAMS,
    EGO_RIG,
    EXO_SLOTS,
    SIDECAR,
    BaseInputs,
    FrameSidecar,
    read_sidecar,
    write_base,
    write_projections,
    write_sidecar,
)
from dataforge.datasets.egoexo4d_source import FPS, Take, TakeClock, localized, read_gopro_calibs, read_take_clock, read_takes
from dataforge.identity import SequenceIdentity
from dataforge.transports import FetchReport, HfFileInfo
from dataforge.vrs import VrsFile

TRAJECTORY_FILES: tuple[str, ...] = ("closed_loop_trajectory.csv", "gopro_calibs.csv")
"""The files of a take's ``trajectory/`` the base layer reads (the part also ships point-cloud summaries, calibration logs …)."""
FETCH_WORKERS: int = 8
"""Concurrent S3 transfers per take; the 4K GoPro MP4s are most of its bytes."""
SCENE_EYE: rrb.EyeControls3D = rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)
"""Orbital, placed by the viewer to fit the scene in the gravity-aligned world: no fixed pose frames both a kitchen and a soccer
pitch, and one rooted at a GoPro inherits its tilt (down to -65 degrees on the piano takes)."""
UNCALIBRATED: list[str] = [f"- {schema.cam_path(EGO_RIG, cam)}/**" for cam, stream in enumerate(ARIA_STREAMS) if not stream.calibrated]
"""3D-view exclusions of the eye-tracking camera: its video has no pinhole, so a 3D view cannot place it."""
SLOTS: list[tuple[int, int]] = [(rig, 0) for rig in range(1, EXO_SLOTS + 1)] + [(EGO_RIG, cam) for cam in range(len(ARIA_STREAMS))]
"""Every (rig, cam) the blueprints lay out: the GoPro rigs, then the Aria's cameras."""


def scene_view(*hidden: str) -> rrb.Spatial3DView:
    """The 3D scene without the 2D keypoint projections and the uncalibrated eye camera, nor ``hidden``."""
    return rrb.Spatial3DView(
        name="Scene",
        origin="/world",
        contents=["+ /world/**", *(f"- {schema.coco133_uv_projected_path(rig, cam)}" for rig, cam in SLOTS), *hidden, *UNCALIBRATED],
        eye_controls=SCENE_EYE,
        line_grid=False,
    )


@dataclass
class Egoexo4dConfig(FrameLimitedConfig):
    """Raw root as `dataforge-download egoexo4d` lays it out: the egoexo CLI tree plus ``hm/`` and ``body_models/``."""

    command: ClassVar[str] = "egoexo4d"
    """CLI and catalog dataset name."""
    _target: type = field(default_factory=lambda: Egoexo4dDataset)
    """Dataset constructor."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "egoexo4d")
    """Raw root: takes.json, takes/, captures/, hm/, body_models/."""
    sequences: tuple[str, ...] | None = None
    """Take names; None selects every take with an HM fit."""
    aws_profile: str = "egoexo4d"
    """AWS profile holding the Ego-Exo4D licence keys (they expire 14 days after issue)."""
    keep_raw: bool = False
    """Keep a take's videos, VRS and trajectory files after base; the fit, models, metadata and sidecars always stay."""
    model_root: Path | None = None
    """Directory holding body_models/smplh/SMPLH_MALE.npz and body_models/smplx/SMPLX_NEUTRAL.npz; defaults to root."""

    @property
    def models(self) -> Path:
        """model_root, falling back to root."""
        return self.model_root or self.root


class Egoexo4dDataset(DataforgeDataset[Egoexo4dConfig, Take]):
    """Fetch, convert and prune one take at a time."""

    layers = (paths.BASE_LAYER, paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER, paths.PROJECTIONS_LAYER)

    def __init__(self, config: Egoexo4dConfig) -> None:
        super().__init__(config)
        self.release: Release = Release(config.aws_profile)  # asks for the keys on its first request, so verbs that fetch nothing need none
        self.model: SmplhModel | None = None

    def trajectory_files(self, take: Take) -> list[ManifestPath]:
        """The two files of the take's ``trajectory/`` base reads."""
        return [path for path in self.release.manifest("take_trajectory")[take.take_uid].paths if Path(path.relative_path).name in TRAJECTORY_FILES]

    def vrs_file(self, take: Take) -> ManifestPath:
        """The take's image-less VRS: the Aria calibration."""
        (vrs,) = self.release.manifest("take_vrs_noimagestream")[take.take_uid].paths
        return vrs

    def take_files(self, take: Take, gopros: Iterable[str]) -> list[ManifestPath]:
        """The release files base reads for this take: the Aria's videos, the videos of ``gopros``, the trajectory files, the VRS."""
        wanted: set[str] = {take.video(take.aria, stream.readable) for stream in ARIA_STREAMS} | {take.video(cam_id, "0") for cam_id in gopros}
        videos: list[ManifestPath] = [path for path in self.release.manifest("takes")[take.take_uid].paths if path.relative_path in wanted]
        return [*videos, *self.trajectory_files(take), self.vrs_file(take)]

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every take with an HM fit, sized by its fit and its take files; needs the metadata and the keys."""
        fits: dict[str, HfFileInfo] = hm_fits()
        takes: dict[str, Take] = read_takes(self.config.root / "takes.json")
        result: list[RemoteSequence] = []
        for name in sorted(set(fits) & set(takes)):  # every exo camera's video: the localized ones are known only after a fetch
            files: list[ManifestPath] = self.take_files(takes[name], takes[name].exo_cameras)
            result.append(
                RemoteSequence(name, fits[name].size_bytes + sum(path.size or 0 for path in files), tuple(path.relative_path for path in files))
            )
        return result

    def download(self) -> None:
        """Fetch the body models, the release metadata and the selected takes' fits; take files come at convert."""
        report: FetchReport = FetchReport()
        fetched_models: int = fetch_models(self.config.models)
        for path in self.release.manifest("metadata")["takes"].paths + self.release.manifest("metadata")["captures"].paths:
            report.count(self.release.fetch(path, self.config.root), path.size or 0)
        fits: dict[str, HfFileInfo] = hm_fits()
        names: list[str] = sorted(fits) if self.config.sequences is None else list(self.config.sequences)
        unknown: list[str] = sorted(set(names) - set(fits))
        if unknown:
            raise ValueError(f"egoexo4d: the HM release has no fit for {unknown}")

        def fetch(name: str) -> bool:
            return transports.hf_fetch_verified(HM_REPO, fits[name], local_dir=self.config.root / HM_DIR, revision=HM_REVISION)

        with ThreadPoolExecutor(max_workers=16) as executor:
            for name, fetched in zip(names, executor.map(fetch, names), strict=True):
                report.count(fetched, fits[name].size_bytes)
        print(f"egoexo4d download: {len(names)} fits, {fetched_models} body models fetched; {report.summary()}")
        print("  convert fetches each take's videos, trajectory and VRS from the release and deletes them after base")

    def discover(self) -> list[tuple[SequenceIdentity, Take]]:
        """Takes that have a local HM fit and a takes.json entry, in name order, narrowed to ``sequences``."""
        takes_json: Path = self.config.root / "takes.json"
        if not takes_json.is_file():
            raise FileNotFoundError(
                f"{takes_json}: run dataforge-download egoexo4d (it needs the Ego-Exo4D keys in AWS profile {self.config.aws_profile!r})"
            )
        takes: dict[str, Take] = read_takes(takes_json)
        names: list[str] = sorted(name for name in takes if fit_path(self.config.root, name).is_file())
        if self.config.sequences is not None:
            missing: set[str] = set(self.config.sequences) - set(names)
            if missing:
                raise ValueError(
                    f"egoexo4d: no local HM fit or takes.json entry for {sorted(missing)}; run dataforge-download egoexo4d --sequences ..."
                )
            names = [name for name in names if name in self.config.sequences]
        return [(SequenceIdentity(self.config.command, (name,)), takes[name]) for name in names]

    def fetch_take(self, take: Take) -> None:
        """Fetch the trajectory files, then concurrently the take's other files and its capture's timesync.csv, unless present.

        ``gopro_calibs.csv`` comes first because it names the localized GoPros: base logs only those, so only their 4K videos
        are fetched.
        """
        trajectory: list[ManifestPath] = self.trajectory_files(take)
        for path in trajectory:
            self.release.fetch(path, self.config.root)
        gopros: list[str] = [
            calib.cam_uid for calib in localized(read_gopro_calibs(self.config.root / take.root_dir / "trajectory/gopro_calibs.csv"))
        ]
        rest: list[ManifestPath] = [path for path in self.take_files(take, gopros) if path not in trajectory]
        rest += [path for path in self.release.manifest("captures")[take.capture_uid].paths if path.relative_path.endswith("/timesync.csv")]
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            list(executor.map(lambda path: self.release.fetch(path, self.config.root), rest))

    def prefetch(self, identity: SequenceIdentity, source: Take, *, force: bool) -> None:
        """Fetch the next take's raw files while this one converts; only base reads them."""
        if not writing.should_skip(self.targets(identity)[paths.BASE_LAYER], force=force):
            self.fetch_take(source)

    def base_inputs(self, take: Take) -> BaseInputs:
        """Read the raw take: localized GoPros, the clock, the Aria pose at each frame and its calibration."""
        take_dir: Path = self.config.root / take.root_dir
        clock: TakeClock = read_take_clock(self.config.root / "captures" / take.capture.capture_name / "timesync.csv", take)
        times: Int64[ndarray, "n"] = clock.times_ns[: self.config.frame_limit]
        return BaseInputs(
            take=take,
            root=self.config.root,
            gopros=localized(read_gopro_calibs(take_dir / "trajectory/gopro_calibs.csv")),
            calib_json=VrsFile(self.config.root / self.vrs_file(take).relative_path).file_tags["calib_json"],
            frames=FrameSidecar(times, aria.read_trajectory(take_dir / "trajectory/closed_loop_trajectory.csv").at(times)),
            take_frames=len(clock.times_ns),
            clock_filled=clock.filled,
        )

    def convert(self, identity: SequenceIdentity, source: Take, *, force: bool) -> Path:
        """Base first (fetch, encode, publish with its sidecar, prune the take), then the fit's layers from the sidecar."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        base: Path = targets[paths.BASE_LAYER]
        sidecar: Path = paths.sidecar_path(base.parents[1], identity, SIDECAR)  # a preview's root holds its own sidecar
        if paths.BASE_LAYER in pending or not sidecar.is_file():  # every derived layer is stamped on base's clock
            pending = list(targets)
        if not pending:
            return base
        if paths.BODY_MESH_LAYER in pending and self.model is None:
            self.model = SmplhModel(self.config.models)  # fails on a missing model before any fetch or encode
        if paths.BASE_LAYER in pending:
            with self.timer.stage("fetch"):
                self.fetch_take(source)
                inputs: BaseInputs = self.base_inputs(source)
            # The sidecar is staged around base's recording, so both are published (base first) or neither is.
            with (
                writing.atomic_write(sidecar) as staged,
                self.timer.stage(f"write:{paths.BASE_LAYER}"),
                self.layer_recording(identity, paths.BASE_LAYER, base) as recording,
            ):
                write_sidecar(staged, write_base(recording, identity, inputs, self.timer), inputs.frames)
            if not self.config.keep_raw:
                for path in self.take_files(source, [calib.cam_uid for calib in inputs.gopros]):
                    (self.config.root / path.relative_path).unlink(missing_ok=True)

        frames, cameras = read_sidecar(sidecar)
        fit: HmFit = read_fit(fit_path(self.config.root, source.take_name))
        count: int = min(len(fit.trans), len(frames.times_ns))
        if len(fit.trans) != len(frames.times_ns) and self.config.frame_limit is None:
            print(f"  {source.take_name}: the HM fit has {len(fit.trans)} frames, the take {len(frames.times_ns)}; converting {count}")
        rows: Int64[ndarray, "t"] = np.arange(count, dtype=np.int64)
        model: SmplhModel | None = self.model

        def body_mesh(recording: rr.RecordingStream) -> None:
            assert model is not None
            write_body_mesh(recording, model, fit, frames.times_ns[rows], rows)

        self.write_layers(
            identity,
            targets,
            [layer for layer in pending if layer != paths.BASE_LAYER],
            {
                paths.BODY_POSE_LAYER: lambda recording: write_body_pose(recording, fit, frames.times_ns[rows], rows),
                paths.BODY_MESH_LAYER: body_mesh,
                paths.PROJECTIONS_LAYER: lambda recording: write_projections(recording, fit, frames, cameras, rows),
            },
        )
        self.timer.capture_s = len(frames.times_ns) / FPS
        return base

    def default_blueprint(self) -> rrb.Blueprint:
        """Scene, z up; the Aria's cameras in a column; the GoPros along the bottom (lens-projected keypoints)."""
        return blueprints.exoego_blueprint(
            scene_view(),
            ego_panes=[blueprints.camera_view(stream.name, EGO_RIG, cam) for cam, stream in enumerate(ARIA_STREAMS)],
            exo_panes=[blueprints.camera_view(f"GoPro {rig}", rig, 0) for rig in range(1, EXO_SLOTS + 1)],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the scene without video beside GoPro 1."""
        return blueprints.exoego_table_blueprint(scene_view(*blueprints.video_exclusions(SLOTS)), blueprints.camera_view("GoPro 1", 1, 0))

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("activity", "task", "university")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
