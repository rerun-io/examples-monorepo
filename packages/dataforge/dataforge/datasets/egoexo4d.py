"""Ego-Exo4D takes with the Ego-Exo4D-HM body fits: four layers per take on the Aria device clock.

``base`` is the raw take (localized GoPros at 1080p, the Aria's RGB, SLAM and eye cameras, the Aria
trajectory); ``body_pose``, ``body_mesh`` and ``projections`` come from the take's HM fit. Only takes the HM
release covers (2,649) are converted. Take files are fetched on demand from the Ego-Exo4D release (licence keys
as an AWS profile) and deleted once base is written, unless ``keep_raw``; the fits, the body models, the
release metadata and base's small sidecars stay. See ``docs/egoexo4d.md``.
"""

import shutil
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from jaxtyping import Int64
from numpy import ndarray

from dataforge import blueprints, paths, schema, writing
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig, RemoteSequence
from dataforge.datasets.egoexo4d_body import HmFit, SmplhModel, read_fit, write_body_mesh, write_body_pose
from dataforge.datasets.egoexo4d_download import ManifestEntry, ManifestPath, Release, fetch_fit, fetch_models, fit_path, hm_fits
from dataforge.datasets.egoexo4d_layers import (
    ARIA_STREAMS,
    EGO_RIG,
    EXO_SLOTS,
    FRAMES_SIDECAR,
    BaseInputs,
    CamerasSidecar,
    FrameSidecar,
    read_sidecars,
    write_base,
    write_projections,
    write_sidecars,
)
from dataforge.datasets.egoexo4d_source import Take, localized, read_gopro_calibs, read_take_clock, read_takes, read_trajectory
from dataforge.identity import SequenceIdentity
from dataforge.transports import FetchReport
from dataforge.vrs import VrsFile

TRAJECTORY_FILES: tuple[str, ...] = ("closed_loop_trajectory.csv", "gopro_calibs.csv")
"""The files of a take's ``trajectory/`` the base layer reads (the part also ships point-cloud summaries, calibration logs …)."""
SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((0.0, -1.2, -1.5), (0.0, 0.3, 3.0), (0.0, -1.0, 0.0))
"""Eye behind and above GoPro 1, looking along its axis, in its (RDF) frame: the take's world frame has an arbitrary
origin, but GoPro 1 always faces the activity."""

UNCALIBRATED: list[str] = [f"- {schema.cam_path(EGO_RIG, cam)}/**" for cam, (_, _, label, _) in enumerate(ARIA_STREAMS) if label is None]
"""3D-view exclusions of the eye-tracking camera: its video has no pinhole, so a 3D view cannot place it."""


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
        self.release: Release | None = None
        self.model: SmplhModel | None = None

    def open_release(self) -> Release:
        """The S3 release, opened on first use so verbs that need no take files never ask for keys."""
        if self.release is None:
            self.release = Release(self.config.aws_profile)
        return self.release

    def take_files(self, take: Take) -> list[ManifestPath]:
        """Every release file only this take's conversion reads: frame-aligned videos, two trajectory files, the image-less VRS."""
        release: Release = self.open_release()
        videos: list[ManifestPath] = [
            path
            for path in release.manifest("takes")[take.take_uid].paths
            if "/frame_aligned_videos/" in path.relative_path and path.relative_path.endswith(".mp4")
        ]
        wanted: set[str] = {take.video(take.aria, stream) for stream, _, _, _ in ARIA_STREAMS}
        wanted |= {take.video(cam_id, "0") for cam_id in take.frame_aligned_videos if cam_id.startswith("cam")}
        trajectory: ManifestEntry = release.manifest("take_trajectory")[take.take_uid]
        vrs: ManifestEntry = release.manifest("take_vrs_noimagestream")[take.take_uid]
        return [
            *(path for path in videos if path.relative_path in wanted),
            *(path for path in trajectory.paths if Path(path.relative_path).name in TRAJECTORY_FILES),
            *vrs.paths,
        ]

    def timesync(self, take: Take) -> list[ManifestPath]:
        """The capture's ``timesync.csv``, shared by every take of the capture."""
        return [path for path in self.open_release().manifest("captures")[take.capture_uid].paths if path.relative_path.endswith("/timesync.csv")]

    def covered(self) -> dict[str, Take]:
        """Takes that have an HM fit and a takes.json entry, in name order, narrowed to ``sequences``."""
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
        return {name: takes[name] for name in names}

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every take with an HM fit, sized by its fit and its take files; needs the metadata and the keys."""
        fits = hm_fits()
        takes: dict[str, Take] = read_takes(self.config.root / "takes.json")
        result: list[RemoteSequence] = []
        for name in sorted(set(fits) & set(takes)):
            files: list[ManifestPath] = self.take_files(takes[name])
            size: int = fits[name].size_bytes + sum(path.size or 0 for path in files)
            result.append(RemoteSequence(name, size, tuple(path.relative_path for path in files)))
        return result

    def download(self) -> None:
        """Fetch the body models, the release metadata and the selected takes' fits; take files come at convert."""
        report: FetchReport = FetchReport()
        fetched_models: int = fetch_models(self.config.models)
        release: Release = self.open_release()
        for path in release.manifest("metadata")["takes"].paths + release.manifest("metadata")["captures"].paths:
            report.count(release.fetch(path, self.config.root), path.size or 0)
        fits = hm_fits()
        names: list[str] = sorted(fits) if self.config.sequences is None else list(self.config.sequences)
        unknown: list[str] = sorted(set(names) - set(fits))
        if unknown:
            raise ValueError(f"egoexo4d: the HM release has no fit for {unknown}")
        for name in names:
            report.count(fetch_fit(self.config.root, fits[name]), fits[name].size_bytes)
        print(f"egoexo4d download: {len(names)} fits, {fetched_models} body models fetched; {report.summary()}")
        print("  convert fetches each take's videos, trajectory and VRS from the release and deletes them after base")

    def discover(self) -> list[tuple[SequenceIdentity, Take]]:
        return [(SequenceIdentity(self.config.command, (name,)), take) for name, take in self.covered().items()]

    def fetch_take(self, take: Take) -> None:
        """Fetch the take's files and its capture's timesync unless already present."""
        release: Release = self.open_release()
        for path in [*self.take_files(take), *self.timesync(take)]:
            release.fetch(path, self.config.root)

    def prefetch(self, identity: SequenceIdentity, source: Take, *, force: bool) -> None:
        """Fetch the next take's raw files while this one converts; only base reads them."""
        if not writing.should_skip(self.targets(identity)[paths.BASE_LAYER], force=force):
            self.fetch_take(source)

    def base_inputs(self, take: Take) -> BaseInputs:
        """Read the raw take: localized GoPros, the clock, the Aria pose at each frame and its calibration."""
        root: Path = self.config.root
        take_dir: Path = root / take.root_dir
        times: Int64[ndarray, "n"] = read_take_clock(root / "captures" / take.capture.capture_name / "timesync.csv", take)[: self.config.frame_limit]
        (vrs_path,) = take_dir.glob("*noimagestreams.vrs")
        return BaseInputs(
            take=take,
            root=root,
            gopros=localized(read_gopro_calibs(take_dir / "trajectory/gopro_calibs.csv")),
            calib_json=VrsFile(vrs_path).file_tags["calib_json"],
            times_ns=times,
            world_T_device=read_trajectory(take_dir / "trajectory/closed_loop_trajectory.csv").at(times),
        )

    def convert(self, identity: SequenceIdentity, source: Take, *, force: bool) -> Path:
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        sidecars: Path = paths.sidecar_path(targets[paths.BASE_LAYER].parent.parent, identity, FRAMES_SIDECAR).parent
        work: Path = self.config.root / "work" / identity.recording_id
        base: bool = paths.BASE_LAYER in pending
        inputs: BaseInputs | None = None
        if base:
            with self.timer.stage("fetch"):
                self.fetch_take(source)
                inputs = self.base_inputs(source)

        def write(recording: rr.RecordingStream) -> None:
            assert inputs is not None
            work.mkdir(parents=True, exist_ok=True)
            try:
                cameras: CamerasSidecar = write_base(recording, identity, inputs, self.timer, work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            write_sidecars(sidecars, cameras, inputs.times_ns, inputs.world_T_device)

        @cache
        def fit_rows() -> tuple[FrameSidecar, CamerasSidecar, HmFit, Int64[ndarray, "t"]]:
            """The sidecars, the fit, and the frames both cover; read once, after base."""
            frames, cameras = read_sidecars(sidecars)
            fit: HmFit = read_fit(fit_path(self.config.root, source.take_name))
            count: int = min(len(fit.trans), len(frames.times_ns))
            if len(fit.trans) != len(frames.times_ns) and self.config.frame_limit is None:
                print(f"  {source.take_name}: the HM fit has {len(fit.trans)} frames, the take {len(frames.times_ns)}; converting {count}")
            return frames, cameras, fit, np.arange(count, dtype=np.int64)

        def body_pose(recording: rr.RecordingStream) -> None:
            frames, _, fit, rows = fit_rows()
            write_body_pose(recording, fit, frames.times_ns[rows], rows)

        def body_mesh(recording: rr.RecordingStream) -> None:
            frames, _, fit, rows = fit_rows()
            if self.model is None:
                self.model = SmplhModel(self.config.models)
            write_body_mesh(recording, self.model, fit, frames.times_ns[rows], rows)

        def projections(recording: rr.RecordingStream) -> None:
            frames, cameras, fit, rows = fit_rows()
            write_projections(recording, fit, frames, cameras, rows)

        self.write_layers(
            identity,
            targets,
            pending,
            {paths.BASE_LAYER: write, paths.BODY_POSE_LAYER: body_pose, paths.BODY_MESH_LAYER: body_mesh, paths.PROJECTIONS_LAYER: projections},
        )
        frames, _, _, rows = fit_rows()
        self.timer.capture_s = float(frames.times_ns[rows[-1]] - frames.times_ns[0]) / 1e9 + 1 / 30
        if base and not self.config.keep_raw:
            for path in self.take_files(source):
                (self.config.root / path.relative_path).unlink(missing_ok=True)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """Scene from behind GoPro 1; the Aria's cameras in a column; the GoPros along the bottom (lens-projected keypoints)."""
        projected: list[str] = [f"- {schema.coco133_uv_projected_path(rig, 0)}" for rig in range(1, EXO_SLOTS + 1)]
        projected += [f"- {schema.coco133_uv_projected_path(EGO_RIG, cam)}" for cam in range(len(ARIA_STREAMS))]
        projected += UNCALIBRATED
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Scene", origin=schema.cam_path(1, 0), contents=["+ /world/**", *projected], eye_controls=SCENE_EYE, line_grid=False
            ),
            ego_panes=[blueprints.camera_view(label or "camera-et", EGO_RIG, cam) for cam, (_, _, label, _) in enumerate(ARIA_STREAMS)],
            exo_panes=[blueprints.camera_view(f"GoPro {rig}", rig, 0) for rig in range(1, EXO_SLOTS + 1)],
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Card: the scene without video beside GoPro 1."""
        slots: list[tuple[int, int]] = [(rig, 0) for rig in range(1, EXO_SLOTS + 1)] + [(EGO_RIG, cam) for cam in range(len(ARIA_STREAMS))]
        return blueprints.exoego_table_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin=schema.cam_path(1, 0),
                contents=[
                    "+ /world/**",
                    *(f"- {schema.coco133_uv_projected_path(rig, cam)}" for rig, cam in slots),
                    *blueprints.video_exclusions(slots),
                    *UNCALIBRATED,
                ],
                eye_controls=SCENE_EYE,
                line_grid=False,
            ),
            blueprints.camera_view("GoPro 1", 1, 0),
        )

    def table_fields(self) -> writing.TableFields:
        fields = tuple(writing.TableField(f"property:episode:{name}", name) for name in ("activity", "task", "university")) + (
            writing.TableField("property:capture:num_frames", "frames"),
        )
        return writing.TableFields(cards=fields, table=fields)
