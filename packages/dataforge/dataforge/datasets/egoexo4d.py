"""Ego-Exo4D takes with the Ego-Exo4D-HM body fits: four layers per take on the Aria device clock.

``base`` is the raw take (localized GoPros at 1080p, the Aria's RGB, SLAM and eye cameras, the Aria
trajectory); ``body_pose``, ``body_mesh`` and ``projections`` come from the take's HM fit. Only takes the HM
release covers (2,649) are converted. Take files are fetched on demand from the Ego-Exo4D release (licence keys
as an AWS profile) and deleted once base is written, unless ``keep_raw``; the fits, the body models, the
release metadata and base's small sidecars stay. See ``docs/egoexo4d.md``.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
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
    CamerasSidecar,
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
SCENE_EYE: rrb.EyeControls3D = blueprints.eye_controls_from_pose((0.0, -1.2, -1.5), (0.0, 0.3, 3.0), (0.0, -1.0, 0.0))
"""Eye behind and above GoPro 1, looking along its axis, in its (RDF) frame: the take's world frame has an arbitrary
origin, but GoPro 1 always faces the activity."""
UNCALIBRATED: list[str] = [f"- {schema.cam_path(EGO_RIG, cam)}/**" for cam, stream in enumerate(ARIA_STREAMS) if stream.label is None]
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
        self.release_lock = threading.Lock()  # threading.Lock is a factory, not a type beartype can check
        self.model: SmplhModel | None = None

    def open_release(self) -> Release:
        """The S3 release, opened on first use (by convert or its prefetch thread) so verbs that need no take files never ask for keys."""
        with self.release_lock:
            if self.release is None:
                self.release = Release(self.config.aws_profile)
            return self.release

    def take_files(self, take: Take) -> list[ManifestPath]:
        """Every release file only this take's conversion reads: frame-aligned videos, two trajectory files, the image-less VRS."""
        release: Release = self.open_release()
        wanted: set[str] = {take.video(take.aria, stream.readable) for stream in ARIA_STREAMS}
        wanted |= {take.video(cam_id, "0") for cam_id in take.frame_aligned_videos if cam_id.startswith("cam")}
        return [
            *(path for path in release.manifest("takes")[take.take_uid].paths if path.relative_path in wanted),
            *(path for path in release.manifest("take_trajectory")[take.take_uid].paths if Path(path.relative_path).name in TRAJECTORY_FILES),
            *release.manifest("take_vrs_noimagestream")[take.take_uid].paths,
        ]

    def remote_sequences(self) -> list[RemoteSequence]:
        """Every take with an HM fit, sized by its fit and its take files; needs the metadata and the keys."""
        fits: dict[str, HfFileInfo] = hm_fits()
        takes: dict[str, Take] = read_takes(self.config.root / "takes.json")
        result: list[RemoteSequence] = []
        for name in sorted(set(fits) & set(takes)):
            files: list[ManifestPath] = self.take_files(takes[name])
            result.append(
                RemoteSequence(name, fits[name].size_bytes + sum(path.size or 0 for path in files), tuple(path.relative_path for path in files))
            )
        return result

    def download(self) -> None:
        """Fetch the body models, the release metadata and the selected takes' fits; take files come at convert."""
        report: FetchReport = FetchReport()
        fetched_models: int = fetch_models(self.config.models)
        release: Release = self.open_release()
        for path in release.manifest("metadata")["takes"].paths + release.manifest("metadata")["captures"].paths:
            report.count(release.fetch(path, self.config.root), path.size or 0)
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
        """Fetch the take's files and its capture's timesync.csv, concurrently, unless already present."""
        release: Release = self.open_release()
        timesync: list[ManifestPath] = [
            path for path in release.manifest("captures")[take.capture_uid].paths if path.relative_path.endswith("/timesync.csv")
        ]
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
            list(executor.map(lambda path: release.fetch(path, self.config.root), [*self.take_files(take), *timesync]))

    def prefetch(self, identity: SequenceIdentity, source: Take, *, force: bool) -> None:
        """Fetch the next take's raw files while this one converts; only base reads them."""
        if not writing.should_skip(self.targets(identity)[paths.BASE_LAYER], force=force):
            self.fetch_take(source)

    def base_inputs(self, take: Take) -> BaseInputs:
        """Read the raw take: localized GoPros, the clock, the Aria pose at each frame and its calibration."""
        take_dir: Path = self.config.root / take.root_dir
        clock: TakeClock = read_take_clock(self.config.root / "captures" / take.capture.capture_name / "timesync.csv", take)
        times: Int64[ndarray, "n"] = clock.times_ns[: self.config.frame_limit]
        (vrs,) = [path for path in self.take_files(take) if path.relative_path.endswith(".vrs")]
        return BaseInputs(
            take=take,
            root=self.config.root,
            gopros=localized(read_gopro_calibs(take_dir / "trajectory/gopro_calibs.csv")),
            calib_json=VrsFile(self.config.root / vrs.relative_path).file_tags["calib_json"],
            frames=FrameSidecar(times, aria.read_trajectory(take_dir / "trajectory/closed_loop_trajectory.csv").at(times)),
            take_frames=len(clock.times_ns),
            clock_filled=clock.filled,
        )

    def convert(self, identity: SequenceIdentity, source: Take, *, force: bool) -> Path:
        """Base first (fetch, encode, publish, sidecar, prune the take), then the fit's layers from the sidecar."""
        targets, pending = self.pending_layers(identity, force=force, roots=[paths.NAS_ROOT, self.config.root])
        if not pending:
            return targets[paths.BASE_LAYER]
        base: Path = targets[paths.BASE_LAYER]
        if paths.BASE_LAYER in pending:
            pending = list(targets)  # every derived layer is stamped on base's clock: a new base rebuilds them all
        if paths.BODY_MESH_LAYER in pending and self.model is None:
            self.model = SmplhModel(self.config.models)  # fails on a missing model before any fetch or encode
        sidecar: Path = paths.sidecar_path(base.parents[1], identity, SIDECAR)  # a preview's root holds its own sidecar
        if paths.BASE_LAYER in pending:
            with self.timer.stage("fetch"):
                self.fetch_take(source)
                inputs: BaseInputs = self.base_inputs(source)
            written: list[CamerasSidecar] = []
            self.write_layers(
                identity,
                targets,
                [paths.BASE_LAYER],
                {paths.BASE_LAYER: lambda recording: written.append(write_base(recording, identity, inputs, self.timer))},
            )
            write_sidecar(sidecar, written[0], inputs.frames)
            if not self.config.keep_raw:
                for path in self.take_files(source):
                    (self.config.root / path.relative_path).unlink(missing_ok=True)
            self.timer.capture_s = len(inputs.frames.times_ns) / FPS

        @cache
        def fit_rows() -> tuple[FrameSidecar, CamerasSidecar, HmFit, Int64[ndarray, "t"]]:
            """The sidecars, the fit, and the frames both cover; read once, after base."""
            frames, cameras = read_sidecar(sidecar, base)
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
            assert self.model is not None
            write_body_mesh(recording, self.model, fit, frames.times_ns[rows], rows)

        def projections(recording: rr.RecordingStream) -> None:
            frames, cameras, fit, rows = fit_rows()
            write_projections(recording, fit, frames, cameras, rows)

        derived: list[str] = [layer for layer in pending if layer != paths.BASE_LAYER]
        self.write_layers(
            identity, targets, derived, {paths.BODY_POSE_LAYER: body_pose, paths.BODY_MESH_LAYER: body_mesh, paths.PROJECTIONS_LAYER: projections}
        )
        if paths.BASE_LAYER not in pending:
            self.timer.capture_s = len(fit_rows()[0].times_ns) / FPS
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        """Scene from behind GoPro 1; the Aria's cameras in a column; the GoPros along the bottom (lens-projected keypoints)."""
        projected: list[str] = [f"- {schema.coco133_uv_projected_path(rig, 0)}" for rig in range(1, EXO_SLOTS + 1)]
        projected += [f"- {schema.coco133_uv_projected_path(EGO_RIG, cam)}" for cam in range(len(ARIA_STREAMS))]
        return blueprints.exoego_blueprint(
            rrb.Spatial3DView(
                name="Scene",
                origin=schema.cam_path(1, 0),
                contents=["+ /world/**", *projected, *UNCALIBRATED],
                eye_controls=SCENE_EYE,
                line_grid=False,
            ),
            ego_panes=[blueprints.camera_view(stream.label or "camera-et", EGO_RIG, cam) for cam, stream in enumerate(ARIA_STREAMS)],
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
