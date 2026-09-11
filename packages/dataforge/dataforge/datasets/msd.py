"""Monado SLAM Datasets (MSD): download-on-demand VR headset captures → a base and a gt rrd per sequence.

Upstream is the HuggingFace dataset ``collabora/monado-slam-datasets`` (CC-BY 4.0),
a tree of per-sequence zip archives::

    M_monado_datasets/<device_dir>/<collection>/<SEQ>.zip     (or .z01 .z02 … .zip)
    M_monado_datasets/<device_dir>/extras/calibration.json

Inside an archive: ``<SEQ>/mav0/cam<N>/data/<ts>.png`` grayscale frames plus a
``cam<N>/data.csv`` index, ``imu0/data.csv`` (~1 kHz), ``gt/data.csv`` (~1 kHz),
and on the Reverb G2 / Odyssey+ a 50 Hz ``mag0/data.csv``. Where a stream also
ships ``data.raw.csv`` / ``data.extra.csv`` siblings, the converter reads only
``data.csv``.

Five invariants hold across everything below, and every one of them is a thing
this converter would break silently rather than loudly:

* **Raw is never kept.** The corpus is ~350 GB of PNG sequences, so ``discover``
  enumerates the *remote* tree and ``convert`` fetches exactly one sequence,
  streams its PNGs through the AV1 encoder, and deletes the archive again.
  ``enforce_raw_budget`` caps the scratch directory so a batch run cannot fill
  the NVMe with leftovers from failed sequences.
* **One clock, no resampling.** Every csv timestamp is nanoseconds on one
  monotonic device clock (values around 1e13, not a Unix epoch). ``video_time``
  is that clock minus ``start_time_ns``, the earliest sample of any stream
  *including* ``gt``, because the two layers must share an origin.
* **The rig frame is the IMU frame.** ``rig_T_cam`` is basalt's ``T_imu_cam`` with
  no inversion, and the rig node states ``reference = "imu_00"``.
* **Frames are encoded upright.** Each camera is rotated by the quarter turn that
  aligns its image-up with the headset's up, both derived from the calibration,
  and its calibration is rotated with it; the turn is stored as
  ``image_rotation_cw_deg`` on the camera node. One uniform rule, so the Index's
  and the Odyssey+'s upright cameras answer zero turns and are untouched, while
  all four of the Reverb G2's rolled cameras stop showing the room sideways.
* **Two per-device claims are not in the corpus** — the world up axis and the
  follow frame. ``MSD_DEVICES`` holds both; ``convert`` re-checks each against
  the sequence in front of it and *warns*, because every rrd of a device must
  agree and one disagreeing is news rather than a reason to reorient it alone.

Each sequence becomes a ``base`` rrd (video, IMU, magnetometer), a ``gt`` rrd
(``world_T_rig`` at the full ~1 kHz rate, plus its path and trail) and a
``gt.csv`` sidecar, all under one recording id, following the rule in
``packages/dataforge/README.md#the-layer-rule``. ``--device`` picks the corpus
*and* the catalog dataset (``msd-index``, ``msd-g2``, ``msd-odyssey``).

What was measured, what it answered, and why each format decision went the way
it did is in ``packages/dataforge/docs/msd.md``. Three modules hold what is not
MSD-specific: ``dataforge.archives`` reads archive members,
``dataforge.basalt`` validates the calibration, and ``dataforge.euroc`` decodes
the csv streams. What is left here is the device table, the world-up measurement
its claims rest on, and the two layers.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import rerun.blueprint as rrb
from huggingface_hub import HfApi, RepoFile
from jaxtyping import Float64
from numpy import ndarray

from dataforge import blueprints, paths, schema, transports, writing
from dataforge.archives import group_archives, open_member_reader, remove_tree
from dataforge.basalt import CalibratedCamera, FollowFrame, follow_frame, load_calibration
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.datasets.msd_layers import (
    GT_SIDECAR_NAME,
    IMU,
    MAG,
    RIG,
    GtSummary,
    MeasuredUp,
    MsdDevice,
    MsdDeviceChoice,
    SequenceStreams,
    read_sequence,
    write_base_layer,
    write_gt_layer,
)
from dataforge.identity import SequenceIdentity
from dataforge.video_encoding import require_av1_nvenc, resolve_ffmpeg

REPO_ID: str = "collabora/monado-slam-datasets"
"""HuggingFace dataset repo holding every MSD device."""
REPO_ROOT: str = "M_monado_datasets"
"""Top-level directory inside the repo; every device tree hangs off it."""
FOLLOW_BACK_M: float = 0.9
"""How far behind the headset the follow eye sits, along the device's own forward."""
FOLLOW_UP_M: float = 0.45
"""How far above the headset the follow eye sits, along the device's own up."""
FOLLOW_AHEAD_M: float = 0.3
"""How far ahead of the headset the follow eye aims, so the shot leads the motion."""
FOLLOW_FRAME_TOLERANCE_DEG: float = 5.0
"""How far a declared ``FollowFrame`` axis may sit from the calibration's before ``convert`` warns."""


MSD_DEVICES: dict[MsdDeviceChoice, MsdDevice] = {
    "index": MsdDevice(
        hf_dir="MI_valve_index",
        collections=(
            "MIO_others",
            "MIP_playing/MIPB_beat_saber",
            "MIP_playing/MIPP_pistol_whip",
            "MIP_playing/MIPT_thrill_of_the_fight",
        ),
        num_cameras=2,
        has_magnetometer=False,
        label="valve-index",
        # Measured on MIO09_short_1_updown: +y at 0.96 of |g|. OpenVR worlds are
        # right-handed Y-up, so Lighthouse ground truth agreeing is expected. The
        # measurement is not a restatement of the raw samples: over that window the
        # accelerometer's own mean points along the headset's -x, and only the gt
        # rotation (123..142 deg from identity there) turns it into world +y.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06: the front pair looks along rig +z
        # and its 13.4 cm baseline runs along rig y, so up is rig -x — the axis MIO09's
        # raw accelerometer mean already picked out. The Index alone carries no camera
        # roll: image-up lands within 1.2 deg of this up.
        follow=FollowFrame(forward=(0.009, 0.0, 1.0), up=(-1.0, 0.001, 0.009)),
        gt_source="lighthouse",
    ),
    "g2": MsdDevice(
        hf_dir="MG_reverb_g2",
        collections=("MGO_others",),
        num_cameras=4,
        has_magnetometer=True,
        label="reverb-g2",
        # Measured on MGO09_short_1_updown: +y at 0.98 of |g|. The MoCap rig
        # documents no convention, and the sign comes entirely from the gt
        # rotation — the raw accelerometer mean points along the headset's -y.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06 over cam0/cam1, the front pair
        # (cam2 and cam3 look along rig -x and +x, i.e. sideways). Their 10.8 cm baseline
        # runs along rig x, so up is rig -y — again where MGO09's raw accelerometer mean
        # points, and it puts the front pair 15.6 deg below the horizon, where tracking
        # cameras are aimed. All four G2 cameras are mounted rolled: image-up is rig +x,
        # a full 90 deg off this up, which is why the baseline and not image-up fixes it.
        follow=FollowFrame(forward=(-0.001, 0.268, 0.963), up=(0.006, -0.963, 0.268)),
        gt_source="mocap",
    ),
    "odyssey": MsdDevice(
        hf_dir="MO_odyssey_plus",
        collections=("MOO_others",),
        num_cameras=2,
        has_magnetometer=True,
        label="odyssey-plus",
        # Measured on MOO09_short_1_updown: +y at 0.93 of |g|; same undocumented
        # MoCap rig as the G2, and the same Y-up answer.
        world_up="+y",
        # follow_frame(calibration.json) on 2026-09-06: a 10.6 cm baseline along rig x
        # and an optical axis pitched 21.3 deg down from it, so up is rig -y. Like the
        # Index and unlike the G2 these cameras are upright — image-up is 0.3 deg away.
        follow=FollowFrame(forward=(0.002, 0.364, 0.932), up=(0.0, -0.932, 0.364)),
        gt_source="mocap",
    ),
}
"""The three headsets MSD covers, keyed by the ``--device`` literal.

``world_up`` and ``follow`` are claims about data rather than facts read out of
the corpus, and this is where both are settled. Each device's comment above
records what its own answer came from; the evidence behind all three, and why the
baseline and not image-up fixes the follow frame, is in
``packages/dataforge/docs/msd.md``.

They are written down here rather than derived at use because ``register`` builds
a device's default blueprint from the registry alone — no sequence, no
calibration on disk — so the eye and the world axes have to be placeable without
either. ``warn_on_follow_frame`` and ``warn_on_world_up`` re-check them per
sequence and warn rather than re-aiming or reorienting a single rrd out of step
with the rest.
"""


@dataclass(frozen=True, slots=True)
class MsdSource:
    """One remote sequence as discovery found it; ``convert`` needs nothing else."""

    collection: str
    """Leaf name of the collection the archive sits in, e.g. ``MIO_others``; also the identity's first part."""
    sequence: str
    """Archive stem, which is also the top-level directory inside it (e.g. ``MIO09_short_1_updown``)."""
    archive_paths: tuple[str, ...]
    """Repo-relative archive files, ``.z01``… first and the closing ``.zip`` last."""
    archive_bytes: int
    """Total download size of ``archive_paths``, from the repo listing."""


@dataclass
class MsdConfig(DataforgeDatasetConfig):
    """Monado SLAM Datasets: one VR headset's stereo/quad camera + IMU sequences."""

    command: ClassVar[str] = "msd"
    """Registry key and CLI subcommand; ``--device`` picks the catalog dataset."""

    _target: type = field(default_factory=lambda: MsdDataset)
    """Dataset class instantiated by ``setup()``."""
    device: MsdDeviceChoice = "index"
    """Headset to work on; also the catalog dataset (``msd-<device>``)."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "msd")
    """Scratch directory for archives, extractions and temp mp4s. Point it at
    local NVMe: every byte written here is deleted again after the sequence."""
    keep_raw: bool = False
    """Keep the fetched archive and the encoded mp4s instead of deleting them."""
    raw_budget_gb: float = 50.0
    """Cap on what ``root`` may hold. A sequence whose archives alone exceed it is
    processed anyway with a warning; leftovers that would breach it are an error."""
    revision: str | None = None
    """Repo branch, tag or commit to resolve; ``None`` takes the default branch.
    Only an input: what every hub call and every rrd records is the *sha* it
    resolves to (see ``MsdDataset.commit_sha``)."""

    @property
    def name(self) -> str:
        """Catalog dataset and identity ``dataset`` part: one per device layout."""
        return f"{self.command}-{self.device}"


def list_collection_files(repo_id: str, collection_path: str, revision: str | None = None) -> list[tuple[str, int]]:
    """List one remote collection directory as ``(repo-relative path, size in bytes)``.

    The whole HF listing surface of this dataset, isolated so tests can replace
    it with a literal tree.

    Args:
        repo_id: Hub dataset repo, normally ``REPO_ID``.
        collection_path: Repo-relative directory to list (not recursive).
        revision: Branch, tag or commit; ``None`` takes the default branch.

    Returns:
        One entry per *file* directly inside the directory; subdirectories are dropped.
    """
    entries = HfApi().list_repo_tree(repo_id, path_in_repo=collection_path, repo_type="dataset", revision=revision)
    return [(entry.path, entry.size) for entry in entries if isinstance(entry, RepoFile)]


def repo_revision(repo_id: str, revision: str | None = None) -> str | None:
    """Resolve a branch/tag to the commit sha stamped into every converted rrd.

    Isolated from ``list_collection_files`` because ``convert`` needs it without
    a listing, and a test needs it without a network.
    """
    return HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha


def follow_eye(follow: FollowFrame) -> rrb.EyeControls3D:
    """The device's chase camera: its own forward and up at this package's distances."""
    return blueprints.follow_eye_controls(follow.forward, follow.up, back_m=FOLLOW_BACK_M, up_m=FOLLOW_UP_M, ahead_m=FOLLOW_AHEAD_M)


def camera_views(num_cameras: int) -> list[rrb.Spatial2DView]:
    """One 2D pane per camera, labelled the way the archives name them (``cam0``…)."""
    return [blueprints.camera_view(f"cam{index}", RIG, index) for index in range(num_cameras)]


def build_blueprint(num_cameras: int, *, has_magnetometer: bool, follow: FollowFrame) -> rrb.Blueprint:
    """Default layout: the rig in 3D beside a camera grid, over the sensor plots.

    Args:
        num_cameras: Cameras the device carries; the grid holds one pane each.
        has_magnetometer: Whether to add the third plot pane.
        follow: The device's forward and up in the rig frame; it orients the Follow view.

    Returns:
        The blueprint embedded in every base-layer rrd of this device and
        registered as the catalog dataset's default.
    """
    plots: list[rrb.TimeSeriesView] = [
        blueprints.sensor_plot(name, origin, contents)
        for name, origin, contents in (
            ("Gyroscope", schema.imu_path(RIG, IMU), schema.gyro_path(RIG, IMU)),
            ("Accelerometer", schema.imu_path(RIG, IMU), schema.accel_path(RIG, IMU)),
        )
    ]
    if has_magnetometer:
        plots.append(blueprints.sensor_plot("Magnetometer", schema.mag_path(RIG, MAG), schema.field_path(RIG, MAG)))
    return blueprints.rig_blueprint(
        camera_views(num_cameras),
        rig=RIG,
        run_source=schema.GT_RUN_SOURCE,
        eye_controls=follow_eye(follow),
        plots=plots,
    )


def build_table_blueprint(num_cameras: int, *, follow: FollowFrame) -> rrb.Blueprint:
    """Segment-table preview card: the 3D rig with no video textures, plus ``cam0``.

    Args:
        num_cameras: Cameras the device carries; all but ``cam0``'s video are excluded.
        follow: The device's forward and up in the rig frame; it orients the Follow view.
    """
    return blueprints.table_blueprint(
        num_cameras,
        rig=RIG,
        run_source=schema.GT_RUN_SOURCE,
        eye_controls=follow_eye(follow),
        front_pane=blueprints.camera_view("cam0", RIG, 0),
    )


@dataclass(frozen=True, slots=True)
class SequencePaths:
    """Where one sequence's conversion reads and writes; ``convert`` works these out once."""

    archives: tuple[Path, ...]
    """Local archive volume paths, parts first and the closing ``.zip`` last."""
    work_dir: Path
    """Scratch directory for the encoded mp4s and any extraction; deleted unless ``--keep-raw``."""
    target: Path
    """Final base-layer rrd path."""
    gt_target: Path
    """Final gt-layer rrd path, a sibling of the same recording id."""
    sidecar: Path
    """Final ``gt.csv`` sidecar path: the one archive member the gt layer still needs
    once the archive itself is gone."""


class MsdDataset(DataforgeDataset[MsdConfig, MsdSource]):
    """Converts one MSD headset's sequences into a base and a gt layer each."""

    def __init__(self, config: MsdConfig) -> None:
        super().__init__(config)
        self.device: MsdDevice = MSD_DEVICES[config.device]

    def discover(self) -> list[tuple[SequenceIdentity, MsdSource]]:
        """Enumerate the **remote** tree: raw archives never stay on disk to walk.

        Collections are listed in the device's declared order and sequences
        sorted within each, so the same repo revision always yields the same list.
        """
        pairs: list[tuple[SequenceIdentity, MsdSource]] = []
        for collection in self.device.collections:
            collection_path: str = f"{REPO_ROOT}/{self.device.hf_dir}/{collection}"
            entries: list[tuple[str, int]] = list_collection_files(REPO_ID, collection_path, revision=self.commit_sha)
            leaf: str = collection.rsplit("/", 1)[-1]
            for stem, volumes in group_archives(entries).items():
                pairs.append(
                    (
                        SequenceIdentity(dataset=self.config.name, parts=(leaf, stem)),
                        MsdSource(
                            collection=leaf,
                            sequence=stem,
                            archive_paths=tuple(path for path, _ in volumes),
                            archive_bytes=sum(size for _, size in volumes),
                        ),
                    )
                )
        return pairs

    def default_blueprint(self) -> rrb.Blueprint:
        """Device-wide layout: every sequence of one headset has the same sensors."""
        return build_blueprint(self.device.num_cameras, has_magnetometer=self.device.has_magnetometer, follow=self.device.follow)

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the device's segment table."""
        return build_table_blueprint(self.device.num_cameras, follow=self.device.follow)

    @property
    def calibration_path(self) -> str:
        """Repo-relative path of the device's basalt calibration."""
        return f"{REPO_ROOT}/{self.device.hf_dir}/extras/calibration.json"

    @functools.cached_property
    def commit_sha(self) -> str:
        """The one repo commit this whole run reads, resolved once per dataset instance.

        ``config.revision`` is only the *input*: a branch name moves, so listing
        a collection on ``main``, fetching an archive on ``main`` an hour later
        and stamping a third answer into the rrd could describe three different
        trees. Resolving the branch to a sha up front and passing that sha to
        every hub call makes the whole conversion one commit, and makes the
        recorded ``hf_revision`` the sha the bytes actually came from.

        Lazily, and deliberately not inside a recording: a batch run asks the hub
        once instead of once per sequence, and ``convert`` warms it before it
        encodes anything, so a transient network failure cannot land on a
        finished encode.

        Raises:
            RuntimeError: The hub named no sha for ``config.revision``, so there
                is nothing to pin the conversion to.
        """
        resolved: str | None = repo_revision(REPO_ID, self.config.revision)
        if resolved is None:
            named: str = self.config.revision or "the default branch"
            raise RuntimeError(f"{REPO_ID} resolved no commit sha for {named}; a conversion has to name the tree it read")
        return resolved

    def fetch_calibration(self) -> Path:
        """Fetch the device's ``calibration.json`` into ``root`` and return where it landed.

        The one place the calibration is pulled from the hub: ``download`` calls
        it to have the file up front, and ``calibration()`` calls it when a
        ``convert`` runs against a scratch dir that ``download`` never touched.
        """
        transports.hf_fetch(REPO_ID, allow_patterns=[self.calibration_path], local_dir=self.config.root, revision=self.commit_sha)
        return self.config.root / self.calibration_path

    def calibration(self) -> tuple[CalibratedCamera, ...]:
        """Load and validate the device calibration, fetching it if ``download`` was skipped.

        The camera count is checked against the device table here rather than at
        each use: a file that lists a different number of cameras is the wrong
        file or a corpus change, and either way the per-camera loop below would
        otherwise walk off the end of one list or quietly ignore a camera.
        """
        local_path: Path = self.config.root / self.calibration_path
        if not local_path.is_file():
            local_path = self.fetch_calibration()
        return load_calibration(local_path, expected_cameras=self.device.num_cameras)

    def download(self) -> None:
        """Fetch the calibration, prove the machine can encode, and print the plan.

        Deliberately **not** a bulk fetch: the corpus is hundreds of gigabytes of
        PNG archives and ``convert`` pulls one sequence at a time, so the only
        thing worth having up front is the few-kilobyte calibration. The NVENC
        check is here because a machine that cannot encode AV1 should find out
        now rather than after the first multi-gigabyte download.
        """
        local_calibration: Path = self.fetch_calibration()
        require_av1_nvenc(resolve_ffmpeg())
        discovered: list[tuple[SequenceIdentity, MsdSource]] = self.discover()
        per_collection: dict[str, list[MsdSource]] = {}
        for _, source in discovered:
            per_collection.setdefault(source.collection, []).append(source)
        total_bytes: int = sum(source.archive_bytes for _, source in discovered)
        print(f"{self.config.name}: {len(discovered)} sequence(s), {total_bytes / 1e9:.1f} GB of archives to stream through convert")
        for collection, sources in per_collection.items():
            print(f"  {collection}: {len(sources)} sequence(s), {sum(source.archive_bytes for source in sources) / 1e9:.1f} GB")
        print(f"  calibration → {local_calibration}; archives are fetched and deleted one sequence at a time")

    def sequence_paths(self, identity: SequenceIdentity, source: MsdSource) -> SequencePaths:
        """Everywhere this sequence's conversion touches disk, worked out in one place."""
        output_root: Path = paths.output_root()
        return SequencePaths(
            archives=tuple(self.config.root / archive_path for archive_path in source.archive_paths),
            work_dir=self.config.root / "work" / source.sequence,
            target=paths.rrd_path(output_root, layer=paths.BASE_LAYER, identity=identity),
            gt_target=paths.rrd_path(output_root, layer=paths.GT_LAYER, identity=identity),
            sidecar=paths.sidecar_path(output_root, identity, GT_SIDECAR_NAME),
        )

    def enforce_raw_budget(self, source: MsdSource, archives: Sequence[Path]) -> None:
        """Refuse to fetch when ``root`` plus this sequence would breach the budget.

        The cap covers this sequence's archives **plus** whatever else is already
        in ``root``. Two things ride on top of it and are not counted: one
        camera's extracted PNGs (split archives only) and the temp mp4s, both of
        which live in ``work/`` for the length of one sequence.

        A sequence whose archives *alone* exceed the budget (the Index and G2 long
        sessions) is an accepted exception and only warns — there is no smaller
        unit to convert — but its leftovers are still tallied, because leftovers
        that breach the cap on their own are an error whatever is being fetched.
        HuggingFace's own bookkeeping under ``root/.cache`` (including resumable
        ``.incomplete`` blobs) is excluded: it is not a leftover anyone should delete.

        Args:
            source: Sequence about to be fetched.
            archives: Its local volume paths, excluded from the on-disk tally so
                a retry of an already-downloaded sequence is not double-counted.

        Raises:
            RuntimeError: What is in ``root`` would push the fetch over the budget.
        """
        budget_bytes: int = int(self.config.raw_budget_gb * 1e9)
        oversized: bool = source.archive_bytes > budget_bytes
        if oversized:
            print(
                f"  warning: {source.sequence} needs {source.archive_bytes / 1e9:.1f} GB of archives, above the "
                f"{self.config.raw_budget_gb:g} GB raw budget; converting it anyway as an accepted exception"
            )
        if not self.config.root.is_dir():
            return

        # One walk, one stat per file: this runs before every fetch, and a scratch
        # dir mid-batch holds a whole sequence's PNGs.
        cache_dir: Path = self.config.root / ".cache"
        own: set[Path] = set(archives)
        leftovers: list[tuple[Path, int]] = []
        for directory, subdirectories, names in os.walk(self.config.root):
            here: Path = Path(directory)
            if here == cache_dir:
                subdirectories.clear()
                continue
            for name in names:
                path: Path = here / name
                if path not in own:
                    leftovers.append((path, os.stat(path).st_size))
        occupied: int = sum(size for _, size in leftovers)
        # An oversized sequence has already been waved through, so only its
        # leftovers are still weighed against the cap.
        needed: int = 0 if oversized else source.archive_bytes
        if occupied + needed <= budget_bytes:
            return
        biggest: list[tuple[Path, int]] = sorted(leftovers, key=lambda entry: entry[1], reverse=True)[:3]
        named: str = ", ".join(f"{path.relative_to(self.config.root)} ({size / 1e9:.2f} GB)" for path, size in biggest)
        raise RuntimeError(
            f"{self.config.root} already holds {occupied / 1e9:.2f} GB of leftovers and {source.sequence} needs "
            f"{source.archive_bytes / 1e9:.2f} GB more, over the {self.config.raw_budget_gb:g} GB raw budget. "
            f"Delete the leftovers of an earlier failed sequence first: {named}"
        )

    def convert(self, identity: SequenceIdentity, source: MsdSource, *, force: bool) -> Path:
        """Write whichever of the two layers is missing, fetching only if base needs it.

        Per README#the-layer-rule the **base** layer is the only step that needs
        the archive: it is a faithful conversion of the raw sequence, and once it
        is published the multi-gigabyte source is deleted. What the **gt** layer
        needs then is a ~1 kHz csv, so that one member is kept verbatim as a
        sidecar and gt is rebuilt from it plus the base rrd — no network, no
        encode, seconds instead of an hour. Three cases follow:

        * both rrds exist → nothing to do.
        * base exists, gt does not, the sidecar does → rebuild gt alone.
        * anything else → fetch, encode, and stage all three.

        A full conversion stages base, gt and the sidecar in temp files and
        publishes them back to back, so a corpus never holds a base rrd whose gt
        pass died halfway; nothing is ever unlinked first, because a catalog
        server may hold the current file open. ``--force`` bypasses the skip
        checks and nothing else.

        Failure keeps the archive (a retry then skips the multi-gigabyte
        download) but removes the mp4s and any extraction directory, so the next
        attempt starts from a clean scratch dir. Nothing outside ``root`` is ever
        deleted.

        Returns:
            The base rrd path, because that is the layer every dataforge verb keys on.
        """
        locations: SequencePaths = self.sequence_paths(identity, source)
        base_done: bool = writing.should_skip(locations.target, force=force)
        gt_done: bool = writing.should_skip(locations.gt_target, force=force)
        if base_done and gt_done:
            print(f"skip {identity.sequence_key} → {locations.target} + {locations.gt_target}")
            return locations.target
        if base_done and locations.sidecar.is_file():
            with writing.atomic_write(locations.gt_target) as staged_gt:
                rebuilt: GtSummary = write_gt_layer(
                    identity, staged_gt, base_rrd=locations.target, sidecar=locations.sidecar, profile=self.device
                )
            self.warn_on_world_up(source, rebuilt.measured)
            print(
                f"rebuilt gt {identity.sequence_key} → {locations.gt_target} "
                f"({rebuilt.num_poses} gt poses from {locations.sidecar} and {locations.target}, no fetch)"
            )
            return locations.target
        if base_done:
            print(f"  {locations.target} exists but {locations.sidecar} does not, so gt cannot be rebuilt from it; fetching the archive again")

        self.enforce_raw_budget(source, locations.archives)
        # Both hub lookups happen before the archives are pulled and long before a
        # frame is encoded: neither a bad calibration nor a transient revision
        # lookup may throw away a multi-gigabyte download and an hour of encoding.
        # (The calibration fetch resolves the sha on its own; this only makes the
        # ordering explicit for a scratch dir that already holds the file.)
        _ = self.commit_sha
        cameras: tuple[CalibratedCamera, ...] = self.calibration()
        self.warn_on_follow_frame(cameras)

        on_disk: int = sum(1 for archive in locations.archives if archive.is_file())
        if on_disk:
            print(f"  {on_disk}/{len(locations.archives)} archive volume(s) already in {self.config.root}; the fetch only verifies them")
        transports.hf_fetch(REPO_ID, allow_patterns=list(source.archive_paths), local_dir=self.config.root, revision=self.commit_sha)

        locations.work_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Every artifact is staged and the three publications happen as these
            # contexts unwind — base first, then gt, then the sidecar. The gt step
            # reads the *staged* base rrd and the *staged* sidecar, so it runs the
            # same code as a rebuild does against the published pair.
            with (
                writing.atomic_write(locations.sidecar) as staged_sidecar,
                writing.atomic_write(locations.gt_target) as staged_gt,
                writing.atomic_write(locations.target) as staged_base,
            ):
                with open_member_reader(locations.archives, locations.work_dir) as reader:
                    streams: SequenceStreams = read_sequence(
                        reader,
                        source.sequence,
                        cameras,
                        profile=self.device,
                        work_dir=locations.work_dir,
                        staged_sidecar=staged_sidecar,
                    )
                write_base_layer(
                    identity,
                    streams,
                    staged_base,
                    profile=self.device,
                    device=self.config.device,
                    collection=source.collection,
                    hf_revision=self.commit_sha,
                    default_blueprint=self.default_blueprint(),
                )
                written: GtSummary = write_gt_layer(
                    identity, staged_gt, base_rrd=staged_base, sidecar=staged_sidecar, profile=self.device
                )
        except BaseException:
            remove_tree(locations.work_dir)
            retained: int = sum(archive.stat().st_size for archive in locations.archives if archive.is_file())
            print(f"  kept {retained / 1e9:.2f} GB of archives in {self.config.root} so a retry skips the download")
            raise
        self.warn_on_world_up(source, written.measured)

        if self.config.keep_raw:
            print(f"  keeping raw: {len(locations.archives)} archive volume(s) and the mp4s in {locations.work_dir}")
        else:
            remove_tree(locations.work_dir)
            for archive in locations.archives:
                archive.unlink(missing_ok=True)
        print(
            f"done {identity.sequence_key} → {locations.target} + {locations.gt_target} + {locations.sidecar} "
            f"({self.device.num_cameras} cameras, {streams.num_frames} frames, {written.num_poses} gt poses, "
            f"world up {written.measured.axis} at {written.measured.fraction:.2f} g)"
        )
        return locations.target

    def warn_on_follow_frame(self, cameras: Sequence[CalibratedCamera]) -> None:
        """Re-derive the declared follow frame from this calibration and say so on a disagreement.

        The claim is not re-applied: every rrd of a device must carry the same
        follow eye, so a calibration disagreeing is news rather than a reason to
        re-aim one rrd. Split from the world-up check because the two rest on
        different inputs — this one needs only the calibration, so it runs before
        the fetch, while the measurement needs a written base layer.
        See ``MSD_DEVICES``.
        """
        # The declared axes are rounded to three decimals and so are a hair short of
        # unit length; dividing by the norms keeps that rounding out of the angles.
        derived: FollowFrame = follow_frame(cameras)
        declared_axes: Float64[ndarray, "2 3"] = np.array([self.device.follow.forward, self.device.follow.up], dtype=np.float64)
        derived_axes: Float64[ndarray, "2 3"] = np.array([derived.forward, derived.up], dtype=np.float64)
        cosines: Float64[ndarray, "2"] = np.einsum("ij,ij->i", declared_axes, derived_axes) / np.linalg.norm(declared_axes, axis=1)
        deviations_deg: Float64[ndarray, "2"] = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
        if float(deviations_deg.max()) > FOLLOW_FRAME_TOLERANCE_DEG:
            print(
                f"  warning: {self.config.device} declares a follow frame its calibration disagrees with — "
                f"forward off by {deviations_deg[0]:.1f} deg, up off by {deviations_deg[1]:.1f} deg, over the "
                f"{FOLLOW_FRAME_TOLERANCE_DEG:g} deg tolerance; the blueprint still uses the declared frame"
            )

    def warn_on_world_up(self, source: MsdSource, measured: MeasuredUp) -> None:
        """Compare this sequence's own gravity measurement with the device's declared axis.

        Not re-applied either: every rrd of a device states the same root
        ``ViewCoordinates``, so one sequence measuring something else is news and
        not a silent per-sequence reorientation. See ``MSD_DEVICES``.
        """
        if measured.axis != self.device.world_up:
            print(
                f"  warning: {self.config.device} declares world_up {self.device.world_up} but {source.sequence} "
                f"measured {measured.axis} carrying {measured.fraction:.2f} of |g|; the rrd still states the declared axis"
            )

