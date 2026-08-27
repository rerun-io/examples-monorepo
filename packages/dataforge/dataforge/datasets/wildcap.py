"""WildCap: in-the-wild exo/ego mp4s — nothing else — → one exoego:v2 base rrd per capture.

Raw layout on disk (user-assembled; ``download`` only verifies it)::

    <root>/<capture-name>/
        exo/*.mp4     one file per static exo camera
        ego/*.mp4     one or more streams from a single head-mounted device

That is the whole contract: no calibration, no per-frame csvs, no IMU, no
poses. The ``.mp4`` match is case-insensitive; ``.mov`` files (iPhones ship
them) are warned about and skipped — Rerun does not support them, so remux
to mp4 first. Each exo video becomes its own single-camera rig (``rig_00..``
in stem order); every ego video lands on one final moving rig as ``cam_00..``
(also in stem order). The base layer therefore only asserts what the mp4s themselves
know:

* Videos go to the canonical ``.../cam_MM/pinhole/video`` paths, but the
  ``pinhole`` node carries **no** ``Pinhole`` — a calibration layer (e.g. a
  gravity-aligned VGGT + MoGe pass) logs intrinsics there later without any
  path change.
* No root ``ViewCoordinates`` and no transforms anywhere: nothing is posed,
  so the localization layer that first establishes a world frame owns both.
* The clock is each file's as-encoded PTS on ``video_time``, unshifted.
  Nothing guarantees the files are mutually synced; a sync layer can retime
  them later. Files trimmed to a common start (like the SelfCap cutting
  pipeline's output) line up as-is.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb

from dataforge import paths, schema, transports, writing
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import log_rig_node, log_video_stream

EGO_RIG_NAME: str = "ego"
"""Device label of the single moving rig; the raw tree names no device."""

def grid_page_size(max_height: int) -> int:
    """Cameras per grid page, by the sharpest stream in the group.

    Higher-resolution streams stay legible in fewer, larger panes: 8 panes for
    1080p and below, 4 for anything sharper (2 per page felt too sparse). A
    group larger than its page becomes tabs of grid pages (see
    ``build_blueprint``).
    """
    return 8 if max_height <= 1080 else 4


def video_height(video_path: Path) -> int | None:
    """Pixel height of the first video stream via the env's ffprobe, or ``None``
    when the file cannot be probed (corrupt, empty, or ffprobe missing)."""
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height", "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if probe.returncode != 0 or not probe.stdout.strip():
        return None
    # Some streams make ffprobe's writers emit trailing separators ("2560,"), so
    # keep only the leading digits of the first line.
    first: str = probe.stdout.strip().splitlines()[0].strip().rstrip(",")
    return int(first) if first.isdigit() else None


def group_page_size(videos: list[Path]) -> int:
    """Grid page size for one device group, from its sharpest probeable stream
    (mixed-resolution groups page for the stream that needs the most pixels).
    Falls back to the 1080p page when nothing probes."""
    heights: list[int] = [height for height in (video_height(path) for path in videos) if height is not None]
    return grid_page_size(max(heights)) if heights else 8


@dataclass
class WildcapConfig(DataforgeDatasetConfig):
    """In-the-wild exo/ego video captures: bare mp4s, no calibration, no metadata.

    One wildcap catalog dataset holds one topology family (one corpus root),
    because the catalog applies a single default blueprint to every segment of
    a dataset. A new corpus with a different camera layout gets its own
    ``--corpus`` (e.g. a 32-camera mocap stage beside the self-collected exoego
    captures), not more segments in an existing dataset.
    """

    _target: type = field(default_factory=lambda: WildcapDataset)
    """Dataset class instantiated by ``setup()``."""
    corpus: str = "selfcollected"
    """Topology family this run works on; the dataset is ``wildcap-<corpus>``."""
    root: Path = Path("data/raw/wildcap-selfcollected")
    """Corpus root holding one ``<capture-name>/{exo,ego}/*.mp4`` tree per capture.
    Pair it with ``corpus``: ``--corpus mamma32 --root data/raw/wildcap-mamma32``."""

    @property
    # pyrefly: ignore[bad-override]  # the ClassVar contract holds for fixed datasets; wildcap's name is per-corpus by design
    def name(self) -> str:
        """Catalog dataset name and identity ``dataset`` part: ``wildcap-<corpus>``."""
        return f"wildcap-{self.corpus}"


def group_videos(capture_dir: Path, group: str) -> list[Path]:
    """Readable mp4s of one device group, in stem order.

    Args:
        capture_dir: A ``<root>/<capture-name>`` directory.
        group: ``exo`` or ``ego``.

    Returns:
        Video paths sorted by filename. Unreadable files and ``.mov`` files
        (which Rerun does not support) are warned about and dropped.
    """
    if not (capture_dir / group).is_dir():
        return []
    videos: list[Path] = []
    for video_path in sorted((capture_dir / group).iterdir()):
        if video_path.suffix.lower() == ".mov":
            print(f"  warning: skipping {video_path} — Rerun does not support .mov; remux it to mp4 first")
            continue
        if video_path.suffix.lower() != ".mp4":
            continue
        if not os.access(video_path, os.R_OK):
            print(f"  warning: skipping unreadable {video_path}")
            continue
        videos.append(video_path)
    return videos


def build_blueprint(exo: list[str], ego: list[str], *, exo_page: int = 8, ego_page: int = 8) -> rrb.Blueprint:
    """Ego cameras over the exo cameras — SelfCap's layout minus the 3D scene
    and IMU strip, which would both be empty here.

    Each device group is a grid of at most one page of cameras; a larger group
    becomes tabs of grid pages, so a 32-camera capture pages instead of
    shrinking every pane into a sliver.

    Args:
        exo: Pane label per exo camera; label ``N`` is rig ``N``, camera 0.
        ego: Pane label per ego stream, all on rig ``len(exo)`` as camera 0..
        exo_page: Cameras per exo grid page (``grid_page_size`` of the group).
        ego_page: Cameras per ego grid page.

    Returns:
        The blueprint embedded at convert (real device names) and registered as
        the dataset default (index labels, see ``default_blueprint``).
    """
    def view(name: str, rig: int, cam: int) -> rrb.Spatial2DView:
        return rrb.Spatial2DView(name=name, origin=schema.pinhole_path(rig, cam), contents=f"{schema.pinhole_path(rig, cam)}/**")

    def paged(views: list[rrb.Spatial2DView], name: str, page: int) -> rrb.Container:
        if len(views) <= page:
            return rrb.Grid(*views, name=name)
        chunks: list[list[rrb.Spatial2DView]] = [views[start : start + page] for start in range(0, len(views), page)]
        return rrb.Tabs(*[rrb.Grid(*chunk, name=f"{name} {i * page + 1}-{i * page + len(chunk)}") for i, chunk in enumerate(chunks)], name=name)

    ego_views: list[rrb.Spatial2DView] = [view(name, len(exo), cam) for cam, name in enumerate(ego)]
    exo_views: list[rrb.Spatial2DView] = [view(name, rig, 0) for rig, name in enumerate(exo)]
    groups: list[rrb.Container] = [
        paged(views, name, page) for name, views, page in (("Ego", ego_views, ego_page), ("Exo", exo_views, exo_page)) if views
    ]
    return rrb.Blueprint(rrb.Vertical(*groups), collapse_panels=True)


class WildcapDataset(DataforgeDataset[WildcapConfig, Path]):
    """Converts bare exo/ego mp4 captures into exoego:v2 base-layer recordings."""

    def default_blueprint(self) -> rrb.Blueprint | None:
        """Corpus-derived dataset default: the first discovered capture's shape.

        The corpus is one topology family (see ``WildcapConfig``), so the first
        capture speaks for all of them. Device names still vary per capture, so
        panes are labeled by index; the per-recording blueprint embedded at
        convert uses real names. ``None`` when the corpus is empty.
        """
        discovered: list[tuple[SequenceIdentity, Path]] = self.discover()
        if not discovered:
            return None
        _, capture_dir = discovered[0]
        exo: list[Path] = group_videos(capture_dir, "exo")
        ego: list[Path] = group_videos(capture_dir, "ego")
        return build_blueprint(
            [f"exo {rig}" for rig in range(len(exo))],
            [f"ego {cam}" for cam in range(len(ego))],
            exo_page=group_page_size(exo),
            ego_page=group_page_size(ego),
        )

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card: the first exo camera's video, nothing else decoded."""
        return rrb.Blueprint(
            rrb.Spatial2DView(name="exo 0", origin=schema.pinhole_path(0, 0), contents=f"{schema.pinhole_path(0, 0)}/**"),
            collapse_panels=True,
        )

    def download(self) -> None:
        """Verify the local tree; WildCap is user-assembled and has no upstream fetch."""
        missing: list[str] = transports.local_verify(self.config.root, required=["*/*/*.mp4"])
        if missing:
            raise FileNotFoundError(f"WildCap tree at {self.config.root} is missing: {', '.join(missing)}")
        print(f"wildcap: {self.config.root} — {len(self.sequences())} convertible capture(s)")

    def discover(self) -> list[tuple[SequenceIdentity, Path]]:
        """Pair every capture directory holding at least one readable mp4 with its path."""
        pairs: list[tuple[SequenceIdentity, Path]] = []
        for capture_dir in sorted(path for path in self.config.root.glob("*") if path.is_dir()):
            if not any(group_videos(capture_dir, group) for group in ("exo", "ego")):
                continue
            pairs.append((SequenceIdentity(dataset=self.config.name, parts=(capture_dir.name,)), capture_dir))
        return pairs

    def convert(self, identity: SequenceIdentity, source: Path, *, force: bool) -> Path:
        """Write one capture's base-layer rrd: the videos, and deliberately nothing else.

        No root ``ViewCoordinates``, no ``Pinhole``, no transforms — see the
        module docstring; those belong to the calibration/localization layers.
        """
        target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
        if writing.should_skip(target, force=force):
            print(f"skip {identity.sequence_key} → {target}")
            return target

        # The blueprint (built before the recording opens) and the loops below derive
        # from the same two lists, so it can only describe cameras that really get logged.
        exo: list[Path] = group_videos(source, "exo")
        ego: list[Path] = group_videos(source, "ego")

        with writing.atomic_recording(
            target,
            application_id="dataforge",
            recording_id=identity.recording_id,
            default_blueprint=build_blueprint(
                [video_path.stem for video_path in exo],
                [f"ego {video_path.stem}" for video_path in ego],
                exo_page=group_page_size(exo),
                ego_page=group_page_size(ego),
            ),
        ) as recording:
            num_frames: int = 0
            for rig, video_path in enumerate(exo):
                log_rig_node(recording, rig, reference="cam_00", num_cameras=1, name=video_path.stem, kind="exo")
                rr.log(schema.cam_path(rig, 0), rr.AnyValues(name=video_path.stem), static=True, recording=recording)
                # Raw PTS is the only clock the raw tree has (see the module docstring), so no shift.
                num_frames = max(num_frames, log_video_stream(recording, video_path, schema.video_path(rig, 0)))
            if ego:
                ego_rig: int = len(exo)
                log_rig_node(recording, ego_rig, reference="cam_00", num_cameras=len(ego), name=EGO_RIG_NAME, kind="ego")
                for cam, video_path in enumerate(ego):
                    rr.log(schema.cam_path(ego_rig, cam), rr.AnyValues(name=video_path.stem), static=True, recording=recording)
                    num_frames = max(num_frames, log_video_stream(recording, video_path, schema.video_path(ego_rig, cam)))
            writing.send_capture_properties(recording, identity, num_cameras=len(exo) + len(ego), num_frames=num_frames)

        print(f"done {identity.sequence_key} → {target} ({len(exo) + len(ego)} cameras, {num_frames} frames)")
        return target
