"""LaMAria: ETH CVG's Aria Gen1 SLAM benchmark → one exoego:v2 base-layer rrd per sequence.

Upstream is a plain Apache-indexed archive at ``cvg-data.inf.ethz.ch/lamaria/``
(github.com/cvg/lamaria), four directories deep::

    raw_data/{training,test}/<seq>.vrs             897 MB … 10 GB per sequence
    aria_calibrations/{training,test}/<seq>.json   ~2.7 kB, the published body-frame calibration
    ground_truth/pseudo_dense/<seq>.txt           training only, 0.2–2.7 MB
    ground_truth/sparse/<seq>.json                only sequences with surveyed control points

The raw tree mirrors the **official** layout, so the upstream evaluation tools
run on it unchanged::

    <root>/<split>/<seq>/raw_data/<seq>.vrs
    <root>/<split>/<seq>/aria_calibrations/<seq>.json
    <root>/<split>/<seq>/ground_truth/pGT/<seq>.txt
    <root>/<split>/<seq>/ground_truth/control_points/<seq>.json

**Raw is scratch.** The default selection alone is 18.2 GB of VRS, so
``download`` fetches only the small files (a few MB) and writes a
``manifest.json`` recording what the archive held; ``convert`` then fetches
**one** VRS, encodes its three camera streams, writes the rrd, and deletes the
VRS and the temp mp4s again. ``--keep-raw`` keeps them. The archive is flaky (it
was down for hours during development), so every fetch resumes and a stalled one
is retried.

**Clocks.** Every VRS timestamp is Aria DEVICE time in nanoseconds, and the
published pGT and control points are stamped on that same clock. ``video_time``
is therefore that clock **unshifted**, so a ground-truth pose lines up with its
frame 1:1 with no retiming anywhere.

**Frames.** The rig frame is imu-right, which is what LaMAria's published
calibration uses as its body frame, so the rig node states
``reference = "imu_00"`` and every ``rig_T_sensor`` in ``dataforge.aria`` is
directly comparable with that file's ``T_b_s``. ``cam_00`` is camera-slam-left,
``cam_01`` camera-slam-right, ``cam_02`` camera-rgb; ``imu_00`` is imu-right
(identity ``rig_T_imu`` by construction) and ``imu_01`` imu-left, 129 mm away.
The base layer logs NO transform on the rig node and NO root
``ViewCoordinates``: the gt layer establishes the world frame and owns both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

import requests
import rerun.blueprint as rrb
import serde
import serde.json

from dataforge import paths, transports
from dataforge.datasets.base import DataforgeDataset, DataforgeDatasetConfig
from dataforge.identity import SequenceIdentity

LamariaSplit: TypeAlias = Literal["training", "test"]
"""Which half of the benchmark a sequence belongs to; only ``training`` ships ground truth."""

LamariaSet: TypeAlias = Literal["controlled", "additional"]
"""Upstream's two collections: the ``R_*`` controlled experimental set, and the
``sequence_<n>_<m>`` additional set recorded around the city."""

DEFAULT_SEQUENCES: tuple[str, ...] = ("R_01_easy", "R_04_medium", "R_11_5cp", "sequence_1_19", "sequence_4_11")
"""The five sequences dataforge converts by default: 18.2 GB of VRS spanning both
sets, all three difficulty tiers, one low-light capture, and every ground-truth
shape (pGT alone, and pGT with 5, 14 and 15 surveyed control points)."""

MANIFEST_NAME: str = "manifest.json"
"""What ``download`` writes under ``root`` and every other verb reads."""

RAW_DATA_DIR: str = "raw_data"
"""Archive (and local) directory holding the VRS files, per split."""
CALIBRATION_DIR: str = "aria_calibrations"
"""Archive (and local) directory holding the published body-frame calibrations, per split."""
PSEUDO_GT_REMOTE_DIR: str = "ground_truth/pseudo_dense"
"""Archive directory of the dense pseudo ground truth; it lands locally in ``ground_truth/pGT``."""
CONTROL_POINTS_REMOTE_DIR: str = "ground_truth/sparse"
"""Archive directory of the surveyed control points; it lands locally in ``ground_truth/control_points``."""
PSEUDO_GT_LOCAL_DIR: str = "ground_truth/pGT"
"""Where the official layout keeps the pGT inside a sequence directory."""
CONTROL_POINTS_LOCAL_DIR: str = "ground_truth/control_points"
"""Where the official layout keeps the control points inside a sequence directory."""

INDEX_TIMEOUT_S: float = 30.0
"""Per-request timeout for an index page; they are a few kilobytes of HTML."""


@serde.serde
@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One sequence as ``download`` resolved it from the archive's index pages."""

    sequence: str
    """Upstream sequence name, e.g. ``R_01_easy``; also the identity's only part."""
    split: LamariaSplit
    """Which ``raw_data/<split>/`` index listed the VRS."""
    vrs_url: str
    """Absolute URL ``convert`` fetches the VRS from."""
    vrs_display_bytes: int
    """Apache's **rounded** display size (``897M`` → 940 572 672). Good for a
    budget or a summary line and useless for verification, so it is never passed
    to ``http_fetch`` as an expected size."""
    has_pseudo_gt: bool
    """Whether ``ground_truth/pseudo_dense/`` lists this sequence (training only)."""
    has_control_points: bool
    """Whether ``ground_truth/sparse/`` lists this sequence (only surveyed ones)."""


@serde.serde
@dataclass(frozen=True, slots=True)
class LamariaManifest:
    """``<root>/manifest.json``: what the archive held when ``download`` last ran."""

    base_url: str
    """Archive root the records were resolved against."""
    sequences: list[SequenceRecord]
    """One record per selected sequence, sorted by name."""


@dataclass(frozen=True, slots=True)
class LamariaSource:
    """One sequence as discovery found it; ``convert`` needs nothing else from the tree."""

    sequence: str
    """Upstream sequence name."""
    split: LamariaSplit
    """``training`` or ``test``; part of the raw path and of the capture properties."""
    vrs_url: str
    """Where the VRS is fetched from on demand."""
    vrs_display_bytes: int
    """Apache's rounded size, for the fetch's progress line only."""
    vrs_path: Path
    """``<root>/<split>/<seq>/raw_data/<seq>.vrs``; deleted after convert unless ``--keep-raw``."""
    calibration_path: Path
    """``.../aria_calibrations/<seq>.json``, the published calibration; always kept."""
    pseudo_gt_path: Path | None
    """``.../ground_truth/pGT/<seq>.txt``, or ``None`` when the archive has none."""
    control_points_path: Path | None
    """``.../ground_truth/control_points/<seq>.json``, or ``None`` when the sequence was not surveyed."""


@dataclass
class LamariaConfig(DataforgeDatasetConfig):
    """LaMAria: Aria Gen1 egocentric SLAM sequences, fetched from ETH CVG on demand."""

    command: ClassVar[str] = "lamaria"
    """Registry key, catalog dataset name, and identity ``dataset`` part."""

    _target: type = field(default_factory=lambda: LamariaDataset)
    """Dataset class instantiated by ``setup()``."""
    root: Path = field(default_factory=lambda: paths.raw_root() / "lamaria")
    """Raw tree in the official layout. Point it at local NVMe: every VRS written
    here is deleted again once its rrd exists."""
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES
    """Sequences to download and convert; a name the archive does not list is an error."""
    keep_raw: bool = False
    """Keep the fetched VRS and the encoded mp4s instead of deleting them."""
    base_url: str = "https://cvg-data.inf.ethz.ch/lamaria/"
    """Archive root; every index page and every file hangs off it."""


def sequence_set(sequence: str) -> LamariaSet:
    """Which upstream collection a sequence name belongs to.

    Args:
        sequence: Upstream sequence name.

    Returns:
        ``controlled`` for the ``R_*`` controlled experimental set, ``additional``
        for the ``sequence_<n>_<m>`` captures.
    """
    return "controlled" if sequence.startswith("R_") else "additional"


def sequence_challenge(sequence: str) -> str | None:
    """The difficulty tier the controlled set encodes in its own name.

    ``R_01_easy`` → ``easy``, ``R_11_5cp`` → ``5cp``. The additional set names
    carry no tier (``sequence_1_19``), so they get ``None`` rather than a guess.

    Args:
        sequence: Upstream sequence name.

    Returns:
        The trailing name token for the controlled set, else ``None``.
    """
    if sequence_set(sequence) != "controlled":
        return None
    return sequence.rsplit("_", 1)[-1]


class LamariaDataset(DataforgeDataset[LamariaConfig, LamariaSource]):
    """Converts LaMAria sequences into exoego:v2 base-layer recordings."""

    def index(self, directory: str) -> list[transports.IndexEntry]:
        """List one Apache index page under ``base_url`` as ``(name, size_bytes)``.

        Args:
            directory: Archive-relative directory, e.g. ``raw_data/training``.

        Returns:
            One entry per listed file, in page order.

        Raises:
            requests.HTTPError: If the archive answered 4xx/5xx — the flaky-archive
                case, and one worth failing on rather than treating as "empty".
        """
        url: str = f"{self.config.base_url.rstrip('/')}/{directory}/"
        page: requests.Response = requests.get(url, timeout=INDEX_TIMEOUT_S)
        page.raise_for_status()
        return transports.parse_apache_index(page.text)

    def manifest(self) -> LamariaManifest:
        """Read ``<root>/manifest.json``, the record ``download`` left behind."""
        path: Path = self.config.root / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"no {MANIFEST_NAME} at {path}; run `dataforge-download lamaria` first")
        return serde.json.from_json(LamariaManifest, path.read_text())

    def source(self, record: SequenceRecord) -> LamariaSource:
        """Place one manifest record in the local official layout."""
        sequence_dir: Path = self.config.root / record.split / record.sequence
        return LamariaSource(
            sequence=record.sequence,
            split=record.split,
            vrs_url=record.vrs_url,
            vrs_display_bytes=record.vrs_display_bytes,
            vrs_path=sequence_dir / RAW_DATA_DIR / f"{record.sequence}.vrs",
            calibration_path=sequence_dir / CALIBRATION_DIR / f"{record.sequence}.json",
            pseudo_gt_path=sequence_dir / PSEUDO_GT_LOCAL_DIR / f"{record.sequence}.txt" if record.has_pseudo_gt else None,
            control_points_path=sequence_dir / CONTROL_POINTS_LOCAL_DIR / f"{record.sequence}.json" if record.has_control_points else None,
        )

    def discover(self) -> list[tuple[SequenceIdentity, LamariaSource]]:
        """Pair every selected sequence whose small files are on disk with its source.

        The manifest is the list of what the archive holds; the small files on
        disk are the evidence that ``download`` finished for a sequence. A
        selected name the manifest never saw is an error (a typo, or a stale
        manifest), while one whose files are missing is announced and skipped so
        a batch convert makes progress on the rest.
        """
        records: dict[str, SequenceRecord] = {record.sequence: record for record in self.manifest().sequences}
        unknown: list[str] = sorted(set(self.config.sequences) - set(records))
        if unknown:
            raise ValueError(f"{MANIFEST_NAME} in {self.config.root} lists no sequence named {', '.join(unknown)}; re-run dataforge-download")
        pairs: list[tuple[SequenceIdentity, LamariaSource]] = []
        for name in sorted(set(self.config.sequences)):
            source: LamariaSource = self.source(records[name])
            missing: list[Path] = [
                path
                for path in (source.calibration_path, source.pseudo_gt_path, source.control_points_path)
                if path is not None and not path.is_file()
            ]
            if missing:
                print(f"  warning: skipping {name}: {len(missing)} small file(s) not on disk, first {missing[0]}; re-run dataforge-download")
                continue
            pairs.append((SequenceIdentity(dataset=self.config.name, parts=(name,)), source))
        return pairs

    def download(self) -> None:
        """Resolve the archive index pages, write the manifest, fetch the small files."""
        raise NotImplementedError("download lands with the index resolution slice")

    def convert(self, identity: SequenceIdentity, source: LamariaSource, *, force: bool) -> Path:
        """Fetch one VRS, encode it, write the base-layer rrd, and delete the raw."""
        raise NotImplementedError("convert lands with the recording slice")

    def default_blueprint(self) -> rrb.Blueprint:
        """Corpus-wide layout: every Aria Gen1 sequence has the same three cameras."""
        raise NotImplementedError("blueprints land with the blueprint slice")

    def table_blueprint(self) -> rrb.Blueprint:
        """Cheap preview card for the dataset's segment table."""
        raise NotImplementedError("blueprints land with the blueprint slice")
