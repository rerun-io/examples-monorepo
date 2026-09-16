"""The synthetic MSD sequence and the fake hub both msd test modules convert against.

Not a test module: it builds one device's remote tree on disk — a sequence zip of
noisy PNG frames plus the csv streams, and the device's **real** calibration — and
wires the HF listing/fetch/revision stubs onto it, so a convert exercises the real
archive reader, the real AV1 encoder and the real writers with only the transport
faked. ``test_msd`` drives the verbs against it; ``test_msd_layers`` reads back
what the layers wrote.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
from conftest import calibration_fixture, png_frame
from jaxtyping import Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import transports
from dataforge.datasets import msd
from dataforge.datasets.msd import MSD_DEVICES, MsdConfig, MsdDevice, MsdDeviceChoice

REVISION_SHA: str = "0123456789abcdef0123456789abcdef01234567"
"""Fake resolved repo revision every test's ``repo_revision`` stub returns."""

SEQUENCE: str = "MIO09_short_1_updown"
"""Sequence stem of every synthetic archive below; also its top directory inside the zip."""


FRAME_WIDTH: int = 192
"""Frame width; NVENC refuses anything much smaller, so the fixture is not tiny."""
FRAME_HEIGHT: int = 160
"""Frame height, likewise above NVENC's minimum."""


GT_NUM_POSES: int = 40
"""gt rows the synthetic tree writes; the gt layer logs one pose per row."""
GT_PERIOD_NS: int = 1_000_000
"""gt sample period in the synthetic tree — 1 kHz, as the real files ship."""
GT_DROPOUT_ROW: int = 17
"""Row the synthetic tree writes a degenerate quaternion into, as a real dropout is written."""
FIXTURE_WORLD_R_RIG: Rotation = Rotation.from_euler("x", -90.0, degrees=True)
"""The synthetic sequence's constant gt orientation.

It maps the rig's +z — where the fixture's accelerometer reads gravity — onto
the world's +y, so ``measured_world_up`` must answer ``+y`` for this tree.
"""


def sequence_frame(index: int) -> bytes:
    """One frame of the synthetic sequence, as MSD ships them: a noisy grayscale PNG.

    Noisy because the split-archive fixture needs an archive that really spans
    volumes, and a flat gradient compresses to a couple of kilobytes.
    """
    return png_frame(index, width=FRAME_WIDTH, height=FRAME_HEIGHT, noisy=True)


@dataclass(frozen=True, slots=True)
class StreamClocks:
    """What the synthetic tree actually wrote, so assertions read it back rather than recompute it."""

    firsts: dict[str, int]
    """First timestamp of every stream, by stream name (``cam0``, ``imu0``, ``mag0``, ``gt``)."""
    lasts: dict[str, int]
    """Last timestamp of every stream, same keys."""


def sequence_tree(root: Path, *, num_cameras: int, num_frames: int, with_magnetometer: bool) -> StreamClocks:
    """Write one synthetic ``<SEQ>/mav0/...`` tree and report the clock of every stream."""
    base_ns: int = 13_000_000_000_000
    firsts: dict[str, int] = {}
    lasts: dict[str, int] = {}
    for camera in range(num_cameras):
        data_dir: Path = root / SEQUENCE / "mav0" / f"cam{camera}" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # Irregular steps around 18.5 ms (~54 fps), offset per camera: a real rig's clock.
        stamps: list[int] = [base_ns + camera * 300_000 + index * 18_518_000 + (index * 7919) % 400_000 for index in range(num_frames)]
        firsts[f"cam{camera}"] = stamps[0]
        lasts[f"cam{camera}"] = stamps[-1]
        rows: list[str] = ["#timestamp [ns],filename"]
        for index, stamp in enumerate(stamps):
            (data_dir / f"{stamp}.png").write_bytes(sequence_frame(index))
            rows.append(f"{stamp},{stamp}.png")
        (data_dir.parent / "data.csv").write_text("\n".join(rows) + "\n")
        # A decoy the converter must ignore, exactly as the real archives ship it.
        (data_dir.parent / "data.extra.csv").write_text("#timestamp [ns],exposure\n0,0\n")

    imu_dir: Path = root / SEQUENCE / "mav0" / "imu0"
    imu_dir.mkdir(parents=True, exist_ok=True)
    imu_first: int = base_ns - 5_000_000
    firsts["imu0"] = imu_first
    lasts["imu0"] = imu_first + 59 * 1_000_000
    imu_rows: list[str] = ["#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"]
    for index in range(60):
        stamp: int = imu_first + index * 1_000_000
        imu_rows.append(f"{stamp},{0.01 * index},{-0.02 * index},{0.03 * index},{0.1},{-0.2},{9.81}")
    (imu_dir / "data.csv").write_text("\n".join(imu_rows) + "\n")

    if with_magnetometer:
        mag_dir: Path = root / SEQUENCE / "mav0" / "mag0"
        mag_dir.mkdir(parents=True, exist_ok=True)
        mag_first: int = base_ns - 2_000_000
        firsts["mag0"] = mag_first
        lasts["mag0"] = mag_first + 5 * 20_000_000
        mag_rows: list[str] = ["#timestamp [ns], x, y, z"]
        for index in range(6):
            mag_rows.append(f"{mag_first + index * 20_000_000},{300.0 + index},{-40.0},{12.0 * index}")
        (mag_dir / "data.csv").write_text("\n".join(mag_rows) + "\n")

    gt_dir: Path = root / SEQUENCE / "mav0" / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    # Earlier than every other stream on purpose: gt owns t0.
    gt_first: int = base_ns - 9_000_000
    firsts["gt"] = gt_first
    # gt bounds the base layer's duration through no stream of its own, so it stays out
    # of ``lasts``; the gt layer's own span is GT_NUM_POSES rows at GT_PERIOD_NS.
    quaternion_xyzw: Float64[ndarray, "4"] = np.asarray(FIXTURE_WORLD_R_RIG.as_quat(), dtype=np.float64)
    scalar_first: list[float] = [float(quaternion_xyzw[3]), *(float(term) for term in quaternion_xyzw[:3])]
    gt_rows: list[str] = ["#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"]
    for index in range(GT_NUM_POSES):
        # One degenerate row, exactly as a real tracking dropout is written.
        written: list[float] = [0.0, 0.0, 0.0, 0.0] if index == GT_DROPOUT_ROW else scalar_first
        stamps_and_pose: list[str] = [str(gt_first + index * GT_PERIOD_NS), f"{0.001 * index}", "0.0", "0.0"]
        gt_rows.append(",".join([*stamps_and_pose, *(f"{term}" for term in written)]))
    (gt_dir / "data.csv").write_text("\n".join(gt_rows) + "\n")
    return StreamClocks(firsts=firsts, lasts=lasts)





@dataclass(frozen=True, slots=True)
class FakeHub:
    """One device's remote tree on disk, plus the scratch root a convert works in."""

    remote: Path
    """Mirror of the repo, so ``allow_patterns`` glob against real files."""
    root: Path
    """``MsdConfig.root``: where the fake fetch copies archives to."""
    config: MsdConfig
    """Config already pointed at ``root`` for this device."""
    clocks: StreamClocks
    """What the synthetic tree wrote, so an assertion reads it back rather than recomputes it."""
    fetched: list[tuple[str, ...]]
    """``allow_patterns`` of every ``hf_fetch`` call, in order."""
    revisions: list[str | None]
    """The ``revision`` every listing and fetch asked the hub for, in order."""
    archives: list[Path]
    """The sequence's archive volume(s), as they land under ``root``."""



def build_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: MsdDeviceChoice = "index",
    num_frames: int = 6,
    archive_bytes: int | None = None,
    raw_budget_gb: float = 50.0,
    keep_raw: bool = False,
) -> FakeHub:
    """Build one synthetic sequence and wire the HF listing/fetch/revision stubs to it."""
    profile: MsdDevice = MSD_DEVICES[device]
    collection: str = profile.collections[0]
    collection_path: str = f"M_monado_datasets/{profile.hf_dir}/{collection}"
    remote: Path = tmp_path / "remote"
    root: Path = tmp_path / "root"

    tree: Path = tmp_path / "tree"
    clocks: StreamClocks = sequence_tree(
        tree, num_cameras=profile.num_cameras, num_frames=num_frames, with_magnetometer=profile.has_magnetometer
    )
    archive_dir: Path = remote / collection_path
    archive_dir.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(archive_dir / SEQUENCE), "zip", root_dir=tree)

    # The device's REAL calibration, verbatim. Deriving it from ``profile.follow``
    # instead would make every follow-frame assertion circular, and the model tags
    # and rpmax below are upstream facts no synthetic file should get to invent.
    # Its resolution is the headset's, not the 192x160 of these noise frames: what
    # this fixture exercises is the wiring, and no writer cross-checks the two.
    calibration_file: Path = remote / "M_monado_datasets" / profile.hf_dir / "extras" / "calibration.json"
    calibration_file.parent.mkdir(parents=True, exist_ok=True)
    calibration_file.write_bytes(calibration_fixture(device).read_bytes())

    size: int = archive_bytes if archive_bytes is not None else (archive_dir / f"{SEQUENCE}.zip").stat().st_size
    listing: list[tuple[str, int]] = [(f"{collection_path}/{SEQUENCE}.zip", size), (f"{collection_path}/README.md", 12)]
    fetched: list[tuple[str, ...]] = []
    revisions: list[str | None] = []

    def fake_fetch(
        repo_id: str, *, allow_patterns: Sequence[str], local_dir: Path, repo_type: str = "dataset", revision: str | None = None
    ) -> Path:
        fetched.append(tuple(allow_patterns))
        revisions.append(revision)
        for pattern in allow_patterns:
            for match in sorted(remote.glob(pattern)):
                destination: Path = Path(local_dir) / match.relative_to(remote)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(match, destination)
        return Path(local_dir)

    def fake_listing(repo_id: str, path: str, revision: str | None = None) -> list[tuple[str, int]]:
        revisions.append(revision)
        return listing if path == collection_path else []

    monkeypatch.setattr(msd, "list_collection_files", fake_listing)
    monkeypatch.setattr(transports, "hf_fetch", fake_fetch)
    monkeypatch.setattr(msd, "repo_revision", lambda repo_id, revision=None: REVISION_SHA)
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "rrd"))

    config: MsdConfig = MsdConfig(device=device, root=root, raw_budget_gb=raw_budget_gb, keep_raw=keep_raw)
    return FakeHub(
        remote=remote,
        root=root,
        config=config,
        clocks=clocks,
        fetched=fetched,
        revisions=revisions,
        archives=[root / f"{collection_path}/{SEQUENCE}.zip"],
    )


def recording_properties(store: rr.experimental.ChunkStore, group: str) -> dict[str, object]:
    """One property group's values (``property:<group>:*``), unwrapped from their one-row lists.

    Properties live on the static ``/__properties`` entity, off every index, so
    they need their own content-filtered read.
    """
    table: pa.Table = store.reader(index=None, contents="/__properties/**").to_arrow_table()
    row: dict[str, list[object] | None] = table.to_pylist()[0]
    prefix: str = f"property:{group}:"
    return {name.removeprefix(prefix): values[0] for name, values in row.items() if name.startswith(prefix) and values}

