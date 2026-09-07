"""Monado SLAM Datasets: remote discovery, the device table, the blueprints, and a full convert.

Nothing here touches the network. The one HF listing helper and ``hf_fetch`` are
monkeypatched onto a synthetic sequence zip built in ``tmp_path``, so a convert
exercises the real archive reader, the real AV1 encoder and the real writers —
only the transport is faked. What is not MSD-specific is tested next door:
``test_archives`` (the archive readers), ``test_basalt`` (the calibration) and
``test_euroc`` (the csv streams). The world-up measurement is here, because the
axis it answers is a claim about MSD's own corpus.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
from conftest import calibration_fixture, column_rows, png_frame, read_back
from jaxtyping import Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import paths, schema, transports
from dataforge.basalt import BasaltPose, CalibratedCamera, FollowFrame, follow_frame, load_calibration
from dataforge.datasets import msd
from dataforge.datasets.msd import (
    MSD_DEVICES,
    MeasuredUp,
    MsdConfig,
    MsdDataset,
    MsdDevice,
    MsdDeviceChoice,
    MsdSource,
    build_blueprint,
    follow_eye,
    measured_world_up,
)
from dataforge.euroc import GtTrajectory, TimestampedSamples, gt_trajectory
from dataforge.identity import SequenceIdentity
from dataforge.logging_toolkit import ImuChannel

REVISION_SHA: str = "0123456789abcdef0123456789abcdef01234567"
"""Fake resolved repo revision every test's ``repo_revision`` stub returns."""


def test_every_device_is_one_catalog_dataset_named_after_it() -> None:
    devices: tuple[MsdDeviceChoice, ...] = ("index", "g2", "odyssey")
    assert [MsdConfig(device=device).name for device in devices] == ["msd-index", "msd-g2", "msd-odyssey"]
    # The calibration collections (MIC/MGC/MOC) are deliberately not convertible sequences.
    assert not any(collection.endswith("C_calibration") for device in MSD_DEVICES.values() for collection in device.collections)


VIEWER_AXIS_VECTORS: dict[int, tuple[float, float, float]] = {
    rr.encodings.ViewDir.Right.value: (1.0, 0.0, 0.0),
    rr.encodings.ViewDir.Left.value: (-1.0, 0.0, 0.0),
    rr.encodings.ViewDir.Up.value: (0.0, 1.0, 0.0),
    rr.encodings.ViewDir.Down.value: (0.0, -1.0, 0.0),
    rr.encodings.ViewDir.Back.value: (0.0, 0.0, 1.0),
    rr.encodings.ViewDir.Forward.value: (0.0, 0.0, -1.0),
}
"""Each ``ViewDir`` as a vector in one right-handed viewer basis (x right, y up, z back)."""


def test_each_world_up_axis_maps_to_a_right_handed_frame_with_that_axis_up() -> None:
    """The axis name is the whole decision; handedness then fixes the other two."""
    assert set(msd.WORLD_UP_VIEW_COORDINATES) == {"+x", "-x", "+y", "-y", "+z", "-z"}
    for axis, coordinates in msd.WORLD_UP_VIEW_COORDINATES.items():
        directions: list[int] = [int(direction.value) for direction in coordinates.coordinates]
        column: int = "xyz".index(axis[1])
        expected: int = (rr.encodings.ViewDir.Up if axis[0] == "+" else rr.encodings.ViewDir.Down).value
        assert directions[column] == expected, f"{axis} does not put that axis up"
        basis: Float64[ndarray, "3 3"] = np.array([VIEWER_AXIS_VECTORS[direction] for direction in directions], dtype=np.float64)
        assert np.linalg.det(basis) > 0.0, f"{axis} maps to a left-handed frame"


def test_every_device_declares_a_world_up_axis_and_names_its_gt_source() -> None:
    for device, profile in MSD_DEVICES.items():
        assert profile.world_up in msd.WORLD_UP_VIEW_COORDINATES, device
    # The Index is tracked by SteamVR Lighthouse; the other two by a MoCap rig.
    assert MSD_DEVICES["index"].gt_source == "lighthouse"
    assert {MSD_DEVICES[device].gt_source for device in ("g2", "odyssey")} == {"mocap"}


def test_discover_groups_split_parts_and_orders_by_collection_then_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    listing: dict[str, list[tuple[str, int]]] = {
        "M_monado_datasets/MI_valve_index/MIO_others": [
            ("M_monado_datasets/MI_valve_index/MIO_others/MIO10_short_2_panorama.zip", 20),
            ("M_monado_datasets/MI_valve_index/MIO_others/MIO09_short_1_updown.zip", 10),
            ("M_monado_datasets/MI_valve_index/MIO_others/README.md", 5),
        ],
        "M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber": [
            ("M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber/MIPB08_long.zip", 4),
            ("M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber/MIPB08_long.z02", 2),
            ("M_monado_datasets/MI_valve_index/MIP_playing/MIPB_beat_saber/MIPB08_long.z01", 1),
        ],
    }
    monkeypatch.setattr(msd, "list_collection_files", lambda repo_id, path, revision=None: listing.get(path, []))
    monkeypatch.setattr(msd, "repo_revision", lambda repo_id, revision=None: REVISION_SHA)

    discovered: list[tuple[SequenceIdentity, MsdSource]] = MsdDataset(MsdConfig(device="index")).discover()
    keys: list[str] = [identity.sequence_key for identity, _ in discovered]
    assert keys[:3] == ["MIO_others/MIO09_short_1_updown", "MIO_others/MIO10_short_2_panorama", "MIPB_beat_saber/MIPB08_long"]
    assert discovered[0][0].recording_id == "msd-index__MIO_others__MIO09_short_1_updown"

    split: MsdSource = discovered[2][1]
    # Parts first, ascending, then the closing .zip — the order 7z needs the volumes in.
    assert [Path(path).suffix for path in split.archive_paths] == [".z01", ".z02", ".zip"]
    assert split.archive_bytes == 7
    assert split.sequence == "MIPB08_long"
    assert split.collection == "MIPB_beat_saber"


def test_discover_ignores_collections_of_other_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def listing(repo_id: str, path: str, revision: str | None = None) -> list[tuple[str, int]]:
        asked.append(path)
        return []

    monkeypatch.setattr(msd, "list_collection_files", listing)
    monkeypatch.setattr(msd, "repo_revision", lambda repo_id, revision=None: REVISION_SHA)
    dataset: MsdDataset = MsdDataset(MsdConfig(device="g2"))
    assert dataset.discover() == []
    assert asked == ["M_monado_datasets/MG_reverb_g2/MGO_others"]



def eye_vector(batch: rr.components.Position3DBatch | rr.components.Vector3DBatch | None) -> list[float]:
    """Read one three-component field back out of an ``EyeControls3D`` archetype.

    Every field of the archetype is optional, so an unset one is a wiring failure
    rather than a value worth asserting on.
    """
    assert batch is not None, "the follow eye sets every field it is read for"
    return [float(value) for value in batch.as_arrow_array().flatten().to_pylist()]


def test_the_follow_eye_chases_the_headset_from_behind_and_above() -> None:
    """A chase camera: back along forward, up along up, aimed just ahead of the rig.

    The Index's frame goes in, so the numbers are readable by hand: 0.9 m back
    along +z and 0.45 m up along -x is (-0.45, 0, -0.9), looking at 0.3 m ahead.
    """
    eye: rrb.EyeControls3D = follow_eye(FollowFrame(forward=(0.0, 0.0, 1.0), up=(-1.0, 0.0, 0.0)))

    assert eye_vector(eye.position) == pytest.approx([-0.45, 0.0, -0.9], abs=1e-6)
    assert eye_vector(eye.look_target) == pytest.approx([0.0, 0.0, 0.3], abs=1e-6)
    assert eye_vector(eye.eye_up) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-6)
    kind: rrb.components.Eye3DKindBatch | None = eye.kind
    spin_speed: rrb.components.AngularSpeedBatch | None = eye.spin_speed
    assert kind is not None and spin_speed is not None
    assert kind.as_arrow_array().to_pylist() == [rrb.Eye3DKind.FirstPerson.value]
    assert spin_speed.as_arrow_array().to_pylist() == [0.0]



# ── archive member reader ─────────────────────────────────────────────────

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



# ── a fake hub in tmp_path ────────────────────────────────────────────────


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


FOLLOW_FRAME_AGREEMENT_DEG: float = 0.05
"""How far a declared ``MSD_DEVICES`` axis may sit from the real calibration's.

Far tighter than ``convert``'s 5 deg warning tolerance: the constants were read
off these very files, so the only gap left is the three decimals they are
rounded to.
"""


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


@pytest.fixture(scope="module")
def converted_index(tmp_path_factory, nvenc_ffmpeg: Path) -> Iterator[tuple[FakeHub, Path, Path]]:
    """One real Index convert, shared by every test that only reads back what it wrote.

    A convert encodes a whole synthetic sequence through NVENC, so the tests that
    only inspect its two rrds share one. Anything whose *setup* differs — another
    device, a budget, a patched registry, a captured warning — still converts for
    itself. ``MonkeyPatch.context()`` because the module-scoped fixture outlives
    the function-scoped ``monkeypatch``.

    Yields:
        The hub the convert ran against, its base rrd, and its gt rrd.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        hub: FakeHub = build_hub(tmp_path_factory.mktemp("converted"), monkeypatch)
        dataset: MsdDataset = MsdDataset(hub.config)
        identity, source = dataset.discover()[0]
        base_target: Path = dataset.convert(identity, source, force=False)
        yield hub, base_target, paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)


# ── download ──────────────────────────────────────────────────────────────


def test_download_fetches_only_the_calibration_and_prints_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc_ffmpeg: Path
) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    MsdDataset(hub.config).download()
    output: str = capsys.readouterr().out
    assert hub.fetched == [("M_monado_datasets/MI_valve_index/extras/calibration.json",)]
    assert (hub.root / "M_monado_datasets/MI_valve_index/extras/calibration.json").is_file()
    # Bulk data stays remote: a single sequence is fetched by convert, then deleted.
    assert not hub.archives[0].exists()
    assert "msd-index: 1 sequence(s)" in output
    assert "MIO_others: 1 sequence(s)" in output


# ── convert ───────────────────────────────────────────────────────────────


def test_one_resolved_commit_serves_the_listing_the_fetches_and_the_rrd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """A branch name moves under a conversion; a sha cannot.

    ``--revision main`` is only the input: listing a collection on ``main``,
    fetching an archive on it minutes later and stamping a third answer into the
    rrd could describe three different trees, and nothing in the rrd would say
    so. Every hub call takes the resolved sha instead, and it is the sha the
    recording reports.
    """
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    monkeypatch.setattr(msd, "repo_revision", lambda repo_id, revision=None: REVISION_SHA)
    dataset: MsdDataset = MsdDataset(replace(hub.config, revision="main"))
    identity, source = dataset.discover()[0]

    target: Path = dataset.convert(identity, source, force=False)

    assert hub.revisions, "the listing and both fetches all go through the hub"
    assert set(hub.revisions) == {REVISION_SHA}, f"a call used something other than the sha: {hub.revisions}"
    assert recording_properties(read_back(target), "capture")["hf_revision"] == REVISION_SHA


def test_a_revision_the_hub_resolves_to_nothing_stops_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a sha there is no tree to name, so the conversion has nothing honest to record."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    monkeypatch.setattr(msd, "repo_revision", lambda repo_id, revision=None: None)
    dataset: MsdDataset = MsdDataset(replace(hub.config, revision="no-such-branch"))

    with pytest.raises(RuntimeError, match="no-such-branch"):
        dataset.discover()




@pytest.mark.parametrize("device", ["index", "g2"])
def test_convert_writes_one_replayable_recording_and_deletes_the_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: MsdDeviceChoice, nvenc_ffmpeg: Path
) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch, device=device)
    dataset: MsdDataset = MsdDataset(hub.config)
    discovered: list[tuple[SequenceIdentity, MsdSource]] = dataset.discover()
    identity, source = discovered[0]

    target: Path = dataset.convert(identity, source, force=False)

    assert target.is_file()
    assert target.name == f"msd-{device}__{MSD_DEVICES[device].collections[0]}__{SEQUENCE}.rrd"
    store: rr.experimental.ChunkStore = read_back(target)
    profile: MsdDevice = MSD_DEVICES[device]

    # t0 is gt's first stamp: it is earlier than every other stream in the fixture.
    start_time_ns: int = hub.clocks.firsts["gt"]
    assert start_time_ns == min(hub.clocks.firsts.values())
    for index in range(profile.num_cameras):
        samples: pa.Table = column_rows(store, f"{schema.video_path(0, index)}:VideoStream:sample")
        assert samples.num_rows == 6, f"cam{index} lost samples"
        first_ns: int = samples.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_pylist()[0]
        assert first_ns == hub.clocks.firsts[f"cam{index}"] - start_time_ns

    assert column_rows(store, f"{schema.gyro_path(0, 0)}:Scalars:scalars").num_rows == 60
    assert column_rows(store, f"{schema.accel_path(0, 0)}:Scalars:scalars").num_rows == 60
    if profile.has_magnetometer:
        assert column_rows(store, f"{schema.field_path(0, 0)}:Scalars:scalars").num_rows == 6

    capture: dict[str, object] = recording_properties(store, "capture")
    assert capture["start_time_ns"] == start_time_ns
    assert capture["num_cameras"] == profile.num_cameras
    assert capture["num_frames"] == 6
    assert capture["device"] == device
    assert capture["device_label"] == profile.label
    assert capture["collection"] == profile.collections[0]
    assert capture["hf_revision"] == REVISION_SHA
    assert capture["duration_ns"] == max(hub.clocks.lasts.values()) - start_time_ns

    # Raw is scratch: the archive and every temp mp4 are gone once the rrd exists.
    assert not hub.archives[0].exists()
    assert not list(hub.root.rglob("*.mp4"))
    assert (hub.root / "M_monado_datasets" / profile.hf_dir / "extras" / "calibration.json").is_file()


def test_keep_raw_leaves_the_archive_and_the_encoded_mp4s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch, keep_raw=True)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)
    assert hub.archives[0].is_file()
    assert sorted(path.name for path in hub.root.rglob("*.mp4")) == ["cam0.mp4", "cam1.mp4"]


def test_a_failed_encode_keeps_the_archive_and_clears_the_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch)

    def explode(*arguments: object, **keywords: object) -> int:
        raise RuntimeError("nvenc fell over")

    monkeypatch.setattr(msd, "encode_frames_to_mp4", explode)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    with pytest.raises(RuntimeError, match="nvenc fell over"):
        dataset.convert(identity, source, force=False)

    # The download is the expensive half, so it survives; the scratch does not.
    assert hub.archives[0].is_file()
    assert not (hub.root / "work" / SEQUENCE).exists()
    assert not paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity).exists()
    assert "kept 0.0" in capsys.readouterr().out


def test_a_sequence_with_both_layers_already_written_is_skipped_without_fetching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    target: Path = paths.rrd_path(paths.output_root(), layer=paths.BASE_LAYER, identity=identity)
    for layer in (paths.BASE_LAYER, paths.GT_LAYER):
        written: Path = paths.rrd_path(paths.output_root(), layer=layer, identity=identity)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_bytes(b"already done")

    assert dataset.convert(identity, source, force=False) == target
    assert hub.fetched == []


def test_the_logged_camera_node_carries_rig_T_cam(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """``T_imu_cam`` is the camera's pose in the rig frame, and that is what lands on the node.

    ``log_pinhole`` stores the child-from-parent step, so the recording holds
    ``cam_T_rig``; inverting it must give back the calibration's pose.
    """
    hub, target, _ = converted_index

    cameras: tuple[CalibratedCamera, ...] = load_calibration(hub.remote / "M_monado_datasets/MI_valve_index/extras/calibration.json")
    store: rr.experimental.ChunkStore = read_back(target)
    for index in range(2):
        node: str = schema.cam_path(0, index)
        row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
        assert row[f"{node}:Transform3D:relation"][0] == rr.components.TransformRelation.ChildFromParent.value
        cam_R_rig: Float64[ndarray, "3 3"] = np.asarray(row[f"{node}:Transform3D:mat3x3"][0], dtype=np.float64).reshape(3, 3).T
        cam_t_rig: Float64[ndarray, "3"] = np.asarray(row[f"{node}:Transform3D:translation"][0], dtype=np.float64)

        pose: BasaltPose = cameras[index].rig_pose
        rig_R_cam: Float64[ndarray, "3 3"] = Rotation.from_quat([pose.qx, pose.qy, pose.qz, pose.qw]).as_matrix()
        rig_t_cam: Float64[ndarray, "3"] = np.array([pose.px, pose.py, pose.pz])
        # float32 on the wire, so a loose tolerance is the honest one.
        np.testing.assert_allclose(cam_R_rig.T, rig_R_cam, atol=1e-6)
        np.testing.assert_allclose(-cam_R_rig.T @ cam_t_rig, rig_t_cam, atol=1e-6)


def test_a_radtan8_camera_node_names_its_projection_and_carries_its_validity_radius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """A consumer reads the projection off the node, plus where the rational model stops holding."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch, device="odyssey")
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    target: Path = dataset.convert(identity, source, force=False)

    expected: float | None = load_calibration(calibration_fixture("odyssey"))[0].distortion_valid_radius
    store: rr.experimental.ChunkStore = read_back(target)
    node: str = schema.cam_path(0, 0)
    row: dict[str, list[object]] = store.reader(index=None, contents=node).to_arrow_table().to_pylist()[0]
    assert row[f"{node}:camera_model"][0] == "pinhole-radtan8"
    assert row[f"{node}:distortion_valid_radius"][0] == pytest.approx(expected)


def test_a_kb4_camera_node_names_its_projection_and_claims_no_validity_radius(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """kb4 is valid over the whole fisheye, so it declares no radius at all."""
    _, target, _ = converted_index

    store: rr.experimental.ChunkStore = read_back(target)
    node: str = schema.cam_path(0, 0)
    table: pa.Table = store.reader(index=None, contents=node).to_arrow_table()
    assert table.to_pylist()[0][f"{node}:camera_model"][0] == "kb4"
    # AnyValues only *omits* a None key while it is untyped: a radtan8 convert earlier in
    # this process types it, and later Nones then arrive as nulls. Assert on the value.
    radius: str = f"{node}:distortion_valid_radius"
    assert radius not in table.column_names or table.column(radius).null_count == table.num_rows


# ── gt layer ──────────────────────────────────────────────────────────────


def test_the_gt_layer_is_a_sibling_rrd_of_the_same_recording(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """One convert writes both layers: same recording id, own layer directory."""
    _, base_target, gt_target = converted_index

    assert gt_target.is_file()
    assert gt_target.name == base_target.name
    assert (gt_target.parent.name, base_target.parent.name) == (paths.GT_LAYER, paths.BASE_LAYER)


def test_the_gt_layer_animates_the_rig_node_at_the_full_gt_rate(converted_index: tuple[FakeHub, Path, Path]) -> None:
    hub, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    poses: pa.Table = column_rows(store, f"{schema.rig_path(0)}:Transform3D:translation")
    assert poses.num_rows == GT_NUM_POSES, "gt is logged raw: no resampling, one row per csv row"
    times_ns: list[int] = poses.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_pylist()
    # gt is the earliest stream in the fixture, so it owns t0 and starts at video_time 0.
    assert hub.clocks.firsts["gt"] == min(hub.clocks.firsts.values())
    assert times_ns[0] == 0
    assert times_ns[-1] == (GT_NUM_POSES - 1) * GT_PERIOD_NS


def test_the_rig_quaternion_is_the_file_quaternion_reordered_to_xyzw(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The csv writes the scalar first; a viewer reading the rrd must see it last."""
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    stored: list[list[list[float]]] = column_rows(store, f"{schema.rig_path(0)}:Transform3D:quaternion").column(1).to_pylist()
    expected_xyzw: Float64[ndarray, "4"] = np.asarray(FIXTURE_WORLD_R_RIG.as_quat(), dtype=np.float64)
    # float32 on the wire, so a loose tolerance is the honest one.
    np.testing.assert_allclose(np.asarray(stored[0][0], dtype=np.float64), expected_xyzw, atol=1e-6)
    # The dropout row keeps its translation but loses its rotation, per slam-evals' repair.
    np.testing.assert_allclose(np.asarray(stored[GT_DROPOUT_ROW][0], dtype=np.float64), [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_the_gt_layer_carries_a_full_path_and_a_per_pose_trail(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The overview strip is static and whole; the trail is one point per pose, for the cursor window."""
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    trajectory: str = schema.trajectory_path("gt")
    strips: list[list[list[float]]] = (
        store.reader(index=None, contents=trajectory).to_arrow_table().to_pylist()[0][f"{trajectory}:LineStrips3D:strips"]
    )
    assert len(strips) == 1, "the whole trajectory is one strip"
    assert len(strips[0]) == GT_NUM_POSES
    assert column_rows(store, f"{schema.trail_path('gt')}:Points3D:positions").num_rows == GT_NUM_POSES


def test_only_the_gt_layer_states_the_world_axes(converted_index: tuple[FakeHub, Path, Path]) -> None:
    """The pose layer establishes a world frame, so it owns the root ViewCoordinates."""
    _, base_target, gt_target = converted_index

    gt_root: pa.Table = read_back(gt_target).reader(index=None, contents="/").to_arrow_table()
    assert "/:ViewCoordinates:xyz" in gt_root.column_names
    declared: list[int] = [int(direction.value) for direction in msd.WORLD_UP_VIEW_COORDINATES[MSD_DEVICES["index"].world_up].coordinates]
    assert [int(value) for value in gt_root.to_pylist()[0]["/:ViewCoordinates:xyz"][0]] == declared
    assert "/:ViewCoordinates:xyz" not in read_back(base_target).reader(index=None, contents="/").to_arrow_table().column_names


def test_the_gt_properties_report_the_poses_the_repairs_and_the_measured_axis(converted_index: tuple[FakeHub, Path, Path]) -> None:
    _, _, gt_target = converted_index

    store: rr.experimental.ChunkStore = read_back(gt_target)
    gt: dict[str, object] = recording_properties(store, "gt")
    assert gt["num_poses"] == GT_NUM_POSES
    assert gt["duration_ns"] == (GT_NUM_POSES - 1) * GT_PERIOD_NS
    assert gt["num_sanitized"] == 1
    assert gt["source"] == MSD_DEVICES["index"].gt_source
    assert gt["world_up"] == MSD_DEVICES["index"].world_up
    assert gt["measured_up"] == "+y"
    measured_fraction: object = gt["measured_up_fraction"]
    assert isinstance(measured_fraction, float) and measured_fraction > 0.9


# ── the world up measurement ────────────────────────────────────


def constant_pose_gt(times_ns: Int64[ndarray, "n_poses"], quaternion_xyzw: Float64[ndarray, "4"]) -> TimestampedSamples:
    """A gt table holding one fixed orientation at the origin, in the file's wxyz order."""
    return TimestampedSamples(
        times_ns=times_ns,
        values=np.column_stack([np.zeros((times_ns.size, 3)), np.tile(quaternion_xyzw[[3, 0, 1, 2]], (times_ns.size, 1))]),
    )


def test_the_world_up_axis_is_measured_by_rotating_the_accelerometer_into_the_world() -> None:
    """An accelerometer at rest reads +g pointing *up*, so ``world_R_rig @ a_rig`` averages to the up axis."""
    # -90 deg about x maps the rig's +z onto the world's +y, so a headset held level
    # in a Y-up world reads gravity along its own +z.
    world_R_rig: Rotation = Rotation.from_euler("x", -90.0, degrees=True)
    times_ns: Int64[ndarray, "n_poses"] = np.arange(4_000, dtype=np.int64) * 1_000_000
    rig_accel_xyz: Float64[ndarray, "n_samples 3"] = np.tile([0.1, -0.2, 9.81], (times_ns.size, 1))
    # The second half of the capture points the other way; the 2 s window must ignore it.
    rig_accel_xyz[times_ns >= msd.MEASURED_UP_WINDOW_NS] = [0.1, -0.2, -9.81]
    gt: GtTrajectory = gt_trajectory(constant_pose_gt(times_ns, np.asarray(world_R_rig.as_quat(), dtype=np.float64)))

    measured: MeasuredUp = measured_world_up(gt, ImuChannel(times_ns=times_ns, values_xyz=rig_accel_xyz))

    assert measured.axis == "+y"
    # At rest the whole of gravity lands on that one axis.
    assert measured.fraction == pytest.approx(1.0, abs=0.01)


def test_a_world_up_the_data_disagrees_with_is_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc_ffmpeg: Path
) -> None:
    """A declared axis is a claim about the data; convert re-measures it every sequence."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    monkeypatch.setitem(MSD_DEVICES, "index", replace(MSD_DEVICES["index"], world_up="+z"))
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)

    output: str = capsys.readouterr().out
    assert "declares world_up +z" in output
    assert "measured +y" in output


def test_a_follow_frame_the_calibration_disagrees_with_is_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc_ffmpeg: Path
) -> None:
    """A declared follow frame is a claim about the calibration; convert re-derives it."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    rolled: FollowFrame = FollowFrame(forward=MSD_DEVICES["index"].follow.forward, up=(0.0, 1.0, 0.0))
    monkeypatch.setitem(MSD_DEVICES, "index", replace(MSD_DEVICES["index"], follow=rolled))
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)

    output: str = capsys.readouterr().out
    assert "follow frame" in output
    assert "up off by 90.0 deg" in output, "a quarter-turn roll is what the tolerance exists to catch"


def test_a_follow_frame_the_calibration_agrees_with_stays_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc_ffmpeg: Path
) -> None:
    """The synthetic calibration is built from the Index's declared frame, so nothing is said."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)

    assert "follow frame" not in capsys.readouterr().out


def test_both_layers_are_skipped_together_and_rebuilt_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    """Both layers come out of one archive fetch, so a half-converted sequence is redone."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    base_target: Path = dataset.convert(identity, source, force=False)
    gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
    fetches: int = len(hub.fetched)

    assert dataset.convert(identity, source, force=False) == base_target
    assert len(hub.fetched) == fetches, "both layers exist, so nothing is downloaded"

    gt_target.unlink()
    assert dataset.convert(identity, source, force=False) == base_target
    assert len(hub.fetched) > fetches, "a missing layer means another fetch"
    assert gt_target.is_file()


# ── raw budget ────────────────────────────────────────────────────────────


def test_a_sequence_bigger_than_the_budget_is_an_announced_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], nvenc_ffmpeg: Path
) -> None:
    # The Index and G2 long sessions are 66 GB and 55 GB of split archives: there is
    # no smaller unit to convert, so the budget warns and the sequence goes through.
    hub: FakeHub = build_hub(tmp_path, monkeypatch, archive_bytes=60_000_000_000, raw_budget_gb=50.0)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)
    output: str = capsys.readouterr().out
    assert "above the 50 GB raw budget" in output
    assert "accepted exception" in output


def test_leftovers_that_would_breach_the_budget_stop_the_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch, archive_bytes=30_000_000_000, raw_budget_gb=50.0)
    leftover: Path = hub.root / "M_monado_datasets/MI_valve_index/MIO_others/MIO12_moving_screens.zip"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.touch()
    # 25 GB of a previous failure, plus the 30 GB this sequence needs, is over the
    # cap. The file is sparse, so it costs a byte of disk and the real stat() runs.
    os.truncate(leftover, 25_000_000_000)

    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    with pytest.raises(RuntimeError, match="MIO12_moving_screens.zip"):
        dataset.convert(identity, source, force=False)
    assert hub.fetched == []


# ── blueprints ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_both_blueprints_serialize_for_every_device_layout(tmp_path: Path, device: MsdDeviceChoice) -> None:
    dataset: MsdDataset = MsdDataset(MsdConfig(device=device))
    default_path: Path = tmp_path / f"{device}.rbl"
    table_path: Path = tmp_path / f"{device}-table.rbl"

    dataset.default_blueprint().save(f"msd-{device}", str(default_path))
    dataset.table_blueprint().save(f"msd-{device}", str(table_path))

    assert default_path.stat().st_size > 0
    assert table_path.stat().st_size > 0


def blueprint_views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Every view in a blueprint, depth-first, whatever containers nest them."""
    found: list[rrb.View] = []

    def walk(node: rrb.View | rrb.Container) -> None:
        if isinstance(node, rrb.View):
            found.append(node)
            return
        for child in node.contents or ():
            walk(child)

    walk(blueprint.root_container)
    return found


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_every_declared_follow_frame_is_two_orthogonal_unit_vectors(device: MsdDeviceChoice) -> None:
    """The constants are typed in by hand from a calibration, so the invariant is checked here."""
    follow: FollowFrame = MSD_DEVICES[device].follow
    assert np.linalg.norm(follow.forward) == pytest.approx(1.0, abs=1e-3)
    assert np.linalg.norm(follow.up) == pytest.approx(1.0, abs=1e-3)
    assert float(np.dot(follow.forward, follow.up)) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_every_declared_follow_frame_is_the_real_calibration_own(device: MsdDeviceChoice) -> None:
    """The constants exist only because ``register`` has no sequence to derive them from.

    That makes them a copy of a derivation, and a copy can rot: this reads the
    device's **real** ``calibration.json`` and re-derives the pair. ``convert``
    re-checks the same thing per sequence but only warns past 5 deg; here the two
    must agree to 0.05 deg, because nothing but the constants' three decimals
    separates them.
    """
    declared: FollowFrame = MSD_DEVICES[device].follow
    derived: FollowFrame = follow_frame(load_calibration(calibration_fixture(device), expected_cameras=MSD_DEVICES[device].num_cameras))

    for axis, (stated, real) in (("forward", (declared.forward, derived.forward)), ("up", (declared.up, derived.up))):
        stated_xyz: Float64[ndarray, "3"] = np.asarray(stated, dtype=np.float64)
        cosine: float = float(np.dot(stated_xyz, real) / np.linalg.norm(stated_xyz))
        deviation_deg: float = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        assert deviation_deg < FOLLOW_FRAME_AGREEMENT_DEG, f"{device}'s declared {axis} is {deviation_deg:.3f} deg off its calibration"


@pytest.mark.parametrize("device", ["index", "g2", "odyssey"])
def test_both_follow_views_are_oriented_by_the_device_own_frame(device: MsdDeviceChoice) -> None:
    """Every headset carries its IMU differently, so one shared eye would tilt two of three."""
    follow: FollowFrame = MSD_DEVICES[device].follow
    dataset: MsdDataset = MsdDataset(MsdConfig(device=device))

    for blueprint in (dataset.default_blueprint(), dataset.table_blueprint()):
        view: rrb.View = next(each for each in blueprint_views(blueprint) if each.name == "Follow")
        eye: object = view.properties["EyeControls3D"]
        assert isinstance(eye, rrb.EyeControls3D)
        assert eye_vector(eye.eye_up) == pytest.approx(list(follow.up), abs=1e-6)
        assert eye_vector(eye.look_target) == pytest.approx([0.3 * axis for axis in follow.forward], abs=1e-6)
        # Behind the headset and above it: the eye leans against forward and with up.
        position: list[float] = eye_vector(eye.position)
        assert float(np.dot(position, follow.forward)) < 0.0
        assert float(np.dot(position, follow.up)) > 0.0


UPRIGHT_FOLLOW: FollowFrame = FollowFrame(forward=(0.0, 0.0, 1.0), up=(0.0, -1.0, 0.0))
"""A stand-in frame for the layout tests, which are about panes and not orientation."""


def test_only_a_magnetometer_device_gets_the_third_plot_pane() -> None:
    with_magnetometer: list[rrb.View] = blueprint_views(build_blueprint(4, has_magnetometer=True, follow=UPRIGHT_FOLLOW))
    without: list[rrb.View] = blueprint_views(build_blueprint(2, has_magnetometer=False, follow=UPRIGHT_FOLLOW))

    assert [view.name for view in with_magnetometer if isinstance(view, rrb.TimeSeriesView)] == [
        "Gyroscope",
        "Accelerometer",
        "Magnetometer",
    ]
    assert [view.name for view in without if isinstance(view, rrb.TimeSeriesView)] == ["Gyroscope", "Accelerometer"]
    assert [view.name for view in with_magnetometer if isinstance(view, rrb.Spatial2DView)] == ["cam0", "cam1", "cam2", "cam3"]
    assert [view.name for view in without if isinstance(view, rrb.Spatial2DView)] == ["cam0", "cam1"]
    magnetometer_pane: rrb.View = next(view for view in with_magnetometer if view.name == "Magnetometer")
    assert str(magnetometer_pane.contents) == schema.field_path(0, 0)

    # The gt overrides are inert in a base-only rrd but must already name the gt paths.
    follow: rrb.View = next(view for view in without if view.name == "Follow")
    assert set(follow.visualizer_overrides) == {schema.trajectory_path("gt"), schema.trail_path("gt")}
    rig: rrb.View = next(view for view in without if view.name == "Rig")
    assert set(rig.visualizer_overrides) == {schema.trail_path("gt")}
