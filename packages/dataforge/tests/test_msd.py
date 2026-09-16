"""Monado SLAM Datasets: the device table, the verbs, the skip rules and the blueprints.

Nothing here touches the network — the HF listing helper, ``hf_fetch`` and the
revision lookup are stubbed onto ``msd_hub``'s synthetic sequence, so a convert
exercises the real archive reader, the real AV1 encoder and the real writers with
only the transport faked. What each layer *wrote* is read back next door in
``test_msd_layers``; what is not MSD-specific is tested in ``test_archives``
(the readers), ``test_basalt`` (the calibration) and ``test_euroc`` (the csvs).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
from conftest import calibration_fixture, column_rows, read_back
from jaxtyping import Float64
from msd_hub import REVISION_SHA, SEQUENCE, FakeHub, build_hub, recording_properties
from numpy import ndarray

from dataforge import blueprints, paths, schema
from dataforge.basalt import FollowFrame, follow_frame, load_calibration
from dataforge.datasets import msd, msd_layers
from dataforge.datasets.msd import (
    MSD_DEVICES,
    MsdConfig,
    MsdDataset,
    MsdDevice,
    MsdDeviceChoice,
    MsdSource,
    build_blueprint,
    follow_eye,
)
from dataforge.datasets.msd_layers import WORLD_UP_VIEW_COORDINATES
from dataforge.identity import SequenceIdentity


def test_every_device_is_one_catalog_dataset_named_after_it() -> None:
    devices: tuple[MsdDeviceChoice, ...] = ("index", "g2", "odyssey")
    assert [MsdConfig(device=device).name for device in devices] == ["msd-index", "msd-g2", "msd-odyssey"]
    # The calibration collections (MIC/MGC/MOC) are deliberately not convertible sequences.
    assert not any(collection.endswith("C_calibration") for device in MSD_DEVICES.values() for collection in device.collections)


def test_every_device_declares_a_world_up_axis_and_names_its_gt_source() -> None:
    for device, profile in MSD_DEVICES.items():
        assert profile.world_up in WORLD_UP_VIEW_COORDINATES, device
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



FOLLOW_FRAME_AGREEMENT_DEG: float = 0.05
"""How far a declared ``MSD_DEVICES`` axis may sit from the real calibration's.

Far tighter than ``convert``'s 5 deg warning tolerance: the constants were read
off these very files, so the only gap left is the three decimals they are
rounded to.
"""


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

    # The encoder is the layer module's now, so that is where the failure is injected.
    monkeypatch.setattr(msd_layers, "encode_frames_to_mp4", explode)
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


# ── the layer rule ────────────────────────────────────────────────────────


def gt_rows(gt_rrd: Path) -> tuple[list[int], list[list[float]]]:
    """Every pose time and translation in a gt rrd, index-sorted.

    What a gt rebuild has to reproduce exactly: same clock, same positions.
    Read through the public reader, so it is what a consumer sees.
    """
    poses: pa.Table = column_rows(read_back(gt_rrd), f"{schema.rig_path(0)}:Transform3D:translation")
    times_ns: list[int] = poses.column(schema.TIMELINE).combine_chunks().cast(pa.int64()).to_pylist()
    return times_ns, [row[0] for row in poses.column(1).to_pylist()]


def test_a_convert_publishes_the_gt_csv_verbatim_beside_the_two_rrds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    """The archive is deleted, so the one member gt still needs is kept as a sidecar."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    dataset.convert(identity, source, force=False)

    sidecar: Path = paths.sidecar_path(paths.output_root(), identity, msd.GT_SIDECAR_NAME)
    assert sidecar.is_file(), f"no sidecar at {sidecar}"
    assert sidecar == paths.output_root() / paths.SIDECAR_DIR / identity.recording_id / "gt.csv"
    # Verbatim: byte for byte what the archive shipped, not re-serialized columns.
    assert sidecar.read_bytes() == (tmp_path / "tree" / SEQUENCE / "mav0" / "gt" / "data.csv").read_bytes()


def test_a_missing_gt_layer_is_rebuilt_from_the_sidecar_without_fetching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This is what the sidecar is for: ``rm gt/*.rrd`` and a convert, no network, no encode.

    The rebuilt layer has to be the *same* layer — it reads the same csv and the
    same base rrd — so its rows are compared against the ones the full
    conversion wrote, not merely counted.
    """
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    base_target: Path = dataset.convert(identity, source, force=False)
    gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
    before: tuple[list[int], list[list[float]]] = gt_rows(gt_target)
    base_bytes: bytes = base_target.read_bytes()
    fetches: int = len(hub.fetched)
    capsys.readouterr()

    gt_target.unlink()
    assert dataset.convert(identity, source, force=False) == base_target

    assert len(hub.fetched) == fetches, "base and the sidecar are on disk, so the archive is not fetched again"
    assert gt_rows(gt_target) == before, "a rebuilt gt layer is the same layer"
    assert base_target.read_bytes() == base_bytes, "rebuilding gt must not touch the base rrd"
    assert "no fetch" in capsys.readouterr().out


def test_a_missing_gt_layer_with_no_sidecar_falls_back_to_the_archive_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the sidecar there is nothing to rebuild from, and paying for the download is the only option."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    base_target: Path = dataset.convert(identity, source, force=False)
    gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
    sidecar: Path = paths.sidecar_path(paths.output_root(), identity, msd.GT_SIDECAR_NAME)
    fetches: int = len(hub.fetched)
    capsys.readouterr()

    gt_target.unlink()
    sidecar.unlink()
    assert dataset.convert(identity, source, force=False) == base_target

    assert len(hub.fetched) > fetches, "no sidecar means the archive is the only source left"
    assert gt_target.is_file() and sidecar.is_file(), "the fallback republishes both"
    assert "cannot be rebuilt" in capsys.readouterr().out


def test_both_layers_and_the_sidecar_are_skipped_when_all_three_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path
) -> None:
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    base_target: Path = dataset.convert(identity, source, force=False)
    fetches: int = len(hub.fetched)

    assert dataset.convert(identity, source, force=False) == base_target
    assert len(hub.fetched) == fetches, "everything exists, so nothing is downloaded"


def test_force_republishes_both_layers_and_the_sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nvenc_ffmpeg: Path) -> None:
    """``--force`` bypasses the skip checks and nothing else: all three come back."""
    hub: FakeHub = build_hub(tmp_path, monkeypatch)
    dataset: MsdDataset = MsdDataset(hub.config)
    identity, source = dataset.discover()[0]
    base_target: Path = dataset.convert(identity, source, force=False)
    gt_target: Path = paths.rrd_path(paths.output_root(), layer=paths.GT_LAYER, identity=identity)
    sidecar: Path = paths.sidecar_path(paths.output_root(), identity, msd.GT_SIDECAR_NAME)
    before: tuple[list[int], list[list[float]]] = gt_rows(gt_target)
    fetches: int = len(hub.fetched)

    assert dataset.convert(identity, source, force=True) == base_target

    assert len(hub.fetched) > fetches, "a forced convert re-reads the archive"
    for published in (base_target, gt_target, sidecar):
        assert published.is_file(), f"{published} was not republished"
        assert not list(published.parent.glob("*.tmp")), f"a staged temp survived beside {published}"
    assert gt_rows(gt_target) == before, "the same inputs give the same layer"




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


def test_the_two_3d_views_are_complementary_views_of_one_path() -> None:
    """The overview hides the trail; the Follow view keeps the whole path as dim context.

    Hiding the trajectory in the Follow view left the highlighted trail floating
    with nothing to place it against, so the path stays and is overridden thin and
    dim instead — context behind the highlight rather than a competing stroke.
    """
    views: list[rrb.View] = blueprint_views(build_blueprint(2, has_magnetometer=False, follow=UPRIGHT_FOLLOW))
    rig: rrb.View = next(view for view in views if view.name == "Rig")
    follow: rrb.View = next(view for view in views if view.name == "Follow")

    assert rig.visualizer_overrides[schema.trail_path("gt")] == rrb.EntityBehavior(visible=False), "the overview still hides the trail"

    dimmed: object = follow.visualizer_overrides[schema.trajectory_path("gt")]
    assert isinstance(dimmed, rr.LineStrips3D), "the Follow view styles the path rather than hiding it"
    assert dimmed.radii is not None and dimmed.colors is not None
    assert dimmed.radii.as_arrow_array().to_pylist() == [-blueprints.DIM_TRAJECTORY_RADIUS_UI_POINTS], "thin, and in ui points"
    packed: int = dimmed.colors.as_arrow_array().to_pylist()[0]
    assert ((packed >> 24) & 0xFF, (packed >> 16) & 0xFF, (packed >> 8) & 0xFF) == blueprints.DIM_TRAJECTORY_COLOR
    trail_override: object = follow.visualizer_overrides[schema.trail_path("gt")]
    assert isinstance(trail_override, rrb.VisibleTimeRanges), "the trail is still the cursor-relative window"
