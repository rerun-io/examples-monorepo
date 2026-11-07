"""Gen-HumanEgo: what differs from Gen-HumanWholeBody, checked on worked examples and synthetic inputs."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from jaxtyping import Float32, Float64, Int64, UInt8, UInt16
from numpy import ndarray
from scipy.spatial.transform import Rotation
from simplecv.ops.mano.mano_np import MANOLayerNP

from dataforge.datasets.gen_humanego import GenHumanEgoConfig, GenHumanEgoDataset
from dataforge.datasets.gen_humanego_depth import decode_predz, zstd_content_size
from dataforge.datasets.genrobot import HubEpisode, shard, stratified_subset
from dataforge.datasets.genrobot_pose import MIRROR_X, PoseTrack, check_mano, skin
from dataforge.datasets.genrobot_source import parse_hub_path
from dataforge.identity import SequenceIdentity


def test_hub_paths_with_spaces_and_capitals_name_the_episode() -> None:
    domain, uuid = parse_hub_path("data/domestic_services/bedroom/organization/organize sewing supplies/01/018d1972e1c34b00866ed15e09c97f9f.mcap")
    assert domain == ("domestic_services", "bedroom", "organization", "organize sewing supplies")
    assert uuid == "018d1972e1c34b00866ed15e09c97f9f"
    assert parse_hub_path("data/domestic_services/living_room/other/Sewing_kit_assembly/2c/2cebb4e942004093b8e401f325ab719c.mcap")[0][3] == "Sewing_kit_assembly"


def zigzag(values: Int64[ndarray, "n"]) -> Int64[ndarray, "n"]:
    return np.where(values < 0, -2 * values - 1, 2 * values)


def varints(values: Int64[ndarray, "n"]) -> bytes:
    out: bytearray = bytearray()
    for value in values.tolist():
        while value >= 0x80:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out)


def encode_predz(bits: UInt16[ndarray, "h w"]) -> bytes:
    """The DAS-Ego depth encoder as das-datakit's decoder implies: left + up - up_left prediction, zigzag varints, zstd."""
    image: Int64[ndarray, "h w"] = bits.astype(np.int64)
    padded: Int64[ndarray, "h1 w1"] = np.pad(image, ((1, 0), (1, 0)))
    prediction: Int64[ndarray, "h w"] = (padded[1:, :-1] + padded[:-1, 1:] - padded[:-1, :-1]) & 0xFFFF
    residual: Int64[ndarray, "h w"] = (image - prediction + 0x8000) % 0x10000 - 0x8000
    mode_bytes: int = (bits.size + 7) // 8
    body: bytes = bytes(mode_bytes) + varints(zigzag(residual.reshape(-1)))
    payload: bytes = struct.pack("<II", mode_bytes, len(body)) + body  # the size counts the mode bytes too
    return pa.Codec("zstd").compress(payload, asbytes=True)


def test_predz_decodes_what_the_encoder_wrote_bit_for_bit() -> None:
    rng: np.random.Generator = np.random.default_rng(7)
    depth: Float32[ndarray, "5 6"] = rng.uniform(0.1, 3.0, size=(5, 6)).astype(np.float32)
    depth[0, 0] = np.nan  # the fisheye border ships as NaN
    bits: UInt16[ndarray, "5 6"] = depth.astype(np.float16).view(np.uint16)
    decoded: UInt8[ndarray, "60"] = decode_predz(encode_predz(bits), height=5, width=6)
    np.testing.assert_array_equal(decoded.view(np.uint16).reshape(5, 6), bits)


def test_a_zstd_frame_without_its_content_size_is_refused() -> None:
    frame: bytes = pa.Codec("zstd").compress(b"\x00" * 64, asbytes=True)
    assert zstd_content_size(frame) == 64
    header_without_size: bytes = frame[:4] + bytes([frame[4] & 0x1F & ~0x20]) + frame[5:]
    with pytest.raises(ValueError, match="content size"):
        zstd_content_size(header_without_size)


def episode(task: str, uuid_head: str, size_gb: float) -> HubEpisode:
    uuid: str = uuid_head.ljust(32, "0")
    return HubEpisode(path=f"data/home/room/{task}/{uuid[:2]}/{uuid}.mcap", size_bytes=int(size_gb * 1e9), sha256="0" * 64)


def test_the_subset_covers_every_task_once_before_any_task_twice() -> None:
    episodes: list[HubEpisode] = [
        *(episode("big/wash", f"a{index}", 1.0) for index in range(6)),
        episode("big/dry", "b0", 1.0),
        *(episode("small/fold", f"c{index}", 1.0) for index in range(3)),
    ]
    chosen: list[HubEpisode] = stratified_subset(episodes, budget_bytes=int(5.5e9))
    tasks: list[str] = [item.path.split("/")[3] + "/" + item.path.split("/")[4] for item in chosen]
    assert len(chosen) == 5, "the budget holds five 1 GB episodes"
    assert {"big/wash", "big/dry", "small/fold"} <= set(tasks)
    assert tasks.count("big/wash") == 2 and tasks.count("small/fold") == 2, "rounds alternate tasks; dry runs out after one"
    assert chosen == stratified_subset(list(reversed(episodes)), budget_bytes=int(5.5e9)), "the listing order does not matter"
    assert [item.uuid for item in chosen if "wash" in item.path] == ["a0".ljust(32, "0"), "a1".ljust(32, "0")], "uuid order within a task"


def test_shards_split_the_selection_without_overlap() -> None:
    items: list[int] = list(range(10))
    parts: list[list[int]] = [shard(items, index, 3) for index in range(3)]
    assert sorted(value for part in parts for value in part) == items
    with pytest.raises(ValueError):
        shard(items, 3, 3)


def left_track(root: Float64[ndarray, "3 3"], joints: Float32[ndarray, "21 3"], betas: Float64[ndarray, "10"]) -> PoseTrack:
    """One frame of a left hand whose shipped bones are ``joints``, with ``root`` as the shipped MANO root and straight fingers."""
    nan: float = float("nan")
    track_betas: Float64[ndarray, "1 2 10"] = np.full((1, 2, 10), nan)
    track_betas[0, 0] = betas
    orient: Float64[ndarray, "1 2 3 3"] = np.full((1, 2, 3, 3), nan)
    orient[0, 0] = root
    pose: Float64[ndarray, "1 2 15 3 3"] = np.full((1, 2, 15, 3, 3), nan)
    pose[0, 0] = np.eye(3)
    transl: Float64[ndarray, "1 2 3"] = np.full((1, 2, 3), nan)
    transl[0, 0] = 0.0
    bones: Float64[ndarray, "1 2 21 3"] = np.full((1, 2, 21, 3), nan)
    bones[0, 0] = joints
    present: np.ndarray = np.array([[True, False]])
    return PoseTrack(
        times_ns=np.zeros(1, dtype=np.int64), frame_indices=np.zeros(1, dtype=np.int64), body=np.full((1, 22, 3), nan), bones=bones,
        wrists=np.full((1, 2, 4, 4), nan), hand_present=present.copy(), mano_present=present, betas=track_betas, global_orient=orient,
        hand_pose=pose, global_transl=transl, is_valid=present.copy(), reproj_loss=np.zeros((1, 2)), bbox_touches_edge=np.zeros((1, 2), dtype=bool),
    )  # fmt: skip


@pytest.mark.parametrize("mirrored", [False, True])
def test_the_left_root_convention_is_found_per_hand_and_used_for_its_mesh(mirrored: bool) -> None:
    shipped: Float64[ndarray, "3 3"] = Rotation.from_rotvec([0.4, -1.1, 0.7]).as_matrix()
    betas: Float64[ndarray, "10"] = np.linspace(-0.5, 0.5, 10)
    model_root: Float64[ndarray, "3 3"] = MIRROR_X @ shipped @ MIRROR_X if mirrored else shipped
    pose: Float32[ndarray, "1 48"] = np.zeros((1, 48), dtype=np.float32)
    pose[0, :3] = Rotation.from_matrix(model_root).as_rotvec()
    layer: MANOLayerNP = MANOLayerNP(side="left", betas=betas.astype(np.float32), use_pca=False, fix_left_shapedirs=True)
    joints: Float32[ndarray, "1 21 3"] = layer(pose, np.zeros((1, 3), dtype=np.float32))[1]
    track: PoseTrack = left_track(shipped, joints[0], betas)
    check = check_mano(track, 0)
    assert check.passed and check.mirror_root == mirrored and check.median_m < 1e-5
    vertices: Float32[ndarray, "1 778 3"] = skin(track, 0, np.array([0], dtype=np.int64), mirror_root=check.mirror_root)[0]
    np.testing.assert_allclose(vertices, layer(pose, np.zeros((1, 3), dtype=np.float32))[0], atol=1e-5)


def test_the_config_selects_a_budgeted_shard(tmp_path: Path) -> None:
    config: GenHumanEgoConfig = GenHumanEgoConfig(root=tmp_path, budget_tb=0.001, shard_index=1, num_shards=2)
    assert config.name == "gen-humanego"
    assert isinstance(config.setup(), GenHumanEgoDataset)
    with pytest.raises(ValueError):
        GenHumanEgoConfig(root=tmp_path, shard_index=2, num_shards=2)


def test_a_failed_episode_keeps_its_mcap_and_leaves_no_partial_layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataforge import paths
    from dataforge.datasets import genrobot

    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset: GenHumanEgoDataset = GenHumanEgoDataset(GenHumanEgoConfig(root=tmp_path / "raw"))
    item: HubEpisode = episode("home/wash", "ab", 0.001)
    identity: SequenceIdentity = SequenceIdentity(dataset.config.name, (item.uuid,))
    mcap: Path = tmp_path / "raw" / item.path
    mcap.parent.mkdir(parents=True)
    mcap.write_bytes(b"mcap")
    monkeypatch.setattr(dataset, "fetch", lambda source: mcap)
    base: Path = dataset.targets(identity)[paths.BASE_LAYER]

    def half_written_sidecar(source: Path, target: Path, **_: object) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"sidecar")
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_bytes(b"base")
        raise RuntimeError("the reader failed half way")

    monkeypatch.setattr(genrobot, "write_sidecar", half_written_sidecar)
    with pytest.raises(RuntimeError, match="half way"):
        dataset.convert(identity, item, force=False)
    assert mcap.is_file(), "the MCAP stays for the retry"
    assert not base.exists() and not dataset.sidecar(identity).exists(), "no partial layer is left to be registered"


def test_every_published_layer_must_hold_bytes_before_the_mcap_goes(tmp_path: Path) -> None:
    from dataforge.datasets.genrobot import require_published

    good: Path = tmp_path / "good.rrd"
    good.write_bytes(b"x")
    require_published({"base": good}, ["base"])
    empty: Path = tmp_path / "empty.rrd"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        require_published({"base": good, "depth": empty}, ["base", "depth"])
    with pytest.raises(ValueError, match="missing"):
        require_published({"pose": tmp_path / "absent.rrd"}, ["pose"])


def test_the_pinhole_resampling_keeps_each_point_at_its_depth() -> None:
    from dataforge.datasets.gen_humanego_depth import PinholeGrid, pinhole_lookup
    from dataforge.datasets.genrobot_source import DoubleSphere

    lens: DoubleSphere = DoubleSphere(fx=172.8, fy=172.8, cx=259.2, cy=207.5, xi=-0.0044, alpha=0.574)
    grid: PinholeGrid = PinholeGrid(width=192, height=156, fov_deg=120.0)
    rows, cols = pinhole_lookup(lens, grid, source_width=512, source_height=416)
    assert rows.shape == (156, 192) and cols.shape == (156, 192)
    # The principal ray maps to the depth image's principal point.
    assert abs(cols[78, 96] - 259) <= 1 and abs(rows[78, 96] - 207) <= 1
    # A ray 45 degrees off axis lands where the double-sphere model sends it.
    u: int = 96 + round(grid.focal)  # tan(45°) = 1
    pixel = lens.project(np.array([[1.0, 0.0, 1.0]]))[0]
    assert abs(cols[78, u] - pixel[0]) <= 1.0
    depth: np.ndarray = np.full((416, 512), 0.7, dtype=np.float16)
    from dataforge.datasets.gen_humanego_depth import resample_depth

    resampled: np.ndarray = resample_depth(depth, rows, cols)
    assert resampled.dtype == np.float16 and resampled.shape == (156, 192)
    assert np.all(resampled[rows >= 0] == np.float16(0.7)), "z is the same for the point whatever the lens"


def test_exposure_is_read_only_from_fields_the_camera_schema_carries() -> None:
    from dataforge.datasets.genrobot_layers import EXPOSURE_FIELDS, exposure_fields

    old: pa.StructType = pa.struct([("data", pa.binary()), ("format", pa.string()), ("frame_id", pa.string())])
    assert exposure_fields(old) == (), "older firmware ships no exposure fields"
    new: pa.StructType = pa.struct([("data", pa.binary()), *((name, pa.float64()) for name in EXPOSURE_FIELDS)])
    assert exposure_fields(new) == EXPOSURE_FIELDS
