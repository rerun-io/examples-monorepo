"""The six-face cubemap of an Insta360 frameset: face geometry, the seam, the joined tables through the Rust Mei model."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import av
import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

LENS: int = 384
"""Synthetic lens image size: a fifth of the X4 Air's 1920."""
FACE: int = 160
"""Face size of the synthetic tests."""
XI: float = 2.315
"""An X4 Air lens's xi."""
FOCAL: float = 1835.37 * LENS / 1920
"""An X4 Air lens's focal at the synthetic lens size."""
HALF_FIELD: float = np.deg2rad(100.0)
"""The X4 Air's half field."""


def camera() -> insta360_io.MeiCamera:
    """A distortion-free Mei lens shaped like an X4 Air lens, so its inverse is closed form."""
    centre: float = (LENS - 1) / 2.0
    return insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [XI, HALF_FIELD] + [0.0] * 13)


def unproject(u: Float64[ndarray, "h w"], v: Float64[ndarray, "h w"]) -> Float64[ndarray, "h w 3"]:
    """Unit rays of lens pixels for ``camera()`` (the Mei inverse without distortion)."""
    centre: float = (LENS - 1) / 2.0
    mx, my = (u - centre) / FOCAL, (v - centre) / FOCAL
    r2: Float64[ndarray, "h w"] = mx * mx + my * my
    factor: Float64[ndarray, "h w"] = (XI + np.sqrt(np.maximum(1.0 + (1.0 - XI * XI) * r2, 0.0))) / (1.0 + r2)
    return np.stack([factor * mx, factor * my, factor - XI], axis=-1)


def colour(rays: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3"]:
    """A smooth colour per direction: ``127.5 + 120 d`` per channel of the unit direction."""
    unit: Float64[ndarray, "... 3"] = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return 127.5 + 120.0 * unit


LENS1_FROM_LENS0: Float64[ndarray, "3 3"] = np.diag([-1.0, 1.0, -1.0]) @ np.array(
    [[1.0, 0.0, 0.0], [0.0, np.cos(0.01), -np.sin(0.01)], [0.0, np.sin(0.01), np.cos(0.01)]]
)
"""Lens 1 looks the other way (half a turn about y), tilted 0.6 deg so the rotation path is exercised."""


def test_the_default_face_size_is_800_at_1920_and_a_face_size_must_be_a_multiple_of_32() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, default_face_size

    assert default_face_size(1920) == 800
    assert default_face_size(3840) == 1600
    assert [face.size for face in cube_faces(800)] == [800] * 6
    with pytest.raises(ValueError, match="multiple of 32"):
        cube_faces(810)


def test_the_six_faces_are_proper_rotations_that_look_along_each_cube_axis_once() -> None:
    from dataforge.datasets.insta360_cube import cube_faces

    faces = cube_faces(FACE)
    assert [face.name for face in faces] == ["cube6_front0", "cube6_front1", "cube6_left", "cube6_right", "cube6_up", "cube6_down"]
    looks: list[tuple[float, ...]] = []
    for face in faces:
        rotation: Float64[ndarray, "3 3"] = face.cube_from_face
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        looks.append(tuple(rotation[:, 2]))
        np.testing.assert_allclose(face.intrinsics, [[FACE / 2, 0, (FACE - 1) / 2], [0, FACE / 2, (FACE - 1) / 2], [0, 0, 1]])
    assert looks == [(0, 0, 1), (0, 0, -1), (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
    # Every direction is seen by exactly one face: the largest |coordinate| of a ray picks its face.
    rng = np.random.default_rng(0)
    for ray in rng.normal(size=(200, 3)):
        seen: list[bool] = []
        for face in faces:
            local: Float64[ndarray, "3"] = face.cube_from_face.T @ ray
            seen.append(bool(local[2] > 0 and abs(local[0]) < local[2] and abs(local[1]) < local[2]))
        assert sum(seen) == 1


def test_the_seam_is_the_centre_line_of_the_four_side_faces() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_table

    for face in cube_faces(FACE):
        use0 = face_table(face, [camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS)).use_lens0
        half: int = FACE // 2
        expected: dict[str, bool] = {"cube6_front0": True, "cube6_front1": False}
        if face.name in expected:
            assert (use0 == expected[face.name]).all()
            continue
        # Lens 0 holds the half toward +Z0: image right on left, image left on right, image bottom on up, top on down.
        sides = {
            "cube6_left": (slice(None), slice(half, None)),
            "cube6_right": (slice(None), slice(None, half)),
            "cube6_up": (slice(half, None), slice(None)),
            "cube6_down": (slice(None, half), slice(None)),
        }
        lens0 = np.zeros_like(use0)
        lens0[sides[face.name]] = True
        np.testing.assert_array_equal(use0, lens0)


def test_a_face_pixel_outside_the_lens_it_is_read_from_is_refused() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_table

    centre: float = (LENS - 1) / 2.0
    narrow = insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, centre, centre, [XI, np.deg2rad(80.0)] + [0.0] * 13)
    with pytest.raises(ValueError, match="outside the lens"):
        face_table(cube_faces(FACE)[2], [narrow, narrow], LENS1_FROM_LENS0, (LENS, LENS))


def lens_pair() -> tuple[UInt8[ndarray, "h w 3"], UInt8[ndarray, "h w 3"]]:
    """Each lens painted from one colour-of-direction function (cube frame)."""
    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_rays: Float64[ndarray, "h w 3"] = unproject(u, v)
    lens0: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays)), 0, 255).astype(np.uint8)
    lens1: UInt8[ndarray, "h w 3"] = np.clip(np.round(colour(lens_rays @ LENS1_FROM_LENS0)), 0, 255).astype(np.uint8)
    return lens0, lens1


def join_faces(device: str, lens0: UInt8[ndarray, "h w 3"], lens1: UInt8[ndarray, "h w 3"]) -> UInt8[ndarray, "f n n 3"]:
    """``CubeJoin.faces`` of one lens pair on ``device``, back as ``f n n 3``."""
    import torch

    from dataforge.datasets.insta360_cube import CubeJoin, cube_faces, face_table

    tables = [face_table(face, [camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS)) for face in cube_faces(FACE)]
    join = CubeJoin(tables, (LENS, LENS), torch.device(device))
    pair = [torch.from_numpy(image).permute(2, 0, 1)[None].to(device) for image in (lens0, lens1)]
    return join.faces(pair[0], pair[1])[:, 0].permute(0, 2, 3, 1).cpu().numpy()


def test_a_scene_at_infinity_joins_into_the_colour_of_each_face_s_own_rays() -> None:
    """Every joined face shows the colour of its own rays."""
    from dataforge.datasets.insta360_cube import cube_faces

    for face, joined in zip(cube_faces(FACE), join_faces("cpu", *lens_pair()), strict=True):
        expected: Float64[ndarray, "h w 3"] = colour(face.rays() @ face.cube_from_face.T)
        error: Float64[ndarray, "h w 3"] = np.abs(joined.astype(np.float64) - expected)
        # Bilinear interpolation of a smooth field plus the 0.5 rounding of the painted lenses; no seam step.
        assert error.max() < 2.0, f"{face.name}: {error.max():.2f}"


def test_one_remap_of_the_stacked_pair_is_cv2_s_remap_of_the_lens_each_pixel_reads() -> None:
    """Noise lenses, so every interpolation weight shows: within 1 of ``cv2.remap`` (fixed-point weights) everywhere."""
    import cv2

    from dataforge.datasets.insta360_cube import cube_faces, face_table

    rng = np.random.default_rng(0)
    lenses = rng.integers(0, 256, size=(2, LENS, LENS, 3), dtype=np.uint8)
    for face, joined in zip(cube_faces(FACE), join_faces("cpu", lenses[0], lenses[1]), strict=True):
        table = face_table(face, [camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS))
        remapped = [cv2.remap(lens, table.map_x, table.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for lens in lenses]
        expected: UInt8[ndarray, "n n 3"] = np.where(table.use_lens0[..., None], remapped[0], remapped[1])
        assert np.abs(joined.astype(np.int16) - expected).max() <= 1, face.name


@pytest.mark.parametrize("mask_size", [LENS, LENS // 4])
def test_a_mask_face_is_cv2_s_nearest_remap_of_the_lens_mask_upsampled_by_contract_v2(mask_size: int) -> None:
    """Random labels at lens size and at a quarter of it: identical to ``cv2.INTER_NEAREST`` of the contract-v2 upsample."""
    import cv2

    from dataforge.datasets.insta360_cube import MaskJoin, cube_faces, face_table

    rng = np.random.default_rng(1)
    masks = rng.integers(0, 4, size=(2, mask_size, mask_size), dtype=np.uint8)
    pick: Int64[ndarray, " h"] = (2 * np.arange(LENS) + 1) * mask_size // (2 * LENS)
    tables = [face_table(face, [camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS)) for face in cube_faces(FACE)]
    joined: UInt8[ndarray, "f n n"] = MaskJoin(tables, (LENS, LENS)).faces(masks[0], masks[1])
    for table, face_mask in zip(tables, joined, strict=True):
        remapped = [cv2.remap(mask[np.ix_(pick, pick)], table.map_x, table.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for mask in masks]
        np.testing.assert_array_equal(face_mask, np.where(table.use_lens0, remapped[0], remapped[1]))


@pytest.mark.integration
def test_the_cuda_join_of_noise_is_cv2_s_remap_within_one_level() -> None:
    """Noise, so every interpolation weight shows: ``grid_sample`` against ``cv2.remap`` of the lens each pixel reads."""
    import cv2
    import torch

    from dataforge.datasets.insta360_cube import cube_faces, face_table

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(2)
    lenses = rng.integers(0, 256, size=(2, LENS, LENS, 3), dtype=np.uint8)
    for face, joined in zip(cube_faces(FACE), join_faces("cuda", lenses[0], lenses[1]), strict=True):
        table = face_table(face, [camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS))
        remapped = [cv2.remap(lens, table.map_x, table.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) for lens in lenses]
        expected: UInt8[ndarray, "n n 3"] = np.where(table.use_lens0[..., None], remapped[0], remapped[1])
        assert np.abs(joined.astype(np.int16) - expected).max() <= 1, face.name


def test_every_face_sits_at_the_midpoint_of_the_two_lens_centres() -> None:
    from dataforge.datasets.insta360_cube import cube_faces, face_from_rig

    lens_from_rig: list[Float64[ndarray, "4 4"]] = [np.eye(4), np.eye(4)]
    lens_from_rig[0][:3, :3] = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    lens_from_rig[1][:3, :3] = LENS1_FROM_LENS0 @ lens_from_rig[0][:3, :3]
    centres: list[Float64[ndarray, "3"]] = [np.array([0.01, -0.0166, 0.002]), np.array([0.01, 0.0166, 0.0])]
    for pose, centre in zip(lens_from_rig, centres, strict=True):
        pose[:3, 3] = -pose[:3, :3] @ centre
    for face in cube_faces(FACE):
        pose = face_from_rig(face, lens_from_rig)
        np.testing.assert_allclose(-pose[:3, :3].T @ pose[:3, 3], (centres[0] + centres[1]) / 2, atol=1e-12)
        np.testing.assert_allclose(pose[:3, :3].T, lens_from_rig[0][:3, :3].T @ face.cube_from_face, atol=1e-12)



def test_a_face_pixel_projected_past_the_lens_image_edge_is_an_error() -> None:
    """The Mei model accepts the ray, but an off-centre sensor ends before it lands: no silent black (or keep) pixels."""
    from dataforge.datasets.insta360_cube import cube_faces, face_table

    shifted = insta360_io.MeiCamera("mei_v6", FOCAL, FOCAL, (LENS - 1) / 2.0 + 120.0, (LENS - 1) / 2.0, [XI, HALF_FIELD] + [0.0] * 13)
    with pytest.raises(ValueError, match="outside the lens"):
        face_table(cube_faces(FACE)[0], [shifted, shifted], LENS1_FROM_LENS0, (LENS, LENS))

def run_with_timeout(target: Callable[[], object], seconds: float) -> object:
    """``target()``'s result or exception; a test failure, not a hung run, when it is still going after ``seconds``."""
    outcome: list[object] = []
    worker = threading.Thread(target=lambda: outcome.append(_call(target)), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"still running after {seconds} s: the face encoders deadlocked"
    return outcome[0]


def _call(target: Callable[[], object]) -> object:
    try:
        return target()
    except ValueError as error:
        return error


SLOTS_HELD = threading.local()
"""Encoder slots the current thread holds through ``insta360.nvenc_slot``."""


class FakeEncoder:
    """torchcodec's ``Encoder`` surface that ``encode_videos`` uses, without NVENC; it opens only while all six slots are held."""

    def add_video(self, **_: object) -> FakeEncoder:
        return self

    def add_frames(self, frames: object) -> None:
        del frames

    def open_file(self, dest: Path) -> FakeEncoder:
        assert getattr(SLOTS_HELD, "count", 0) == 6, "an encoder opened before all six slots were reserved"
        dest.write_bytes(b"")
        return self

    def __enter__(self) -> FakeEncoder:
        return self

    def __exit__(self, *exc: object) -> None:
        assert getattr(SLOTS_HELD, "count", 0) == 6, "the slots were released before an encoder closed"


@pytest.fixture
def slot_encoder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``encode_videos`` with its slot protocol (``nvenc_slot`` in ``tmp_path``, tracked per thread) and a fake torchcodec encoder."""
    import contextlib

    from dataforge import video_encoding
    from dataforge.datasets import insta360

    @contextlib.contextmanager
    def tracked_slot(count: int = 1) -> Iterator[tuple[int, ...]]:
        with video_encoding.nvenc_slot(count) as fds:
            SLOTS_HELD.count = count
            try:
                yield fds
            finally:
                SLOTS_HELD.count = 0

    monkeypatch.setattr(video_encoding, "NVENC_SLOT_DIR", tmp_path / "slots")
    monkeypatch.setattr(insta360, "nvenc_slot", tracked_slot)
    monkeypatch.setattr("torchcodec.encoders.Encoder", FakeEncoder)


def face_batches() -> Iterable[object]:
    """20 batches of one 8 x 8 frame for each of six faces."""
    import torch

    return (torch.zeros((6, 1, 3, 8, 8), dtype=torch.uint8) for _ in range(20))


@pytest.mark.usefixtures("slot_encoder")
def test_six_faces_on_two_encoder_slots_fail_fast_instead_of_hanging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    from dataforge.datasets.insta360 import encode_videos

    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "2")
    videos: list[Path] = [tmp_path / f"{k}.mp4" for k in range(6)]
    outcome = run_with_timeout(lambda: encode_videos(face_batches(), videos, width=8, height=8, gop=10, cq=14, device=torch.device("cpu"), codec="av1_nvenc"), 20.0)
    assert isinstance(outcome, ValueError) and "capacity is 2" in str(outcome)
    assert not any(video.exists() for video in videos), "an encoder opened although the slots were refused"


@pytest.mark.usefixtures("slot_encoder")
def test_two_concurrent_cube_encodes_share_six_slots_without_deadlock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    from dataforge.datasets.insta360 import encode_videos

    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "6")

    def one_run(name: str) -> int:
        return encode_videos(face_batches(), [tmp_path / f"{name}-{k}.mp4" for k in range(6)], width=8, height=8, gop=10, cq=14, device=torch.device("cpu"), codec="av1_nvenc")

    results: list[object] = []
    runs = [threading.Thread(target=lambda name=name: results.append(one_run(name)), daemon=True) for name in ("a", "b")]
    for run in runs:
        run.start()
    for run in runs:
        run.join(30.0)
    assert not any(run.is_alive() for run in runs), "two concurrent cube encodes deadlocked on the encoder slots"
    assert results == [20, 20]


def test_faces_encode_in_software_av1_with_the_nvenc_contract(tmp_path: Path) -> None:
    """A host without NVENC: SVT-AV1 writes yuv420p AV1, a keyframe every ``gop``, every frame in order."""
    import torch

    from dataforge.datasets.insta360 import encode_videos

    size: int = 96
    ramp = torch.arange(size, dtype=torch.uint8).expand(size, size)
    frames = torch.stack([torch.stack([ramp + 2 * k] * 3) for k in range(25)])
    batches = (frames[i : i + 5].unsqueeze(0).repeat(2, 1, 1, 1, 1) for i in range(0, 25, 5))
    videos: list[Path] = [tmp_path / f"{k}.mp4" for k in range(2)]
    assert encode_videos(batches, videos, width=size, height=size, gop=10, cq=14, device=torch.device("cpu"), codec="libsvtav1") == 25
    for video in videos:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            assert stream.codec_context.codec.name in {"av1", "libdav1d", "libaom-av1"} and stream.codec_context.pix_fmt == "yuv420p"
            assert [k for k, packet in enumerate(p for p in container.demux(stream) if p.size) if packet.is_keyframe] == [0, 10, 20]
        with av.open(str(video)) as container:
            means = [float(frame.to_ndarray(format="gray").mean()) for frame in container.decode(video=0)]
        np.testing.assert_allclose(means, [float(ramp.float().mean()) + 2 * k for k in range(25)], atol=2.0)
