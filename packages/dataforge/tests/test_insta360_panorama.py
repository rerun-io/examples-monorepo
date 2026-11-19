"""The equirectangular panorama of an Insta360 frameset: the tables, the feathered seam band, the optical-flow seam."""

from __future__ import annotations

import math

import insta360_io
import numpy as np
import pytest
from jaxtyping import Float64, UInt8
from numpy import ndarray
from test_insta360_cube import LENS, LENS1_FROM_LENS0, camera, colour, lens_pair, unproject
from test_insta360_io import factory_camera

PANO: int = 2 * LENS
"""Panorama width of the synthetic tests: the default, twice the lens width."""
BAND_DEG: float = 5.0
"""Seam band half-width of the synthetic tests (the default)."""


def join_panoramas(device: str, lens0: UInt8[ndarray, "h w 3"], lens1: UInt8[ndarray, "h w 3"], *, frames: int = 1) -> UInt8[ndarray, "ph pw 3"]:
    """The panorama of one lens pair (synthetic cameras), joined ``frames`` times in a row on ``device`` (``cuda``: the
    flow seam; ``cpu``: the plain feather); the last one."""
    import torch

    from dataforge.datasets.insta360_panorama import PanoramaJoin, panorama_table

    table = panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, BAND_DEG)
    pair = [torch.from_numpy(image).permute(2, 0, 1)[None].to(device) for image in (lens0, lens1)]
    join = PanoramaJoin(table, (LENS, LENS), torch.device(device))
    try:
        joined = join.panoramas(pair[0], pair[1])
        for _ in range(frames - 1):
            joined = join.panoramas(pair[0], pair[1])
    finally:
        join.close()
    return joined[0].permute(1, 2, 0).cpu().numpy()


def seam_offset_rad(width: int) -> Float64[ndarray, "h w"]:
    """Signed angle of each panorama pixel's ray off the seam plane ``z = 0``, positive on lens 0's side."""
    from dataforge.datasets.insta360_panorama import panorama_rays

    return np.arcsin(np.clip(panorama_rays(width)[..., 2], -1.0, 1.0))


def test_the_default_panorama_is_twice_the_lens_width_and_a_width_must_be_a_multiple_of_32() -> None:
    from dataforge.datasets.insta360_panorama import default_panorama_width, panorama_rays

    assert default_panorama_width(1920) == 3840
    assert default_panorama_width(3840) == 7680
    rays: Float64[ndarray, "h w 3"] = panorama_rays(64)
    assert rays.shape == (32, 64, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0)
    # Column W/4 looks along lens 0's +Z, column 3W/4 along -Z; row 0 is near -Y (up).
    np.testing.assert_allclose(rays[16, 16], [math.sin(math.pi / 64), math.sin(math.pi / 64), math.cos(math.pi / 64)], atol=0.02)
    assert rays[16, 48, 2] < -0.99
    assert rays[0, 0, 1] < -0.99
    with pytest.raises(ValueError, match="multiple of 32"):
        panorama_rays(100)


def test_a_seam_band_beyond_either_lens_s_field_is_refused() -> None:
    from dataforge.datasets.insta360_panorama import panorama_table

    # The lenses see 100 deg off their axes: 10 deg past the seam plane.
    panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, 9.0)
    with pytest.raises(ValueError, match="outside lens"):
        panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, 11.0)
    with pytest.raises(ValueError, match="positive"):
        panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, 0.0)


def test_a_table_built_in_row_blocks_is_the_table_built_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataforge.datasets import insta360_panorama
    from dataforge.datasets.insta360_panorama import panorama_table

    whole = panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, BAND_DEG)
    monkeypatch.setattr(insta360_panorama, "TABLE_BLOCK_RAYS", 7 * PANO)
    blocks = panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, BAND_DEG)
    for name in ("pano_xy", "band_index", "band_weight", "band_lens_xy", "band_strip_xy", "strip_lens_xy"):
        np.testing.assert_array_equal(getattr(blocks, name), getattr(whole, name), err_msg=name)


def test_a_scene_at_infinity_stitches_into_the_colour_of_each_panorama_ray() -> None:
    """Both lenses agree everywhere, so the blend band is invisible: every pixel is the colour of its own ray."""
    from dataforge.datasets.insta360_panorama import panorama_rays

    joined: UInt8[ndarray, "h w 3"] = join_panoramas("cpu", *lens_pair())
    error: Float64[ndarray, "h w 3"] = np.abs(joined.astype(np.float64) - colour(panorama_rays(PANO)))
    assert error.max() < 2.0, f"{error.max():.2f}"


def test_the_blend_ramps_from_lens_0_to_lens_1_across_the_band_and_each_lens_holds_its_side() -> None:
    bright: UInt8[ndarray, "h w 3"] = np.full((LENS, LENS, 3), 200, dtype=np.uint8)
    dark: UInt8[ndarray, "h w 3"] = np.full((LENS, LENS, 3), 40, dtype=np.uint8)
    joined: Float64[ndarray, "h w"] = join_panoramas("cpu", bright, dark)[..., 0].astype(np.float64)
    offset: Float64[ndarray, "h w"] = seam_offset_rad(PANO)
    band: float = math.radians(BAND_DEG)
    lens1_weight: Float64[ndarray, "h w"] = np.clip((band - offset) / (2.0 * band), 0.0, 1.0)
    assert np.all(joined[offset > band] == 200)
    assert np.all(joined[offset < -band] == 40)
    np.testing.assert_allclose(joined, 200.0 - 160.0 * lens1_weight, atol=1.0)


def test_away_from_the_band_the_panorama_is_insta360_io_s_rust_stitch() -> None:
    """The X4 Air factory cameras, a smooth random scene: off the band the RGB join of a gray pair is the CPU stitch."""
    import cv2
    import torch

    from dataforge.datasets.insta360_panorama import PanoramaJoin, panorama_table

    rng = np.random.default_rng(3)
    coarse: UInt8[ndarray, "2 h w"] = rng.integers(0, 256, size=(2, 48, 48), dtype=np.uint8)
    lenses: list[UInt8[ndarray, "h w"]] = [cv2.resize(image, (1920, 1920), interpolation=cv2.INTER_CUBIC) for image in coarse]
    lens1_from_lens0: Float64[ndarray, "3 3"] = np.diag([-1.0, 1.0, -1.0])
    stitch = insta360_io.PanoramaStitch(factory_camera(0), factory_camera(1), [[-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, -0.0333], [0.0, 0.0, 0.0, 1.0]], (1920, 1920))
    expected: UInt8[ndarray, "960 1920"] = stitch.panorama(lenses[0], lenses[1])
    table = panorama_table([factory_camera(0), factory_camera(1)], lens1_from_lens0, (1920, 1920), 1920, BAND_DEG)
    join = PanoramaJoin(table, (1920, 1920), torch.device("cpu"))
    pair = [torch.from_numpy(np.repeat(image[None, None], 3, axis=1)) for image in lenses]
    joined: UInt8[ndarray, "960 1920"] = join.panoramas(pair[0], pair[1])[0, 0].numpy()
    outside: np.ndarray = np.abs(seam_offset_rad(1920)) > math.radians(BAND_DEG) + 2 * math.pi / 1920
    difference: Float64[ndarray, " n"] = np.abs(joined.astype(np.float64) - expected.astype(np.float64))[outside]
    # The Rust stitch rounds its fixed-point bilinear weights; a smooth scene keeps that within a few levels.
    assert difference.mean() < 0.5 and difference.max() <= 4, f"mean {difference.mean():.2f} max {difference.max():.0f}"


@pytest.mark.integration
def test_nvof_finds_a_known_shift_of_a_texture() -> None:
    import torch
    import torch.nn.functional as F

    from dataforge.nvof import OpticalFlow

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    generator = torch.Generator(device="cuda").manual_seed(0)
    texture = F.interpolate(torch.rand(1, 1, 40, 160, device="cuda", generator=generator), size=(180, 660), mode="bicubic")[0, 0]
    image = (texture.clamp(0, 1) * 255).to(torch.uint8)
    flow = OpticalFlow(640, 160, grid=4)
    try:
        # reference(x) = image(x - (7, -3)), so input(x) = reference(x + (7, -3)).
        found = flow.flow(image[10:170, 10:650], image[13:173, 3:643])
    finally:
        flow.close()
    with pytest.raises(RuntimeError, match="closed"):
        flow.flow(image[10:170, 10:650], image[13:173, 3:643])
    assert found.shape == (40, 160, 2)
    median = found[2:-2, 2:-2].reshape(-1, 2).median(0).values.tolist()
    assert median == pytest.approx([7.0, -3.0], abs=0.25)


@pytest.mark.integration
def test_off_the_band_the_cuda_join_is_the_cpu_join_and_a_scene_at_infinity_stays_its_colour_across_the_flow_seam() -> None:
    """Noise lenses off the band (every interpolation weight shows): within 1 level of ``cv2.remap``. The scene at infinity:
    both lenses agree, the flow is about zero, and the flow seam keeps each band pixel the colour of its ray."""
    import torch

    from dataforge.datasets.insta360_panorama import panorama_rays

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(4)
    lenses = rng.integers(0, 256, size=(2, LENS, LENS, 3), dtype=np.uint8)
    outside: np.ndarray = np.abs(seam_offset_rad(PANO)) >= math.radians(BAND_DEG)
    cpu = join_panoramas("cpu", lenses[0], lenses[1])
    cuda = join_panoramas("cuda", lenses[0], lenses[1])
    assert np.abs(cpu.astype(np.int16) - cuda)[outside].max() <= 1
    joined: UInt8[ndarray, "h w 3"] = join_panoramas("cuda", *lens_pair(), frames=3)
    error: Float64[ndarray, "h w 3"] = np.abs(joined.astype(np.float64) - colour(panorama_rays(PANO)))
    assert error[~outside].mean() < 1.0 and error.max() < 4.0, f"band mean {error[~outside].mean():.2f}, max {error.max():.2f}"


def texture(rays: Float64[ndarray, "... 3"]) -> Float64[ndarray, "... 3"]:
    """A smooth random gray texture on the sphere (40 plane waves, wavelengths 3 to 6 deg), as RGB."""
    rng = np.random.default_rng(5)
    directions: Float64[ndarray, "k 3"] = rng.normal(size=(40, 3))
    waves: Float64[ndarray, "k 3"] = directions / np.linalg.norm(directions, axis=1, keepdims=True) * rng.uniform(60.0, 120.0, size=(40, 1))
    phases: Float64[ndarray, " k"] = rng.uniform(0.0, 2 * math.pi, size=40)
    unit: Float64[ndarray, "... 3"] = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    value: Float64[ndarray, "..."] = 127.5 + 110.0 * np.cos(unit @ waves.T + phases).mean(-1) * math.sqrt(2.0 * 40) / 2.0
    return np.repeat(np.clip(value, 0.0, 255.0)[..., None], 3, axis=-1)


def rotation_y(angle: float) -> Float64[ndarray, "3 3"]:
    return np.array([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])


@pytest.mark.integration
def test_the_flow_seam_pulls_a_misregistered_lens_into_register_across_the_band() -> None:
    """Lens 1 sees the scene turned 1.5 deg about y, which moves it across the seam (like parallax): in the band the flow
    seam shows the scene turned by ``t * 1.5 deg``, the view between the lenses, much closer than the plain blend does.

    Scored where the turn moves the scene across the seam most (``|cos phi| > 0.7`` along the seam circle); near the poles
    nothing moves and both show only the interpolation error of a texture this fine at this lens size (about 3 levels)."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from dataforge.datasets.insta360_panorama import panorama_rays

    turn: float = math.radians(1.5)
    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_rays: Float64[ndarray, "h w 3"] = unproject(u, v)
    lens0: UInt8[ndarray, "h w 3"] = np.round(texture(lens_rays)).astype(np.uint8)
    lens1: UInt8[ndarray, "h w 3"] = np.round(texture(lens_rays @ LENS1_FROM_LENS0 @ rotation_y(turn).T)).astype(np.uint8)
    offset: Float64[ndarray, "h w"] = seam_offset_rad(PANO)
    band: float = math.radians(BAND_DEG)
    rays: Float64[ndarray, "h w 3"] = panorama_rays(PANO)
    inside: np.ndarray = (np.abs(offset) < band) & (np.abs(rays[..., 0]) > 0.7 * np.hypot(rays[..., 0], rays[..., 1]))
    lens1_weight: Float64[ndarray, " n"] = np.clip((band - offset[inside]) / (2.0 * band), 0.0, 1.0)
    pano_rays: Float64[ndarray, "n 3"] = rays[inside]
    angle: Float64[ndarray, " n"] = lens1_weight * turn
    turned: Float64[ndarray, "n 3"] = np.stack(
        [np.cos(angle) * pano_rays[:, 0] + np.sin(angle) * pano_rays[:, 2], pano_rays[:, 1], -np.sin(angle) * pano_rays[:, 0] + np.cos(angle) * pano_rays[:, 2]], axis=-1
    )
    between: Float64[ndarray, " n"] = texture(turned)[:, 0]
    errors: dict[str, float] = {}
    for name, device in (("blend", "cpu"), ("flow", "cuda")):
        joined: UInt8[ndarray, "h w 3"] = join_panoramas(device, lens0, lens1, frames=3)
        errors[name] = float(np.abs(joined[..., 0][inside].astype(np.float64) - between).mean())
    print(errors)
    assert errors["flow"] < 0.5 * errors["blend"]


@pytest.mark.integration
def test_a_closed_flow_seam_refuses_instead_of_falling_back_to_the_blend() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from dataforge.datasets.insta360_panorama import PanoramaJoin, panorama_table

    table = panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, BAND_DEG)
    join = PanoramaJoin(table, (LENS, LENS), torch.device("cuda"))
    pair = [torch.from_numpy(image).permute(2, 0, 1)[None].to("cuda") for image in lens_pair()]
    assert join.panoramas(pair[0], pair[1]).shape == (1, 3, PANO // 2, PANO)
    join.close()
    with pytest.raises(RuntimeError, match="closed"):
        join.panoramas(pair[0], pair[1])


@pytest.mark.integration
def test_a_batch_across_the_sub_batch_boundary_is_the_same_framesets_joined_one_at_a_time() -> None:
    """Five lens pairs whose lens 1 turns 0.5 deg more each frameset: one batch of five (sub-batches of 4 and 1, the flow
    smoothed across them) and five calls of one, on fresh joins, give the same panoramas."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from dataforge.datasets.insta360_panorama import SUB_BATCH, PanoramaJoin, panorama_table

    assert SUB_BATCH < 5
    u, v = np.meshgrid(np.arange(LENS, dtype=np.float64), np.arange(LENS, dtype=np.float64))
    lens_rays: Float64[ndarray, "h w 3"] = unproject(u, v)
    lens0 = torch.from_numpy(np.round(texture(lens_rays)).astype(np.uint8)).permute(2, 0, 1)
    lens1 = torch.stack(
        [torch.from_numpy(np.round(texture(lens_rays @ LENS1_FROM_LENS0 @ rotation_y(math.radians(0.5 * k)).T)).astype(np.uint8)).permute(2, 0, 1) for k in range(5)]
    ).cuda()
    lens0 = lens0[None].expand(5, -1, -1, -1).contiguous().cuda()
    table = panorama_table([camera(), camera()], LENS1_FROM_LENS0, (LENS, LENS), PANO, BAND_DEG)
    batched = PanoramaJoin(table, (LENS, LENS), torch.device("cuda"))
    single = PanoramaJoin(table, (LENS, LENS), torch.device("cuda"))
    try:
        together = batched.panoramas(lens0, lens1)
        apart = torch.cat([single.panoramas(lens0[k : k + 1], lens1[k : k + 1]) for k in range(5)])
    finally:
        batched.close()
        single.close()
    assert torch.equal(together, apart)
