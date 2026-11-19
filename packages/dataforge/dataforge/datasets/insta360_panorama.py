"""The equirectangular panorama of an Insta360 frameset, with the two lenses joined across a seam band.

Panorama frame ``E`` (the ``insta360`` module docstring): lens 0's axes at the rig centre, ``W x W/2`` pixels, pixel
``(u, v)`` (centres at integers) along longitude ``lon = (u + 0.5) 2 pi / W - pi / 2`` (``lon = atan2(x, z)``) and
latitude ``lat = (v + 0.5) 2 pi / W - pi / 2`` (``+y`` down). The default ``W`` is twice the lens width, the lens's
own density at its axis (3840 at 1920).

Seam. The two lenses meet on the plane ``z = 0`` (lens 0 sees ``z > 0``); each sees 100 deg off its axis, so both see
the rays within 10 deg of that plane. Outside the band ``|delta| < band`` (``delta = asin(z)``, the angle off the seam
plane) a pixel reads its own lens, as ``insta360_io.PanoramaStitch`` does. Inside it, lens 1's weight
``t = (band - delta) / (2 band)`` ramps from 0 on lens 0's edge to 1 on lens 1's (a feather). Both lenses are read as if
seen from the rig centre (only their rotations enter), so a near scene shows the 33 mm lens baseline as parallax in the
band: the blend ghosts.

Flow seam (``cuda``). Each frameset both lenses are remapped into the seam strip, a band around the seam circle
(columns: angle ``phi`` along the circle, ``x = cos phi, y = sin phi``, at the panorama's density, wrapped
``STRIP_PAD`` columns past each end; rows: ``delta`` from ``+band`` to ``-band``). NVIDIA's optical-flow engine
(``dataforge.nvof``) estimates the flow ``F01`` from lens 0's strip to lens 1's and ``F10`` back. A band pixel at strip
point ``x`` with weight ``t`` reads lens 0 at ``x - t F01(x)`` and lens 1 at ``x - (1 - t) F10(x)`` (Surround360's
novel view between the two lenses, the flow sampled at ``x``), each through the strip's lens table, then blends them
``(1 - t, t)``: lens 0 alone on its edge, lens 1 alone on its, the scene moved part way between in the band. Temporal
stability: NVOF seeds each frame's search from the last (temporal hints), and the flow is smoothed over frames
(``FLOW_SMOOTHING``), so a join must see one clip's framesets in order.

Seam kinds (``SeamKind``): ``flow`` (the flow seam, CUDA only), ``blend`` (the feather alone, no flow) and ``hard`` (each
pixel from its own lens, no band work: the fastest, the seam a visible step). The flow's quality and cost are NVOF's
preset (``slow`` is the best) and output grid (one vector per 1, 2 or 4 pixels square). ``cpu`` runs ``blend`` and
``hard`` through the same API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import insta360_io
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from dataforge.datasets.insta360_cube import ALIGN, project_pair, stack_pair, stack_pair_hwc, stacked_grid
from dataforge.nvof import OpticalFlow, PerfLevel

SEAM_BAND_DEG: float = 5.0
"""Default half-width of the seam band: inside the 10 deg both lenses see, clear of the camera body each lens sees past it."""
STRIP_PAD: int = 64
"""Strip columns wrapped past each end of the seam circle, so the flow sees no edge at ``phi = 0``."""
MIN_STRIP_ROWS: int = 32
"""NVOF's smallest image height."""
MAX_STRIP_COLUMNS: int = 8192
"""NVOF's largest image width."""
SeamKind: TypeAlias = Literal["flow", "blend", "hard"]
"""How the band joins the two lenses: across the flow seam, by the feather alone, or not at all (each pixel's own lens)."""
FlowGrid: TypeAlias = Literal[1, 2, 4]
"""NVOF output grid: one flow vector per ``grid x grid`` block, sampled bilinearly in between."""
FLOW_SMOOTHING: float = 0.5
"""Weight of the new flow in the running flow (exponential smoothing over frames)."""
SUB_BATCH: int = 4
"""Framesets remapped together on CUDA: one 3840x1920 remap already fills the GPU, so a larger step only raises the float peak."""
BAND_COLUMNS: int = 4096
"""Band pixels are remapped on the CPU as rows of this many."""
LUMA: tuple[float, float, float] = (0.299, 0.587, 0.114)
"""RGB weights of the gray strips the flow sees."""


def default_panorama_width(lens_width: int) -> int:
    """``W`` for a lens image ``lens_width`` pixels wide: twice it, to a multiple of 32 (1920 -> 3840)."""
    return max(ALIGN, ALIGN * round(2 * lens_width / ALIGN))


def panorama_rays(width: int) -> Float64[ndarray, "h w 3"]:
    """The unit ray of every panorama pixel in frame ``E``.

    Raises:
        ValueError: If ``width`` is not a positive multiple of ``ALIGN``.
    """
    if width <= 0 or width % ALIGN:
        raise ValueError(f"panorama width {width} is not a positive multiple of {ALIGN}")
    step: float = 2.0 * math.pi / width
    lon: Float64[ndarray, " w"] = (np.arange(width) + 0.5) * step - math.pi / 2.0
    lat: Float64[ndarray, " h"] = (np.arange(width // 2) + 0.5) * step - math.pi / 2.0
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat_grid) * np.sin(lon_grid), np.sin(lat_grid), np.cos(lat_grid) * np.cos(lon_grid)], axis=-1)


@dataclass(frozen=True, slots=True)
class PanoramaTable:
    """Where each panorama pixel reads the stacked lens pair (``insta360_cube.stack_pair``)."""

    pano_xy: Float32[ndarray, "ph pw 2"]
    """Stacked-pair pixel of each panorama pixel in its own lens (lens 0 where ``z >= 0``): the read outside the band."""
    band_index: Int64[ndarray, " n"]
    """Flat panorama index of each band pixel (``|delta| < band``)."""
    band_weight: Float32[ndarray, " n"]
    """Lens 1's weight ``t`` of each band pixel."""
    band_lens_xy: Float32[ndarray, "2 n 2"]
    """Stacked-pair pixel of each band pixel in lens 0 and in lens 1."""
    band_strip_xy: Float32[ndarray, "n 2"]
    """Strip pixel of each band pixel."""
    strip_lens_xy: Float32[ndarray, "2 sh sw 2"]
    """Stacked-pair pixel of each strip pixel in lens 0 and in lens 1."""

    @property
    def size(self) -> tuple[int, int]:
        """Panorama width and height."""
        return self.pano_xy.shape[1], self.pano_xy.shape[0]

    @property
    def strip_size(self) -> tuple[int, int]:
        """Strip width and height."""
        return self.strip_lens_xy.shape[2], self.strip_lens_xy.shape[1]


def panorama_table(
    cameras: list[insta360_io.MeiCamera], lens1_from_lens0: Float64[ndarray, "3 3"], lens_size: tuple[int, int], width: int, band_deg: float
) -> PanoramaTable:
    """The table of a ``width``-pixel panorama with a ``band_deg`` seam band on the two lens cameras (built once per clip).

    Raises:
        ValueError: If ``band_deg`` is not positive, or a pixel a lens must supply lies outside it.
    """
    if band_deg <= 0.0:
        raise ValueError(f"seam band {band_deg} deg must be positive")
    lens1_row: Float64[ndarray, "2"] = np.array([0.0, lens_size[1] + 1.0])
    band: float = math.radians(band_deg)
    step: float = 2.0 * math.pi / width
    rays: Float64[ndarray, "n 3"] = panorama_rays(width).reshape(-1, 3)
    pixels, held = project_pair(cameras, lens1_from_lens0, lens_size, rays)
    pixels[1] += lens1_row
    lens1: Bool[ndarray, " n"] = rays[:, 2] < 0.0
    outside: int = int((~np.where(lens1, held[1], held[0])).sum())
    if outside:
        raise ValueError(f"{outside} panorama pixels lie outside the lens they are read from")
    delta: Float64[ndarray, " n"] = np.arcsin(np.clip(rays[:, 2], -1.0, 1.0))
    index: Int64[ndarray, " k"] = np.flatnonzero(np.abs(delta) < band)
    weight: Float64[ndarray, " k"] = (band - delta[index]) / (2.0 * band)
    columns: int = width + 2 * STRIP_PAD
    rows: int = max(MIN_STRIP_ROWS, 4 * round(2.0 * band / step / 4))
    phi: Float64[ndarray, " sw"] = (np.arange(columns) - STRIP_PAD + 0.5) * step
    strip_delta: Float64[ndarray, " sh"] = band - (np.arange(rows) + 0.5) * 2.0 * band / rows
    phi_grid, delta_grid = np.meshgrid(phi, strip_delta)
    strip_rays: Float64[ndarray, "m 3"] = np.stack([np.cos(delta_grid) * np.cos(phi_grid), np.cos(delta_grid) * np.sin(phi_grid), np.sin(delta_grid)], axis=-1).reshape(-1, 3)
    strip_pixels, strip_held = project_pair(cameras, lens1_from_lens0, lens_size, strip_rays)
    strip_pixels[1] += lens1_row
    for lens in range(2):
        if not (strip_held[lens].all() and held[lens][index].all()):
            raise ValueError(f"the seam band of +-{band_deg} deg reaches outside lens {lens}")
    band_phi: Float64[ndarray, " k"] = np.mod(np.arctan2(rays[index, 1], rays[index, 0]), 2.0 * math.pi)
    return PanoramaTable(
        pano_xy=np.where(lens1[:, None], pixels[1], pixels[0]).reshape(width // 2, width, 2).astype(np.float32),
        band_index=index,
        band_weight=weight.astype(np.float32),
        band_lens_xy=pixels[:, index].astype(np.float32),
        band_strip_xy=np.stack([band_phi / step - 0.5 + STRIP_PAD, weight * rows - 0.5], axis=-1).astype(np.float32),
        strip_lens_xy=strip_pixels.reshape(2, rows, columns, 2).astype(np.float32),
    )


class FlowSeam:
    """The flow seam's state on CUDA: two NVOF sessions (``F01``, ``F10``) on the seam strip and the running flow.

    Raises:
        ValueError: If the strip is wider than NVOF takes.
    """

    def __init__(self, table: PanoramaTable, lens_size: tuple[int, int], device: torch.device, *, preset: PerfLevel, grid: FlowGrid) -> None:
        """``lens_size``: lens image ``(width, height)``; ``preset`` and ``grid``: NVOF's quality preset and output grid."""
        strip_width, strip_height = table.strip_size
        if strip_width > MAX_STRIP_COLUMNS:
            raise ValueError(f"a {strip_width}-column seam strip is wider than NVOF's {MAX_STRIP_COLUMNS}; use a narrower panorama")
        self.strip_height: int = strip_height
        strip_grid: Float32[Tensor, "2 sh sw 2"] = torch.from_numpy(stacked_grid(table.strip_lens_xy, lens_size)).to(device)
        self.strip_grid: Float32[Tensor, "1 sh2 sw 2"] = strip_grid.reshape(1, 2 * strip_height, strip_width, 2)
        self.strip_lens_grid: Float32[Tensor, "2 2 sh sw"] = strip_grid.permute(0, 3, 1, 2).contiguous()
        self.band_strip_xy: Float32[Tensor, "n 2"] = torch.from_numpy(table.band_strip_xy).to(device)
        self.band_weight: Float32[Tensor, "n 1"] = torch.from_numpy(table.band_weight).to(device)[:, None]
        self.strip_scale: Float32[Tensor, "2"] = torch.tensor([2.0 / (strip_width - 1), 2.0 / (strip_height - 1)], device=device)
        coarse_size: Float32[Tensor, "2"] = torch.tensor([-(-strip_width // grid), -(-strip_height // grid)], dtype=torch.float32, device=device)
        # Flow block j covers strip pixels [gj, gj + g): its centre is pixel gj + (g - 1) / 2.
        coarse_xy: Float32[Tensor, "n 2"] = (self.band_strip_xy - (grid - 1) / 2.0) / grid
        self.band_flow_grid: Float32[Tensor, "1 1 n 2"] = (coarse_xy * 2.0 / (coarse_size - 1.0) - 1.0)[None, None]
        self.luma: Float32[Tensor, "1 3 1 1"] = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)
        self.running: Float32[Tensor, "4 gh gw"] | None = None
        """The smoothed ``F01`` (channels 0-1) and ``F10`` (2-3) on the flow grid; None before the first frameset."""
        self.sessions: list[OpticalFlow] = []
        try:
            for _ in range(2):
                self.sessions.append(OpticalFlow(strip_width, strip_height, grid=grid, perf=preset))
        except BaseException:
            self.close()
            raise

    def band_grid(self, stacked: Float32[Tensor, "b 3 r w"]) -> Float32[Tensor, "b 2 n 2"]:
        """Where each band pixel reads lens 0 and lens 1 in each frameset, as stacked-pair grid coordinates.

        Raises:
            RuntimeError: After ``close``.
        """
        if not self.sessions:
            raise RuntimeError("the flow seam is closed")
        b: int = stacked.shape[0]
        strips: Float32[Tensor, "b 3 sh2 sw"] = F.grid_sample(stacked, self.strip_grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        gray: UInt8[Tensor, "b sh2 sw"] = (strips * self.luma).sum(1).round_().to(torch.uint8)
        smoothed: list[Float32[Tensor, "4 gh gw"]] = []
        for frame in gray:
            lens0, lens1 = frame[: self.strip_height], frame[self.strip_height :]
            flow: Float32[Tensor, "4 gh gw"] = torch.cat([self.sessions[0].flow(lens0, lens1), self.sessions[1].flow(lens1, lens0)], dim=-1).permute(2, 0, 1)
            self.running = flow if self.running is None else torch.lerp(self.running, flow, FLOW_SMOOTHING)
            smoothed.append(self.running)
        at_band: Float32[Tensor, "b 4 n"] = F.grid_sample(torch.stack(smoothed), self.band_flow_grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0]
        # Surround360's novel view at t: lens 0 moved t of the way along F01, lens 1 (1 - t) of the way along F10.
        read0: Float32[Tensor, "b n 2"] = self.band_strip_xy - self.band_weight * at_band[:, 0:2].transpose(1, 2)
        read1: Float32[Tensor, "b n 2"] = self.band_strip_xy - (1.0 - self.band_weight) * at_band[:, 2:4].transpose(1, 2)
        lens_grids: list[Float32[Tensor, "b n 2"]] = [
            F.grid_sample(self.strip_lens_grid[lens : lens + 1].expand(b, -1, -1, -1), (read * self.strip_scale - 1.0)[:, None], mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0].transpose(1, 2)
            for lens, read in enumerate((read0, read1))
        ]
        return torch.stack(lens_grids, dim=1)

    def close(self) -> None:
        """Free the NVOF sessions."""
        for session in self.sessions:
            session.close()
        self.sessions = []


class PanoramaJoin:
    """The panoramas of lens pairs on one device, in frame order (the flow seam carries state from frame to frame).

    ``cuda``: ``grid_sample`` remaps; ``cpu``: ``cv2.remap``. ``seam`` picks how the band is joined (module docstring).
    Call ``close`` to free the NVOF sessions.

    Raises:
        ValueError: If the flow seam is asked for off CUDA, or its strip is wider than NVOF takes.
    """

    def __init__(
        self, table: PanoramaTable, lens_size: tuple[int, int], device: torch.device, *, seam: SeamKind, flow_preset: PerfLevel = "slow", flow_grid: FlowGrid = 4
    ) -> None:
        """``lens_size``: lens image ``(width, height)``; ``flow_preset``, ``flow_grid``: NVOF's, for the flow seam."""
        if seam == "flow" and device.type != "cuda":
            raise ValueError("the flow seam runs on CUDA (NVOF); on the cpu the seam is blend or hard")
        self.table: PanoramaTable = table
        self.device: torch.device = device
        self.kind: SeamKind = seam
        self.seam: FlowSeam | None = None
        if device.type == "cpu":
            count: int = table.band_index.size
            padded: Float32[ndarray, "2 m 2"] = np.zeros((2, -(-count // BAND_COLUMNS) * BAND_COLUMNS, 2), dtype=np.float32)
            padded[:, :count] = table.band_lens_xy
            band_maps: Float32[ndarray, "2 r c 2"] = padded.reshape(2, -1, BAND_COLUMNS, 2)
            # x and y maps contiguous once, so cv2.remap does not copy them every frame.
            self.pano_maps: tuple[Float32[ndarray, "ph pw"], Float32[ndarray, "ph pw"]] = (
                np.ascontiguousarray(table.pano_xy[..., 0]),
                np.ascontiguousarray(table.pano_xy[..., 1]),
            )
            self.band_maps: list[tuple[Float32[ndarray, "r c"], Float32[ndarray, "r c"]]] = [
                (np.ascontiguousarray(lens_map[..., 0]), np.ascontiguousarray(lens_map[..., 1])) for lens_map in band_maps
            ]
            return
        self.pano_grid: Float32[Tensor, "1 ph pw 2"] = torch.from_numpy(stacked_grid(table.pano_xy, lens_size)).to(device)[None]
        self.band_index: Int64[Tensor, " n"] = torch.from_numpy(table.band_index).to(device)
        self.band_weight: Float32[Tensor, " n"] = torch.from_numpy(table.band_weight).to(device)
        self.band_lens_grid: Float32[Tensor, "1 2 n 2"] = torch.from_numpy(stacked_grid(table.band_lens_xy, lens_size)).to(device)[None]
        if seam == "flow":
            self.seam = FlowSeam(table, lens_size, device, preset=flow_preset, grid=flow_grid)

    def panoramas(self, lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[Tensor, "b 3 ph pw"]:
        """The panoramas of ``b`` consecutive lens pairs (RGB, channels first, on the device).

        Raises:
            RuntimeError: If the join's flow seam was closed.
        """
        if self.device.type == "cpu":
            return self._cpu_panoramas(lens0, lens1)
        b, c = lens0.shape[:2]
        width, height = self.table.size
        out: UInt8[Tensor, "b 3 ph pw"] = torch.empty((b, c, height, width), dtype=torch.uint8, device=self.device)
        for first in range(0, b, SUB_BATCH):
            stacked: Float32[Tensor, "s 3 r w"] = stack_pair(lens0[first : first + SUB_BATCH], lens1[first : first + SUB_BATCH])
            s, _, rows, columns = stacked.shape
            # Static grids: the framesets ride in the channels, so the bilinear weights are computed once per pixel.
            folded: Float32[Tensor, "1 sc r w"] = stacked.view(1, s * c, rows, columns)
            pano: Float32[Tensor, "s 3 ph pw"] = F.grid_sample(folded, self.pano_grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(s, c, height, width)
            if self.kind == "hard":
                out[first : first + s] = pano.round_()
                continue
            if self.seam is None:
                reads: Float32[Tensor, "s 3 2 n"] = F.grid_sample(folded, self.band_lens_grid, mode="bilinear", padding_mode="zeros", align_corners=True).view(s, c, 2, -1)
            else:
                reads = F.grid_sample(stacked, self.seam.band_grid(stacked), mode="bilinear", padding_mode="zeros", align_corners=True)
            pano.view(s, c, -1)[:, :, self.band_index] = torch.lerp(reads[:, :, 0], reads[:, :, 1], self.band_weight)
            # Bilinear reads and a lerp of [0, 255] values stay in [0, 255].
            out[first : first + s] = pano.round_()
        return out

    def _cpu_panoramas(self, lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[Tensor, "b 3 ph pw"]:
        """``cv2.remap`` of the stacked pair, the band blended without flow (``blend``) or left as read (``hard``)."""
        count: int = self.table.band_index.size
        weight: Float32[ndarray, "n 1"] = self.table.band_weight[:, None]
        stacked: UInt8[ndarray, "b r w 3"] = stack_pair_hwc(lens0, lens1)
        panoramas: list[UInt8[ndarray, "ph pw 3"]] = []
        for pair in stacked:
            pano: UInt8[ndarray, "ph pw 3"] = cv2.remap(pair, *self.pano_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            panoramas.append(pano)
            if self.kind == "hard":
                continue
            reads: list[Float32[ndarray, "n 3"]] = [
                cv2.remap(pair, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, pair.shape[-1])[:count].astype(np.float32)
                for map_x, map_y in self.band_maps
            ]
            pano.reshape(-1, pair.shape[-1])[self.table.band_index] = np.rint(reads[0] + weight * (reads[1] - reads[0])).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(np.stack(panoramas).transpose(0, 3, 1, 2)))

    def close(self) -> None:
        """Free the flow seam's NVOF sessions; a later ``panoramas`` call with the flow seam raises."""
        if self.seam is not None:
            self.seam.close()
