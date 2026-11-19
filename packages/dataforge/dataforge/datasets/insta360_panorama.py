"""The equirectangular panorama of an Insta360 frameset, with the two lenses joined across a seam band.

Panorama frame ``E`` (the ``insta360`` module docstring): lens 0's axes at the rig centre, ``W x W/2`` pixels, pixel
``(u, v)`` (centres at integers) along longitude ``lon = (u + 0.5) 2 pi / W - pi / 2`` (``lon = atan2(x, z)``) and
latitude ``lat = (v + 0.5) 2 pi / W - pi / 2`` (``+y`` down). The default ``W`` is twice the lens width, the lens's
own density at its axis (3840 at 1920).

Seam. The two lenses meet on the plane ``z = 0`` (lens 0 sees ``z > 0``); each sees 100 deg off its axis, so both see
the rays within 10 deg of that plane. Outside the band ``|delta| < band`` (``delta = asin(z)``, the angle off the seam
plane) a pixel reads its own lens, as today's CPU stitch does. Inside it, lens 1's weight ``t = (band - delta) / (2 band)``
ramps from 0 on lens 0's edge to 1 on lens 1's (a feather). Both lenses are read as if seen from the rig centre (only
their rotations enter), so a near scene shows the 33 mm lens baseline as parallax in the band: the blend ghosts.

Flow seam (``cuda``). Each frameset both lenses are remapped into the seam strip, a band around the seam circle
(columns: angle ``phi`` along the circle, ``x = cos phi, y = sin phi``, at the panorama's density, wrapped
``STRIP_PAD`` columns past each end; rows: ``delta`` from ``+band`` to ``-band``). NVIDIA's optical-flow engine
(``dataforge.nvof``) estimates the flow ``F01`` from lens 0's strip to lens 1's and ``F10`` back. A band pixel at strip
point ``x`` with weight ``t`` reads lens 0 at ``x - t F01(x)`` and lens 1 at ``x - (1 - t) F10(x)`` (Surround360's
novel view between the two lenses, the flow sampled at ``x``), each through the strip's lens table, then blends them
``(1 - t, t)``: lens 0 alone on its edge, lens 1 alone on its, the scene moved part way between in the band. Temporal
stability: NVOF seeds each frame's search from the last (temporal hints), and the flow is smoothed over frames
(``FLOW_SMOOTHING``), so a join must see one clip's framesets in order.

``cpu`` blends the band without flow (``flow=False``), through the same API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import insta360_io
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Float64, Int64, UInt8
from numpy import ndarray
from torch import Tensor

from dataforge.nvof import OpticalFlow

ALIGN: int = 32
"""The panorama width is a multiple of this."""
SEAM_BAND_DEG: float = 5.0
"""Default half-width of the seam band: inside the 10 deg both lenses see, clear of the camera body each lens sees past it."""
STRIP_PAD: int = 64
"""Strip columns wrapped past each end of the seam circle, so the flow sees no edge at ``phi = 0``."""
MIN_STRIP_ROWS: int = 32
"""NVOF's smallest image height."""
MAX_STRIP_COLUMNS: int = 8192
"""NVOF's largest image width."""
FLOW_GRID: int = 4
"""NVOF output grid: one vector per 4 x 4 block, sampled bilinearly in between."""
FLOW_SMOOTHING: float = 0.5
"""Weight of the new flow in the running flow (exponential smoothing over frames)."""
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
    """Where each panorama pixel reads the stacked lens pair (lens 0 on rows ``[0, h)``, lens 1 on ``[h + 1, 2h + 1)``)."""

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
    lens_width, lens_height = lens_size

    def stacked(rays: Float64[ndarray, "n 3"]) -> tuple[Float64[ndarray, "2 n 2"], Bool[ndarray, "2 n"]]:
        """Each ray's pixel in lens 0 and lens 1 (stacked-pair rows), and whether it is inside the lens's field and image."""
        pixels0, valid0 = cameras[0].project(np.ascontiguousarray(rays))
        pixels1, valid1 = cameras[1].project(np.ascontiguousarray(rays @ lens1_from_lens0.T))
        pixels: Float64[ndarray, "2 n 2"] = np.stack([pixels0, pixels1])
        on_sensor: Bool[ndarray, "2 n"] = (pixels[..., 0] >= 0.0) & (pixels[..., 0] <= lens_width - 1) & (pixels[..., 1] >= 0.0) & (pixels[..., 1] <= lens_height - 1)
        pixels[1, :, 1] += lens_height + 1.0
        return pixels, np.stack([valid0, valid1]) & on_sensor

    band: float = math.radians(band_deg)
    step: float = 2.0 * math.pi / width
    rays: Float64[ndarray, "n 3"] = panorama_rays(width).reshape(-1, 3)
    pixels, valid = stacked(rays)
    lens1: Bool[ndarray, " n"] = rays[:, 2] < 0.0
    if not np.where(lens1, valid[1], valid[0]).all():
        raise ValueError(f"{int((~np.where(lens1, valid[1], valid[0])).sum())} panorama pixels lie outside the lens they are read from")
    delta: Float64[ndarray, " n"] = np.arcsin(np.clip(rays[:, 2], -1.0, 1.0))
    index: Int64[ndarray, " k"] = np.flatnonzero(np.abs(delta) < band)
    columns: int = width + 2 * STRIP_PAD
    rows: int = max(MIN_STRIP_ROWS, 4 * round(2.0 * band / step / 4))
    phi: Float64[ndarray, " sw"] = (np.arange(columns) - STRIP_PAD + 0.5) * step
    strip_delta: Float64[ndarray, " sh"] = band - (np.arange(rows) + 0.5) * 2.0 * band / rows
    phi_grid, delta_grid = np.meshgrid(phi, strip_delta)
    strip_rays: Float64[ndarray, "m 3"] = np.stack([np.cos(delta_grid) * np.cos(phi_grid), np.cos(delta_grid) * np.sin(phi_grid), np.sin(delta_grid)], axis=-1).reshape(-1, 3)
    strip_pixels, strip_valid = stacked(strip_rays)
    for lens in range(2):
        if not (strip_valid[lens].all() and valid[lens][index].all()):
            raise ValueError(f"the seam band of +-{band_deg} deg reaches outside lens {lens}")
    band_phi: Float64[ndarray, " k"] = np.mod(np.arctan2(rays[index, 1], rays[index, 0]), 2.0 * math.pi)
    band_strip_xy: Float64[ndarray, "k 2"] = np.stack([band_phi / step - 0.5 + STRIP_PAD, (band - delta[index]) / (2.0 * band) * rows - 0.5], axis=-1)
    return PanoramaTable(
        pano_xy=np.where(lens1[:, None], pixels[1], pixels[0]).reshape(width // 2, width, 2).astype(np.float32),
        band_index=index,
        band_weight=((band - delta[index]) / (2.0 * band)).astype(np.float32),
        band_lens_xy=pixels[:, index].astype(np.float32),
        band_strip_xy=band_strip_xy.astype(np.float32),
        strip_lens_xy=strip_pixels.reshape(2, rows, columns, 2).astype(np.float32),
    )


class PanoramaJoin:
    """The panoramas of lens pairs on one device, in frame order (the flow seam carries state from frame to frame).

    ``cuda``: ``grid_sample`` remaps, the band blended with (``flow=True``) or without the flow seam. ``cpu``: ``cv2.remap``
    and the plain blend. Call ``close`` to free the NVOF sessions.

    Raises:
        ValueError: If ``flow`` is asked for off CUDA, or the strip is wider than NVOF takes.
    """

    def __init__(self, table: PanoramaTable, lens_size: tuple[int, int], device: torch.device, *, flow: bool) -> None:
        """``lens_size``: lens image ``(width, height)``."""
        if flow and device.type != "cuda":
            raise ValueError("the flow seam runs on CUDA (NVOF); the cpu join blends the band without flow")
        lens_width, lens_height = lens_size
        self.device: torch.device = device
        self.size: tuple[int, int] = table.size
        self.flows: list[OpticalFlow] = []
        self.flow: Float32[Tensor, "4 gh gw"] | None = None
        if device.type == "cpu":
            self.table: PanoramaTable = table
            count: int = table.band_index.size
            padded: Float32[ndarray, "2 m 2"] = np.zeros((2, -(-count // BAND_COLUMNS) * BAND_COLUMNS, 2), dtype=np.float32)
            padded[:, :count] = table.band_lens_xy
            self.band_maps: Float32[ndarray, "2 r c 2"] = padded.reshape(2, -1, BAND_COLUMNS, 2)
            return
        # Grid coordinates (align_corners) of the stacked pair: x over [0, w - 1], y over [0, 2h].
        scale: Float32[Tensor, "2"] = torch.tensor([2.0 / (lens_width - 1), 2.0 / (2 * lens_height)], device=device)

        def grid(pixels: Float32[ndarray, "... 2"]) -> Float32[Tensor, "... 2"]:
            return torch.from_numpy(pixels).to(device) * scale - 1.0

        self.pano_grid: Float32[Tensor, "1 ph pw 2"] = grid(table.pano_xy)[None]
        self.band_index: Int64[Tensor, " n"] = torch.from_numpy(table.band_index).to(device)
        self.band_weight: Float32[Tensor, " n"] = torch.from_numpy(table.band_weight).to(device)
        self.band_lens_grid: Float32[Tensor, "1 2 n 2"] = grid(table.band_lens_xy)[None]
        if not flow:
            return
        strip_width, strip_height = table.strip_size
        if strip_width > MAX_STRIP_COLUMNS:
            raise ValueError(f"a {strip_width}-column seam strip is wider than NVOF's {MAX_STRIP_COLUMNS}; use a narrower panorama")
        strip_grid: Float32[Tensor, "2 sh sw 2"] = grid(table.strip_lens_xy)
        self.strip_grid: Float32[Tensor, "1 sh2 sw 2"] = strip_grid.reshape(1, 2 * strip_height, strip_width, 2)
        self.strip_lens_grid: Float32[Tensor, "2 2 sh sw"] = strip_grid.permute(0, 3, 1, 2).contiguous()
        self.band_strip_xy: Float32[Tensor, "n 2"] = torch.from_numpy(table.band_strip_xy).to(device)
        self.strip_scale: Float32[Tensor, "2"] = torch.tensor([2.0 / (strip_width - 1), 2.0 / (strip_height - 1)], device=device)
        coarse_size: Float32[Tensor, "2"] = torch.tensor([-(-strip_width // FLOW_GRID), -(-strip_height // FLOW_GRID)], dtype=torch.float32, device=device)
        # Flow block j covers strip pixels [4j, 4j + 4): its centre is pixel 4j + 1.5.
        coarse_xy: Float32[Tensor, "n 2"] = (self.band_strip_xy - (FLOW_GRID - 1) / 2.0) / FLOW_GRID
        self.band_flow_grid: Float32[Tensor, "1 1 n 2"] = (coarse_xy * 2.0 / (coarse_size - 1.0) - 1.0)[None, None]
        self.luma: Float32[Tensor, "1 3 1 1"] = torch.tensor(LUMA, device=device).view(1, 3, 1, 1)
        self.flows = [OpticalFlow(strip_width, strip_height, grid=FLOW_GRID) for _ in range(2)]

    def panoramas(self, lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[Tensor, "b 3 ph pw"]:
        """The panoramas of ``b`` consecutive lens pairs (RGB, channels first, on the device)."""
        b, c, h, w = lens0.shape
        if self.device.type == "cpu":
            return self._cpu_panoramas(lens0, lens1)
        stacked: Float32[Tensor, "b 3 r w"] = torch.empty((b, c, 2 * h + 1, w), dtype=torch.float32, device=self.device)
        stacked[:, :, :h] = lens0
        stacked[:, :, h] = 0.0
        stacked[:, :, h + 1 :] = lens1
        pano: Float32[Tensor, "b 3 ph pw"] = F.grid_sample(stacked, self.pano_grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        band_grid: Float32[Tensor, "b 2 n 2"] = self._flow_band_grid(stacked) if self.flows else self.band_lens_grid.expand(b, -1, -1, -1)
        reads: Float32[Tensor, "b 3 2 n"] = F.grid_sample(stacked, band_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        flat: Float32[Tensor, "b 3 m"] = pano.view(b, c, -1)
        flat[:, :, self.band_index] = torch.lerp(reads[:, :, 0], reads[:, :, 1], self.band_weight)
        return pano.round_().clamp_(0.0, 255.0).to(torch.uint8)

    def _flow_band_grid(self, stacked: Float32[Tensor, "b 3 r w"]) -> Float32[Tensor, "b 2 n 2"]:
        """Where each band pixel reads lens 0 and lens 1 under the flow seam, as stacked-pair grid coordinates."""
        b: int = stacked.shape[0]
        strip_height: int = self.strip_grid.shape[1] // 2
        strips: Float32[Tensor, "b 3 sh2 sw"] = F.grid_sample(stacked, self.strip_grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=True)
        gray: UInt8[Tensor, "b sh2 sw"] = (strips * self.luma).sum(1).round_().clamp_(0.0, 255.0).to(torch.uint8)
        smoothed: list[Float32[Tensor, "4 gh gw"]] = []
        for frame in gray:
            lens0, lens1 = frame[:strip_height], frame[strip_height:]
            flow: Float32[Tensor, "4 gh gw"] = torch.cat([self.flows[0].flow(lens0, lens1), self.flows[1].flow(lens1, lens0)], dim=-1).permute(2, 0, 1)
            self.flow = flow if self.flow is None else torch.lerp(self.flow, flow, FLOW_SMOOTHING)
            smoothed.append(self.flow)
        at_band: Float32[Tensor, "b 4 n"] = F.grid_sample(torch.stack(smoothed), self.band_flow_grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0]
        weight: Float32[Tensor, "n 1"] = self.band_weight[:, None]
        # Surround360's novel view at t: lens 0 moved t of the way along F01, lens 1 (1 - t) of the way along F10.
        read0: Float32[Tensor, "b n 2"] = self.band_strip_xy - weight * at_band[:, 0:2].transpose(1, 2)
        read1: Float32[Tensor, "b n 2"] = self.band_strip_xy - (1.0 - weight) * at_band[:, 2:4].transpose(1, 2)
        lens_grids: list[Float32[Tensor, "b n 2"]] = [
            F.grid_sample(self.strip_lens_grid[lens : lens + 1].expand(b, -1, -1, -1), (read * self.strip_scale - 1.0)[:, None], mode="bilinear", padding_mode="border", align_corners=True)[:, :, 0].transpose(1, 2)
            for lens, read in enumerate((read0, read1))
        ]
        return torch.stack(lens_grids, dim=1)

    def _cpu_panoramas(self, lens0: UInt8[Tensor, "b 3 h w"], lens1: UInt8[Tensor, "b 3 h w"]) -> UInt8[Tensor, "b 3 ph pw"]:
        """``cv2.remap`` of the stacked pair, the band blended without flow."""
        b, c, h, w = lens0.shape
        width, height = self.size
        count: int = self.table.band_index.size
        weight: Float32[ndarray, "n 1"] = self.table.band_weight[:, None]
        stacked: UInt8[ndarray, "r w 3"] = np.zeros((2 * h + 1, w, c), dtype=np.uint8)
        panoramas: UInt8[ndarray, "b ph pw 3"] = np.empty((b, height, width, c), dtype=np.uint8)
        for k in range(b):
            stacked[:h] = lens0[k].permute(1, 2, 0).numpy()
            stacked[h + 1 :] = lens1[k].permute(1, 2, 0).numpy()
            panoramas[k] = cv2.remap(stacked, self.table.pano_xy[..., 0], self.table.pano_xy[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            reads: list[Float32[ndarray, "n 3"]] = [
                cv2.remap(stacked, lens_map[..., 0], lens_map[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).reshape(-1, c)[:count].astype(np.float32)
                for lens_map in self.band_maps
            ]
            blended: Float32[ndarray, "n 3"] = reads[0] + weight * (reads[1] - reads[0])
            panoramas[k].reshape(-1, c)[self.table.band_index] = np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(panoramas.transpose(0, 3, 1, 2)))

    def close(self) -> None:
        """Free the NVOF sessions; the join blends without flow afterwards."""
        for flow in self.flows:
            flow.close()
        self.flows = []
