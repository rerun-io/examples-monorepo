"""Six-view contact sheet of one time, for labeling agents that read images.

The sheet is a 3x2 grid of 640x360 cells (cam_00..cam_02 on top, cam_03..cam_05 below), so 1920 px wide,
plus a footer that lists every numbered proposal; it stays within 1920x1080. Each proposal is drawn with
its number ``#k`` (left hand blue, right hand orange; thick = EgoHand, thin = hands layer); stored labels
are drawn dashed green. EgoHand tags sit above their box, hands-layer tags below. ``SheetInfo`` carries the same numbers and each cell's offset and scale:
``native_px = (sheet_px - (cell.x, cell.y)) / cell.scale``. Numbers start at 1 and depend only on the
proposals, so the same (segment, time) always gets the same numbers. The machine track of that time (``TrackBox``) is
drawn dashed in the side colour with a ``T L``/``T R`` tag inside its top-left corner (``T L trk`` when a visual
tracker carried it).

The playback filmstrip (``filmstrip``) shows consecutive grid times as rows and the cameras as columns, each cell
320x180, with the machine boxes drawn as the reviewer judges them: left blue, right orange; ``reviewed`` thick with a
``*``; ``held`` dashed; ``tracked`` (carried by the visual tracker) drawn like ``chained`` with a ``trk`` tag. A header row names the cameras; a label column on the left reads ``r<i> <mm:ss.ss>``, the
time since the segment's first grid time (Cap B ``video_time`` values are days long, so absolute times do not read). ``PlaybackInfo`` gives each cell's ``x``, each row's ``y`` and the per-camera ``scale``:
``native_px = (strip_px - (cell.x, row.y)) / cell.scale``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from serde import serde

from posekit.labeler.schema import Box, CameraInfo, HandSide, Proposal, TrackBox, TrackFrame, ViewLabel

CELL_W: int = 640
"""Cell width in sheet px."""
CELL_H: int = 360
"""Cell height in sheet px."""
COLS: int = 3
"""Cells per row."""
MAX_H: int = 1080
"""Sheet height limit, px."""
FOOTER_LINE: int = 20
"""Footer line height, px."""
FOOTER_COLS: int = 4
"""Footer text columns."""
FOOTER_HEADER: int = 28
"""Footer header line height, px."""
SIDE_RGB: dict[HandSide, tuple[int, int, int]] = {"left": (40, 110, 255), "right": (255, 140, 0)}
"""Box colour per hand side."""
LABEL_RGB: tuple[int, int, int] = (0, 220, 0)
"""Colour of stored labels."""
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
STRIP_CELL_W: int = 320
"""Filmstrip cell width, px."""
STRIP_CELL_H: int = 180
"""Filmstrip cell height, px."""
STRIP_HEADER_H: int = 24
"""Filmstrip header row (camera names) height, px."""
STRIP_LABEL_W: int = 118
"""Filmstrip row-label column width, px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SheetCell:
    """Where one camera's image sits on the sheet."""

    cam: int
    """Camera index."""
    name: str
    """Rig name of the camera."""
    x: int
    """Left edge of the image on the sheet, px."""
    y: int
    """Top edge of the image on the sheet, px."""
    w: int
    """Image width on the sheet, px."""
    h: int
    """Image height on the sheet, px."""
    scale: float
    """Sheet px per native px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class NumberedProposal:
    """A proposal with its number on the sheet."""

    k: int
    """Number drawn on the sheet, from 1."""
    cam: int
    """Camera index."""
    side: HandSide
    """Hand side the model assigned."""
    score: float
    """Model confidence."""
    model: Literal["egohand", "hands_layer"]
    """Which proposer made it."""
    box: Box
    """Box in native image px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class SheetInfo:
    """Layout and numbering of one contact sheet (``GET /api/sheet/<sid>/<t_ns>.json``)."""

    segment_id: str
    """Catalog segment id."""
    t_ns: int
    """``video_time``, ns."""
    width: int
    """Sheet width, px."""
    height: int
    """Sheet height, px."""
    cells: list[SheetCell]
    """One cell per camera, rig order."""
    proposals: list[NumberedProposal]
    """Every proposal, in number order."""
    track: list[TrackBox] = field(default_factory=list)
    """The machine track at this time (drawn dashed, tagged ``T L``/``T R``)."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PlaybackRow:
    """One row (grid time) of the playback filmstrip."""

    r: int
    """Row index from 0, drawn as ``r<i>``."""
    t_ns: int
    """``video_time`` of the row, ns."""
    y: int
    """Top edge of the row on the filmstrip, px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PlaybackCell:
    """One camera column of the playback filmstrip (same geometry in every row)."""

    cam: int
    """Camera index."""
    name: str
    """Rig name of the camera."""
    x: int
    """Left edge of the column on the filmstrip, px."""
    w: int
    """Image width in the cell, px."""
    h: int
    """Image height in the cell, px."""
    scale: float
    """Filmstrip px per native px."""


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class PlaybackInfo:
    """Layout of one playback filmstrip (``GET /api/playback/<sid>/<t0_ns>.json``)."""

    segment_id: str
    """Catalog segment id."""
    width: int
    """Filmstrip width, px."""
    height: int
    """Filmstrip height, px."""
    rows: list[PlaybackRow]
    """One row per grid time, in time order."""
    cells: list[PlaybackCell]
    """One column per camera, rig order."""


@dataclass(frozen=True, slots=True)
class ContactSheet:
    """A rendered sheet and its layout."""

    jpeg: bytes
    """JPEG image."""
    info: SheetInfo
    """Layout and numbering."""


def number_proposals(proposals: list[Proposal]) -> list[NumberedProposal]:
    """Number proposals from 1 in a fixed order: camera, EgoHand before hands layer, side, score descending, box."""
    order: list[Proposal] = sorted(proposals, key=lambda p: (p.cam, p.model != "egohand", p.side, -p.score, p.box.x0, p.box.y0, p.box.x1, p.box.y1))
    return [NumberedProposal(k, p.cam, p.side, p.score, p.model, p.box) for k, p in enumerate(order, start=1)]


def sheet_info(segment_id: str, t_ns: int, cams: list[CameraInfo], proposals: list[Proposal], track: list[TrackBox] | None = None) -> SheetInfo:
    """Layout and numbering of the sheet, without decoding any frame (``track``: the machine boxes at ``t_ns``)."""
    cells: list[SheetCell] = []
    for index, camera in enumerate(cams):
        scale: float = min(CELL_W / camera.width, CELL_H / camera.height)
        cells.append(
            SheetCell(
                camera.cam,
                camera.name,
                (index % COLS) * CELL_W,
                (index // COLS) * CELL_H,
                round(camera.width * scale),
                round(camera.height * scale),
                scale,
            )
        )
    numbered: list[NumberedProposal] = number_proposals(proposals)
    grid_h: int = math.ceil(len(cams) / COLS) * CELL_H
    lines: int = math.ceil(len(numbered) / FOOTER_COLS)
    footer_h: int = min(FOOTER_HEADER + max(lines, 1) * FOOTER_LINE + 6, MAX_H - grid_h)
    return SheetInfo(segment_id, t_ns, COLS * CELL_W, grid_h + footer_h, cells, numbered, list(track or []))


def _to_sheet(cell: SheetCell, box: Box) -> tuple[tuple[int, int], tuple[int, int]]:
    """A native box as sheet-px corners in a cell."""
    return (
        (cell.x + round(box.x0 * cell.scale), cell.y + round(box.y0 * cell.scale)),
        (cell.x + round(box.x1 * cell.scale), cell.y + round(box.y1 * cell.scale)),
    )


def _text(canvas: UInt8[ndarray, "h w 3"], text: str, origin: tuple[int, int], rgb: tuple[int, int, int], scale: float = 0.5) -> None:
    """Text with a dark backing box, ``origin`` = bottom-left of the text."""
    measured: tuple[Sequence[int], int] = cv2.getTextSize(text, FONT, scale, 1)
    width: int = int(measured[0][0])
    height: int = int(measured[0][1])
    x: int = origin[0]
    y: int = origin[1]
    cv2.rectangle(canvas, (x - 1, y - height - 3), (x + width + 1, y + 3), (0, 0, 0), thickness=-1)
    cv2.putText(canvas, text, (x, y), FONT, scale, rgb, 1, cv2.LINE_AA)


def _dashed_rect(canvas: UInt8[ndarray, "h w 3"], top_left: tuple[int, int], bottom_right: tuple[int, int], rgb: tuple[int, int, int]) -> None:
    """A dashed rectangle (8 px dashes)."""
    x0, y0 = top_left
    x1, y1 = bottom_right
    for start in range(x0, x1, 12):
        stop: int = min(start + 8, x1)
        cv2.line(canvas, (start, y0), (stop, y0), rgb, 2)
        cv2.line(canvas, (start, y1), (stop, y1), rgb, 2)
    for start in range(y0, y1, 12):
        stop = min(start + 8, y1)
        cv2.line(canvas, (x0, start), (x0, stop), rgb, 2)
        cv2.line(canvas, (x1, start), (x1, stop), rgb, 2)


def contact_sheet(
    segment_id: str,
    t_ns: int,
    cams: list[CameraInfo],
    images: list[UInt8[ndarray, "h w 3"]],
    proposals: list[Proposal],
    labels: list[ViewLabel],
    track: list[TrackBox] | None = None,
) -> ContactSheet:
    """Render the six-view sheet of one time.

    Args:
        segment_id: Catalog segment id.
        t_ns: ``video_time``, ns.
        cams: Cameras in rig order.
        images: RGB frame of each camera, same order as ``cams`` (any size; drawn at the cell size).
        proposals: Every proposal at this time, native px.
        labels: Stored labels at this time, native px.
        track: The machine track at this time, native px.
    """
    if len(images) != len(cams):
        raise ValueError(f"{len(images)} images for {len(cams)} cameras")
    info: SheetInfo = sheet_info(segment_id, t_ns, cams, proposals, track)
    canvas: UInt8[ndarray, "h w 3"] = np.full((info.height, info.width, 3), 24, dtype=np.uint8)
    cell_by_cam: dict[int, SheetCell] = {cell.cam: cell for cell in info.cells}
    for cell, image in zip(info.cells, images, strict=True):
        shown: UInt8[ndarray, "h w 3"] = cv2.resize(image, (cell.w, cell.h), interpolation=cv2.INTER_AREA)
        canvas[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] = shown
    for label in labels:
        cell = cell_by_cam.get(label.cam)
        if cell is not None and label.box is not None:
            top_left, bottom_right = _to_sheet(cell, label.box)
            _dashed_rect(canvas, top_left, bottom_right, LABEL_RGB)
    for machine in info.track:
        cell = cell_by_cam.get(machine.cam)
        if cell is not None and machine.box is not None:
            top_left, bottom_right = _to_sheet(cell, machine.box)
            _dashed_rect(canvas, top_left, bottom_right, SIDE_RGB[machine.side])
            machine_tag: str = f"T {machine.side[0].upper()}{' trk' if machine.origin == 'tracked' else ''}"
            _text(canvas, machine_tag, (top_left[0] + 4, top_left[1] + 16), SIDE_RGB[machine.side], 0.45)
    for proposal in info.proposals:
        cell = cell_by_cam.get(proposal.cam)
        if cell is None:
            continue
        top_left, bottom_right = _to_sheet(cell, proposal.box)
        rgb: tuple[int, int, int] = SIDE_RGB[proposal.side]
        cv2.rectangle(canvas, top_left, bottom_right, rgb, 2 if proposal.model == "egohand" else 1)
        tag: str = f"#{proposal.k} {proposal.side[0].upper()} {proposal.score:.2f}{'' if proposal.model == 'egohand' else ' H'}"
        # EgoHand tags sit above the box, hands-layer tags below it, so the two models' tags rarely collide.
        tag_y: int = max(top_left[1] - 4, cell.y + 14) if proposal.model == "egohand" else min(bottom_right[1] + 14, cell.y + cell.h - 4)
        _text(canvas, tag, (top_left[0], tag_y), rgb, 0.45)
    labels_by_cam: dict[int, list[ViewLabel]] = {}
    for label in labels:
        labels_by_cam.setdefault(label.cam, []).append(label)
    for cell in info.cells:
        cv2.rectangle(canvas, (cell.x, cell.y), (cell.x + CELL_W - 1, cell.y + CELL_H - 1), (90, 90, 90), 1)
        _text(canvas, f"cam_{cell.cam:02d} {cell.name}", (cell.x + 6, cell.y + 20), (255, 255, 255), 0.6)
        stored: list[ViewLabel] = sorted(labels_by_cam.get(cell.cam, []), key=lambda label: label.side)
        if stored:
            status: str = "  ".join(f"{label.side[0].upper()}:{label.status}" for label in stored)
            _text(canvas, status, (cell.x + 6, cell.y + cell.h - 8), LABEL_RGB, 0.5)
    grid_h: int = max(cell.y for cell in info.cells) + CELL_H if info.cells else 0
    _text(
        canvas,
        "blue=left orange=right  thick=EgoHand thin(H)=hands layer  dashed T L/T R=machine track  green dashed=stored label",
        (8, grid_h + 20),
        (220, 220, 220),
        0.5,
    )
    lines: int = max(1, (info.height - grid_h - FOOTER_HEADER - 6) // FOOTER_LINE)
    column_w: int = info.width // FOOTER_COLS
    for index, proposal in enumerate(info.proposals):
        column: int = index // lines
        if column >= FOOTER_COLS:
            _text(canvas, f"... {len(info.proposals) - index} more (see .json)", (info.width - 260, info.height - 6), (255, 255, 255), 0.45)
            break
        row: int = index % lines
        entry: str = f"#{proposal.k} cam_{proposal.cam:02d} {proposal.side} {proposal.score:.2f} {proposal.model}"
        _text(canvas, entry, (column * column_w + 8, grid_h + FOOTER_HEADER + (row + 1) * FOOTER_LINE - 4), SIDE_RGB[proposal.side], 0.45)
    return ContactSheet(encode_jpeg(canvas), info)


def playback_info(segment_id: str, cams: list[CameraInfo], times_ns: list[int]) -> PlaybackInfo:
    """Layout of a filmstrip of ``times_ns`` (rows) x ``cams`` (columns), without decoding any frame."""
    cells: list[PlaybackCell] = []
    for index, camera in enumerate(cams):
        scale: float = min(STRIP_CELL_W / camera.width, STRIP_CELL_H / camera.height)
        cells.append(
            PlaybackCell(
                camera.cam, camera.name, STRIP_LABEL_W + index * STRIP_CELL_W, round(camera.width * scale), round(camera.height * scale), scale
            )
        )
    rows: list[PlaybackRow] = [PlaybackRow(r, t_ns, STRIP_HEADER_H + r * STRIP_CELL_H) for r, t_ns in enumerate(times_ns)]
    return PlaybackInfo(segment_id, STRIP_LABEL_W + len(cams) * STRIP_CELL_W, STRIP_HEADER_H + len(times_ns) * STRIP_CELL_H, rows, cells)


def strip_cell(image: UInt8[ndarray, "h w 3"], cell: PlaybackCell) -> UInt8[ndarray, "ch cw 3"]:
    """A native frame downscaled to its filmstrip cell size."""
    return cv2.resize(image, (cell.w, cell.h), interpolation=cv2.INTER_AREA)


def filmstrip(info: PlaybackInfo, images: list[list[UInt8[ndarray, "h w 3"]]], track: list[TrackFrame], start_ns: int = 0) -> bytes:
    """Render the playback filmstrip as JPEG.

    Args:
        info: Layout (``playback_info``).
        images: ``images[r][c]``: the frame of row ``r`` and camera column ``c``, at cell size (other sizes are resized).
        track: The machine track of each row, same order as ``info.rows``.
        start_ns: Segment start (its first grid time); row labels show the time since it.
    """
    if len(images) != len(info.rows) or len(track) != len(info.rows):
        raise ValueError(f"{len(images)} image rows and {len(track)} track frames for {len(info.rows)} rows")
    canvas: UInt8[ndarray, "h w 3"] = np.full((info.height, info.width, 3), 24, dtype=np.uint8)
    for cell in info.cells:
        _text(canvas, f"cam_{cell.cam:02d} {cell.name}", (cell.x + 6, STRIP_HEADER_H - 7), (255, 255, 255), 0.5)
    cell_by_cam: dict[int, PlaybackCell] = {cell.cam: cell for cell in info.cells}
    for row, row_images, frame in zip(info.rows, images, track, strict=True):
        seconds: float = (row.t_ns - start_ns) / 1e9
        _text(canvas, f"r{row.r} {int(seconds // 60):02d}:{seconds % 60:05.2f}", (6, row.y + STRIP_CELL_H // 2 + 5), (255, 255, 255), 0.5)
        for cell, image in zip(info.cells, row_images, strict=True):
            shown: UInt8[ndarray, "ch cw 3"] = image if image.shape[:2] == (cell.h, cell.w) else strip_cell(image, cell)
            canvas[row.y : row.y + cell.h, cell.x : cell.x + cell.w] = shown
            cv2.rectangle(canvas, (cell.x, row.y), (cell.x + STRIP_CELL_W - 1, row.y + STRIP_CELL_H - 1), (70, 70, 70), 1)
        for machine in frame.boxes:
            placed: PlaybackCell | None = cell_by_cam.get(machine.cam)
            if placed is None or machine.box is None:
                continue
            top_left: tuple[int, int] = (placed.x + round(machine.box.x0 * placed.scale), row.y + round(machine.box.y0 * placed.scale))
            bottom_right: tuple[int, int] = (placed.x + round(machine.box.x1 * placed.scale), row.y + round(machine.box.y1 * placed.scale))
            rgb: tuple[int, int, int] = SIDE_RGB[machine.side]
            if machine.origin == "held":
                _dashed_rect(canvas, top_left, bottom_right, rgb)
            else:
                cv2.rectangle(canvas, top_left, bottom_right, rgb, 3 if machine.origin == "reviewed" else 2)
            tag: str = f"{machine.side[0].upper()}{'*' if machine.origin == 'reviewed' else ''}{' trk' if machine.origin == 'tracked' else ''}"
            _text(canvas, tag, (top_left[0] + 3, max(top_left[1] - 3, row.y + 12)), rgb, 0.4)
    return encode_jpeg(canvas)


def encode_jpeg(image: UInt8[ndarray, "h w 3"], max_side: int | None = None, quality: int = 85) -> bytes:
    """JPEG bytes of an RGB image, downscaled so its longer side is at most ``max_side``."""
    if max_side is not None and max(image.shape[:2]) > max_side:
        scale: float = max_side / max(image.shape[:2])
        size: tuple[int, int] = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
