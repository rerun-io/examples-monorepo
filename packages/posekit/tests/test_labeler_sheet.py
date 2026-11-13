"""Contact sheet: layout, stable numbering, rendering and the JSON the labeling agents read."""

from __future__ import annotations

import json

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy import ndarray
from serde.json import from_json, to_json

from posekit.labeler.schema import Box, CameraInfo, Proposal, TrackBox, TrackFrame, ViewLabel
from posekit.labeler.sheet import (
    SIDE_RGB,
    ContactSheet,
    PlaybackInfo,
    SheetInfo,
    contact_sheet,
    encode_jpeg,
    filmstrip,
    number_proposals,
    playback_info,
    sheet_info,
)

NAMES: list[str] = ["left_front", "right_front", "left_eye", "right_eye", "left", "right"]
CAMS: list[CameraInfo] = [CameraInfo(index, name, 1920, 1080) for index, name in enumerate(NAMES)]
PROPOSALS: list[Proposal] = [
    Proposal(4, "right", Box(900.0, 300.0, 1200.0, 600.0), 0.8, "egohand"),
    Proposal(0, "left", Box(300.0, 300.0, 600.0, 600.0), 0.95, "egohand"),
    Proposal(0, "left", Box(310.0, 310.0, 610.0, 610.0), 1.0, "hands_layer"),
    Proposal(0, "right", Box(1200.0, 300.0, 1500.0, 600.0), 0.6, "egohand"),
    Proposal(0, "left", Box(1500.0, 600.0, 1700.0, 800.0), 0.4, "egohand"),
]


def decode(jpeg: bytes) -> UInt8[ndarray, "h w 3"]:
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def test_numbering_is_stable_and_ordered() -> None:
    numbered = number_proposals(PROPOSALS)
    assert [p.k for p in numbered] == [1, 2, 3, 4, 5]
    assert [(p.cam, p.model, p.side, p.score) for p in numbered] == [
        (0, "egohand", "left", 0.95),
        (0, "egohand", "left", 0.4),
        (0, "egohand", "right", 0.6),
        (0, "hands_layer", "left", 1.0),
        (4, "egohand", "right", 0.8),
    ]
    assert number_proposals(list(reversed(PROPOSALS))) == numbered


def test_layout_and_coordinate_mapping() -> None:
    info: SheetInfo = sheet_info("seg", 123, CAMS, PROPOSALS)
    assert info.width == 1920
    assert 720 < info.height <= 1080
    assert [(cell.cam, cell.x, cell.y, cell.w, cell.h) for cell in info.cells] == [
        (0, 0, 0, 640, 360),
        (1, 640, 0, 640, 360),
        (2, 1280, 0, 640, 360),
        (3, 0, 360, 640, 360),
        (4, 640, 360, 640, 360),
        (5, 1280, 360, 640, 360),
    ]
    cell = info.cells[4]
    # native_px = (sheet_px - (x, y)) / scale
    assert ((640 + 300) - cell.x) / cell.scale == 900.0
    assert ((360 + 100) - cell.y) / cell.scale == 300.0


def test_sheet_json_matches_the_agent_contract() -> None:
    info: SheetInfo = sheet_info("seg", 123, CAMS, PROPOSALS)
    payload = json.loads(to_json(info))  # the raw JSON is the contract under test
    assert set(payload) == {"segment_id", "t_ns", "width", "height", "cells", "proposals", "track"}
    assert payload["cells"][0] == {"cam": 0, "name": "left_front", "x": 0, "y": 0, "w": 640, "h": 360, "scale": 1 / 3}
    assert payload["proposals"][0] == {
        "k": 1,
        "cam": 0,
        "side": "left",
        "score": 0.95,
        "model": "egohand",
        "box": {"x0": 300.0, "y0": 300.0, "x1": 600.0, "y1": 600.0},
    }
    assert from_json(SheetInfo, to_json(info)) == info


def test_contact_sheet_renders_boxes() -> None:
    images: list[UInt8[ndarray, "h w 3"]] = [np.full((1080, 1920, 3), 128, dtype=np.uint8) for _ in CAMS]
    labels: list[ViewLabel] = [
        ViewLabel(2, "left", "corrected", Box(300.0, 300.0, 900.0, 900.0), "person", "pablo"),
        ViewLabel(2, "right", "no_hand", None, "person", "pablo"),
    ]
    sheet: ContactSheet = contact_sheet("seg", 123, CAMS, images, PROPOSALS, labels)
    rgb: UInt8[ndarray, "h w 3"] = decode(sheet.jpeg)
    assert rgb.shape == (sheet.info.height, sheet.info.width, 3)
    # The right edge of cam 4's right-hand box (native x=1200 -> sheet x=640+400), mid-height, is orange.
    pixel: UInt8[ndarray, "3"] = rgb[360 + 150, 640 + 400]
    assert np.abs(pixel.astype(np.int64) - np.array(SIDE_RGB["right"])).max() < 60
    # The stored label of cam 2 is drawn green on its left edge (native x=300 -> sheet x=1280+100).
    green: UInt8[ndarray, "3"] = rgb[102, 1280 + 100]
    assert green[1] > 150 and green[0] < 100


def test_contact_sheet_draws_the_machine_track() -> None:
    images: list[UInt8[ndarray, "h w 3"]] = [np.full((1080, 1920, 3), 128, dtype=np.uint8) for _ in CAMS]
    track: list[TrackBox] = [TrackBox(5, "left", Box(300.0, 300.0, 900.0, 900.0), "chained", 0.7), TrackBox(5, "right", None, "none", None)]
    sheet: ContactSheet = contact_sheet("seg", 123, CAMS, images, [], [], track)
    assert sheet.info.track == track
    rgb: UInt8[ndarray, "h w 3"] = decode(sheet.jpeg)
    # cam 5's machine box, left edge (native x=300 -> sheet x=1280+100), dashed blue: a dash starts at the top corner.
    pixel: UInt8[ndarray, "3"] = rgb[360 + 102, 1280 + 100]
    assert np.abs(pixel.astype(np.int64) - np.array(SIDE_RGB["left"])).max() < 60


def test_playback_layout_and_json() -> None:
    info: PlaybackInfo = playback_info("seg", CAMS, [65_000_000_000 + i * 200_000_000 for i in range(10)])
    payload = json.loads(to_json(info))
    assert set(payload) == {"segment_id", "width", "height", "rows", "cells"}
    assert payload["rows"][1] == {"r": 1, "t_ns": 65_200_000_000, "y": info.rows[0].y + 180}
    assert payload["cells"][0] == {"cam": 0, "name": "left_front", "x": info.cells[0].x, "w": 320, "h": 180, "scale": 1 / 6}
    assert [cell.x - info.cells[0].x for cell in info.cells] == [0, 320, 640, 960, 1280, 1600]
    assert info.width == info.cells[-1].x + 320 and info.height == info.rows[-1].y + 180
    assert from_json(PlaybackInfo, to_json(info)) == info


def test_filmstrip_draws_machine_boxes_by_side_and_origin() -> None:
    times: list[int] = [0, 200_000_000]
    info: PlaybackInfo = playback_info("seg", CAMS, times)
    cells: list[list[UInt8[ndarray, "h w 3"]]] = [[np.full((180, 320, 3), 128, dtype=np.uint8) for _ in CAMS] for _ in times]
    big: Box = Box(600.0, 300.0, 1200.0, 900.0)
    track: list[TrackFrame] = [
        TrackFrame(0, [TrackBox(2, "left", big, "reviewed", None), TrackBox(2, "right", None, "none", None)]),
        TrackFrame(200_000_000, [TrackBox(4, "right", big, "held", None)]),
    ]
    rgb: UInt8[ndarray, "h w 3"] = decode(filmstrip(info, cells, track))
    assert rgb.shape == (info.height, info.width, 3)
    # Row 0, cam 2: the reviewed left box is thick blue (native x=600 -> cell x=100), at mid-height of the box.
    x: int = info.cells[2].x + 100
    y: int = info.rows[0].y + 100
    for dx in (-1, 0, 1):
        assert np.abs(rgb[y, x + dx].astype(np.int64) - np.array(SIDE_RGB["left"])).max() < 70
    # Row 1, cam 4: the held right box is dashed orange: the top edge has gaps.
    top: int = info.rows[1].y + 50
    edge: list[bool] = [
        bool(np.abs(rgb[top, info.cells[4].x + 100 + k].astype(np.int64) - np.array(SIDE_RGB["right"])).max() < 70) for k in range(4, 96)
    ]
    assert any(edge) and not all(edge)
    # Nothing is drawn where the machine says no hand.
    assert np.abs(rgb[info.rows[0].y + 100, info.cells[3].x + 100].astype(np.int64) - 128).max() < 20


def test_footer_fits_many_proposals() -> None:
    many: list[Proposal] = [Proposal(index % 6, "left", Box(10.0 * index, 0.0, 10.0 * index + 50.0, 50.0), 0.9, "egohand") for index in range(150)]
    info: SheetInfo = sheet_info("seg", 0, CAMS, many)
    assert info.height == 1080
    sheet: ContactSheet = contact_sheet("seg", 0, CAMS, [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 6, many, [])
    assert decode(sheet.jpeg).shape == (1080, 1920, 3)


def test_encode_jpeg_downscales() -> None:
    image: UInt8[ndarray, "h w 3"] = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert decode(encode_jpeg(image, 960)).shape == (540, 960, 3)
    assert decode(encode_jpeg(image)).shape == (1080, 1920, 3)
