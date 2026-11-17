"""The derived panorama layer: the stitch binding, the pure helpers, and the layer written from the real clip."""

import os
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.chunk as rrc
from jaxtyping import Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.apis.insv_panorama import PANO_ENTITY, Config, PanoLayer, layer_duration_s, layer_path, write_layer
from slam_rs.catalog_feed import LocalSegment, open_segment

V6: str = (
    "2_2.314940_7096.760_7095.390_3834.830_3864.120_0.119_0.407_90.629_0.000000_0.000000_0.000000_0.93861496_-1.68402135_"
    "2.19166017_3.85762072_0.00000000_-0.00495962_0.00156076_-0.00330722_0.00397486_0.00827898_-0.00125315_0.02416405_"
    "-0.00143547_15360_7680_131_2.314940_7149.220_7149.700_11543.300_3858.430_-0.212_0.339_88.279_0.000132_-0.000043_"
    "-0.033291_0.95675713_-1.80496192_2.88544273_1.76948690_0.00000000_0.00131924_0.00023973_0.00279902_-0.00570679_"
    "0.00069985_0.00064811_0.00799088_0.02677096_15360_7680_131_394240"
)
"""The X4 Air clip's V6 factory string, as the Rust lane's tests hold it."""
CROP: tuple[int, int, int, int, int, int] = (7680, 7680, 7424, 7424, 0, 0)

CLIP_RRD: Path | None = Path(clip) if (clip := os.environ.get("SLAM_RS_INSTA360_RRD")) else None
"""The Insta360 clip's recording VID_20261116_125200_00_003.rrd (``slam-rs-insv`` output), from ``SLAM_RS_INSTA360_RRD``; the integration test skips without it."""
MIN_PSNR_DB: float = 38.0
"""Floor for the first decoded layer frame against the stitch at the default cq 34 (measured 41.7 dB on 2026-11-17)."""


def test_the_layer_sits_beside_the_recording() -> None:
    assert layer_path(Path("/data/VID_1.rrd")) == Path("/data/VID_1.pano.rrd")


def test_the_duration_counts_one_period_past_the_last_frame() -> None:
    times: Int64[ndarray, " n"] = np.array([0, 33_366_700, 66_733_400], dtype=np.int64)
    assert layer_duration_s(times) == pytest.approx(3 * 0.0333667)
    assert layer_duration_s(np.array([5], dtype=np.int64)) == pytest.approx(1 / 30)
    assert PanoLayer(Path("x"), 3, 2.0, 1_000_000, 1.0).bitrate_mbps == pytest.approx(4.0)


def test_each_lens_fills_its_own_hemisphere() -> None:
    stitch = _core.PanoramaStitch(V6, CROP, (1920, 1920))
    bright: UInt8[ndarray, "h w"] = np.full((1920, 1920), 200, dtype=np.uint8)
    dark: UInt8[ndarray, "h w"] = np.full((1920, 1920), 40, dtype=np.uint8)
    pano: UInt8[ndarray, "960 1920"] = stitch.panorama(bright, dark)
    assert pano.shape == (960, 1920)
    assert pano.dtype == np.uint8
    # Lens 0's axis is the left half's centre, lens 1's the right half's.
    assert pano[480, 480] == 200
    assert pano[480, 1440] == 40
    swapped: UInt8[ndarray, "960 1920"] = stitch.panorama(dark, bright)
    assert swapped[480, 480] == 40


def test_a_lens_of_the_wrong_size_or_a_bad_string_is_refused() -> None:
    stitch = _core.PanoramaStitch(V6, CROP, (1920, 1920))
    small: UInt8[ndarray, "h w"] = np.zeros((1080, 1920), dtype=np.uint8)
    with pytest.raises(ValueError, match="lens image"):
        stitch.panorama(small, small)
    with pytest.raises(ValueError):
        _core.PanoramaStitch("2_not_a_lens_string", CROP, (1920, 1920))


def test_a_missing_recording_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such recording"):
        write_layer(Config(rrd=tmp_path / "absent.rrd"))


def _column(chunks: list[rrc.Chunk], name: str) -> list[pa.Array]:
    return [chunk.to_record_batch().column(name) for chunk in chunks if name in chunk.to_record_batch().schema.names]


def _psnr_db(a: UInt8[ndarray, "h w"], b: UInt8[ndarray, "h w"]) -> float:
    mse: float = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0**2 / mse)


@pytest.mark.integration
@pytest.mark.usefixtures("av1_nvenc")
def test_the_first_90_frames_of_the_clip_become_an_av1_layer_on_the_frame_times(tmp_path: Path) -> None:
    if CLIP_RRD is None or not CLIP_RRD.is_file():
        pytest.skip(f"the Insta360 clip recording VID_20261116_125200_00_003.rrd is absent: set SLAM_RS_INSTA360_RRD (now {CLIP_RRD})")
    layer: PanoLayer = write_layer(Config(rrd=CLIP_RRD, output=tmp_path / "clip.pano.rrd", max_frames=90))
    assert layer.frames == 90

    reader = rrc.RrdReader(layer.path)
    assert reader.recordings() == rrc.RrdReader(CLIP_RRD).recordings()  # same ids: the viewer overlays the two
    pano: list[rrc.Chunk] = reader.stream().filter(content=PANO_ENTITY).to_chunks()
    codecs: list[int] = [value for column in _column(pano, "VideoStream:codec") for value in column.flatten().to_pylist()]
    assert codecs == [rr.VideoCodec.AV1.value]
    # One blob per row: a row is a one-element list of the sample's bytes.
    samples: list[bytes] = [bytes(row[0]) for column in _column(pano, "VideoStream:sample") for row in column.to_pylist()]
    assert len(samples) == 90
    sample_chunks: list[rrc.Chunk] = [c for c in pano if "VideoStream:sample" in c.to_record_batch().schema.names]
    unsorted_ns: Int64[ndarray, " n"] = np.concatenate([np.asarray(c.to_record_batch().column("video_time").cast(pa.int64())) for c in sample_chunks])
    # Chunks arrive in storage order, not time order.
    order: Int64[ndarray, " n"] = np.argsort(unsorted_ns)
    times_ns: Int64[ndarray, " n"] = unsorted_ns[order]
    # The converter stamps the exposure series and the lens videos with the same frame times.
    exposure: list[rrc.Chunk] = rrc.RrdReader(CLIP_RRD).stream().filter(content="/world/rig_00/exposure", has_timeline="video_time").to_chunks()
    frame_ns: Int64[ndarray, " n"] = np.sort(
        np.concatenate([np.asarray(c.to_record_batch().column("video_time").cast(pa.int64())) for c in exposure])
    )
    np.testing.assert_array_equal(times_ns, frame_ns[:90])

    decoder: av.CodecContext = av.CodecContext.create("libdav1d", "r")
    assert isinstance(decoder, av.video.codeccontext.VideoCodecContext)
    frames = decoder.decode(av.Packet(samples[int(order[0])])) + decoder.decode(None)
    decoded: UInt8[ndarray, "h w"] = frames[0].to_ndarray(format="gray")
    with open_segment(LocalSegment(CLIP_RRD), include_ground_truth=False) as feed:
        assert feed.insta360 is not None
        stitch = _core.PanoramaStitch(feed.insta360.lens_string, feed.insta360.window_crop, (feed.cameras[0].width, feed.cameras[0].height))
        first = next(iter(feed.framesets(int(feed.frame_t_ns[0]))))
        expected: UInt8[ndarray, "960 1920"] = stitch.panorama(first.images[0], first.images[1])
    assert decoded.shape == expected.shape
    psnr: float = _psnr_db(decoded, expected)
    print(f"first frame PSNR {psnr:.2f} dB, AV1 {layer.bitrate_mbps:.2f} Mbit/s")
    assert psnr > MIN_PSNR_DB
