"""A MAMMA-format capture on disk (the public demo files) as the capture graph's video Source: same ports as ``CatalogVideoSource``.

Layout (the ``ma_cap`` contract of the original pipeline)::

    <root>/meta/<camera>.npz   cam_name, cam_int (3x3 K, stored pixels), cam_ext (4x4 cam_T_world; |t| > 200 means mm),
                               cam_img_w, cam_img_h
    <root>/meta/global.npz     seq_name, fps, frame_start, frame_end (the calibrated window)
    <root>/videos_light/<camera>.mp4   (or videos/, videos_crf24/), frame-synchronized
"""

import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from dagkit.node import Event
from dagkit.sources import SegmentRef
from dagkit.time import STATIC
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray
from simplecv.sources.catalog_video import CatalogVideoSource
from simplecv.types.rig import RigCameras
from torch import Tensor
from torchcodec.decoders import VideoDecoder

VIDEO_DIRS: tuple[str, ...] = ("videos_light", "videos", "videos_crf24")


def load_rig(meta: Path, cameras: tuple[str, ...]) -> RigCameras:
    """The cameras' pinholes from their ``meta/<camera>.npz`` (translations in metres)."""
    files = [np.load(meta / f"{camera}.npz") for camera in cameras]
    cam_T_world: Float64[ndarray, "v 4 4"] = np.stack([np.asarray(f["cam_ext"], dtype=np.float64) for f in files])
    cam_T_world[:, :3, 3] /= np.where(np.abs(cam_T_world[:, :3, 3]).max(axis=1, keepdims=True) > 200, 1000.0, 1.0)
    resolution: Int64[ndarray, "v 2"] = np.asarray([[int(f["cam_img_w"]), int(f["cam_img_h"])] for f in files], dtype=np.int64)
    return RigCameras(cameras, np.stack([np.asarray(f["cam_int"], dtype=np.float64) for f in files]), cam_T_world, resolution)


class FileFrames:
    """``SegmentFrames`` over files: one NVDEC decoder per (thread, view), frame times = index / fps."""

    def __init__(self, videos: list[Path], fps: float) -> None:
        self.videos, self.fps = videos, fps
        self._decoders: dict[tuple[int, int], VideoDecoder] = {}

    def _decoder(self, view: int) -> VideoDecoder:
        key: tuple[int, int] = (threading.get_ident(), view)
        if key not in self._decoders:
            self._decoders[key] = VideoDecoder(self.videos[view], device="cuda", seek_mode="exact", num_ffmpeg_threads=0)
        return self._decoders[key]

    def times_ns(self, view: int) -> Int64[ndarray, " n"]:
        return np.round(np.arange(len(self._decoder(view))) * 1e9 / self.fps).astype(np.int64)

    def frames_at(self, view: int, indices: list[int]) -> UInt8[Tensor, "k 3 h w"]:
        return self._decoder(view).get_frames_at(indices).data

    def close(self) -> None:
        self._decoders.clear()


class NpzSequenceSource(CatalogVideoSource):
    def __init__(self, root: Path, cameras: tuple[str, ...], *, resize_hw: tuple[int, int], keep_native: bool) -> None:
        capture = np.load(root / "meta" / "global.npz")
        fps: float = float(capture["fps"])
        window: tuple[int, int] = (round(int(capture["frame_start"]) * 1e9 / fps), round(int(capture["frame_end"]) * 1e9 / fps))
        super().__init__(SegmentRef("files", str(capture["seq_name"]), window_ns=window), cameras, decode="nvdec_rgb", resize_hw=resize_hw, keep_native=keep_native)
        directory: Path = next(root / d for d in VIDEO_DIRS if (root / d).is_dir())
        self.videos: list[Path] = [directory / f"{camera}.mp4" for camera in cameras]
        self.statics, self.frames = {}, FileFrames(self.videos, fps)  # type: ignore[assignment]  # the rig comes from the npz files
        self.rig: RigCameras = load_rig(root / "meta", cameras)

    def events(self) -> Iterator[Event]:
        yield Event(STATIC, "rig", self.rig)
        yield from super().events()
