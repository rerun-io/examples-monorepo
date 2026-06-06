# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from cv2 import (
    CAP_PROP_FOURCC,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_COUNT,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CAP_PROP_POS_FRAMES,
)
from einops import rearrange
from jaxtyping import UInt8

from simplecv.image_types import BGRList, ImageBGR


def rgb_chw_tensor_to_bgr_hwc(rgb_chw: UInt8[torch.Tensor, "3 h w"]) -> ImageBGR:
    """Convert an RGB ``CHW`` uint8 tensor to a BGR ``HWC`` numpy image."""
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] != 3:
        raise ValueError(f"Expected RGB tensor with shape (3, *, *), got {tuple(rgb_chw.shape)}.")

    bgr_chw: UInt8[torch.Tensor, "3 h w"] = torch.flip(rgb_chw.detach(), dims=(0,))
    bgr_hwc_tensor: UInt8[torch.Tensor, "h w 3"] = rearrange(bgr_chw, "c h w -> h w c")
    bgr_hwc: ImageBGR = np.ascontiguousarray(bgr_hwc_tensor.cpu().numpy(), dtype=np.uint8)
    return bgr_hwc


class Cache:
    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache.get(key, default)
        return val


class VideoReader:
    """Video class with similar usage to a list object.

    This video wrapper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.
    Cache is used when decoding videos. So if the same frame is visited for
    the second time, there is no need to decode again if it is stored in the
    cache.

    Examples:
        >>> import mmcv
        >>> v = mmcv.VideoReader('sample.mp4')
        >>> len(v)  # get the total frame number with `len()`
        120
        >>> for img in v:  # v is iterable
        >>>     mmcv.imshow(img)
        >>> v[5]  # get the 6th frame
    """

    def __init__(self, filename: Path, cache_capacity: int = 10):
        # Check whether the video path is a url
        if not str(filename).startswith(("https://", "http://")):
            assert filename.exists(), f"file {filename} does not exist"
        self._vcap = cv2.VideoCapture(str(filename))
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        # get basic info
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self) -> np.ndarray | None:
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
            return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.get_frame(i) for i in range(*index.indices(self.frame_cnt))]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError("index out of range")
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._vcap.release()


class MultiVideoReader:
    def __init__(self, video_paths: list[Path]) -> None:
        # check that all video_paths are valid
        for video_path in video_paths:
            assert video_path.exists(), f"{video_path} does not exist"

        self.video_paths: list[Path] = video_paths
        self.video_readers: list[VideoReader] = [VideoReader(video_path) for video_path in video_paths]

        # TODO: confirm that we actually want this assertion
        # assert all(
        #     reader.height == self.video_readers[0].height and reader.width == self.video_readers[0].width
        #     for reader in self.video_readers
        # )

    @property
    def height(self) -> int:
        return self.video_readers[0].height

    @property
    def width(self) -> int:
        return self.video_readers[0].width

    def __len__(self) -> int:
        # Use minimum length to ensure safe iteration
        return min(len(reader) for reader in self.video_readers)

    def __iter__(self) -> Generator[BGRList | None, None, None]:
        while True:
            bgr_list: BGRList = []
            for reader in self.video_readers:
                bgr_image: ImageBGR | None = reader.read()
                match bgr_image:
                    case _ if bgr_image is not None:
                        bgr_list.append(bgr_image)
                    case None:
                        return
            yield bgr_list

    def __getitem__(self, idx: int) -> BGRList:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # Collect frames from each reader
        bgr_list: BGRList = []
        for reader in self.video_readers:
            frame = reader.get_frame(idx)
            if frame is not None:
                bgr_list.append(frame)
        return bgr_list


class TorchCodecVideoReader:
    """TorchCodec reader that returns RGB uint8 tensors in ``CHW``/``NCHW`` order."""

    def __init__(
        self,
        source: Path | bytes,
        device: str | torch.device | None = None,
        num_ffmpeg_threads: int = 0,
        seek_mode: Literal["exact", "approximate"] = "approximate",
    ) -> None:
        """Initialize the TorchCodec tensor video reader.

        Args:
            source: Path to a video file or raw encoded video bytes.
            device: TorchCodec decode device. Defaults to CUDA when available.
            num_ffmpeg_threads: FFmpeg thread count passed to TorchCodec.
            seek_mode: TorchCodec seek mode.
        """
        from torchcodec.decoders import VideoDecoder

        self._source: Path | bytes = source
        self.device: str = str(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_ffmpeg_threads: int = num_ffmpeg_threads
        self.seek_mode: Literal["exact", "approximate"] = seek_mode

        self._decoder: VideoDecoder = VideoDecoder(
            source,
            device=self.device,
            seek_mode=self.seek_mode,
            num_ffmpeg_threads=self.num_ffmpeg_threads,
            dimension_order="NCHW",
        )

        metadata = self._decoder.metadata
        self._width: int = self._required_int(metadata.width, "width")
        self._height: int = self._required_int(metadata.height, "height")
        self._fps: float = self._required_float(metadata.average_fps, "average_fps")
        self._frame_cnt: int = self._required_int(metadata.num_frames, "num_frames")
        self._position: int = 0
        self._read_chunk_size: int = 32
        self._read_buffer: UInt8[torch.Tensor, "b 3 h w"] | None = None
        self._read_buffer_start: int = 0
        self._read_buffer_stop: int = 0

    @staticmethod
    def _required_int(value: int | None, name: str) -> int:
        if value is None:
            raise ValueError(f"TorchCodec metadata field {name} is missing")
        return int(value)

    @staticmethod
    def _required_float(value: float | None, name: str) -> float:
        if value is None:
            raise ValueError(f"TorchCodec metadata field {name} is missing")
        return float(value)

    @property
    def width(self) -> int:
        """int: Width of video frames."""
        return self._width

    @property
    def height(self) -> int:
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self) -> tuple[int, int]:
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self) -> int:
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def source(self) -> Path | bytes:
        """Path | bytes: Original source (file path or bytes)."""
        return self._source

    def read(self) -> UInt8[torch.Tensor, "3 h w"] | None:
        """Read the next frame sequentially.

        Returns:
            RGB ``CHW`` uint8 tensor or ``None`` if end of video.
        """
        if self._position >= self._frame_cnt:
            return None

        if self._read_buffer is None or self._position < self._read_buffer_start or self._position >= self._read_buffer_stop:
            read_stop: int = min(self._position + self._read_chunk_size, self._frame_cnt)
            self._read_buffer = self.get_frames_in_range(self._position, read_stop)
            self._read_buffer_start = self._position
            self._read_buffer_stop = read_stop

        assert self._read_buffer is not None
        local_idx: int = self._position - self._read_buffer_start
        frame: UInt8[torch.Tensor, "3 h w"] = self._read_buffer[local_idx]
        self._position += 1
        return frame

    def get_frame(self, frame_id: int) -> UInt8[torch.Tensor, "3 h w"]:
        """Get frame by index (random access).

        Note: Random access is slower than sequential iteration due to seeking.

        Args:
            frame_id: Index of the expected frame, 0-based.

        Returns:
            RGB ``CHW`` uint8 tensor.

        Raises:
            IndexError: If frame_id is out of range.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(f'"frame_id" must be between 0 and {self._frame_cnt - 1}')

        frame: UInt8[torch.Tensor, "3 h w"] = self._decoder.get_frame_at(frame_id).data
        return frame

    def get_frames_in_range(self, start: int, stop: int) -> UInt8[torch.Tensor, "b 3 h w"]:
        """Get an RGB ``NCHW`` uint8 frame range."""
        if start < 0:
            raise IndexError("start must be non-negative")
        if stop < start:
            raise ValueError("stop must be greater than or equal to start")
        clamped_stop: int = min(stop, self._frame_cnt)
        if start >= clamped_stop:
            empty: UInt8[torch.Tensor, "b 3 h w"] = torch.empty(
                (0, 3, self._height, self._width),
                dtype=torch.uint8,
                device=self.device,
            )
            return empty
        video: UInt8[torch.Tensor, "b 3 h w"] = self._decoder.get_frames_in_range(start, clamped_stop).data
        return video

    def __len__(self) -> int:
        return self._frame_cnt

    def __getitem__(self, index: int | slice) -> UInt8[torch.Tensor, "3 h w"] | UInt8[torch.Tensor, "b 3 h w"]:
        if isinstance(index, slice):
            indices: list[int] = list(range(*index.indices(self._frame_cnt)))
            if not indices:
                empty: UInt8[torch.Tensor, "b 3 h w"] = torch.empty(
                    (0, 3, self._height, self._width),
                    dtype=torch.uint8,
                    device=self.device,
                )
                return empty
            if index.step in (None, 1):
                return self.get_frames_in_range(indices[0], indices[-1] + 1)
            stepped_frames: UInt8[torch.Tensor, "b 3 h w"] = torch.stack([self.get_frame(i) for i in indices], dim=0)
            return stepped_frames
        if index < 0:
            index += self._frame_cnt
            if index < 0:
                raise IndexError("index out of range")
        frame: UInt8[torch.Tensor, "3 h w"] = self.get_frame(index)
        return frame

    def __iter__(self):
        """Reset iterator for sequential access."""
        self._position = 0
        self._read_buffer = None
        self._read_buffer_start = 0
        self._read_buffer_stop = 0
        return self

    def __next__(self) -> UInt8[torch.Tensor, "3 h w"]:
        frame: UInt8[torch.Tensor, "3 h w"] | None = self.read()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        pass  # No explicit cleanup needed for TorchCodec


class TorchCodecMultiVideoReader:
    """Decode synchronized videos to RGB uint8 tensors on one device.

    The reader accepts file paths and raw encoded video bytes. Indexed access
    returns one ``CHW`` RGB tensor per video; chunked access returns one
    ``BCHW`` RGB tensor per video.
    """

    def __init__(
        self,
        video_sources: list[Path | bytes],
        device: str | torch.device | None = None,
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 0,
        seek_mode: Literal["exact", "approximate"] = "approximate",
    ) -> None:
        """Initialize the TorchCodec multiview tensor reader.

        Args:
            video_sources: Video paths or encoded video bytes in camera order.
            device: TorchCodec decode device. Defaults to CUDA when available.
            num_workers: Number of camera decode workers. Defaults to one worker per video.
            num_ffmpeg_threads: FFmpeg thread count passed to each decoder.
            seek_mode: TorchCodec seek mode.
        """
        if len(video_sources) == 0:
            raise ValueError("video_sources must contain at least one video")
        self.device: str = str(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_ffmpeg_threads: int = num_ffmpeg_threads
        self.num_workers: int = len(video_sources) if num_workers is None else num_workers
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.seek_mode: Literal["exact", "approximate"] = seek_mode

        self._video_readers: list[TorchCodecVideoReader] = [
            TorchCodecVideoReader(
                source,
                device=self.device,
                num_ffmpeg_threads=self.num_ffmpeg_threads,
                seek_mode=self.seek_mode,
            )
            for source in video_sources
        ]
        self._video_paths: list[Path] = [
            source if isinstance(source, Path) else Path(f"<bytes_{idx}>")
            for idx, source in enumerate(video_sources)
        ]
        self._height: int = self._video_readers[0].height
        self._width: int = self._video_readers[0].width
        self._fps: float = self._video_readers[0].fps
        self._frame_cnt: int = min(len(reader) for reader in self._video_readers)

    @property
    def video_paths(self) -> list[Path]:
        """Video file paths, with placeholders for byte-backed sources."""
        return self._video_paths

    @property
    def video_readers(self) -> list[TorchCodecVideoReader]:
        """Individual synchronized video readers."""
        return self._video_readers

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_cnt(self) -> int:
        return self._frame_cnt

    def __len__(self) -> int:
        return self.frame_cnt

    def __iter__(self) -> Generator[list[UInt8[torch.Tensor, "3 h w"]], None, None]:
        """Iterate synchronized frames across all videos."""
        for videos in self.iter_chunks():
            chunk_frame_count: int = min(int(video.shape[0]) for video in videos)
            for local_idx in range(chunk_frame_count):
                rgb_list: list[UInt8[torch.Tensor, "3 h w"]] = [video[local_idx] for video in videos]
                yield rgb_list

    def __getitem__(self, idx: int) -> list[UInt8[torch.Tensor, "3 h w"]]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        rgb_list: list[UInt8[torch.Tensor, "3 h w"]] = [reader.get_frame(idx) for reader in self._video_readers]
        return rgb_list

    @staticmethod
    def _decode_range(
        reader: TorchCodecVideoReader,
        start: int,
        stop: int,
    ) -> UInt8[torch.Tensor, "b 3 h w"]:
        video: UInt8[torch.Tensor, "b 3 h w"] = reader.get_frames_in_range(start, stop)
        return video

    def iter_chunks(
        self,
        chunk_size: int = 32,
        max_frames: int | None = None,
    ) -> Generator[list[UInt8[torch.Tensor, "b 3 h w"]], None, None]:
        """Decode the full multiview sequence in bounded chunks.

        Args:
            chunk_size: Number of frames per video to decode at a time.
            max_frames: Optional cap for benchmarks or smoke tests.

        Yields:
            One RGB ``BCHW`` uint8 tensor per video.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        frame_count: int = len(self) if max_frames is None else min(max_frames, len(self))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start in range(0, frame_count, chunk_size):
                stop: int = min(start + chunk_size, frame_count)
                videos: list[UInt8[torch.Tensor, "b 3 h w"]] = list(
                    executor.map(self._decode_range, self._video_readers, repeat(start), repeat(stop))
                )
                yield videos
