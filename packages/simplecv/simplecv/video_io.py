# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Literal

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
from jaxtyping import UInt8

from simplecv.image_types import BGRList, ImageBGR


@dataclass(frozen=True, slots=True)
class TorchCodecVideoChunk:
    """Synchronized multiview video chunk."""

    start: int
    """Inclusive start frame index."""
    stop: int
    """Exclusive stop frame index."""
    videos: list[UInt8[torch.Tensor, "b 3 h w"]]
    """One RGB BCHW uint8 tensor per video."""


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
    """TorchCodec-based video reader supporting both file paths and in-memory bytes.

    This reader uses TorchCodec for faster decoding compared to OpenCV's VideoCapture.
    It outputs BGR numpy arrays for compatibility with existing OpenCV-based code.

    Supports both file paths and raw video bytes (e.g., extracted from RRD recordings).

    Examples:
        >>> from pathlib import Path
        >>> reader = TorchCodecVideoReader(Path("video.mp4"))
        >>> len(reader)  # total frame count
        300
        >>> for frame in reader:  # iterate sequentially (fastest)
        ...     process(frame)
        >>> reader[50]  # random access (slower, requires seek)
    """

    def __init__(self, source: Path | bytes) -> None:
        """Initialize the TorchCodec video reader.

        Args:
            source: Path to video file or raw video bytes.
        """
        from torchcodec.decoders import VideoDecoder

        self._source: Path | bytes = source
        self._is_bytes: bool = isinstance(source, bytes)

        # Create decoder - TorchCodec accepts str path or bytes
        decoder_source: str | bytes = source if isinstance(source, bytes) else str(source)
        self._decoder: VideoDecoder = VideoDecoder(
            decoder_source,
            device="cpu",
            seek_mode="exact",
            num_ffmpeg_threads=0,  # Auto-managed threading
            dimension_order="NHWC",
        )

        # Cache metadata (type ignores: TorchCodec metadata types are Optional but never None in practice)
        metadata = self._decoder.metadata
        self._width: int = int(metadata.width)  # type: ignore[arg-type]
        self._height: int = int(metadata.height)  # type: ignore[arg-type]
        self._fps: float = float(metadata.average_fps)  # type: ignore[arg-type]
        self._frame_cnt: int = int(metadata.num_frames)  # type: ignore[arg-type]

        # Iterator state
        self._position: int = 0
        self._iterator = None

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

    def _frame_to_bgr(self, frame_tensor) -> ImageBGR:
        """Convert TorchCodec frame tensor (RGB) to BGR numpy array.

        Args:
            frame_tensor: Torch tensor in NHWC format (1, H, W, 3) RGB.

        Returns:
            BGR numpy array (H, W, 3).
        """
        # Squeeze batch dimension and convert to numpy
        rgb_frame: np.ndarray = frame_tensor.squeeze(0).numpy()
        # Convert RGB to BGR for OpenCV compatibility
        bgr_frame: ImageBGR = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return bgr_frame

    def read(self) -> ImageBGR | None:
        """Read the next frame sequentially.

        Returns:
            BGR numpy array or None if end of video.
        """
        if self._position >= self._frame_cnt:
            return None

        if self._iterator is None:
            self._iterator = iter(self._decoder)  # type: ignore[arg-type]

        try:
            frame_tensor = next(self._iterator)
            self._position += 1
            return self._frame_to_bgr(frame_tensor)
        except StopIteration:
            return None

    def get_frame(self, frame_id: int) -> ImageBGR:
        """Get frame by index (random access).

        Note: Random access is slower than sequential iteration due to seeking.

        Args:
            frame_id: Index of the expected frame, 0-based.

        Returns:
            BGR numpy array.

        Raises:
            IndexError: If frame_id is out of range.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(f'"frame_id" must be between 0 and {self._frame_cnt - 1}')

        frame = self._decoder.get_frame_at(frame_id)
        # With dimension_order="NHWC", Frame.data is HWC format, add batch dim for _frame_to_bgr
        frame_tensor = frame.data.unsqueeze(0)
        return self._frame_to_bgr(frame_tensor)

    def __len__(self) -> int:
        return self._frame_cnt

    def __getitem__(self, index: int | slice) -> ImageBGR | list[ImageBGR]:
        if isinstance(index, slice):
            frames: list[ImageBGR] = [self.get_frame(i) for i in range(*index.indices(self._frame_cnt))]
            return frames
        if index < 0:
            index += self._frame_cnt
            if index < 0:
                raise IndexError("index out of range")
        frame: ImageBGR = self.get_frame(index)
        return frame

    def __iter__(self):
        """Reset iterator for sequential access."""
        from torchcodec.decoders import VideoDecoder

        # Create fresh decoder for iteration
        decoder_source: str | bytes = str(self._source) if isinstance(self._source, Path) else self._source
        fresh_decoder: VideoDecoder = VideoDecoder(
            decoder_source,
            device="cpu",
            seek_mode="exact",
            num_ffmpeg_threads=0,
            dimension_order="NHWC",
        )
        self._iter_decoder = fresh_decoder
        self._iter_position = 0
        return self

    def __next__(self) -> ImageBGR:
        if self._iter_position >= self._frame_cnt:
            raise StopIteration

        try:
            # Use iterator on the fresh decoder
            if not hasattr(self, "_iter_iterator"):
                self._iter_iterator = iter(self._iter_decoder)  # type: ignore[arg-type]
            frame_tensor = next(self._iter_iterator)
            self._iter_position += 1
            return self._frame_to_bgr(frame_tensor)
        except StopIteration:
            raise StopIteration from None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        pass  # No explicit cleanup needed for TorchCodec


class TorchCodecMultiVideoReader:
    """Multi-video reader using TorchCodec for synchronized multi-camera setups.

    Supports mixed inputs: file paths and/or raw video bytes.
    """

    def __init__(self, video_sources: list[Path | bytes]) -> None:
        """Initialize with list of video sources.

        Args:
            video_sources: List of video file paths or raw bytes.
        """
        self._video_sources: list[Path | bytes] = video_sources
        self._video_readers: list[TorchCodecVideoReader] = [
            TorchCodecVideoReader(source) for source in video_sources
        ]

        # Extract paths for compatibility with existing code
        self._video_paths: list[Path] = [
            source if isinstance(source, Path) else Path(f"<bytes_{i}>")
            for i, source in enumerate(video_sources)
        ]

    @property
    def video_paths(self) -> list[Path]:
        """list[Path]: Video file paths (placeholder for bytes sources)."""
        return self._video_paths

    @property
    def video_readers(self) -> list[TorchCodecVideoReader]:
        """list[TorchCodecVideoReader]: Individual video readers."""
        return self._video_readers

    @property
    def height(self) -> int:
        """int: Height of first video's frames."""
        return self._video_readers[0].height

    @property
    def width(self) -> int:
        """int: Width of first video's frames."""
        return self._video_readers[0].width

    def __len__(self) -> int:
        """Use minimum length to ensure safe iteration."""
        return min(len(reader) for reader in self._video_readers)

    def __iter__(self) -> Generator[BGRList | None, None, None]:
        """Iterate through all videos frame-by-frame."""
        # Create fresh iterators for each reader
        iterators = [iter(reader) for reader in self._video_readers]

        while True:
            bgr_list: BGRList = []
            for iterator in iterators:
                try:
                    bgr_image: ImageBGR = next(iterator)
                    bgr_list.append(bgr_image)
                except StopIteration:
                    return
            yield bgr_list

    def __getitem__(self, idx: int) -> BGRList:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # Collect frames from each reader
        bgr_list: BGRList = []
        for reader in self._video_readers:
            frame: ImageBGR = reader.get_frame(idx)
            bgr_list.append(frame)
        return bgr_list


class TorchCodecCudaMultiVideoReader:
    """Decode synchronized videos to RGB BCHW uint8 tensors on one device."""

    def __init__(
        self,
        video_paths: list[Path],
        device: str = "cuda",
        num_workers: int | None = None,
        num_ffmpeg_threads: int = 0,
        seek_mode: Literal["exact", "approximate"] = "approximate",
    ) -> None:
        """Initialize the TorchCodec multiview tensor reader.

        Args:
            video_paths: Video paths in camera order.
            device: TorchCodec decode device. Use ``"cuda"`` to keep output frames on GPU.
            num_workers: Number of cameras to decode concurrently. Defaults to one worker per video.
            num_ffmpeg_threads: FFmpeg thread count passed to each decoder.
            seek_mode: TorchCodec seek mode.
        """
        self.video_paths: list[Path] = video_paths
        self.device: str = device
        self.num_ffmpeg_threads: int = num_ffmpeg_threads
        if len(video_paths) == 0:
            raise ValueError("video_paths must contain at least one video")
        self.num_workers: int = len(video_paths) if num_workers is None else num_workers
        self.seek_mode: Literal["exact", "approximate"] = seek_mode

        from torchcodec.decoders import VideoDecoder

        self._decoders: list[Any] = [
            VideoDecoder(
                video_path,
                device=self.device,
                dimension_order="NCHW",
                num_ffmpeg_threads=self.num_ffmpeg_threads,
                seek_mode=self.seek_mode,
            )
            for video_path in video_paths
        ]
        metadata: list[Any] = [decoder.metadata for decoder in self._decoders]
        self._height: int = self._required_int(metadata[0].height, "height")
        self._width: int = self._required_int(metadata[0].width, "width")
        self._fps: float = self._required_float(metadata[0].average_fps, "average_fps")
        self._frame_cnt: int = min(self._required_int(item.num_frames, "num_frames") for item in metadata)

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

    @staticmethod
    def _decode_range(decoder: Any, start: int, stop: int) -> UInt8[torch.Tensor, "b 3 h w"]:
        video: UInt8[torch.Tensor, "b 3 h w"] = decoder.get_frames_in_range(start, stop).data
        return video

    def iter_chunks(
        self,
        chunk_size: int = 32,
        max_frames: int | None = None,
    ) -> Generator[TorchCodecVideoChunk, None, None]:
        """Decode the full multiview sequence in bounded GPU chunks.

        Args:
            chunk_size: Number of frames per video to decode at a time.
            max_frames: Optional cap for benchmarks or smoke tests.

        Yields:
            Synchronized chunks with one ``BCHW`` RGB tensor per video.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        frame_count: int = len(self) if max_frames is None else min(max_frames, len(self))
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for start in range(0, frame_count, chunk_size):
                stop: int = min(start + chunk_size, frame_count)
                videos: list[UInt8[torch.Tensor, "b 3 h w"]] = list(
                    executor.map(self._decode_range, self._decoders, repeat(start), repeat(stop))
                )
                yield TorchCodecVideoChunk(start=start, stop=stop, videos=videos)
