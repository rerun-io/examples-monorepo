"""Where a run's keyframe pixels come from: one `FrameSource`, two backends, frames handed over as GPU tensors.

* `ImageFolderFrames` — the input directory holds the images (Galileo JPEGs, KITTI PNGs). Files are read a bounded
  `READ_AHEAD` ahead on a thread pool; JPEGs decode on the GPU with nvJPEG, anything else with OpenCV and one upload.
* `VideoStreamFrames` — a catalog segment's `VideoStream`s (the catalog door's input), read through simplecv's
  `CatalogVideoReader`: a blob-free index of every stream, then GOP-aligned time chunks of every camera in one query,
  decoded on NVDEC chunk by chunk (`ChunkedVideoDecoder`), so memory is a few chunks whatever the session's length.
  Only the names asked for are decoded, `DECODE_CHUNK` at a time in packet order, into a GPU LRU with a byte budget; a
  frame the LRU evicted is decoded again from the video. No JPEG is written or read: the door records which packet
  each keyframe is (`FrameSourceManifest`) and nothing else.

Both return RGB uint8 `[3, height, width]` CUDA tensors keyed by the image names of `frames_meta.json` and the COLMAP
database. `open_frame_source` picks the backend from the input directory: a `frame_source.json` means the video
streams, its absence means the folder. A source belongs to one run: `colsfm.pipeline.run_pipeline` opens it, hands it
to every stage that reads pixels and to the observer, and closes it however the run ends.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import cv2
import numpy as np
import pycolmap
import torch
from numpy import ndarray
from torch import Tensor
from torchvision.io import ImageReadMode, decode_jpeg

from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSource, FrameSourceManifest, StreamKeyframes, read_manifest

READ_WORKERS: Final[int] = 4
"""Threads reading image files ahead of their decode."""

READ_AHEAD: Final[int] = 2 * READ_WORKERS
"""Encoded files in flight at once: `ThreadPoolExecutor.map` would submit (and so hold) every file of the request."""

DECODE_CHUNK: Final[int] = 64
"""Frames decoded per NVDEC call, and the window a request is served in: bounds the transient GPU memory of one call."""

FRAME_STORE_FRACTION: Final[float] = 0.15
"""Share of the device's total memory the decoded-frame LRU may hold. On the 5090 (32 GB) that is 4.8 GB, so MGO07's
3192 keyframes at 480x640 (2.9 GB) stay resident and every stage after extraction reads them without decoding; a longer
input evicts and decodes again from the video instead of growing, and leaves the rest to the engines and the mapper."""


@runtime_checkable
class FrameDecoder(Protocol):
    """One stream's frames by sample index (simplecv's `ChunkedVideoDecoder`)."""

    def get_frames_at(self, indices: Sequence[int]) -> Tensor:
        """RGB uint8 `[n, 3, h, w]` frames at stream sample indices, in the order asked."""
        ...


def decode_image_bytes(payload: bytes, origin: str, device: torch.device) -> Tensor:
    """Decode one encoded image to RGB uint8 `[3, h, w]` on the GPU: nvJPEG for JPEG, OpenCV and one upload otherwise.

    Args:
        payload: The encoded image.
        origin: Where it came from, for the error message.
        device: The CUDA device.

    Returns:
        The decoded image on the device.

    Raises:
        ValueError: When the bytes do not decode.
    """
    if payload[:2] == b"\xff\xd8":
        decoded: Tensor | list[Tensor] = decode_jpeg(torch.frombuffer(bytearray(payload), dtype=torch.uint8), mode=ImageReadMode.RGB, device=device)
        # One encoded tensor in, one image out; the list form is for batched input.
        return decoded if isinstance(decoded, Tensor) else decoded[0]
    bgr: ndarray | None = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"could not decode {origin}")
    return torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).to(device).permute(2, 0, 1)


class ImageFolderFrames:
    """Frames read from image files under one root directory."""

    def __init__(self, root: Path, device: torch.device | None = None) -> None:
        """Bind the source to a directory.

        Args:
            root: Directory the image names are relative to.
            device: CUDA device the frames land on.

        Raises:
            FileNotFoundError: When `root` is not a directory.
        """
        if not root.is_dir():
            raise FileNotFoundError(f"No image directory at {root}")
        self.root: Path = root
        self.device: torch.device = device or torch.device("cuda")

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        """Yield the named frames in order; files are read on a small thread pool ahead of their decode.

        Args:
            names: Image names relative to the root.

        Yields:
            `(name, rgb)` on the GPU.
        """
        with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
            pending: deque[tuple[str, Future[bytes]]] = deque()
            for name in names:
                pending.append((name, pool.submit((self.root / name).read_bytes)))
                if len(pending) < READ_AHEAD:
                    continue
                first, payload = pending.popleft()
                yield first, decode_image_bytes(payload.result(), str(self.root / first), self.device)
            while pending:
                first, payload = pending.popleft()
                yield first, decode_image_bytes(payload.result(), str(self.root / first), self.device)

    def close(self) -> None:
        """Nothing to release: the folder source holds no frames between requests."""


class VideoStreamFrames:
    """Keyframes decoded from a catalog segment's video streams on NVDEC, held in a byte-bounded GPU LRU.

    A request decodes only the names it asks for that the LRU does not hold, `DECODE_CHUNK` at a time in packet order
    per stream, and yields them in the order asked. Each stream's decoder is opened on first use and kept until `close`.
    """

    def __init__(self, manifest: FrameSourceManifest, device: torch.device | None = None, budget_bytes: int | None = None) -> None:
        """Bind the source to a manifest; nothing is read until frames are asked for.

        Args:
            manifest: Which segment, streams and samples.
            device: CUDA device the frames land on.
            budget_bytes: Decoded bytes the LRU may hold; `FRAME_STORE_FRACTION` of the device's memory when None.
        """
        self.manifest: FrameSourceManifest = manifest
        self.device: torch.device = device or torch.device("cuda")
        self.budget_bytes: int | None = budget_bytes
        """Read from the device on the first decode when None (`FRAME_STORE_FRACTION`)."""
        self.store: OrderedDict[str, Tensor] = OrderedDict()
        self.stored_bytes: int = 0
        self.decoded_frames: int = 0
        """Frames decoded so far, re-decodes of evicted frames included."""
        self.location: dict[str, tuple[int, int]] = {
            name: (stream_index, position) for stream_index, stream in enumerate(manifest.streams) for position, name in enumerate(stream.names)
        }
        self.decoders: dict[int, tuple[ndarray, FrameDecoder]] = {}
        """Per stream: the stream's sample timestamps and its decoder; every stream opens on the first decode."""

    def _decoder(self, stream_index: int) -> tuple[ndarray, FrameDecoder]:
        if not self.decoders:
            self.decoders = dict(enumerate(_open_streams(self.manifest, self.device)))
        return self.decoders[stream_index]

    def _decode(self, names: Sequence[str]) -> dict[str, Tensor]:
        by_stream: dict[int, list[str]] = {}
        for name in names:
            by_stream.setdefault(self.location[name][0], []).append(name)
        decoded: dict[str, Tensor] = {}
        for stream_index, stream_names in by_stream.items():
            stream: StreamKeyframes = self.manifest.streams[stream_index]
            sample_ns, decoder = self._decoder(stream_index)
            wanted: ndarray = np.asarray([stream.t_ns[self.location[name][1]] for name in stream_names], dtype=np.int64)
            indices: ndarray = np.searchsorted(sample_ns, wanted)
            if np.any(indices >= len(sample_ns)) or np.any(sample_ns[np.minimum(indices, len(sample_ns) - 1)] != wanted):
                raise ValueError(f"{stream.entity}: a keyframe timestamp is not a sample of the stream")
            order: ndarray = np.argsort(indices, kind="stable")
            frames: Tensor = decoder.get_frames_at([int(indices[position]) for position in order])
            for row, position in enumerate(order):
                decoded[stream_names[int(position)]] = frames[row].clone()
        self.decoded_frames += len(decoded)
        return decoded

    def _keep(self, name: str, frame: Tensor) -> None:
        if self.budget_bytes is None:
            self.budget_bytes = int(FRAME_STORE_FRACTION * torch.cuda.get_device_properties(frame.device).total_memory)
        self.store[name] = frame
        self.stored_bytes += frame.numel() * frame.element_size()
        while self.stored_bytes > self.budget_bytes and self.store:
            _, evicted = self.store.popitem(last=False)
            self.stored_bytes -= evicted.numel() * evicted.element_size()

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        """Yield the named frames in order, decoding the ones the LRU does not hold.

        Args:
            names: Image names from the manifest.

        Yields:
            `(name, rgb)` on the GPU.

        Raises:
            KeyError: When a name is not one of the manifest's keyframes.
        """
        for first in range(0, len(names), DECODE_CHUNK):
            window: Sequence[str] = names[first : first + DECODE_CHUNK]
            # Take the window's held frames first: inserting the fresh ones can evict them from the store.
            held: dict[str, Tensor] = {name: self.store[name] for name in window if name in self.store}
            for name in held:
                self.store.move_to_end(name)
            missing: list[str] = list(dict.fromkeys(name for name in window if name not in held))
            fresh: dict[str, Tensor] = self._decode(missing) if missing else {}
            for name, frame in fresh.items():
                self._keep(name, frame)
            for name in window:
                yield name, held[name] if name in held else fresh[name]

    def close(self) -> None:
        """Drop every decoded frame and decoder (their GPU memory returns to torch's allocator)."""
        self.store.clear()
        self.stored_bytes = 0
        self.decoders.clear()


def _open_streams(manifest: FrameSourceManifest, device: torch.device) -> list[tuple[ndarray, FrameDecoder]]:
    """Index every stream of the manifest and open a chunked NVDEC decoder per stream over one shared reader.

    One reader serves all streams, so a chunk is fetched for every camera in one query and decoded from its cache.

    Args:
        manifest: The segment and its streams.
        device: The CUDA device.

    Returns:
        Per stream, in manifest order: its sample timestamps (int64 ns) and its decoder.
    """
    # Imported here: the catalog client and torchcodec are only needed by a catalog input.
    from rerun.catalog import CatalogClient
    from simplecv.catalog_video_reader import CatalogVideoReader, ChunkedVideoDecoder, StreamIndex, read_video_index

    dataset = CatalogClient(manifest.catalog_url).get_dataset(name=manifest.dataset)
    entities: list[str] = [stream.entity for stream in manifest.streams]
    index: tuple[StreamIndex, ...] = read_video_index(dataset, manifest.segment, entities, manifest.timeline)
    reader: CatalogVideoReader = CatalogVideoReader(dataset, manifest.segment, index, manifest.timeline)
    # The same int64 view `simplecv.catalog_video_reader.StreamIndex.t_ns` gives the catalog door: no unit change.
    return [(stream.t_ns, ChunkedVideoDecoder(reader, position, device)) for position, stream in enumerate(index)]


def open_frame_source(input_dir: Path, device: torch.device | None = None) -> FrameSource:
    """The frame source an input directory describes: its manifest's video streams, else its image files.

    Args:
        input_dir: The run's input directory.
        device: CUDA device the frames land on.

    Returns:
        A `VideoStreamFrames` when `frame_source.json` is present, else an `ImageFolderFrames` over the directory.
    """
    manifest_path: Path = input_dir / FRAME_SOURCE_FILE
    if manifest_path.is_file():
        return VideoStreamFrames(read_manifest(manifest_path), device)
    return ImageFolderFrames(input_dir, device)


def colour_points(reconstruction: pycolmap.Reconstruction, frames: FrameSource) -> int:
    """Give every 3D point the mean colour of its observations, sampled from the run's GPU frames.

    The frame-source equivalent of COLMAP's `Reconstruction.extract_colors_for_all_images`, which reads image files:
    each observation is sampled bilinearly at `xy - 0.5` (COLMAP's pixel-centre convention), observations whose sample
    would leave the image are skipped, and a point gets the rounded mean over its track.

    Args:
        reconstruction: The model to colour, edited in place.
        frames: The run's frame source.

    Returns:
        Points that ended up with a non-black colour; points with no usable observation keep `(0, 0, 0)`.
    """
    point_ids: ndarray = np.asarray(sorted(reconstruction.points3D), dtype=np.int64)
    if len(point_ids) == 0:
        return 0
    # Each image adds its samples into the observed rows only: a per-image `np.bincount(minlength=points)` costs O(points)
    # per image, which was 170 of the 194 s colsfm spent colouring clip 007 (18,606 images, 1.6 M points). Memory stays
    # O(points) plus one image's observations. The sums stay on the frames' device (`index_add_`, one download at the
    # end), so a GPU source does not sync per image.
    sums: Tensor | None = None
    counts: Tensor | None = None
    image_ids: list[int] = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
    names: list[str] = [reconstruction.image(image_id).name for image_id in image_ids]
    for image_id, (_, rgb) in zip(image_ids, frames.frames(names), strict=True):
        observed: pycolmap.Point2DList = reconstruction.image(image_id).get_observation_points2D()
        if not observed:
            continue
        xy: ndarray = np.asarray([point2D.xy for point2D in observed], dtype=np.float64)
        rows: ndarray = np.searchsorted(point_ids, np.fromiter((point2D.point3D_id for point2D in observed), dtype=np.int64, count=len(observed)))
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        centre: ndarray = xy - 0.5
        inside: ndarray = (centre[:, 0] >= 0) & (centre[:, 0] <= width - 1) & (centre[:, 1] >= 0) & (centre[:, 1] <= height - 1)
        if not inside.any():
            continue
        if sums is None or counts is None:
            sums = torch.zeros((len(point_ids), 3), dtype=torch.float64, device=rgb.device)
            counts = torch.zeros(len(point_ids), dtype=torch.int64, device=rgb.device)
        grid: Tensor = torch.from_numpy((2.0 * xy[inside] / np.array([width, height]) - 1.0).astype(np.float32)).to(rgb.device)[None, None]
        sampled: Tensor = torch.nn.functional.grid_sample(rgb[None].float(), grid, mode="bilinear", align_corners=False)[0, :, 0].T
        # `index_add_`, not `sums[rows] += ...`: a point seen twice in one image adds both samples.
        index: Tensor = torch.from_numpy(rows[inside]).to(rgb.device)
        sums.index_add_(0, index, sampled.double())
        counts.index_add_(0, index, torch.ones_like(index))
    if sums is None or counts is None:
        return 0
    total: ndarray = sums.cpu().numpy()
    seen_counts: ndarray = counts.cpu().numpy()
    coloured: int = 0
    seen: ndarray = np.flatnonzero(seen_counts)
    colours: ndarray = np.clip(np.rint(total[seen] / seen_counts[seen, None]), 0, 255).astype(np.uint8)
    for point_id, colour in zip(point_ids[seen], colours, strict=True):
        reconstruction.points3D[int(point_id)].color = colour
        coloured += int(colour.any())
    return coloured
