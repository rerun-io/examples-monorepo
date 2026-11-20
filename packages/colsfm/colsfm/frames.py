"""Where a run's keyframe pixels come from: one `FrameSource`, two backends, frames handed over as GPU tensors.

* `ImageFolderFrames` — the input directory holds the images (Galileo JPEGs, KITTI PNGs). Files are read a bounded
  `READ_AHEAD` ahead on a thread pool; JPEGs decode on the GPU with nvJPEG, anything else with OpenCV and one upload.
* `VideoStreamFrames` — a catalog segment's `VideoStream`s (the catalog door's input). Each camera's packets are read
  from the catalog once and decoded on NVDEC (simplecv's `open_segment_decoder`, the decoder behind the Rerun
  dataloader's `SegmentNvdecDecoder`). Only the names asked for are decoded, `DECODE_CHUNK` at a time in packet order,
  into a GPU LRU with a byte budget; a frame the LRU evicted is decoded again from the video. No JPEG is written or
  read: the door records which packet each keyframe is (`FrameSourceManifest`) and nothing else.

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
from typing import Any, Final

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

NOMINAL_FPS: Final[int] = 30
"""Frame rate written into the MP4 wrapper; frames are addressed by index, so it only has to be valid."""


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
        self.decoders: dict[int, tuple[ndarray, Any]] = {}
        """Per stream: the stream's sample timestamps and its open decoder."""

    def _decoder(self, stream_index: int) -> tuple[ndarray, Any]:
        if stream_index not in self.decoders:
            self.decoders[stream_index] = _open_stream(self.manifest, self.manifest.streams[stream_index], self.device)
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
            frames: Tensor = decoder.get_frames_at([int(indices[position]) for position in order]).data
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


def _open_stream(manifest: FrameSourceManifest, stream: StreamKeyframes, device: torch.device) -> tuple[ndarray, Any]:
    """Read one stream's packets from the catalog and open its NVDEC decoder.

    Args:
        manifest: The segment the stream belongs to.
        stream: The stream.
        device: The CUDA device.

    Returns:
        The stream's sample timestamps (int64 ns) and its decoder.
    """
    # Imported here: the catalog client and torchcodec are only needed by a catalog input.
    from rerun.catalog import CatalogClient
    from simplecv.rerun_dataloader import open_segment_decoder

    dataset = CatalogClient(manifest.catalog_url).get_dataset(name=manifest.dataset)
    times, _, _, decoder = open_segment_decoder(dataset, manifest.segment, stream.entity, manifest.timeline, device, NOMINAL_FPS)
    # The same int64 view `simplecv.catalog_video.CatalogVideo.t_ns` gives the catalog door: no unit change.
    return np.asarray(times).view(np.int64), decoder


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
    # Rows and samples are gathered per image and summed once at the end: a per-image `np.bincount(minlength=points)`
    # costs O(points) per image, which was 170 of the 194 s colsfm spent colouring clip 007 (18,606 images, 1.6 M points).
    all_rows: list[ndarray] = []
    all_samples: list[ndarray] = []
    image_ids: list[int] = sorted(int(image_id) for image_id in reconstruction.reg_image_ids())
    names: list[str] = [reconstruction.image(image_id).name for image_id in image_ids]
    for image_id, (_, rgb) in zip(image_ids, frames.frames(names), strict=True):
        observed = [(point2D.xy, point2D.point3D_id) for point2D in reconstruction.image(image_id).points2D if point2D.has_point3D()]
        if not observed:
            continue
        xy: ndarray = np.asarray([pixel for pixel, _ in observed], dtype=np.float64)
        rows: ndarray = np.searchsorted(point_ids, np.asarray([point_id for _, point_id in observed], dtype=np.int64))
        height, width = int(rgb.shape[1]), int(rgb.shape[2])
        centre: ndarray = xy - 0.5
        inside: ndarray = (centre[:, 0] >= 0) & (centre[:, 0] <= width - 1) & (centre[:, 1] >= 0) & (centre[:, 1] <= height - 1)
        if not inside.any():
            continue
        grid: Tensor = torch.from_numpy((2.0 * xy[inside] / np.array([width, height]) - 1.0).astype(np.float32)).to(rgb.device)[None, None]
        sampled: ndarray = torch.nn.functional.grid_sample(rgb[None].float(), grid, mode="bilinear", align_corners=False)[0, :, 0].T.cpu().numpy()
        all_rows.append(rows[inside])
        all_samples.append(sampled.astype(np.float64))
    if not all_rows:
        return 0
    stacked_rows: ndarray = np.concatenate(all_rows)
    stacked_samples: ndarray = np.concatenate(all_samples)
    sums: ndarray = np.stack([np.bincount(stacked_rows, weights=stacked_samples[:, channel], minlength=len(point_ids)) for channel in range(3)], axis=1)
    counts: ndarray = np.bincount(stacked_rows, minlength=len(point_ids))
    coloured: int = 0
    seen: ndarray = np.flatnonzero(counts)
    colours: ndarray = np.clip(np.rint(sums[seen] / counts[seen, None]), 0, 255).astype(np.uint8)
    for point_id, colour in zip(point_ids[seen], colours, strict=True):
        reconstruction.points3D[int(point_id)].color = colour
        coloured += int(colour.any())
    return coloured
