"""Where a run's keyframe pixels come from: one `FrameSource`, two backends, frames handed over as GPU tensors.

* `ImageFolderFrames` — the input directory holds the images (Galileo JPEGs, KITTI PNGs). Files are read ahead on a
  thread pool; JPEGs decode on the GPU with nvJPEG, anything else with OpenCV and one upload.
* `VideoStreamFrames` — a catalog segment's `VideoStream`s (the catalog door's input). Each camera's packets are read
  from the catalog once and decoded once on NVDEC (simplecv's `open_segment_decoder`, the decoder behind the Rerun
  dataloader's `SegmentNvdecDecoder`), keeping only the selected keyframes, in sorted order and in bounded chunks. No
  JPEG is written or read: the door records which packet each keyframe is (`FrameSourceManifest`) and nothing else.

Both return RGB uint8 `[3, height, width]` CUDA tensors keyed by the image names of `frames_meta.json` and the COLMAP
database. `open_frame_source` picks the backend from the input directory: a `frame_source.json` means the video
streams, its absence means the folder.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np
import pycolmap
import torch
from numpy import ndarray
from torch import Tensor
from torchvision.io import ImageReadMode, decode_jpeg

from colsfm.frame_manifest import FRAME_SOURCE_FILE, FrameSourceManifest, read_manifest

READ_WORKERS: Final[int] = 4
"""Threads reading image files ahead of their decode."""

DECODE_CHUNK: Final[int] = 64
"""Frames decoded per NVDEC call: bounds the transient GPU memory of one call."""

NOMINAL_FPS: Final[int] = 30
"""Frame rate written into the MP4 wrapper; frames are addressed by index, so it only has to be valid."""


class FrameSource(Protocol):
    """Keyframe pixels by image name, as RGB uint8 `[3, h, w]` tensors on the GPU."""

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        """Yield each named frame once, in the order given.

        Args:
            names: Image names as `frames_meta.json` and the database hold them.

        Yields:
            `(name, rgb)` with `rgb` uint8 `[3, height, width]` on the GPU.
        """
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
            for name, payload in zip(names, pool.map(lambda name: (self.root / name).read_bytes(), names), strict=True):
                yield name, decode_image_bytes(payload, str(self.root / name), self.device)


class VideoStreamFrames:
    """Keyframes decoded from a catalog segment's video streams on NVDEC, once, and kept on the GPU.

    The first request decodes every camera's keyframes (sorted, `DECODE_CHUNK` at a time) into a store keyed by image
    name; later requests are lookups. MGO07's 3192 keyframes at 480x640 take 2.9 GB.
    """

    def __init__(self, manifest: FrameSourceManifest, device: torch.device | None = None) -> None:
        """Bind the source to a manifest; nothing is read until frames are asked for.

        Args:
            manifest: Which segment, streams and samples.
            device: CUDA device the frames land on.
        """
        self.manifest: FrameSourceManifest = manifest
        self.device: torch.device = device or torch.device("cuda")
        self.store: dict[str, Tensor] = {}

    def _decode_all(self) -> None:
        # Imported here: the catalog client and torchcodec are only needed by a catalog input.
        from rerun.catalog import CatalogClient
        from simplecv.rerun_dataloader import open_segment_decoder

        dataset = CatalogClient(self.manifest.catalog_url).get_dataset(name=self.manifest.dataset)
        # Built locally and kept only when every stream decoded: a half-filled store would read as already decoded.
        store: dict[str, Tensor] = {}
        for stream in self.manifest.streams:
            times, _, _, decoder = open_segment_decoder(
                dataset, self.manifest.segment, stream.entity, self.manifest.timeline, self.device, NOMINAL_FPS
            )
            # The same int64 view `simplecv.catalog_video.CatalogVideo.t_ns` gives the catalog door: no unit change.
            sample_ns: ndarray = np.asarray(times).view(np.int64)
            wanted: ndarray = np.asarray(stream.t_ns, dtype=np.int64)
            indices: ndarray = np.searchsorted(sample_ns, wanted)
            if np.any(indices >= len(sample_ns)) or np.any(sample_ns[np.minimum(indices, len(sample_ns) - 1)] != wanted):
                raise ValueError(f"{stream.entity}: a keyframe timestamp is not a sample of the stream")
            order: ndarray = np.argsort(indices, kind="stable")
            for first in range(0, len(order), DECODE_CHUNK):
                chunk: ndarray = order[first : first + DECODE_CHUNK]
                decoded: Tensor = decoder.get_frames_at([int(indices[position]) for position in chunk]).data
                if first == 0:
                    _check_store_fits(decoded[0], sum(len(entry.names) for entry in self.manifest.streams), self.device)
                for row, position in enumerate(chunk):
                    store[stream.names[int(position)]] = decoded[row].clone()
            print(f"[colsfm.frames] {stream.entity}: {len(order)} keyframes decoded on {self.device}")
        self.store = store

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        """Yield the named frames in order, decoding every stream on the first call.

        Args:
            names: Image names from the manifest.

        Yields:
            `(name, rgb)` on the GPU.

        Raises:
            KeyError: When a name is not one of the manifest's keyframes.
        """
        if not self.store:
            self._decode_all()
        for name in names:
            yield name, self.store[name]


def _check_store_fits(frame: Tensor, num_frames: int, device: torch.device) -> None:
    """Fail before decoding the rest when the whole keyframe store cannot fit in free GPU memory.

    Args:
        frame: One decoded frame, for the per-frame size.
        num_frames: Keyframes the store will hold.
        device: The CUDA device.

    Raises:
        MemoryError: When the store would exceed the device's free memory.
    """
    needed: int = frame.numel() * frame.element_size() * num_frames
    free, _ = torch.cuda.mem_get_info(device)
    if needed > free:
        raise MemoryError(
            f"the keyframe store needs {needed / 2**30:.1f} GiB of GPU memory and {free / 2**30:.1f} GiB is free; "
            "sample fewer frames (--source.frame-stride, --source.max-samples)"
        )


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


def frame_source_for(input_dir: Path) -> FrameSource:
    """The one frame source of an input directory for this process, so every stage reads the same decoded store.

    A video-stream input is decoded once however many stages ask (feature extraction, both matching passes, point
    colours); an image folder holds no state, so sharing it costs nothing. The path is resolved before the lookup, so
    two spellings of one directory share one store.

    Args:
        input_dir: The run's input directory, as `PipelineOptions.input_dir` holds it.

    Returns:
        The cached frame source of the resolved directory.
    """
    return _frame_source(input_dir.resolve())


@cache
def _frame_source(resolved_dir: Path) -> FrameSource:
    return open_frame_source(resolved_dir)


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
    sums: ndarray = np.zeros((len(point_ids), 3), dtype=np.float64)
    counts: ndarray = np.zeros(len(point_ids), dtype=np.int64)
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
        for channel in range(3):
            sums[:, channel] += np.bincount(rows[inside], weights=sampled[:, channel], minlength=len(point_ids))
        counts += np.bincount(rows[inside], minlength=len(point_ids))
    coloured: int = 0
    seen: ndarray = np.flatnonzero(counts)
    colours: ndarray = np.clip(np.rint(sums[seen] / counts[seen, None]), 0, 255).astype(np.uint8)
    for point_id, colour in zip(point_ids[seen], colours, strict=True):
        reconstruction.points3D[int(point_id)].color = colour
        coloured += int(colour.any())
    return coloured
