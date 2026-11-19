"""The frame-source manifest (which `VideoStream` sample each keyframe of a catalog input is) and the `FrameSource` contract.

Torch-free on purpose: the catalog door (metadata only), run resolution and the COLMAP-only backends read it, and the
stage signatures name `FrameSource`, without loading the GPU stack that `colsfm.frames` needs to decode the pixels.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from serde import SerdeError, serde
from serde.json import from_json, to_json

if TYPE_CHECKING:
    from torch import Tensor

FRAME_SOURCE_FILE: Final[str] = "frame_source.json"
"""The manifest a catalog input directory carries instead of image files."""


@runtime_checkable
class FrameSource(Protocol):
    """Keyframe pixels by image name, as RGB uint8 `[3, h, w]` tensors on the GPU; `close` releases what it holds."""

    def frames(self, names: Sequence[str]) -> Iterator[tuple[str, Tensor]]:
        """Yield each named frame once, in the order given.

        Args:
            names: Image names as `frames_meta.json` and the database hold them.

        Yields:
            `(name, rgb)` with `rgb` uint8 `[3, height, width]` on the GPU.
        """
        ...

    def close(self) -> None:
        """Release decoded frames and decoders; the source can still be read afterwards, from scratch."""
        ...


def is_catalog_input(input_dir: Path) -> bool:
    """Whether an input directory's keyframes live in a catalog segment's video streams rather than in files.

    Args:
        input_dir: The run's input directory.

    Returns:
        True when it carries a frame-source manifest.
    """
    return (input_dir / FRAME_SOURCE_FILE).is_file()


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class StreamKeyframes:
    """One camera's video stream and the keyframes the run uses from it."""

    entity: str
    """The `VideoStream` entity path, without the leading slash."""
    names: list[str]
    """Image names, as `frames_meta.json` holds them."""
    t_ns: list[int]
    """Each keyframe's sample index value on the manifest's timeline as int64 (`CatalogVideo.t_ns`); the frame is matched by it."""
    mask_entity: str | None = None
    """A `SegmentationImage` entity (no leading slash) on the same stamps, nonzero = drop keypoints there
    (`colsfm.keypoint_masks`); None = unmasked."""

    def __post_init__(self) -> None:
        """Check the two lists pair up."""
        if len(self.names) != len(self.t_ns):
            raise ValueError(f"{self.entity}: {len(self.names)} names for {len(self.t_ns)} timestamps")


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class FrameSourceManifest:
    """Where a catalog input's keyframes live: the segment's video streams, and which samples are keyframes."""

    catalog_url: str
    """The catalog server."""
    dataset: str
    """The catalog dataset."""
    segment: str
    """The segment id."""
    timeline: str
    """The index timeline the timestamps are on."""
    streams: list[StreamKeyframes]
    """One entry per camera."""


def write_manifest(path: Path, manifest: FrameSourceManifest) -> None:
    """Write a frame-source manifest as JSON.

    Args:
        path: Destination, normally `<input_dir>/frame_source.json`.
        manifest: The manifest.
    """
    path.write_text(to_json(manifest))


def read_manifest(path: Path) -> FrameSourceManifest:
    """Read a frame-source manifest.

    Args:
        path: The JSON file.

    Returns:
        The manifest.

    Raises:
        ValueError: When the file does not parse as a manifest, naming it.
    """
    try:
        return from_json(FrameSourceManifest, path.read_text())
    except (SerdeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: not a frame-source manifest ({error})") from error
