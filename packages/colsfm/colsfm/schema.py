"""Runtime protobuf message classes for cuSFM's own schema.

cuSFM ships no `.proto` sources, only prebuilt binaries that embed their
`FileDescriptorProto`s in a `protodesc_cold` ELF section. `tools/extract_cusfm_schema.py`
pulls those out into `data/cusfm_schema/cusfm_protos.fdset`; this module turns
that vendored `FileDescriptorSet` into real generated-message classes at import
time. Everything downstream — `frames_meta.json` parsing, `.pb.txt` config
parsing, keyframe and map I/O — then goes through protobuf's own JSON and text
codecs instead of hand-rolled readers, which is what makes the proto3 defaults
(a missing `camera_params_id` means 0) come out right for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Protocol, TypeAlias, runtime_checkable

from google.protobuf import descriptor, descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

from colsfm import RESOURCES_DIR

MessageClass: TypeAlias = type[Message]
"""A protobuf message class built at runtime from a descriptor."""


@runtime_checkable
class DescriptorPoolLike(Protocol):
    """The part of `DescriptorPool` this module uses.

    Annotated structurally rather than nominally because protobuf's C++/upb
    backend returns a `google._upb._message.DescriptorPool`, which is not a
    subclass of the pure-Python `descriptor_pool.DescriptorPool`.
    """

    def Add(self, file_desc_proto: descriptor_pb2.FileDescriptorProto) -> object:
        """Add one file descriptor to the pool."""
        ...

    def FindMessageTypeByName(self, full_name: str) -> descriptor.Descriptor:
        """Look up a message descriptor by fully qualified name."""
        ...

    def FindEnumTypeByName(self, full_name: str) -> descriptor.EnumDescriptor:
        """Look up an enum descriptor by fully qualified name."""
        ...


DEFAULT_DESCRIPTOR_SET_PATH: Final[Path] = RESOURCES_DIR / "schema" / "cusfm_protos.fdset"
"""Vendored `FileDescriptorSet` covering all 21 cuSFM binaries."""

KEYFRAME: Final[str] = "protos.visual.general.Keyframe"
"""One image's keypoints and descriptors."""

KEYFRAMES_METADATA_COLLECTION: Final[str] = "protos.visual.general.KeyframesMetadataCollection"
"""The `frames_meta.json` root message: poses, calibration, rig extrinsics."""

MAP_KEYPOINTS: Final[str] = "protos.visual.cusfm.MapKeyPoints"
"""The mapper's output: 3-D points, observations and the keyframes themselves."""


@dataclass(frozen=True, slots=True)
class CusfmSchema:
    """cuSFM's protobuf schema, loaded into a private descriptor pool."""

    pool: DescriptorPoolLike
    """Pool holding every proto file from the vendored descriptor set."""
    proto_file_names: tuple[str, ...]
    """Proto file names in the set, in dependency order."""

    def message_class(self, full_name: str) -> MessageClass:
        """Build the message class for a fully qualified protobuf message name.

        Args:
            full_name: Name such as `protos.visual.general.KeyframesMetadataCollection`.

        Returns:
            A generated message class usable with `json_format` and `text_format`.

        Raises:
            KeyError: When the schema holds no such message.
        """
        return message_factory.GetMessageClass(self.pool.FindMessageTypeByName(full_name))

    def enum_value_name(self, full_name: str, number: int) -> str:
        """Look up an enum value's name.

        Args:
            full_name: Fully qualified enum type name.
            number: Enum value number.

        Returns:
            The symbolic name, e.g. `PINHOLE`.

        Raises:
            KeyError: When the schema holds no such enum type or value.
        """
        return self.pool.FindEnumTypeByName(full_name).values_by_number[number].name


def build_schema(descriptor_set_path: Path) -> CusfmSchema:
    """Load a `FileDescriptorSet` into a fresh descriptor pool.

    A private pool (rather than the default one) keeps cuSFM's message names out
    of the process-wide registry, so importing this alongside any other protobuf
    schema cannot collide.

    The set is added in file order and every failure propagates. `tools/extract_cusfm_schema.py`
    already emits the files in dependency order, so a file whose imports do not resolve is a
    corrupt or stale descriptor set rather than an ordering problem — the retry-to-fixpoint
    loop this replaced could only turn that into a confusing `ValueError` several seconds later.

    Args:
        descriptor_set_path: Serialised `FileDescriptorSet` on disk.

    Returns:
        The loaded schema.

    Raises:
        FileNotFoundError: When the descriptor set is missing.
        TypeError: When a file's imports are not already in the pool, i.e. the set
            is not in dependency order.
    """
    if not descriptor_set_path.is_file():
        raise FileNotFoundError(
            f"No cuSFM descriptor set at {descriptor_set_path}; regenerate it with `python tools/extract_cusfm_schema.py`"
        )
    file_set: descriptor_pb2.FileDescriptorSet = descriptor_pb2.FileDescriptorSet.FromString(descriptor_set_path.read_bytes())
    pool: DescriptorPoolLike = descriptor_pool.DescriptorPool()
    for file_descriptor in file_set.file:
        pool.Add(file_descriptor)
    return CusfmSchema(pool=pool, proto_file_names=tuple(file_descriptor.name for file_descriptor in file_set.file))


@cache
def load_schema(descriptor_set_path: Path = DEFAULT_DESCRIPTOR_SET_PATH) -> CusfmSchema:
    """Load the vendored cuSFM schema, caching it per descriptor-set path.

    Args:
        descriptor_set_path: Serialised `FileDescriptorSet` on disk.

    Returns:
        The loaded schema.
    """
    return build_schema(descriptor_set_path)
