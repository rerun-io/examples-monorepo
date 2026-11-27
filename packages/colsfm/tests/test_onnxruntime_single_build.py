"""The installed env holds one ONNX Runtime build: COLMAP's onnxruntime-cpp and the Python bindings agree.

pycolmap's CUDA provider and the ``onnxruntime`` module's both resolve their provider bridge through
``libonnxruntime_providers_shared.so``, opened ``RTLD_GLOBAL``: whichever loads first serves the other. That is safe
only when the two are the same build, so this reads what pixi installed (``conda-meta``, the shared libraries) and never
imports ``onnxruntime``, which would itself load a bridge. ci/tests/test_one_onnxruntime.py checks the lock; this checks
the env that was installed from it.

The copies under ``lib/`` and ``onnxruntime/capi/`` are not byte-identical even from one build: conda rewrites their
RPATH and the prefix strings in ``.rodata``, and the libraries carry no build-id note. Their machine code (``.text``) is
what must match.
"""

import hashlib
import re
import struct
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from serde import serde
from serde.json import from_json

PREFIX: Path = Path(sys.prefix)
SITE_PACKAGES: Path = Path(sysconfig.get_paths()["purelib"])


@serde
@dataclass(frozen=True, slots=True)
class CondaRecord:
    """The fields of a ``conda-meta/<name>-<version>-<build>.json`` record this module compares."""

    version: str
    """Upstream version."""
    build_number: int
    """Feedstock build number."""
    build: str
    """Build string, e.g. ``py312h6654455_201_cuda130``."""
    files: list[str]
    """Files the package installed, relative to the prefix."""

    @property
    def variant(self) -> str:
        """``cpu`` or ``cuda<NNN>``: the build string's last token."""
        return self.build.rsplit("_", maxsplit=1)[-1]


def _conda_record(name: str) -> CondaRecord:
    records: list[Path] = [path for path in (PREFIX / "conda-meta").glob(f"{name}-*.json") if re.fullmatch(rf"{re.escape(name)}-[0-9][^-]*-[^-]+", path.stem)]
    assert len(records) == 1, f"expected one conda {name} in {PREFIX}, found {[path.name for path in records]}"
    return from_json(CondaRecord, records[0].read_text())


def _text_sha256(path: Path) -> str:
    """sha256 of the ``.text`` section of a 64-bit little-endian ELF file."""
    with path.open("rb") as elf:
        header: bytes = elf.read(64)
        assert header[:4] == b"\x7fELF" and header[4:6] == b"\x02\x01", f"{path} is not a 64-bit little-endian ELF file"
        section_offset, = struct.unpack_from("<Q", header, 0x28)
        entry_size, count, names_index = struct.unpack_from("<HHH", header, 0x3A)
        elf.seek(section_offset)
        table: bytes = elf.read(entry_size * count)
        sections: list[tuple[int, int, int]] = [
            (struct.unpack_from("<I", table, i * entry_size)[0], *struct.unpack_from("<QQ", table, i * entry_size + 0x18)) for i in range(count)
        ]
        elf.seek(sections[names_index][1])
        names: bytes = elf.read(sections[names_index][2])
        for name_offset, offset, size in sections:
            if names[name_offset : names.index(b"\0", name_offset)] == b".text":
                elf.seek(offset)
                return hashlib.sha256(elf.read(size)).hexdigest()
    raise AssertionError(f"{path} has no .text section")


def test_the_python_bindings_and_colmaps_library_are_one_build() -> None:
    bindings: CondaRecord = _conda_record("onnxruntime")
    library: CondaRecord = _conda_record("onnxruntime-cpp")
    assert (bindings.version, bindings.build_number, bindings.variant) == (library.version, library.build_number, library.variant)
    # The conda CUDA build ships an onnxruntime_gpu dist-info of its own (upstream's wheel name); a PyPI install adds one
    # that no conda package owns.
    owned: set[str] = {str(Path(file).parent) for file in bindings.files if ".dist-info/" in file}
    installed: set[str] = {str(path.relative_to(PREFIX)) for path in SITE_PACKAGES.glob("onnxruntime*.dist-info")}
    assert installed <= owned, f"ONNX Runtime metadata no conda package owns (a PyPI install): {sorted(installed - owned)}"


@pytest.mark.parametrize("provider", ["shared", "cuda"])
def test_every_copy_of_a_provider_library_holds_the_same_code(provider: Literal["shared", "cuda"]) -> None:
    name: str = f"libonnxruntime_providers_{provider}.so"
    copies: list[Path] = [path for path in (PREFIX / "lib" / name, SITE_PACKAGES / "onnxruntime" / "capi" / name) if path.is_file()]
    assert len(copies) == 2, f"expected the onnxruntime-cpp and the Python copies of {name}, found {copies}"
    digests: dict[str, str] = {str(path.relative_to(PREFIX)): _text_sha256(path) for path in copies}
    assert len(set(digests.values())) == 1, f"{name} copies hold different code: {digests}"
