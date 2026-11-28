"""The DiT checkpoint is read into memory by the prefetch, not left as lazy file-backed pages (CPU)."""

import json
import struct
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from fdanyone.errors import AssetError
from fdanyone.model.loader import read_safetensors


def _tensors() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "blocks.0.weight": torch.randn(64, 48, generator=generator).to(torch.bfloat16),
        "blocks.0.bias": torch.randn(64, generator=generator).to(torch.bfloat16),
        "pose_encoder.conv.weight": torch.randn(8, 3, 3, 3, generator=generator).to(torch.bfloat16),
        "scale": torch.randn(1, 64, generator=generator),
        "half": torch.randn(5, generator=generator).to(torch.float16),
    }


def test_read_safetensors_matches_safetensors_and_filters_prefixes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file(_tensors(), str(path), metadata={"kind": "test"})
    expected = load_file(str(path))

    tensors, metadata = read_safetensors(path, exclude_prefixes=("pose_encoder.",), read_threads=3, chunk_bytes=1000)

    assert metadata == {"kind": "test"}
    assert sorted(tensors) == sorted(key for key in expected if not key.startswith("pose_encoder."))
    for key, tensor in tensors.items():
        assert tensor.dtype == expected[key].dtype
        assert tensor.shape == expected[key].shape
        assert torch.equal(tensor, expected[key])


def test_read_safetensors_owns_its_bytes(tmp_path: Path) -> None:
    """A later change to the file must not reach the loaded tensors: the bytes were read, not mapped."""

    path = tmp_path / "model.safetensors"
    expected = _tensors()
    save_file(expected, str(path))
    tensors, _ = read_safetensors(path)

    size = path.stat().st_size
    with path.open("r+b") as handle:
        handle.seek(size // 2)
        handle.write(bytes(size - size // 2))

    for key, tensor in tensors.items():
        assert torch.equal(tensor, expected[key])


def test_read_safetensors_rejects_a_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file(_tensors(), str(path))
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="truncated"):
        read_safetensors(path)


def test_read_safetensors_names_a_dtype_it_does_not_map(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file({"blocks.0.int16": torch.arange(6, dtype=torch.int16)}, str(path))
    with pytest.raises(AssetError, match="I16"):
        read_safetensors(path)


def test_read_safetensors_rejects_overlapping_tensors(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    save_file({"a": torch.zeros(4), "b": torch.ones(4)}, str(path))
    raw = path.read_bytes()
    (header_size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + header_size])
    header["b"]["data_offsets"] = header["a"]["data_offsets"]  # b now aliases a's bytes
    encoded = json.dumps(header, separators=(",", ":")).encode().ljust(header_size)
    assert len(encoded) == header_size
    path.write_bytes(raw[:8] + encoded + raw[8 + header_size :])
    with pytest.raises(AssetError, match="invalid layout"):
        read_safetensors(path)
