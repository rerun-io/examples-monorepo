from __future__ import annotations

from simplecv.print_utils import format_bytes


def test_format_bytes_uses_binary_units() -> None:
    """Byte counts are formatted with binary units for benchmark output."""
    assert format_bytes(0) == "0.0 B"
    assert format_bytes(1023) == "1023.0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1024 * 1024) == "1.0 MiB"
