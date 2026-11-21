"""Packet views of a ``VideoStream:sample`` column: one zero-copy view per packet, across chunks and slices."""

import pyarrow as pa

from simplecv.catalog_video import packet_views


def test_packet_views_follow_chunks_and_slices() -> None:
    packets: list[bytes] = [b"\x00", b"\x01\x02", b"", b"\x03\x04\x05", b"\x06", b"\x07\x08"]
    rows = pa.array([[p] for p in packets], type=pa.list_(pa.list_(pa.uint8())))
    # A sliced first chunk (offsets that do not start at zero) and a second chunk of its own.
    column = pa.chunked_array([rows.slice(1, 3), rows.slice(4)])
    assert [bytes(view) for view in packet_views(column)] == packets[1:4] + packets[4:]

