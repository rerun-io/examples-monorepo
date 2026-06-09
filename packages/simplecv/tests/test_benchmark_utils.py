from __future__ import annotations

from simplecv._benchmark_utils import TimingResult, build_timing_table, format_timing_table


def test_timing_table_prints_readable_relative_summary() -> None:
    """Timing rows are rendered as one readable comparison table."""
    torchcodec_result: TimingResult = TimingResult(
        label="TorchCodec cuda",
        elapsed_seconds=1.0,
        frames=800,
        detail="chunk_size=32, checksum=1",
    )
    existing_result: TimingResult = TimingResult(
        label="MultiVideoReader -> cuda",
        elapsed_seconds=8.0,
        frames=800,
        detail="RGB NCHW uint8 tensors, checksum=2",
    )

    table_text: str = format_timing_table(
        [torchcodec_result, existing_result],
        throughput_header="Camera FPS",
    )
    rich_table = build_timing_table(
        [torchcodec_result, existing_result],
        throughput_header="Camera FPS",
    )

    assert "Results" in table_text
    assert "Camera FPS" in table_text
    assert "TorchCodec cuda" in table_text
    assert "MultiVideoReader -> cuda" in table_text
    assert "0.12x" in table_text
    assert "RGB NCHW uint8 tensors" in table_text
    assert rich_table.title_style == "bold cyan"
    assert rich_table.header_style == "bold white"
    assert rich_table.columns[0].style == "cyan"
    assert rich_table.columns[2].style == "green"
