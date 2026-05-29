"""Shared tracking helpers for Robocap cuVSLAM entrypoints."""


def bounded_frame_count(total_frames: int, max_frames: int | None) -> int:
    """Return the number of frames to process for a tracking run.

    Args:
        total_frames: Number of frames available in the dataset.
        max_frames: Optional upper bound for validation and reference runs.

    Returns:
        Number of frames to process.
    """
    if max_frames is None:
        return total_frames
    if max_frames <= 0:
        raise ValueError("max_frames must be positive when provided.")
    return min(total_frames, max_frames)
