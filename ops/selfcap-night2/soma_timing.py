"""Measure completed normal SOMA jobs, excluding work already in flight."""


def timing_window(events: list[tuple[float, float, str]], start: float, count: int = 20) -> dict:
    selected = sorted((row for row in events if row[0] - row[1] >= start), key=lambda row: row[0])[:count]
    return {
        "samples": len(selected),
        "target": count,
        "mean_seconds": sum(row[1] for row in selected) / len(selected) if selected else None,
        "complete": len(selected) == count,
        "started_at": start,
        "segments": [row[2] for row in selected],
    }
