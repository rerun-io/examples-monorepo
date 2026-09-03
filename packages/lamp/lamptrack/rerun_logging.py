"""Rerun logging helpers shared by the LAMP replay and catalog tools."""

from collections.abc import Collection
from dataclasses import dataclass, field

import rerun as rr


@dataclass(slots=True)
class LivePeopleLogger:
    """Clear people from latest-at views exactly once when their tracks end."""

    _visible_track_ids: set[int] = field(default_factory=set, init=False)

    def update(self, live_track_ids: Collection[int]) -> None:
        """Record the current live set and recursively clear tracks that ended.

        Call this after setting the current Rerun time and before logging the
        current people.

        Args:
            live_track_ids: Track IDs with a renderable smoothed state at the
                current frameset.
        """
        live_ids = set(live_track_ids)
        for track_id in sorted(self._visible_track_ids - live_ids):
            rr.log(f"world/people/{track_id}", rr.Clear(recursive=True))
        self._visible_track_ids = live_ids


__all__ = ("LivePeopleLogger",)
