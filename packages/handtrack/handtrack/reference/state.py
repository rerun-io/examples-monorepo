"""Per-hand acquisition and track-end policy; the pose stage owns neural memory."""
from dataclasses import dataclass
from typing import Literal, TypeAlias

TrackEnd: TypeAlias = Literal["geometry", "detnet"]


@dataclass(slots=True)
class TrackState:
    policy: TrackEnd
    """Whether DetNet misses can also end a geometrically valid track."""
    miss_frames: int = 3
    """Consecutive all-view misses needed to end a track."""
    tracked: bool = False
    """Previous frame returned a pose."""
    misses: int = 0
    """Current consecutive all-view misses."""

    def __post_init__(self) -> None:
        if self.miss_frames < 1:
            raise ValueError("miss_frames must be positive")

    def choose(self, acquired: list[int], predicted: list[int], probability: list[float]) -> list[int]:
        """Choose acquisition or previous-pose views; an ended track waits until the next frame.

        Empty views must be passed to the pose stage. Upstream clears that hand's history
        after the frame (or calls reset_history when both hands have empty views).
        """
        if not self.tracked:
            return acquired
        self.misses = self.misses + 1 if all(probability[view] < 0.5 for view in predicted) else 0
        if not predicted or (self.policy == "detnet" and self.misses >= self.miss_frames):
            self.accept(False)
            return []
        return predicted

    def accept(self, posed: bool) -> None:
        """Record the pose stage's result, resetting the miss counter on loss."""
        self.tracked = posed
        if not posed:
            self.misses = 0
