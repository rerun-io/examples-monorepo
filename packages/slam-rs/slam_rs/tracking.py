"""Driving the estimator over a frameset feed in lockstep, with the D17 hold.

Offline mode consumes one frameset at a time in the calling thread, and a
frameset the estimator refuses for want of inertial samples is **held, not
dropped** (D17). That rule is the contract, not a detail of one caller: the
replay tool draws a Rerun rung from what tracked and the V2 gate asserts numbers
on it, and both have to hold and retry identically or the gate stops measuring
the tool. It therefore lives here once, and :class:`Lockstep` is what both
drive.
"""

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from slam_rs import _core
from slam_rs.catalog_feed import Frameset

MAX_HELD_FRAMESETS: int = 2
"""Most framesets the hold may ever carry: one refused, plus the one whose samples unblock it."""


@dataclass(slots=True)
class Lockstep:
    """One estimator, driven frameset by frameset, holding a refused frameset.

    Everything a driver needs to report afterwards is on this value: what was
    pushed, what tracked and how long it took, how many retries it cost, and
    what is still held.
    """

    vio: _core.Vio
    """The estimator every frameset and every inertial sample goes through."""
    pending: list[Frameset] = field(default_factory=list)
    """Framesets refused for want of IMU, oldest first, waiting for the samples that cover them."""
    imu_samples: int = 0
    """Inertial samples pushed so far."""
    statuses: dict[str, int] = field(default_factory=dict)
    """How many times ``track`` answered each status, retries included."""
    elapsed_ms: list[float] = field(default_factory=list)
    """Wall time each ``track`` call that tracked took, in the order they tracked."""

    def push(self, frameset: Frameset) -> Iterator[tuple[Frameset, _core.VioResult]]:
        """Push the frameset's inertial samples, then yield everything they now cover.

        A frameset the estimator refuses is **held, not dropped** (D17): the next
        frameset's batch runs one sample past its own frame time and therefore
        past this one's, so the held frameset tracks then — and before the
        frameset whose samples unblocked it, because time order is the
        trajectory. Nothing moved on the refusal, so the retry produces the pose
        a run that had the samples all along would have produced.

        Args:
            frameset: The frameset to track, with the samples since the previous one.

        Yields:
            Each frameset that tracked and what ``track`` returned for it, in
            trajectory order. A refused frameset yields nothing until its samples
            arrive.

        Raises:
            ValueError: If the hold ever carries more than
                :data:`MAX_HELD_FRAMESETS` framesets, which the feed's one-sample
                lead makes impossible.
        """
        if len(frameset.imu):
            # The feed already hands over the samples since the previous frameset,
            # running one past this frame time: a backend that integrates up to the
            # frame and blocks until it can deadlocks on the first frameset otherwise.
            self.vio.push_imu_batch(
                frameset.imu.t_ns,
                np.ascontiguousarray(frameset.imu.gyro_rad_s),
                np.ascontiguousarray(frameset.imu.accel_m_s2),
            )
            self.imu_samples += len(frameset.imu)
        self.pending.append(frameset)
        if len(self.pending) > MAX_HELD_FRAMESETS:
            # Every batch runs one sample past its own frame time and therefore
            # past the previous frameset's, so a second refusal in a row cannot
            # happen; a deeper hold would silently retain whole decoded framesets
            # (~1.8 MB each) until the end of the segment.
            raise ValueError(
                f"frameset {frameset.t_ns} takes the hold to {len(self.pending)} framesets, past the {MAX_HELD_FRAMESETS} "
                f"the feed's one-sample lead allows; held {[held.t_ns for held in self.pending]}"
            )
        while self.pending:
            held: Frameset = self.pending[0]
            started: float = time.monotonic()
            result: _core.VioResult = self.vio.track(held.t_ns, held.images)
            elapsed_ms: float = 1e3 * (time.monotonic() - started)
            # A PyO3 enum has no ``name`` and is unhashable (see ``_core.pyi``), so the
            # repr is both the only name it has and the only thing that keys a dict.
            status_name: str = str(result.status)
            self.statuses[status_name] = self.statuses.get(status_name, 0) + 1
            if result.status != _core.VioStatus.Tracking:
                return
            self.pending.pop(0)
            self.elapsed_ms.append(elapsed_ms)
            yield held, result
