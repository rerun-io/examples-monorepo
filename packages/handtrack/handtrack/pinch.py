"""Pinch clicks for a consumer (the Steam Frame's select gesture): a hysteresis state machine on a per-frame pinch distance.

The distance comes from the tracker: the fitted mesh's thumb-pad to index contact (``labels.pinch.contact_mm`` on the fitted pose),
optionally gated by KeyNet's pinch head (its probability averaged over the views it ran on). ``pinch_signal`` builds it,
``pinch_state`` runs the machine over a whole sequence and ``PinchDetector`` runs the same machine one frame at a time.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float32
from numpy import ndarray

OPEN_MM: float = 30.0
"""The distance a head-driven signal reports for 'not pinching' (well over any leave threshold)."""


@dataclass(frozen=True, slots=True)
class PinchConfig:
    """How a pinch is called."""

    source: Literal["fit", "head", "fit_and_head"] = "fit"
    """``fit``: the fitted contact distance; ``head``: the pinch head alone (p >= ``head_threshold`` reads 0 mm, else ``OPEN_MM``);
    ``fit_and_head``: the fitted contact where the head agrees (p >= ``head_threshold``), else ``OPEN_MM``."""
    enter_mm: float = 10.0
    """Enter a pinch after ``frames`` consecutive frames under this..."""
    leave_mm: float = 16.0
    """...and release after ``frames`` consecutive frames over this."""
    frames: int = 2
    hold: int = 3
    """An untracked stretch (NaN) keeps the state for up to this many frames, then releases."""
    head_threshold: float = 0.5


def pinch_signal(contact_mm: Float32[ndarray, "f"], head: Float32[ndarray, "f"] | None, config: PinchConfig) -> Float32[ndarray, "f"]:
    """The per-frame distance the state machine reads (NaN = untracked). ``contact_mm`` is NaN where the hand is untracked; ``head``
    is NaN where KeyNet did not run."""
    if config.source == "fit":
        return contact_mm.astype(np.float32)
    if head is None:
        raise ValueError(f"pinch source {config.source!r} needs the pinch head's probabilities")
    agrees: Bool[ndarray, "f"] = np.nan_to_num(head, nan=0.0) >= config.head_threshold
    if config.source == "head":
        return np.where(np.isfinite(contact_mm) & np.isfinite(head), np.where(agrees, 0.0, OPEN_MM), np.nan).astype(np.float32)
    return np.where(np.isfinite(contact_mm), np.where(agrees, contact_mm, OPEN_MM), np.nan).astype(np.float32)


def pinch_state(distance: np.ndarray, enter: float = 10.0, leave: float = 16.0, frames: int = 2, hold: int = 3) -> np.ndarray:
    """The pinch state machine on a per-frame distance (NaN = untracked): enter after ``frames`` frames under ``enter``, release after
    ``frames`` frames over ``leave``; an untracked stretch holds the state for up to ``hold`` frames, then releases."""
    detector: PinchDetector = PinchDetector(PinchConfig(enter_mm=enter, leave_mm=leave, frames=frames, hold=hold))
    return np.array([detector.update(float(d)) for d in distance], dtype=bool)


DEFAULT_PINCH_CONFIG: PinchConfig = PinchConfig()


@dataclass(frozen=True, slots=True)
class PinchEvent:
    pinched: bool
    onset: bool
    """This frame starts a pinch: the click."""
    release: bool


class PinchDetector:
    """``pinch_state`` one frame at a time, for a live consumer."""

    def __init__(self, config: PinchConfig = DEFAULT_PINCH_CONFIG) -> None:
        self.config: PinchConfig = config
        self.on: bool = False
        self._below: int = 0
        self._above: int = 0
        self._missing: int = 0
        self._last: PinchEvent = PinchEvent(False, False, False)

    def update(self, distance: float) -> bool:
        """Feed one frame's distance (NaN = untracked); returns the state after it."""
        was: bool = self.on
        if not np.isfinite(distance):
            self._missing += 1
            if self._missing > self.config.hold:
                self.on = False
            self._below = self._above = 0
        else:
            self._missing = 0
            self._below = self._below + 1 if distance < self.config.enter_mm else 0
            self._above = self._above + 1 if distance > self.config.leave_mm else 0
            if not self.on and self._below >= self.config.frames:
                self.on = True
            elif self.on and self._above >= self.config.frames:
                self.on = False
        self._last = PinchEvent(self.on, self.on and not was, was and not self.on)
        return self.on

    def step(self, contact_mm: float, head: float | None = None) -> PinchEvent:
        """Feed one frame's fitted contact (NaN = untracked) and head probability (None or NaN = not run); returns the event."""
        signal: Float32[ndarray, "1"] = pinch_signal(np.array([contact_mm], dtype=np.float32),
                                                     None if head is None else np.array([head], dtype=np.float32), self.config)
        self.update(float(signal[0]))
        return self._last
