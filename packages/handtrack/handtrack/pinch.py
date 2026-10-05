"""Pinch clicks for a consumer (the Steam Frame's select gesture): a hysteresis state machine on a per-frame pinch distance.

The distance comes from the tracker: the fitted mesh's thumb-pad to index contact (``labels.pinch.contact_mm`` on the fitted pose),
optionally gated by KeyNet's pinch head (its probability averaged over the views it ran on). ``pinch_signal`` builds it,
``pinch_state`` runs the machine over a whole sequence and ``PinchDetector`` runs the same machine one frame at a time.

With ``veto_threshold`` the head vetoes whole pinches instead: the machine proposes each pinch and the head's mean probability over
its first frames keeps or drops it. On HOT3D object grasps this cut the fit's false clicks from 8.7 to 6.3 per minute with the
UmeTrack-only head, where frame-by-frame gating added clicks (each dip of the head splits a pinch).
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
    veto_threshold: float | None = None
    """With a head: a pinch the machine enters is reported only if the head's mean probability over its first ``veto_window`` frames
    is >= this (the click waits ``veto_window - 1`` frames); otherwise the whole pinch is dropped. A pinch shorter than the window never
    clicks. None: no veto."""
    veto_window: int = 3


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


def pinch_states(contact_mm: Float32[ndarray, "f"], head: Float32[ndarray, "f"] | None, config: PinchConfig) -> Bool[ndarray, "f"]:
    """``PinchDetector.step`` over a sequence: the reported pinch state per frame under ``config`` (source, thresholds, veto)."""
    detector: PinchDetector = PinchDetector(config)
    heads: Float32[ndarray, "f"] = np.full(len(contact_mm), np.nan, dtype=np.float32) if head is None else head
    return np.array([detector.step(float(c), None if head is None else float(h)).pinched for c, h in zip(contact_mm, heads, strict=True)], dtype=bool)


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
        """The reported state."""
        self.raw: bool = False
        """The machine's state before the head's veto."""
        self._votes: list[float] = []
        self._vetoed: bool = False
        self._below: int = 0
        self._above: int = 0
        self._missing: int = 0
        self._last: PinchEvent = PinchEvent(False, False, False)

    def update(self, distance: float, head: float = float("nan")) -> bool:
        """Feed one frame's distance (NaN = untracked) and head probability (NaN = not run; read only by the veto); returns the reported state."""
        was: bool = self.on
        if not np.isfinite(distance):
            self._missing += 1
            if self._missing > self.config.hold:
                self.raw = False
            self._below = self._above = 0
        else:
            self._missing = 0
            self._below = self._below + 1 if distance < self.config.enter_mm else 0
            self._above = self._above + 1 if distance > self.config.leave_mm else 0
            if not self.raw and self._below >= self.config.frames:
                self.raw = True
            elif self.raw and self._above >= self.config.frames:
                self.raw = False
        if self.config.veto_threshold is None:
            self.on = self.raw
        elif not self.raw:
            self.on, self._votes, self._vetoed = False, [], False
        elif not self.on and not self._vetoed:
            self._votes.append(head)
            if len(self._votes) >= self.config.veto_window:
                finite: list[float] = [vote for vote in self._votes if np.isfinite(vote)]
                self.on = bool(finite) and float(np.mean(finite)) >= self.config.veto_threshold
                self._vetoed = not self.on
        self._last = PinchEvent(self.on, self.on and not was, was and not self.on)
        return self.on

    def step(self, contact_mm: float, head: float | None = None) -> PinchEvent:
        """Feed one frame's fitted contact (NaN = untracked) and head probability (None or NaN = not run); returns the event."""
        signal: Float32[ndarray, "1"] = pinch_signal(np.array([contact_mm], dtype=np.float32),
                                                     None if head is None else np.array([head], dtype=np.float32), self.config)
        self.update(float(signal[0]), float("nan") if head is None else head)
        return self._last
