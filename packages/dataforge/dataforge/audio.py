"""A recording's sound track, read from its mp4: the samples as FLAC and their band loudness.

FLAC, not the source codec: the 0.41 viewer plays FLAC but cannot decode the Opus that phones record, and FLAC is
lossless over the decoded samples. The loudness curve makes a sound visible on the timeline, where a tap that several
microphones heard lines up once their clocks agree.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from scipy.signal import butter, sosfiltfilt

LOUDNESS_RATE_HZ: int = 16_000
"""Rate the loudness is measured at; the band below stays well under its Nyquist frequency."""
LOUDNESS_HOP: int = 80
"""Samples per loudness value at ``LOUDNESS_RATE_HZ``: one value every 5 ms."""
LOUDNESS_BAND_HZ: tuple[float, float] = (300.0, 6000.0)
"""Pass band: taps, knocks and speech, without hum and rumble."""
SILENCE_RMS: float = 1e-5
"""RMS floor (-100 dBFS, full scale 1.0): digital silence reads as this, not as minus infinity."""


@dataclass(frozen=True, slots=True)
class AudioTrack:
    """One mp4's first audio stream, decoded once."""

    start_ns: int
    """Presentation time of the first sample, including the container's edit-list start."""
    flac: bytes
    """Every decoded sample, FLAC-encoded at the source rate and channel layout."""
    loudness_db: Float64[ndarray, "n"]
    """Band level every ``LOUDNESS_HOP`` samples, in dB over the track's median level (never negative)."""

    def loudness_times_ns(self) -> Int64[ndarray, "n"]:
        """Presentation time of each loudness value: the centre of its 5 ms window."""
        step_ns: float = LOUDNESS_HOP / LOUDNESS_RATE_HZ * 1e9
        return self.start_ns + np.rint(np.arange(self.loudness_db.size) * step_ns).astype(np.int64)


def band_loudness_db(samples: Float32[ndarray, "n"]) -> Float64[ndarray, "m"]:
    """Band level of mono ``LOUDNESS_RATE_HZ`` samples, in dB over its own median, one value per ``LOUDNESS_HOP``."""
    if samples.size < LOUDNESS_HOP:
        return np.zeros(0)  # shorter than one window; the filter needs more samples than its padding
    sos = butter(4, LOUDNESS_BAND_HZ, btype="band", fs=LOUDNESS_RATE_HZ, output="sos")
    band: Float64[ndarray, "n"] = sosfiltfilt(sos, samples.astype(np.float64))
    power: Float64[ndarray, "n"] = np.convolve(band**2, np.ones(LOUDNESS_HOP) / LOUDNESS_HOP, mode="same")[::LOUDNESS_HOP]
    level: Float64[ndarray, "m"] = 20.0 * np.log10(np.maximum(np.sqrt(power), SILENCE_RMS))
    return np.clip(level - np.median(level), 0.0, None)


def read_audio_track(video_path: Path) -> AudioTrack | None:
    """Decode an mp4's first audio stream into FLAC bytes and its band loudness; None when it has no audio.

    Args:
        video_path: The mp4 to read.

    Returns:
        The decoded track, or None for a file without an audio stream.
    """
    with av.open(str(video_path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        start_ns: int = 0
        if stream.start_time is not None and stream.time_base is not None:
            start_ns = round(stream.start_time * Fraction(stream.time_base.numerator, stream.time_base.denominator) * 1_000_000_000)
        layout: str = stream.layout.name
        to_flac = av.AudioResampler(format="s16", layout=layout, rate=stream.rate)
        to_level = av.AudioResampler(format="flt", layout="mono", rate=LOUDNESS_RATE_HZ)
        buffer = io.BytesIO()
        levels: list[Float32[ndarray, "k"]] = []
        with av.open(buffer, "w", format="flac") as output:
            flac: av.audio.stream.AudioStream = cast(av.audio.stream.AudioStream, output.add_stream("flac", rate=stream.rate, layout=layout))
            for frame in itertools.chain(container.decode(stream), (None,)):  # None flushes the resamplers
                for converted in to_flac.resample(frame):
                    output.mux(flac.encode(converted))
                levels.extend(converted.to_ndarray().reshape(-1) for converted in to_level.resample(frame))
            output.mux(flac.encode(None))
    samples: Float32[ndarray, "n"] = np.concatenate(levels) if levels else np.zeros(0, dtype=np.float32)
    return AudioTrack(start_ns=start_ns, flac=buffer.getvalue(), loudness_db=band_loudness_db(samples))
