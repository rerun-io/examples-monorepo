"""Insta360 ``.insv`` clips: the reader, the calibrated rig and the CPU panorama stitch.

``InsvReader`` reads a clip as Rerun chunks, in the layout of ``rerun.experimental.Mp4Reader``, from ``InsvRig``
(the clip with its device profile applied, in Rust). ``InsvClip`` is the clip as the file states it, without a
profile. ``PanoramaStitch`` turns the two lens images of a frameset into the viewing panorama.
"""

from insta360_io._insta360_io import ImuNoise, InsvClip, InsvLens, InsvRig, MeiCamera, PanoramaStitch
from insta360_io.reader import InsvReader

__all__ = ["ImuNoise", "InsvClip", "InsvLens", "InsvReader", "InsvRig", "MeiCamera", "PanoramaStitch"]
