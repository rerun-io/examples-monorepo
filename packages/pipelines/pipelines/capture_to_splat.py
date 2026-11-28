"""The capture_to_splat flow, from a registered base: lens masks, then SLAM.

Each step is a stage package's pixi task, which reads its inputs from the catalog and writes, registers and records one
layer of the segment. The flow passes only the catalog URL, the segment and an output directory. The catalog dataset is
the segment's prefix (``insta360__<clip>``: dataset ``insta360``), which is also how slam-rs finds its rig.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

InputProfile: TypeAlias = Literal["insta360", "robocap"]
"""What the base came from. ``insta360``: two fisheye lenses, masked (lens masks, then masked SLAM). ``robocap``: the
cap's fisheye cameras, no masks (SLAM alone)."""


@dataclass(frozen=True, slots=True)
class Step:
    """One step: a stage's pixi task and the layer it writes."""

    name: str
    """Short name, for logs."""
    layer: str
    """The catalog layer the task registers, with its run record ``run_<layer>``."""
    environment: str
    """The pixi environment of the task."""
    task: str
    """The pixi task."""
    arguments: tuple[str, ...]
    """The task's arguments."""


def dataset_of(segment: str) -> str:
    """The catalog dataset of a segment: its prefix before ``__``."""
    return segment.split("__", 1)[0]


def flow_steps(profile: InputProfile, *, segment: str, catalog_url: str, output_root: Path) -> tuple[Step, ...]:
    """The steps of one run of the flow, in order.

    Args:
        profile: The input profile.
        segment: The base's segment id.
        catalog_url: The catalog holding the base.
        output_root: Where each step writes its layer file (``<output_root>/<layer>/<segment>.rrd``); the catalog server
            must read it at the same path.

    Returns:
        The steps.
    """
    root: Path = output_root.resolve()
    slam: tuple[str, ...] = ("--segment", segment, "--catalog", catalog_url, "--output-dir", str(root / "slam_rs"))
    if profile == "robocap":
        return (Step("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer", slam),)
    lens_masks: tuple[str, ...] = ("--dataset", dataset_of(segment), "--segment", segment, "--catalog-url", catalog_url, "--output-dir", str(root / "lens_mask"))
    return (
        Step("lens_masks", "lens_mask", "posekit", "posekit-lens-masks", lens_masks),
        # The lens masks are the cameras' people_mask entities in the lens_mask layer.
        Step("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer", (*slam, "--mask-entity", "people_mask")),
    )
