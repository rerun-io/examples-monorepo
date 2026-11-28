"""The capture_to_splat flow, from a registered base: masks, SLAM, then colsfm.

Each step is a stage package's pixi task, which reads its inputs from the catalog and writes, registers and records one
layer of the segment. The flow passes only the catalog URL, the segment and an output directory. The catalog dataset is
the segment's prefix (``insta360__<clip>``: dataset ``insta360``), which is also how slam-rs finds its rig.
"""

from pathlib import Path
from typing import Literal, TypeAlias

from pipelines.step import Step

InputProfile: TypeAlias = Literal["insta360", "robocap"]
"""What the base came from. ``insta360``: two fisheye lenses and their cube faces, masked (lens masks, masked SLAM, face
masks, colsfm on the six faces). ``robocap``: the cap's fisheye cameras, no masks (SLAM, colsfm on four cameras)."""


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
    dataset: str = dataset_of(segment)
    slam: tuple[str, ...] = ("--segment", segment, "--catalog", catalog_url, "--output-dir", str(root / "slam_rs"))
    # colsfm reads the slam_rs layer as its prior; its run directory and inputs stay beside its layer.
    colsfm: tuple[str, ...] = (
        "--source.dataset", dataset, "--source.segment", segment, "--source.catalog-url", catalog_url,
        "--source.output-root", str(root / "colsfm" / "inputs"), "--options.output-dir", str(root / "colsfm" / "run"),
        "--layer-path", str(root / "colsfm" / f"{segment}.rrd"),
    )
    if profile == "robocap":
        return (
            Step("slam", "slam_rs", dataset, segment, "slam-rs-cuda", "slam-rs-catalog-layer", slam),
            # The colsfm-robocap task's camera rig: the four fisheyes, the front pair stereo, every 4th frameset.
            Step("colsfm", "colsfm", dataset, segment, "colsfm", "colsfm-catalog", (
                *colsfm, "--source.cameras", "left_front", "right_front", "left", "right", "--source.stereo-pairs", "left_front", "right_front",
                "--source.frame-stride", "4", "--source.trim-start-seconds", "2", "--source.trim-end-seconds", "2",
            )),
        )
    lens_masks: tuple[str, ...] = ("--dataset", dataset, "--segment", segment, "--catalog-url", catalog_url, "--output-dir", str(root / "lens_mask"))
    return (
        Step("lens_masks", "lens_mask", dataset, segment, "posekit", "posekit-lens-masks", lens_masks),
        # The lens masks are the cameras' people_mask entities in the lens_mask layer; the face masks, the faces' in masks.
        Step("slam", "slam_rs", dataset, segment, "slam-rs-cuda", "slam-rs-catalog-layer", (*slam, "--mask-entity", "people_mask")),
        Step("face_masks", "masks", dataset, segment, "dataforge-cuda", "dataforge-insta360-face-masks",
             ("--segment", segment, "--catalog-url", catalog_url, "--output-dir", str(root / "masks"))),
        Step("colsfm", "colsfm", dataset, segment, "colsfm", "colsfm-catalog", (
            *colsfm, "--source.cameras", "cube6_front0", "cube6_front1", "cube6_left", "cube6_right", "cube6_up", "cube6_down",
            "--source.mask-entity", "people_mask", "--source.frame-stride", "1",
            "--options.config-dir", "colsfm/resources/configs/insta360-cube", "--options.align-to-prior",
        )),
    )
