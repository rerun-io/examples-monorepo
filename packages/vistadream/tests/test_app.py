"""The generated gradio form (unit lane, no GPU: ``build`` never runs): every Config field has its widget or is hidden on
purpose, and the example is the VERIFY short case. dagkit's own tests own the form round trip."""

import dataclasses
from typing import get_args

import gradio as gr
from dagkit.testing import check_form
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.apis.single_image import PACKAGE_DIR, Config, build, short_case
from vistadream.types import Stage

RR = RerunTyroConfig(headless=True)
WIDGETS: dict[str, str] = {
    "image_path": "File",
    "stage": "Dropdown",
    "max_resolution": "Dropdown",
    "expansion_percent": "Number",
    "n_frames": "Number",
    "output_ply": "Checkbox",  # "save it": the run writes a temporary PLY and offers it as a download
    "seed": "Number",
    "flux.offload": "Checkbox",
    "flux.num_steps": "Number",
    "flux.guidance": "Number",
    "flux.seed": "Number",
    "flux.local_files_only": "Checkbox",
    "runner": "Dropdown",
    "mode": "Dropdown",
}
HIDDEN: dict[str, str] = {"rr_config": "the app renders into its own Rerun pane: no viewer, save or connect flags"}


def test_every_config_field_has_its_widget_or_is_hidden() -> None:
    demo = check_form(Config, build, WIDGETS, HIDDEN, examples=[short_case(RR)])
    shown: dict[str, gr.components.Component] = {str(b.label): b for b in demo.blocks.values() if isinstance(b, gr.components.Component) and b.label}
    stage, max_resolution = shown["stage"], shown["max_resolution"]
    assert isinstance(stage, gr.Dropdown) and isinstance(max_resolution, gr.Dropdown)
    assert [value for _, value in stage.choices] == list(get_args(Stage)) and [value for _, value in max_resolution.choices] == [512, 1024, 1920]
    [examples] = [b for b in demo.blocks.values() if isinstance(b, gr.Dataset)]
    assert examples.samples[0][:3] == ["IMG_4029.jpg", "no-outpaint", 512]


def test_the_example_is_the_verify_short_case() -> None:
    case: Config = short_case(RR)
    assert (case.image_path, case.stage, case.max_resolution, case.seed) == (PACKAGE_DIR / "data/office/IMG_4029.jpg", "no-outpaint", 512, 0)
    assert dataclasses.replace(case, rr_config=RR) == Config(RR)  # the defaults are the short case: a Run without the example works too
