"""The generated gradio form (unit lane, no GPU: ``build`` never runs): every Config field has its widget or is hidden on
purpose, the example is the VERIFY short case, and widget values come back as the typed Config the CLI would build."""

import dataclasses
import tempfile
from pathlib import Path
from typing import get_args

import gradio as gr
from dagkit.app import _assemble, _example, _leaves, _value, _widgets, gradio_app
from dagkit.node import Mode
from hypothesis import given, settings
from hypothesis import strategies as st
from simplecv.rerun_log_utils import RerunTyroConfig

from vistadream.apis.single_image import PACKAGE_DIR, Config, build, short_case
from vistadream.models.flux import FluxConfig
from vistadream.types import Stage

RR = RerunTyroConfig(headless=True)
WIDGETS: dict[str, type] = {
    "image_path": gr.File,
    "stage": gr.Dropdown,
    "max_resolution": gr.Dropdown,
    "expansion_percent": gr.Number,
    "n_frames": gr.Number,
    "output_ply": gr.Checkbox,  # "save it": the run writes a temporary PLY and offers it as a download
    "seed": gr.Number,
    "flux.offload": gr.Checkbox,
    "flux.num_steps": gr.Number,
    "flux.guidance": gr.Number,
    "flux.seed": gr.Number,
    "flux.local_files_only": gr.Checkbox,
    "runner": gr.Dropdown,
    "mode": gr.Dropdown,
}
HIDDEN: set[str] = {"rr_config"}  # the app renders into its own Rerun pane: no viewer, save or connect flags


def fields(cls: type, prefix: str = "") -> list[str]:
    """Every Config field as a dotted path, nested dataclasses opened up (the hidden viewer config stays one field)."""
    out: list[str] = []
    for f in dataclasses.fields(cls):
        path: str = prefix + f.name
        nested: bool = isinstance(f.type, type) and dataclasses.is_dataclass(f.type) and path not in HIDDEN
        out += fields(f.type, path + ".") if nested else [path]  # type: ignore[arg-type]
    return out


def test_every_config_field_has_its_widget_or_is_hidden() -> None:
    assert set(fields(Config)) == set(WIDGETS) | HIDDEN, "a new Config field needs a row in WIDGETS (or HIDDEN)"
    demo = gradio_app(Config, build, title="VistaDream single image", examples=[short_case(RR)])
    components: list[gr.components.Component] = [b for b in demo.blocks.values() if isinstance(b, gr.components.Component) and b.label]
    shown: dict[str, gr.components.Component] = {str(b.label).split(":")[0]: b for b in components}
    assert {name: type(shown.get(name)) for name in WIDGETS} == WIDGETS
    stage, max_resolution = shown["stage"], shown["max_resolution"]
    assert isinstance(stage, gr.Dropdown) and isinstance(max_resolution, gr.Dropdown)
    assert [value for _, value in stage.choices] == list(get_args(Stage)) and [value for _, value in max_resolution.choices] == [512, 1024, 1920]
    [examples] = [b for b in demo.blocks.values() if isinstance(b, gr.Dataset)]
    assert examples.samples[0][:3] == ["IMG_4029.jpg", "no-outpaint", 512]


def test_the_example_is_the_verify_short_case() -> None:
    case: Config = short_case(RR)
    assert (case.image_path, case.stage, case.max_resolution, case.seed) == (PACKAGE_DIR / "data/office/IMG_4029.jpg", "no-outpaint", 512, 0)
    assert dataclasses.replace(case, rr_config=RR) == Config(RR)  # the defaults are the short case: a Run without the example works too


@settings(max_examples=40, deadline=None)
@given(
    stage=st.sampled_from(get_args(Stage)),
    max_resolution=st.sampled_from([512, 1024, 1920]),
    expansion_percent=st.floats(0.0, 1.0),
    n_frames=st.integers(2, 40),
    save=st.booleans(),
    seed=st.integers(0, 2**31 - 1),
    flux=st.builds(FluxConfig, num_steps=st.integers(1, 50), guidance=st.floats(0.0, 50.0), seed=st.integers(0, 2**31 - 1)),
    runner=st.sampled_from(["sequential", "threaded"]),
    mode=st.sampled_from(get_args(Mode)),
)
def test_the_form_gives_back_the_config_it_shows(
    stage: Stage, max_resolution: int, expansion_percent: float, n_frames: int, save: bool, seed: int, flux: FluxConfig, runner: str, mode: Mode
) -> None:
    """Config -> widget values (``gr.Examples``' path) -> the Config the Run button builds (``_value``, ``_assemble``)."""
    shown = Config(RR, PACKAGE_DIR / "data/office/IMG_4029.jpg", stage, max_resolution, expansion_percent, n_frames, Path("x.ply") if save else None, seed, flux, runner, mode)  # type: ignore[arg-type]
    scratch = Path(tempfile.gettempdir())
    given_: dict[tuple[str, ...], object] = {}
    for leaf in _leaves(Config, None):
        if n := len(_widgets(leaf)):
            value: object = _value(leaf, _example(leaf, shown)[:n], scratch)
            given_ |= {} if value is dataclasses.MISSING else {leaf[0]: value}
    back: Config = _assemble(Config, given_, None)
    assert back.output_ply == (scratch / "output.ply" if save else None)
    assert dataclasses.replace(back, rr_config=RR, output_ply=shown.output_ply) == shown
    assert type(back.max_resolution) is int and type(back.expansion_percent) is float and type(back.flux.guidance) is float
