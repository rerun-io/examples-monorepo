"""The generated gradio app: every Config field has a widget or is hidden on purpose, the example is the VERIFY short case,
and whatever the form sends comes back as the typed Config the CLI would build (no GPU, no catalog: ``build`` is a fake)."""

import dataclasses
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, get_args

import rerun as rr
from dagkit.app import HardwareClass
from dagkit.graph import Graph
from dagkit.node import Mode
from dagkit.sinks import Register
from dagkit.sources import SegmentRef
from dagkit.testing import check_form
from dagkit.verify import load_card
from hypothesis import given, settings
from hypothesis import strategies as st

from exo_calib.apis.calibrate import Config
from exo_calib.verify import CARD, examples

WIDGETS: dict[str, str] = {
    "segment.dataset": "Textbox", "segment.segment": "Textbox", "segment.window_ns": "Textbox", "frame_index": "Number", "frame_budget": "Number", "output_dir": "Checkbox",
    "register": "Dropdown", "resume_from": "Dropdown", "hardware": "Dropdown", "runner": "Dropdown", "mode": "Dropdown", "draw": "Dropdown",
}  # fmt: skip
"""Form label (without the "save it" note) -> widget class, in form order."""
HIDDEN: dict[str, str] = {
    "rr_config": "the app draws into its own Rerun pane (the form passes a headless viewer config)",
    "exo_rigs": "explicit rig names are a CLI override; the form leaves them to the catalog metadata",
}
"""Config fields the form leaves out on purpose."""
NAME: st.SearchStrategy[str] = st.text("abcxyz0129_-", min_size=1, max_size=12)


def form(built: list[Config]) -> tuple[list[Any], Callable[..., Iterator[tuple[object, ...]]], Any]:
    """The form's input widgets, its Run handler and the app, around a fake ``build`` that records the Config it gets.

    ``check_form``: every Config field has its WIDGETS widget, in order, or is HIDDEN on purpose."""

    def build(config: Config, recording: rr.RecordingStream) -> Iterator[Graph]:
        built.append(config)
        return iter(())

    demo = check_form(Config, build, WIDGETS, HIDDEN, examples=examples())
    run = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    return list(run.inputs), run.fn, demo


def test_every_config_field_has_a_widget_or_is_hidden_on_purpose() -> None:
    inputs, _, _ = form([])
    choices: dict[str, list[object]] = {w.label: [c[1] for c in w.choices] for w in inputs if type(w).__name__ == "Dropdown"}
    assert choices["hardware"] == [*get_args(HardwareClass), "auto"] and choices["register"] == list(get_args(Register))
    assert choices["resume_from"] == ["init", "kp2d"] and choices["mode"] == list(get_args(Mode)) and choices["draw"] == ["all", "outputs"]


def test_the_example_is_the_verify_short_case_and_runs_from_one_click() -> None:
    short = next(case for block in load_card(CARD).graph for case in block.cases if case.clip == "short")
    (example,) = examples()
    assert (example.segment.dataset, example.segment.segment, example.frame_budget, example.output_dir) == (short.dataset, short.segment, 200, None)
    built: list[Config] = []
    inputs, run, demo = form(built)
    dataset = next(b for b in demo.blocks.values() if type(b).__name__ == "Dataset")  # gr.Examples: what one click fills in
    clicked: dict[str, object] = dict(zip([c.label for c in dataset._components], dataset.samples[0], strict=True))
    list(run(*(clicked.get(w.label, w.value) for w in inputs)))  # the widgets the example leaves alone keep their defaults
    assert dataclasses.replace(built[0], rr_config=example.rr_config) == example and built[0].rr_config.headless


@settings(max_examples=25, deadline=None)
@given(
    segment=st.tuples(NAME, NAME),
    numbers=st.tuples(st.integers(0, 10_000), st.integers(1, 100_000)),
    as_float=st.booleans(),
    save=st.booleans(),
    choices=st.tuples(
        st.sampled_from(get_args(Register)), st.sampled_from([None, "init", "kp2d"]), st.sampled_from([*get_args(HardwareClass), "auto"]),
        st.sampled_from(["sequential", "threaded"]), st.sampled_from(get_args(Mode)), st.sampled_from(["all", "outputs"]),
    ),  # fmt: skip
)
def test_whatever_the_form_sends_comes_back_as_the_typed_config(
    segment: tuple[str, str], numbers: tuple[int, int], as_float: bool, save: bool, choices: tuple
) -> None:
    """A checked save box is a scratch ``output.dir`` that is never registered; an unchecked one is the viewer only."""
    built: list[Config] = []
    _, run, _ = form(built)
    sent: tuple[float, ...] = tuple(float(n) for n in numbers) if as_float else numbers  # a browser may send 200.0 for 200
    list(run(*segment, "", *sent, save, *choices))  # "": no window, the whole segment
    (config,) = built
    default = Config(config.rr_config)
    expected = dataclasses.replace(
        default,
        segment=dataclasses.replace(default.segment, dataset=segment[0], segment=segment[1]),
        frame_index=numbers[0], frame_budget=numbers[1], output_dir=config.output_dir if save else None,
        register="off" if save else choices[0], resume_from=choices[1], hardware=choices[2], runner=choices[3], mode=choices[4], draw=choices[5],
    )  # fmt: skip
    assert config == expected and (not save or (isinstance(config.output_dir, Path) and config.output_dir.name == "output.dir"))
    assert type(config.frame_index) is int and type(config.frame_budget) is int and isinstance(config.segment, SegmentRef)
