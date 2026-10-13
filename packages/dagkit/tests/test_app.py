"""Profiles (hand-written TOML, coerced, strict) and the generated gradio app."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest
import rerun as rr
from serde import coerce, serde

from dagkit.app import NodeProfile, Profile, detect_hardware, gradio_app, load_profile
from dagkit.graph import Graph
from dagkit.node import Mode
from dagkit.sources import SegmentRef

PROFILE = """
hardware = "x86_64-cuda"

[nodes.fit]
cpus = [6, 7]
params = { iterations = 30, lr = 1 }

[presets.fast.fit]
params = { iterations = 5 }

[presets.fast.track]
placement = "process"
"""


@serde(type_check=coerce)
@dataclass(frozen=True, slots=True)
class FitConfig:
    iterations: int = 100
    lr: float = 0.5
    polish: bool = True


def test_profile_merges_presets_and_configures_nodes(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text(PROFILE)
    profile: Profile = load_profile(tmp_path, "x86_64-cuda")
    assert profile.node("fit") == NodeProfile("thread", [6, 7], {"iterations": 30, "lr": 1})
    assert profile.node("fit", "fast") == NodeProfile("thread", [6, 7], {"iterations": 5, "lr": 1})
    assert profile.node("track", "fast").placement == "process"
    assert profile.configure("fit", FitConfig(), "fast") == FitConfig(iterations=5, lr=1.0, polish=True)
    with pytest.raises(KeyError, match="unknown preset 'slow'"):
        profile.node("fit", "slow")


def test_profile_refuses_unknown_keys_and_the_wrong_class(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text(PROFILE + "\nhostname = 'box'\n")
    with pytest.raises(ValueError, match="x86_64-cuda.toml"):
        load_profile(tmp_path, "x86_64-cuda")
    (tmp_path / "aarch64-cuda.toml").write_text('hardware = "x86_64-cuda"\n')
    with pytest.raises(ValueError, match="declares hardware 'x86_64-cuda'"):
        load_profile(tmp_path, "aarch64-cuda")


def test_detect_hardware_names_a_class_or_explains() -> None:
    try:
        assert detect_hardware() in ("x86_64-cuda", "aarch64-cuda", "robocap-cap")
    except RuntimeError as exc:
        assert "pass the hardware class explicitly" in str(exc)


@dataclass
class Inner:
    headless: bool = False


@dataclass
class DemoConfig:
    rr_config: Inner
    segment: SegmentRef | None = None
    image: Path | None = None
    frames: int = 10
    scale: float = 1.0
    name: str = "demo"
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"
    output_ply: Path | None = None


def test_gradio_app_builds_a_form_from_config_fields() -> None:
    def build(config: DemoConfig, recording: rr.RecordingStream) -> Graph:
        return Graph("empty")

    demo = gradio_app(DemoConfig, build, title="Demo", examples=[DemoConfig(Inner(), SegmentRef("ds", "seg"))])
    labels: list[str] = [getattr(block, "label", None) or "" for block in demo.blocks.values()]
    for expected in ("segment.dataset", "segment.segment", "image", "frames", "scale", "name", "runner", "mode", "output_ply"):
        assert expected in labels, expected
    assert "rr_config" not in labels


def test_a_preset_can_put_a_process_node_back_on_a_thread(tmp_path: Path) -> None:
    (tmp_path / "x86_64-cuda.toml").write_text('hardware = "x86_64-cuda"\n[nodes.fit]\nplacement = "process"\n[presets.debug.fit]\nplacement = "thread"\ncpus = []\n')
    profile: Profile = load_profile(tmp_path, "x86_64-cuda")
    assert profile.node("fit").placement == "process" and profile.node("fit", "debug") == NodeProfile("thread", [], {})


@dataclass
class FormConfig:
    rr_config: Inner = field(default_factory=Inner)
    segment: SegmentRef | None = None
    frames: int = 10
    label: str = field(default_factory=lambda: "factory value")
    runner: Literal["sequential", "threaded"] = "threaded"
    mode: Mode = "lossless"


def test_generated_run_keeps_defaults_none_segments_and_a_headless_viewer_config() -> None:
    built: list[FormConfig] = []

    def build(config: FormConfig, recording: rr.RecordingStream) -> Graph:
        built.append(config)
        return Graph("empty")

    demo = gradio_app(FormConfig, build, title="Form")
    run = next(f.fn for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "run")
    # widget order: segment.dataset, segment.segment, frames, label, runner, mode
    list(run("", "", 10, None, "threaded", "lossless"))
    list(run("ds", "seg", 10, "typed", "threaded", "lossless"))
    assert built[0].segment is None and built[0].label == "factory value" and built[0].rr_config.headless
    assert built[1].segment == SegmentRef("ds", "seg") and built[1].label == "typed"
