"""person_to_4d's steps from a person video: the views, the mattes, then the trained person, and what each task runs."""

from pathlib import Path


def _option(arguments: tuple[str, ...], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def test_a_video_becomes_views_then_mattes_then_the_trained_person(tmp_path: Path) -> None:
    from pipelines.person_to_4d import flow_steps
    from pipelines.step import Step

    video: Path = tmp_path / "person.mp4"
    steps: tuple[Step, ...] = flow_steps(video=video, clip="pexels-1", seed=7, catalog_url="rerun+http://127.0.0.1:1", output_root=tmp_path / "out")
    assert [(s.name, s.layer, s.dataset, s.segment, [t.task for t in s.tasks]) for s in steps] == [
        ("views", "base", "fdanyone-views", "pexels-1__v06-seed7", ["fdanyone-run", "fdanyone-register-catalog"]),
        ("mattes", "people_matte", "fdanyone-views", "pexels-1__v06-seed7", ["open4dgs-masks"]),
        ("person", "asog", "open4dgs-runs", "pexels-1__v06-seed7", ["open4dgs-train", "open4dgs-export"]),
    ]
    assert {t.environment for s in steps for t in s.tasks} == {"fdanyone-rerun"}, "one env: the one 4DAnyone's app trains in"

    generate, register = steps[0].tasks
    run_dir: str = str(tmp_path / "out" / "fdanyone_run" / "pexels-1__v06-seed7")
    assert (_option(generate.arguments, "--run.video"), _option(generate.arguments, "--run.seed")) == (str(video), "7")
    assert _option(generate.arguments, "--run.output-dir") == run_dir and "--run.overwrite" in generate.arguments
    assert _option(generate.arguments, "--rr-config.save") == _option(register.arguments, "--rrd") == run_dir + ".rrd", (
        "the run recording sits beside the run directory, which --run.overwrite deletes when the step starts"
    )
    assert (_option(register.arguments, "--run-dir"), _option(register.arguments, "--clip")) == (run_dir, "pexels-1")
    assert _option(register.arguments, "--catalog-url") == "rerun+http://127.0.0.1:1"

    (mattes,) = steps[1].tasks
    assert (_option(mattes.arguments, "--dataset"), _option(mattes.arguments, "--segment-id")) == ("fdanyone-views", "pexels-1__v06-seed7")
    assert _option(mattes.arguments, "--num-frames") == "121", "every generated frame"

    train, export = steps[2].tasks
    assert (_option(train.arguments, "--segment-id"), _option(train.arguments, "--run-name")) == ("pexels-1__v06-seed7", "pexels-1__v06-seed7")
    assert _option(export.arguments, "--run") == str(tmp_path / "out" / "runs" / "pexels-1__v06-seed7")
    assert _option(train.arguments, "--runs-catalog-url") == _option(export.arguments, "--runs-catalog-url") == "rerun+http://127.0.0.1:1"


def test_the_trained_person_uses_the_settings_of_the_apps_4d_person(tmp_path: Path) -> None:
    """The app's 4D person (fdanyone_rerun.person4d.PersonConfig): all frames at the generated size, 25,000 steps of 8
    cameras, 8,000 init samples per person, the MoGe-2 prior on."""
    from pipelines.person_to_4d import flow_steps

    train = flow_steps(video=tmp_path / "v.mp4", clip="c", seed=42, catalog_url="u", output_root=tmp_path)[2].tasks[0]
    settings: dict[str, str] = {name: _option(train.arguments, name) for name in (
        "--num-frames", "--max-side", "--train.steps", "--train.cameras-per-step", "--samples-per-person", "--prior",
    )}  # fmt: skip
    assert settings == {
        "--num-frames": "121", "--max-side": "None", "--train.steps": "25000", "--train.cameras-per-step": "8", "--samples-per-person": "8000", "--prior": "auto",
    }  # fmt: skip
    quick = flow_steps(video=tmp_path / "v.mp4", clip="c", seed=42, catalog_url="u", output_root=tmp_path, train_steps=500)[2].tasks[0]
    assert _option(quick.arguments, "--train.steps") == "500", "a short proof run trains fewer steps"


def test_every_output_lands_under_the_output_root(tmp_path: Path, monkeypatch) -> None:
    """The stages default to NAS paths; a run of the flow writes only under its own output root, given relative or not."""
    from pipelines.person_to_4d import flow_steps

    monkeypatch.chdir(tmp_path)
    paths: list[str] = [
        _option(task.arguments, name)
        for step in flow_steps(video=Path("v.mp4"), clip="c", seed=1, catalog_url="u", output_root=Path("out"))
        for task in step.tasks
        for name in ("--run.output-dir", "--rr-config.save", "--output-dir", "--runs-root", "--prior-output-dir", "--run")
        if name in task.arguments
    ]
    assert len(paths) == 7 and all(path.startswith(str(tmp_path / "out")) for path in paths), paths
