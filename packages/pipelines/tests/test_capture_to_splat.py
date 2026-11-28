"""The flow's steps per input profile, and the command each step runs as."""

from pathlib import Path


def test_an_insta360_base_gets_lens_masks_then_masked_slam(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import Step, flow_steps

    steps: tuple[Step, ...] = flow_steps("insta360", segment="insta360__clip", catalog_url="rerun+http://127.0.0.1:1", output_root=tmp_path)
    assert [(s.name, s.layer, s.environment, s.task) for s in steps] == [
        ("lens_masks", "lens_mask", "posekit", "posekit-lens-masks"),
        ("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer"),
    ]
    assert steps[0].arguments == (
        "--dataset", "insta360", "--segment", "insta360__clip", "--catalog-url", "rerun+http://127.0.0.1:1", "--output-dir", str(tmp_path / "lens_mask"),
    )
    assert steps[1].arguments == (
        "--segment", "insta360__clip", "--catalog", "rerun+http://127.0.0.1:1", "--output-dir", str(tmp_path / "slam_rs"), "--mask-entity", "people_mask",
    )


def test_a_robocap_base_gets_slam_alone_unmasked(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import flow_steps

    (step,) = flow_steps("robocap", segment="robocap__cap__s1", catalog_url="rerun+http://127.0.0.1:1", output_root=tmp_path)
    assert (step.task, step.layer) == ("slam-rs-catalog-layer", "slam_rs")
    assert "--mask-entity" not in step.arguments and step.arguments[:2] == ("--segment", "robocap__cap__s1")


def test_a_relative_output_root_reaches_the_stages_absolute(tmp_path: Path, monkeypatch) -> None:
    """Each stage task runs in its own package directory, so a relative path would land in a different place per step."""
    from pipelines.capture_to_splat import flow_steps

    monkeypatch.chdir(tmp_path)
    (step,) = flow_steps("robocap", segment="robocap__cap__s1", catalog_url="u", output_root=Path("out"))
    assert step.arguments[step.arguments.index("--output-dir") + 1] == str(tmp_path / "out" / "slam_rs")


def test_a_step_runs_its_pixi_task_in_a_memory_scope_inside_the_gpu_lock(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import Step
    from pipelines.runner import Launch, step_command

    step: Step = Step("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer", ("--segment", "s"))
    assert step_command(step, Launch(manifest=tmp_path / "pixi.toml", gpu_lock=Path("/tmp/gpu.lock"), memory_max="48G")) == [
        "flock", "-x", "/tmp/gpu.lock", "systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=48G",
        "pixi", "run", "--manifest-path", str(tmp_path / "pixi.toml"), "--frozen", "-e", "slam-rs-cuda", "slam-rs-catalog-layer", "--segment", "s",
    ]
    assert step_command(step, Launch(manifest=tmp_path / "pixi.toml", gpu_lock=None, memory_max=None))[:2] == ["pixi", "run"]
