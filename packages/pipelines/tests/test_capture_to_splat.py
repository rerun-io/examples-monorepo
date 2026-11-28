"""The flow's steps per input profile, and the command each step runs as."""

from pathlib import Path


def test_an_insta360_base_gets_lens_masks_masked_slam_face_masks_then_colsfm(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import Step, flow_steps

    steps: tuple[Step, ...] = flow_steps("insta360", segment="insta360__clip", catalog_url="rerun+http://127.0.0.1:1", output_root=tmp_path)
    assert [(s.name, s.layer, s.environment, s.task) for s in steps] == [
        ("lens_masks", "lens_mask", "posekit", "posekit-lens-masks"),
        ("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer"),
        ("face_masks", "masks", "dataforge-cuda", "dataforge-insta360-face-masks"),
        ("colsfm", "colsfm", "colsfm", "colsfm-catalog"),
    ]
    assert steps[0].arguments == (
        "--dataset", "insta360", "--segment", "insta360__clip", "--catalog-url", "rerun+http://127.0.0.1:1", "--output-dir", str(tmp_path / "lens_mask"),
    )
    assert steps[1].arguments == (
        "--segment", "insta360__clip", "--catalog", "rerun+http://127.0.0.1:1", "--output-dir", str(tmp_path / "slam_rs"), "--mask-entity", "people_mask",
    )
    assert steps[2].arguments == ("--segment", "insta360__clip", "--catalog-url", "rerun+http://127.0.0.1:1", "--output-dir", str(tmp_path / "masks"))
    colsfm: tuple[str, ...] = steps[3].arguments
    assert colsfm[:6] == ("--source.dataset", "insta360", "--source.segment", "insta360__clip", "--source.catalog-url", "rerun+http://127.0.0.1:1")
    assert colsfm[colsfm.index("--layer-path") + 1] == str(tmp_path / "colsfm" / "insta360__clip.rrd")
    assert colsfm[colsfm.index("--source.cameras") + 1 : colsfm.index("--source.mask-entity")] == (
        "cube6_front0", "cube6_front1", "cube6_left", "cube6_right", "cube6_up", "cube6_down",
    )
    assert colsfm[colsfm.index("--options.config-dir") + 1] == "colsfm/resources/configs/insta360-cube" and "--options.align-to-prior" in colsfm


def test_a_robocap_base_gets_unmasked_slam_then_colsfm_on_its_fisheyes(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import flow_steps

    slam, colsfm = flow_steps("robocap", segment="robocap__cap__s1", catalog_url="rerun+http://127.0.0.1:1", output_root=tmp_path)
    assert (slam.task, slam.layer, colsfm.task, colsfm.layer) == ("slam-rs-catalog-layer", "slam_rs", "colsfm-catalog", "colsfm")
    assert "--mask-entity" not in slam.arguments and "--source.mask-entity" not in colsfm.arguments
    cameras: tuple[str, ...] = colsfm.arguments[colsfm.arguments.index("--source.cameras") + 1 : colsfm.arguments.index("--source.stereo-pairs")]
    assert cameras == ("left_front", "right_front", "left", "right") and colsfm.arguments[colsfm.arguments.index("--source.frame-stride") + 1] == "4"


def test_a_relative_output_root_reaches_the_stages_absolute(tmp_path: Path, monkeypatch) -> None:
    """Each stage task runs in its own package directory, so a relative path would land in a different place per step."""
    from pipelines.capture_to_splat import flow_steps

    monkeypatch.chdir(tmp_path)
    slam, colsfm = flow_steps("robocap", segment="robocap__cap__s1", catalog_url="u", output_root=Path("out"))
    assert slam.arguments[slam.arguments.index("--output-dir") + 1] == str(tmp_path / "out" / "slam_rs")
    assert colsfm.arguments[colsfm.arguments.index("--layer-path") + 1] == str(tmp_path / "out" / "colsfm" / "robocap__cap__s1.rrd")


def test_a_step_runs_its_pixi_task_in_a_memory_scope_inside_the_gpu_lock(tmp_path: Path) -> None:
    from pipelines.capture_to_splat import Step
    from pipelines.runner import Launch, step_command

    step: Step = Step("slam", "slam_rs", "slam-rs-cuda", "slam-rs-catalog-layer", ("--segment", "s"))
    assert step_command(step, Launch(manifest=tmp_path / "pixi.toml", gpu_lock=Path("/tmp/gpu.lock"), memory_max="48G")) == [
        "flock", "-x", "/tmp/gpu.lock", "systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=48G",
        "pixi", "run", "--manifest-path", str(tmp_path / "pixi.toml"), "--frozen", "-e", "slam-rs-cuda", "slam-rs-catalog-layer", "--segment", "s",
    ]
    assert step_command(step, Launch(manifest=tmp_path / "pixi.toml", gpu_lock=None, memory_max=None))[:2] == ["pixi", "run"]


def test_every_step_of_a_profile_runs_in_one_solve_group(tmp_path: Path) -> None:
    """One version of each package across the stages a flow chains (Pablo, study Q3): their envs share a solve group."""
    import tomllib

    from pipelines.apis.capture_to_splat import WORKSPACE_MANIFEST
    from pipelines.capture_to_splat import flow_steps

    environments: dict[str, dict[str, object]] = tomllib.loads(WORKSPACE_MANIFEST.read_text())["environments"]
    for profile in ("insta360", "robocap"):
        steps = flow_steps(profile, segment=f"{profile}__s", catalog_url="u", output_root=tmp_path)
        groups: set[object] = {environments[step.environment].get("solve-group") for step in steps}
        assert len(groups) == 1 and None not in groups, f"{profile}: {groups}"
