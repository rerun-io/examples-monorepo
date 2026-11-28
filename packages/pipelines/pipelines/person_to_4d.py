"""The person_to_4d flow, from a person video: 4DAnyone novel views, Open4DGS mattes, then the trained 4D person.

The views go on dataset ``fdanyone-views`` as the ``base`` of segment ``<clip>__v06-seed<seed>`` (one ring of six
views), the mattes onto the same segment, and the person, encoded as Animated SOG, onto dataset
``open4dgs-runs`` as the ``asog`` layer of the training run, named after the views' segment. Two stages do not publish
their output as a layer yet: fdanyone-run writes a run directory and open4dgs-train a checkpoint that the export reads
by path. Each shares a step with the stage that registers its output (fdanyone-register-catalog, open4dgs-export), so
those files are scratch of the step and only layers pass between steps. Every task runs in env ``fdanyone-rerun``, which
composes feature ``open4dgs``: the env in which 4DAnyone's app trains its 4D person.
"""

import re
from pathlib import Path

from pipelines.step import Step, Task

ENVIRONMENT: str = "fdanyone-rerun"
VIEWS_DATASET: str = "fdanyone-views"
"""fdanyone_rerun.catalog.DATASET."""
RUNS_DATASET: str = "open4dgs-runs"
"""open4dgs.run_log.RUNS_DATASET."""
VIEWS_PER_LAYER: int = 6
"""The views of the one ring the flow asks fdanyone-run for (fdanyone_rerun.core.ViewLayout): the segment id names this
count, so the flow passes the layout rather than relying on the stage's default."""
LAYER_PITCH: int = 15
"""The ring's pitch, degrees above the subject."""
NUM_FRAMES: int = 121
"""Frames 4DAnyone generates per view."""
RUNS_SCRATCH: str = "fdanyone_run"
"""Under the output root: fdanyone-run's run directories and recordings, scratch of the views step."""


def views_segment(clip: str, seed: int) -> str:
    """The views' segment id (fdanyone_rerun.catalog.segment_id, with its check of the clip name, run before the views
    are generated rather than at their registration).

    Raises:
        ValueError: If the clip name is not lower-case letters, digits and dashes.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", clip):
        raise ValueError(f"clip name {clip!r}: use lower-case letters, digits and dashes")
    return f"{clip}__v{VIEWS_PER_LAYER:02d}-seed{seed}"


def flow_steps(*, video: Path, clip: str, seed: int, catalog_url: str, output_root: Path, train_steps: int = 25_000) -> tuple[Step, ...]:
    """The steps of one run of the flow, in order.

    Args:
        video: The person video (one person, at least 121 frames).
        clip: Short name of the clip (lower-case letters, digits and dashes), the first part of the segment id.
        seed: 4DAnyone's noise seed.
        catalog_url: The catalog every step registers on; the server must read the layer files at their paths.
        output_root: Where the tasks write: the scratch run directories, the layer files and the training runs.
        train_steps: TRiGS steps; the app trains 25,000 (fewer for a short proof).

    Returns:
        The steps.
    """
    root: Path = output_root.resolve()
    segment: str = views_segment(clip, seed)
    run_dir: Path = root / RUNS_SCRATCH / segment
    run_rrd: str = f"{run_dir}.rrd"  # beside the run directory, which --run.overwrite deletes when the task starts
    views: Step = Step("views", "base", VIEWS_DATASET, segment, (
        Task(ENVIRONMENT, "fdanyone-run", (
            "--run.video", str(video.resolve()), "--run.seed", str(seed), "--run.output-dir", str(run_dir), "--run.overwrite",
            "--run.layout.views-per-layer", str(VIEWS_PER_LAYER), "--run.layout.layer-pitches", str(LAYER_PITCH),
            "--no-previews", "--rr-config.headless", "--rr-config.save", run_rrd,
        )),
        Task(ENVIRONMENT, "fdanyone-register-catalog", (
            "--rrd", run_rrd, "--run-dir", str(run_dir), "--clip", clip, "--output-dir", str(root / "views"), "--catalog-url", catalog_url,
        )),
    ))  # fmt: skip
    mattes: Step = Step("mattes", "people_matte", VIEWS_DATASET, segment, (
        Task(ENVIRONMENT, "open4dgs-masks", (
            "--catalog-url", catalog_url, "--dataset", VIEWS_DATASET, "--segment-id", segment, "--num-frames", str(NUM_FRAMES),
            "--register-url", catalog_url, "--register-dataset", VIEWS_DATASET, "--output-dir", str(root / "people_matte"),
        )),
    ))  # fmt: skip
    # The app's 4D person (fdanyone_rerun.person4d): every view and frame at the generated size, 8 cameras a step, 8,000
    # init samples per person, the MoGe-2 prior (computed and registered by the train task when missing). Each layer's
    # catalog and dataset are named: the stages send a catalog they do not call local to the shared server's open4dgs-mamma.
    person: Step = Step("person", "asog", RUNS_DATASET, segment, (
        Task(ENVIRONMENT, "open4dgs-train", (
            "--catalog-url", catalog_url, "--dataset", VIEWS_DATASET, "--segment-id", segment, "--num-frames", str(NUM_FRAMES),
            "--num-holdout", "0", "--max-side", "None", "--samples-per-person", "8000",
            "--train.steps", str(train_steps), "--train.cameras-per-step", "8",
            "--prior", "auto", "--prior-catalog-url", catalog_url, "--prior-dataset", VIEWS_DATASET, "--prior-output-dir", str(root / "moge2_prior"),
            "--run-name", segment, "--runs-root", str(root / "runs"), "--runs-catalog-url", catalog_url,
        )),
        Task(ENVIRONMENT, "open4dgs-export", ("--run", str(root / "runs" / segment), "--runs-catalog-url", catalog_url)),
    ))  # fmt: skip
    return views, mattes, person
