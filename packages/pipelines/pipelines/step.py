"""One step of a flow: stage pixi tasks and the catalog layer the step publishes. Standard library only, so the ci env can
import the flows' step lists (ci/tests/test_flow_solve_groups.py)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Task:
    """One stage pixi task, run as its own process."""

    environment: str
    """The pixi environment of the task."""
    task: str
    """The pixi task."""
    arguments: tuple[str, ...]
    """The task's arguments."""


@dataclass(frozen=True, slots=True)
class Step:
    """One step: its tasks, run in order, and the layer the last of them registers."""

    name: str
    """Short name, for logs."""
    layer: str
    """The catalog layer the step's last task registers, with its run record ``run_<layer>``."""
    dataset: str
    """The catalog dataset the layer goes on."""
    segment: str
    """The segment the layer goes on."""
    tasks: tuple[Task, ...]
    """Usually one. More when a stage's output is not yet a catalog layer: what an earlier task writes is scratch of the
    step (person_to_4d: the 4DAnyone run directory, the trained checkpoint), and only the last task's layer is handed on."""
