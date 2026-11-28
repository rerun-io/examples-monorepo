"""One step of a flow: a stage's pixi task and the catalog layer it publishes. Standard library only, so the ci env can
import the flows' step lists (ci/tests/test_flow_solve_groups.py)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Step:
    """One step: a stage's pixi task and the layer it writes."""

    name: str
    """Short name, for logs."""
    layer: str
    """The catalog layer the task registers, with its run record ``run_<layer>``."""
    dataset: str
    """The catalog dataset the layer goes on."""
    segment: str
    """The segment the layer goes on."""
    environment: str
    """The pixi environment of the task."""
    task: str
    """The pixi task."""
    arguments: tuple[str, ...]
    """The task's arguments."""
