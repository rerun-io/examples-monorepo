"""dagkit.testing.check_node_contract on fakes."""

from dataclasses import dataclass

import numpy as np
import pytest
from dagkit_fakes import Add, Batch

from dagkit.node import Emit, Fenced, RunContext, Tick
from dagkit.testing import check_node_contract, host


def ticks(n: int) -> list[Tick]:
    return [Tick(t, "x", t, {}) for t in range(n)]


def test_pure_and_late_nodes_meet_the_contract() -> None:
    assert check_node_contract(Add(1), ticks(3)) == [("y", t, t + 1) for t in range(3)]
    assert check_node_contract(Batch(2), ticks(5)) == [("y", t, 10 * t) for t in range(5)]


def test_state_that_survives_reset_is_caught() -> None:
    class Leaky(Add):
        count: int = 0

        def reset(self, ctx: RunContext) -> None:  # forgets to clear the count
            pass

        def step(self, tick: Tick, out: Emit) -> None:
            self.count += 1
            out.put("y", tick.t_ns, self.count)

    with pytest.raises(AssertionError, match="reset is not idempotent"):
        check_node_contract(Leaky(0), ticks(3))


def test_a_reset_that_rewrites_an_output_the_first_run_kept_is_caught() -> None:
    """Runs compare outputs as they were put: a reset that writes into an output the first run still holds cannot make
    both runs look the same."""

    class Rewrites(Add):
        def __init__(self) -> None:
            super().__init__(0)
            self.buffer: list[int] = [0]

        def reset(self, ctx: RunContext) -> None:  # not idempotent, and it writes into the buffer every run emitted
            self.resets += 1
            self.buffer[0] = self.resets

        def step(self, tick: Tick, out: Emit) -> None:
            out.put("y", tick.t_ns, self.buffer)

    with pytest.raises(AssertionError, match="reset is not idempotent"):
        check_node_contract(Rewrites(), ticks(2))


def test_an_output_written_again_after_its_put_is_caught() -> None:
    """A node emits owned values: one buffer rewritten by every step makes every kept output read the last value."""

    class Reuses(Add):
        def reset(self, ctx: RunContext) -> None:
            self.buffer = np.zeros(1)

        def step(self, tick: Tick, out: Emit) -> None:
            self.buffer[0] = float(tick.value)  # type: ignore[arg-type]
            out.put("y", tick.t_ns, self.buffer)

    with pytest.raises(AssertionError, match="changed after its put"):
        check_node_contract(Reuses(0), ticks(3))


@dataclass(frozen=True)
class Cloud:
    xyz: np.ndarray


def test_array_valued_outputs_compare_by_value() -> None:
    class Lift(Add):
        def step(self, tick: Tick, out: Emit) -> None:
            out.put("y", tick.t_ns, Cloud(np.full(3, float(tick.value))))  # type: ignore[arg-type]

    class LeakyLift(Lift):
        count: float = 0.0

        def reset(self, ctx: RunContext) -> None:  # forgets to clear the count
            pass

        def step(self, tick: Tick, out: Emit) -> None:
            self.count += 1.0
            out.put("y", tick.t_ns, Cloud(np.full(3, self.count)))

    outputs = check_node_contract(Lift(0), ticks(2))
    assert [(port, t) for port, t, _ in outputs] == [("y", 0), ("y", 1)]
    with pytest.raises(AssertionError, match="reset is not idempotent"):
        check_node_contract(LeakyLift(0), ticks(2))


class Tensorish:
    """Duck-typed like a torch tensor (dagkit never imports torch): equal values, a new object every run."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def detach(self) -> "Tensorish":
        return self

    def cpu(self) -> "Tensorish":
        return self

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values)


@dataclass(frozen=True)
class GpuCloud:
    xyz: Tensorish
    views: tuple[Tensorish, ...]


def test_fenced_and_tensor_like_outputs_compare_by_value() -> None:
    class OnGpu(Add):
        def step(self, tick: Tick, out: Emit) -> None:
            cloud = GpuCloud(Tensorish([float(tick.value)] * 3), (Tensorish([1.0]),))  # type: ignore[arg-type]
            out.put("y", tick.t_ns, Fenced(cloud, None))

    outputs = check_node_contract(OnGpu(0), ticks(2))
    assert [type(value) for _, _, value in outputs] == [Fenced, Fenced]  # the caller still gets the values the node put
    np.testing.assert_equal(host(outputs[1][2]), {"xyz": np.ones(3), "views": [np.ones(1)]})
