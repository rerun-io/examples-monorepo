"""dagkit.testing.check_node_contract on fakes."""

import pytest
from dagkit_fakes import Add, Batch

from dagkit.node import Emit, RunContext, Tick
from dagkit.testing import check_node_contract


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
