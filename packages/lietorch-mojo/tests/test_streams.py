"""CUDA stream ordering between PyTorch and the Mojo kernels.

The Mojo backend enqueues every kernel on PyTorch's current CUDA stream and
never synchronises, so it must be ordered with PyTorch's work exactly as a
native op is. The stress tests make rapid-fire calls where every input block
is freed right after the call and grabbed by a junk tensor, PyTorch consumes
each output at once, and random `torch.cuda._sleep` delays let an unordered
launch overtake the stream. They run on the default stream, on a non-default
stream, and alternating between the two with no host sync (which rebuilds the
cached Mojo stream wrapper while kernels are still in flight). A build that
launched on Mojo's own stream without a sync failed 14-21 of 600 iterations
on the side and alternating lanes, and the graph test.

The reference for every op is the Mojo CPU path on the same inputs; it is
synchronous and shares the element code, so a mismatch means a GPU result
read or wrote the wrong memory or the wrong moment.
"""

import random
import time
from collections.abc import Callable

import pytest
import torch
from conftest import require_mojo_extension
from torch import Tensor

require_mojo_extension()

from lietorch_mojo import SE3, backends  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]

LANES: tuple[str, ...] = ("default", "side", "alternate")
ITERS: int = 600
SIZES: tuple[int, ...] = (1, 7, 256, 1000, 4097, 65536)
# Ordering faults give garbage (junk 777s, stale blocks), not rounding-level differences.
ATOL: float = 1e-4


def _stream_for(lane: str, it: int, side: torch.cuda.Stream) -> torch.cuda.Stream:
    default: torch.cuda.Stream = torch.cuda.default_stream()
    return {"default": default, "side": side, "alternate": side if it % 2 else default}[lane]


def _se3_data(n: int, gen: torch.Generator) -> Tensor:
    q: Tensor = torch.randn(n, 4, generator=gen)
    return torch.cat([torch.randn(n, 3, generator=gen), q / q.norm(dim=-1, keepdim=True)], dim=-1)


# Each op: (inputs on CPU) -> (CUDA inputs) -> outputs, built only from backends calls.
OPS: dict[str, tuple[Callable[[int, torch.Generator], list[Tensor]], Callable[..., list[Tensor]]]] = {
    "inv": (lambda n, g: [_se3_data(n, g)], lambda X: [backends.inv(3, X)]),
    "mul": (lambda n, g: [_se3_data(n, g), _se3_data(n, g)], lambda X, Y: [backends.mul(3, X, Y)]),
    "exp": (lambda n, g: [0.5 * torch.randn(n, 6, generator=g)], lambda a: [backends.expm(3, a)]),
    "log": (lambda n, g: [_se3_data(n, g)], lambda X: [backends.logm(3, X)]),
    "act": (lambda n, g: [_se3_data(n, g), torch.randn(n, 3, generator=g)], lambda X, p: [backends.act(3, X, p)]),
    "act_backward": (
        lambda n, g: [torch.randn(n, 3, generator=g), _se3_data(n, g), torch.randn(n, 3, generator=g)],
        lambda grad, X, p: backends.act_backward(3, grad, X, p),
    ),
    "mul_backward": (
        lambda n, g: [torch.randn(n, 7, generator=g), _se3_data(n, g), _se3_data(n, g)],
        lambda grad, X, Y: backends.mul_backward(3, grad, X, Y),
    ),
}


def run_lane(lane: str, iters: int, seed: int) -> list[str]:
    """Rapid-fire random ops on `lane`; returns a description of every mismatching iteration."""
    rng: random.Random = random.Random(seed)
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    side: torch.cuda.Stream = torch.cuda.Stream()
    pending: list[tuple[str, int, list[Tensor], list[Tensor]]] = []
    for it in range(iters):
        op: str = rng.choice(list(OPS))
        make, call = OPS[op]
        n: int = rng.choice((*SIZES, rng.randint(1, 200_000)))
        inputs_cpu: list[Tensor] = make(n, gen)
        expected: list[Tensor] = call(*inputs_cpu)
        with torch.cuda.stream(_stream_for(lane, it, side)):
            inputs: list[Tensor] = [t.to("cuda") for t in inputs_cpu]
            if rng.random() < 0.3:
                torch.cuda._sleep(rng.randint(10_000, 200_000))
            outputs: list[Tensor] = call(*inputs)
            del inputs  # free the input blocks while the kernel may still read them...
            junk: list[Tensor] = [torch.full_like(t, 777.0, device="cuda") for t in inputs_cpu]  # ...and reuse them
            consumed: list[Tensor] = [o * 2.0 for o in outputs]  # PyTorch reads the outputs at once
            del outputs, junk
        pending.append((op, n, consumed, expected))
    torch.cuda.synchronize()
    bad: list[str] = []
    for it, (op, n, consumed, expected) in enumerate(pending):
        for got, ref in zip(consumed, expected, strict=True):
            if not torch.allclose(got.cpu(), 2.0 * ref, rtol=ATOL, atol=ATOL):
                bad.append(f"iter {it} {op} n={n}: max err {(got.cpu() - 2.0 * ref).abs().max().item():.3g}")
                break
    return bad


@pytest.mark.parametrize("lane", LANES)
def test_rapid_fire_allocator_reuse(lane: str) -> None:
    bad: list[str] = run_lane(lane, ITERS, seed=LANES.index(lane))
    assert not bad, f"{len(bad)}/{ITERS} iterations wrong on the {lane} lane: {bad[:5]}"


@pytest.mark.parametrize("lane", LANES)
def test_rapid_fire_autograd(lane: str) -> None:
    """Forward + backward through the group API (inv, mul, act and their backward kernels) on each lane."""
    rng: random.Random = random.Random(100 + LANES.index(lane))
    gen: torch.Generator = torch.Generator().manual_seed(100 + LANES.index(lane))
    side: torch.cuda.Stream = torch.cuda.Stream()

    def loss_grads(X: Tensor, Y: Tensor, p: Tensor) -> tuple[Tensor, ...]:
        X, Y, p = (t.requires_grad_(True) for t in (X, Y, p))
        q: Tensor = (SE3(X) * SE3(Y).inv()).act(p)
        return torch.autograd.grad((q * q).sum(), [X, Y, p])

    pending: list[tuple[int, tuple[Tensor, ...], tuple[Tensor, ...]]] = []
    for it in range(ITERS // 3):
        n: int = rng.choice((1, 33, 4097, rng.randint(1, 50_000)))
        cpu: list[Tensor] = [_se3_data(n, gen), _se3_data(n, gen), torch.randn(n, 3, generator=gen)]
        expected: tuple[Tensor, ...] = loss_grads(*[t.clone() for t in cpu])
        with torch.cuda.stream(_stream_for(lane, it, side)):
            if rng.random() < 0.3:
                torch.cuda._sleep(rng.randint(10_000, 200_000))
            grads: tuple[Tensor, ...] = loss_grads(*[t.to("cuda") for t in cpu])
            junk: Tensor = torch.full((n, 16), 777.0, device="cuda")
            consumed: tuple[Tensor, ...] = tuple(g * 1.0 for g in grads)
            del grads, junk
        pending.append((n, consumed, expected))
    torch.cuda.synchronize()
    bad: list[int] = [
        n for n, got, ref in pending if not all(torch.allclose(g.cpu(), r, rtol=1e-3, atol=1e-3) for g, r in zip(got, ref, strict=True))
    ]
    assert not bad, f"{len(bad)}/{len(pending)} autograd iterations wrong on the {lane} lane (sizes {bad[:5]})"


def test_launch_does_not_synchronise() -> None:
    """A Mojo call behind ~100 ms of queued GPU work returns long before that work ends."""
    X: Tensor = _se3_data(1024, torch.Generator().manual_seed(0)).cuda()
    backends.inv(3, X)
    torch.cuda.synchronize()
    torch.cuda._sleep(300_000_000)
    t0: float = time.perf_counter()
    Y: Tensor = backends.inv(3, X)
    host_s: float = time.perf_counter() - t0
    torch.cuda.synchronize()
    wall_s: float = time.perf_counter() - t0
    assert wall_s > 0.02, f"the sleep kernel was too short to test anything ({wall_s:.4f} s)"
    assert host_s < wall_s / 4, f"the Mojo call blocked the host for {host_s:.4f} s of {wall_s:.4f} s"
    assert torch.allclose(backends.inv(3, Y), X, atol=1e-5)


def test_cuda_graph_capture_and_replay() -> None:
    gen: torch.Generator = torch.Generator().manual_seed(1)
    X: Tensor = _se3_data(4096, gen).cuda()
    p: Tensor = torch.randn(4096, 3, generator=gen).cuda()
    expected: Tensor = backends.act(3, backends.inv(3, X), p)
    graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
    side: torch.cuda.Stream = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        backends.act(3, backends.inv(3, X), p)  # warm-up on the capture stream, as PyTorch recommends
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(graph):
        out: Tensor = backends.act(3, backends.inv(3, X), p)
    X.copy_(_se3_data(4096, gen).cuda())
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(out, backends.act(3, backends.inv(3, X), p), atol=1e-5)
    assert not torch.allclose(out, expected, atol=1e-3)  # the replay used the new X


def test_inputs_written_by_pending_torch_work_are_seen() -> None:
    n: int = 1 << 16
    a_cpu: Tensor = 0.5 * torch.randn(n, 6, dtype=torch.float32)
    expected: Tensor = backends.expm(3, a_cpu)
    for _ in range(5):
        a: Tensor = torch.zeros(n, 6, device="cuda")
        torch.cuda._sleep(20_000_000)
        a.copy_(a_cpu.cuda())  # lands after the sleep
        X: Tensor = backends.expm(3, a)
        assert torch.allclose(X.cpu(), expected, atol=1e-5)
