"""Broadcasting of batch dimensions: a port of lietorch `broadcasting.py`.

Behaviour is upstream's, including one quirk kept on purpose: the fast path
compares `x.shape[:-1]` (a torch.Size) with `y.shape[-1]` (an int), which is
never equal, so binary ops always take the `repeat` path. Changing that would
change which tensors autograd sees, so the port keeps it.
"""

from torch import Tensor


def check_broadcastable(x: Tensor, y: Tensor) -> None:
    assert len(x.shape) == len(y.shape)
    for n, m in zip(x.shape[:-1], y.shape[:-1], strict=False):
        assert n == m or n == 1 or m == 1


def broadcast_inputs(x: Tensor, y: Tensor | None) -> tuple[tuple[Tensor, ...], tuple[int, ...]]:
    """Automatic broadcasting of missing dimensions"""
    if y is None:
        xd: int = x.shape[-1]
        return (x.view(-1, xd).contiguous(),), x.shape[:-1]

    check_broadcastable(x, y)

    xs: tuple[int, ...] = x.shape[:-1]
    xd = x.shape[-1]
    ys: tuple[int, ...] = y.shape[:-1]
    yd: int = y.shape[-1]
    out_shape: list[int] = [max(n, m) for (n, m) in zip(xs, ys, strict=False)]

    if x.shape[:-1] == y.shape[-1]:
        x1: Tensor = x.view(-1, xd)
        y1: Tensor = y.view(-1, yd)

    else:
        x_expand: list[int] = [m if n == 1 else 1 for (n, m) in zip(xs, ys, strict=False)]
        y_expand: list[int] = [n if m == 1 else 1 for (n, m) in zip(xs, ys, strict=False)]
        x1 = x.repeat(x_expand + [1]).reshape(-1, xd).contiguous()
        y1 = y.repeat(y_expand + [1]).reshape(-1, yd).contiguous()

    return (x1, y1), tuple(out_shape)
