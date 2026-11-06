"""Finite-difference gradient checks: a port of lietorch `gradcheck.py` (VSLAM-LAB/lietorch @ 3e2695dd).

Upstream ships a copy of torch 1.8's `torch.autograd.gradcheck` whose
`gradcheck` returns `(analytical, numerical)`, the two Jacobians of the first
differentiable output, instead of a bool (`run_tests.py` compares them
itself). This port keeps that behaviour and the public names and signatures:
`gradcheck`, `gradgradcheck`, `get_analytical_jacobian`,
`get_numerical_jacobian`, `make_jacobian`, `iter_tensors`, `zero_gradients`.

Deviations, all on paths a Lie group never takes:
- Sparse and MKLDNN inputs raise `ValueError` (upstream has partial support).
- `gradgradcheck` draws random grad outputs with `torch.randn_like`;
  upstream calls `torch.testing.randn_like` and `make_non_contiguous`, which
  torch 2 removed, so upstream raises `AttributeError` there.
"""

import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import product
from typing import TypeAlias

import torch
from torch import Tensor
from torch.overrides import is_tensor_like

TORCH_MAJOR: int = int(torch.__version__.split(".")[0])
TORCH_MINOR: int = int(torch.__version__.split(".")[1])

TensorOrTensors: TypeAlias = Tensor | Sequence[Tensor]
JacobianTree: TypeAlias = Tensor | Sequence[Tensor] | None
GradcheckResult: TypeAlias = bool | tuple[tuple[Tensor, ...], tuple[Tensor, ...]]


def zero_gradients(x: object) -> None:
    if isinstance(x, Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, Iterable):
        for elem in x:
            zero_gradients(elem)


def make_jacobian(input: object, num_out: int) -> JacobianTree:
    if is_tensor_like(input):
        if not input.is_floating_point() and not input.is_complex():
            return None
        if not input.requires_grad:
            return None
        return input.new_zeros((input.nelement(), num_out), dtype=input.dtype, layout=torch.strided)
    if isinstance(input, Iterable) and not isinstance(input, str):
        jacobians: list[Tensor] = [j for j in (make_jacobian(elem, num_out) for elem in input) if j is not None]
        if not jacobians:
            return None
        return type(input)(jacobians)
    return None


def iter_tensors(x: object, only_requiring_grad: bool = False) -> Iterator[Tensor]:
    if is_tensor_like(x):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, Iterable) and not isinstance(x, str):
        for elem in x:
            yield from iter_tensors(elem, only_requiring_grad)


def _check_dense(x: Tensor) -> None:
    if x.is_sparse or x.layout == torch._mkldnn:
        raise ValueError("lietorch_mojo.gradcheck supports dense (strided) tensors only")


def get_numerical_jacobian(
    fn: Callable[[Sequence[Tensor]], Tensor],
    input: Sequence[Tensor],
    target: Sequence[Tensor] | None = None,
    eps: float = 1e-3,
    grad_out: float | complex = 1.0,
) -> JacobianTree:
    """Central-difference Jacobian of `fn(input)` with respect to `target` (default `input`)."""
    if target is None:
        target = input
    output_size: int = fn(input).numel()
    jacobian: JacobianTree = make_jacobian(target, output_size)

    def compute_gradient(x: Tensor, idx: tuple[int, ...], delta: float | complex) -> Tensor:
        orig: float | complex = x[idx].item()
        x[idx] = orig - delta
        outa: Tensor = fn(input).clone()
        x[idx] = orig + delta
        outb: Tensor = fn(input).clone()
        x[idx] = orig
        return ((outb - outa) / (2 * eps)).detach().reshape(-1)

    for x_tensor, d_tensor in zip(iter_tensors(target, True), iter_tensors(jacobian), strict=False):
        _check_dense(x_tensor)
        x_data: Tensor = x_tensor.data  # .data gets around the autograd version check
        for d_idx, x_idx in enumerate(product(*[range(m) for m in x_data.size()])):
            ds_dx: Tensor = compute_gradient(x_data, x_idx, eps)
            if x_data.is_complex():  # C -> C, C -> R: Wirtinger derivatives
                ds_dy: Tensor = compute_gradient(x_data, x_idx, eps * 1j)
                conj_w_d: Tensor = 0.5 * (ds_dx + ds_dy * 1j)
                w_d: Tensor = 0.5 * (ds_dx - ds_dy * 1j)
                d_tensor[d_idx] = complex(grad_out).conjugate() * conj_w_d + grad_out * w_d.conj()
            elif ds_dx.is_complex():  # R -> C
                d_tensor[d_idx] = torch.real(complex(grad_out).conjugate() * ds_dx)
            else:  # R -> R
                d_tensor[d_idx] = ds_dx * grad_out
    return jacobian


def get_analytical_jacobian(
    input: Sequence[Tensor], output: Tensor, nondet_tol: float = 0.0, grad_out: float | complex = 1.0
) -> tuple[JacobianTree, bool, bool, bool]:
    """Autograd Jacobian of `output`; also reports reentrancy and whether gradient sizes and dtypes are right."""
    _check_dense(output)
    diff_input_list: list[Tensor] = list(iter_tensors(input, True))
    jacobian: JacobianTree = make_jacobian(input, output.numel())
    jacobian_reentrant: JacobianTree = make_jacobian(input, output.numel())
    grad_output: Tensor = torch.zeros_like(output, memory_format=torch.legacy_contiguous_format)
    flat_grad_output: Tensor = grad_output.view(-1)
    reentrant: bool = True
    correct_grad_sizes: bool = True
    correct_grad_types: bool = True

    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = grad_out
        for jacobian_c in (jacobian, jacobian_reentrant):
            grads_input: tuple[Tensor | None, ...] = torch.autograd.grad(output, diff_input_list, grad_output, retain_graph=True, allow_unused=True)
            for jacobian_x, d_x, x in zip(jacobian_c, grads_input, diff_input_list, strict=False):
                if d_x is not None and d_x.size() != x.size():
                    correct_grad_sizes = False
                elif d_x is not None and d_x.dtype != x.dtype:
                    correct_grad_types = False
                elif jacobian_x.numel() != 0:
                    if d_x is None:
                        jacobian_x[:, i].zero_()
                    else:
                        d_x_dense: Tensor = d_x.to_dense() if d_x.layout != torch.strided else d_x
                        assert jacobian_x[:, i].numel() == d_x_dense.numel()
                        jacobian_x[:, i] = d_x_dense.contiguous().view(-1)

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant, strict=False):
        if jacobian_x.numel() != 0 and (jacobian_x - jacobian_reentrant_x).abs().max() > nondet_tol:
            reentrant = False

    return jacobian, reentrant, correct_grad_sizes, correct_grad_types


def _as_tuple(x: object) -> tuple:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def _differentiable_outputs(x: object) -> tuple[Tensor, ...]:
    return tuple(o for o in _as_tuple(x) if o.requires_grad)


def gradcheck(
    func: Callable[..., TensorOrTensors],
    inputs: TensorOrTensors,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
    check_sparse_nnz: bool = False,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
) -> GradcheckResult:
    """Analytical and numerical Jacobians of `func`'s first differentiable output, as upstream returns them.

    `atol`, `rtol`, `check_undefined_grad` and `check_grad_dtypes` are
    accepted for signature compatibility; upstream returns before it uses
    them. With no differentiable output, returns True when the numerical
    gradient is zero (else fails like upstream).
    """

    def fail_test(msg: str) -> bool:
        if raise_exception:
            raise RuntimeError(msg)
        return False

    tupled_inputs: tuple = _as_tuple(inputs)
    if not check_sparse_nnz and any(t.is_sparse for t in tupled_inputs if isinstance(t, Tensor)):
        return fail_test("gradcheck expects all tensor inputs are dense when check_sparse_nnz is set to False.")

    any_input_requiring_grad: bool = False
    for idx, inp in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if inp.dtype not in (torch.float64, torch.complex128):
                warnings.warn(
                    f"Input #{idx} requires gradient and is not a double precision floating point or complex. "
                    "This check will likely fail if all the inputs are not of double precision floating point or complex. ",
                    stacklevel=2,
                )
            _check_dense(inp)
            if not all(st > 0 or sz <= 1 for st, sz in zip(inp.stride(), inp.size(), strict=True)):
                raise RuntimeError(
                    f"The {idx}th input has a dimension with stride 0. gradcheck only supports inputs that are non-overlapping "
                    "to be able to compute the numerical gradients correctly. You should call .contiguous on the input "
                    "before passing it to gradcheck."
                )
            any_input_requiring_grad = True
            inp.retain_grad()
    if not any_input_requiring_grad:
        raise ValueError("gradcheck expects at least one input tensor to require gradient, but none of the them have requires_grad=True.")

    func_out: TensorOrTensors = func(*tupled_inputs)
    output: tuple[Tensor, ...] = _differentiable_outputs(func_out)

    if not output:
        for i in range(len(_as_tuple(func_out))):

            def fn_i(input: Sequence[Tensor], i: int = i) -> Tensor:
                return _as_tuple(func(*input))[i]

            numerical: JacobianTree = get_numerical_jacobian(fn_i, tupled_inputs, eps=eps)
            for n in iter_tensors(numerical):
                if torch.ne(n, 0).sum() > 0:
                    return fail_test("Numerical gradient for function expected to be zero")
        return True

    def fn(input: Sequence[Tensor]) -> Tensor:
        return _as_tuple(func(*input))[0]

    analytical_result: tuple[JacobianTree, bool, bool, bool] = get_analytical_jacobian(tupled_inputs, output[0], nondet_tol=nondet_tol)
    analytical: tuple[Tensor, ...] = tuple(iter_tensors(analytical_result[0]))
    numerical_first: tuple[Tensor, ...] = tuple(iter_tensors(get_numerical_jacobian(fn, tupled_inputs, eps=eps)))
    return analytical, numerical_first


def gradgradcheck(
    func: Callable[..., TensorOrTensors],
    inputs: TensorOrTensors,
    grad_outputs: TensorOrTensors | None = None,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    gen_non_contig_grad_outputs: bool = False,
    raise_exception: bool = True,
    nondet_tol: float = 0.0,
    check_undefined_grad: bool = True,
    check_grad_dtypes: bool = False,
) -> GradcheckResult:
    """`gradcheck` of the gradients of `func` with respect to its inputs, for the given (or random) grad outputs."""
    tupled_inputs: tuple = _as_tuple(inputs)

    if grad_outputs is None:

        def randn_like(x: Tensor) -> Tensor:
            y: Tensor = torch.randn_like(x if (x.is_floating_point() or x.is_complex()) else x.double(), memory_format=torch.legacy_contiguous_format)
            if gen_non_contig_grad_outputs:
                y = torch.stack([y, torch.zeros_like(y)], dim=-1)[..., 0]
            return y.requires_grad_()

        tupled_grad_outputs: tuple = tuple(randn_like(x) for x in _as_tuple(func(*tupled_inputs)))
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)

    num_outputs: int = len(tupled_grad_outputs)

    def new_func(*args: Tensor) -> tuple[Tensor, ...]:
        input_args: tuple[Tensor, ...] = args[:-num_outputs]
        grads: tuple[Tensor, ...] = args[-num_outputs:]
        outputs: tuple[Tensor, ...] = _differentiable_outputs(func(*input_args))
        diff_args: tuple[Tensor, ...] = tuple(x for x in input_args if isinstance(x, Tensor) and x.requires_grad)
        return torch.autograd.grad(outputs, diff_args, grads, create_graph=True)

    return gradcheck(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        eps,
        atol,
        rtol,
        raise_exception,
        nondet_tol=nondet_tol,
        check_undefined_grad=check_undefined_grad,
        check_grad_dtypes=check_grad_dtypes,
    )
