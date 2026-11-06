"""Freeze upstream lietorch's outputs into `fixtures/lietorch_references.rrd` (run once, needs lietorch and a CUDA GPU).

Run through `pixi run -e lietorch-mojo-dev lietorch-mojo-freeze-references` (a `pixi exec`
environment with lietorch; it never enters the workspace lock). This module draws every input
the parity tests use, with the seeds and distributions the tests have always used, runs lietorch
on CPU and CUDA in float32 and float64, writes inputs and outputs as static records
(`references.py`), and reads the file back to check that every tensor round-trips bit-exactly.

Raw-op input sets keep every `ROW_STRIDE`-th row of their 256 rows. Each edge value of
`cases.py` fills a block of at least 26 consecutive rows, so every edge value stays in. The
purely random sets (`RANDOM_CASES`, no edge values) keep every `RANDOM_ROW_STRIDE`-th row.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import rerun as rr
import torch
from cases import OpCase, ill_conditioned_case, op_cases
from groups_common import GROUP_NAMES, random_group_data
from parity import (
    ALL_OPS,
    API_FORWARD_INPUTS,
    BROADCAST_SHAPES,
    COMPOSITE_SUITES,
    CONVERSIONS,
    DTYPE_NAMES,
    F32_CPU_REFERENCE_GROUPS,
    REFERENCE_DEVICES,
    SGD_NODES,
    CompositeSuite,
    OpInputs,
    api_forward_outputs,
    broadcast_gradients,
    call_op,
    composite_gradients,
    lie_group_parameter_gradient,
    make_inputs,
    parity_composites,
    sgd_trajectory,
    shapes_id,
)
from references import REFERENCE_FILE, encode, load
from torch import Tensor

LIETORCH_COMMIT: str = "3e2695dd"
LIETORCH_SOURCE: str = "https://github.com/VSLAM-LAB/lietorch"
ROW_STRIDE: int = 16
RANDOM_ROW_STRIDE: int = 32
RANDOM_CASES: frozenset[str] = frozenset({"random", "random_large_angle", "large_translation", "unnormalised"})
F64: torch.dtype = torch.float64
DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.float64)
Values = dict[str, Tensor | str]


def _unit_group_data(group_name: str, shape: tuple[int, ...], gen: torch.Generator) -> Tensor:
    """The SE3/SO3 and RxSO3/Sim3 API tests' random elements (float64, CPU).

    SE3: t ~ N(0, 1), q normalised Gaussian; SO3: the q of that draw. RxSO3: q, s = exp(0.5 N(0, 1));
    Sim3: q, s, then t.
    """
    if group_name in ("SE3", "SO3"):
        t: Tensor = torch.randn(*shape, 3, generator=gen, dtype=F64)
        q: Tensor = torch.randn(*shape, 4, generator=gen, dtype=F64)
        data: Tensor = torch.cat([t, q / q.norm(dim=-1, keepdim=True)], dim=-1)
        return data if group_name == "SE3" else data[..., 3:].contiguous()
    q = torch.randn(*shape, 4, generator=gen, dtype=F64)
    q = q / q.norm(dim=-1, keepdim=True)
    s: Tensor = torch.exp(0.5 * torch.randn(*shape, 1, generator=gen, dtype=F64))
    parts: list[Tensor] = [q, s] if group_name == "RxSO3" else [torch.randn(*shape, 3, generator=gen, dtype=F64), q, s]
    return torch.cat(parts, dim=-1)


def _half_group_data(group_name: str, shape: tuple[int, ...], gen: torch.Generator) -> Tensor:
    return random_group_data(group_name, shape, gen, F64, "cpu")


SUITE_DATA: dict[str, Callable[[str, tuple[int, ...], torch.Generator], Tensor]] = {"unit": _unit_group_data, "half": _half_group_data}


def _on(inputs: dict[str, Tensor], device: str) -> dict[str, Tensor]:
    return {k: v.to(device) for k, v in inputs.items()}


def _numbered(prefix: str, tensors: list[Tensor]) -> Values:
    return {f"{prefix}/{k}": t.detach() for k, t in enumerate(tensors)}


def _rows(inp: OpInputs, rows: Tensor) -> OpInputs:
    return inp.map(lambda t: t[rows].contiguous())


def _op_record(lb: ModuleType, G: type, case: OpCase, dtypes: tuple[torch.dtype, ...], truth: bool) -> Values:
    """Inputs, lietorch outputs per device and dtype, and (if `truth`) float64 outputs on the float32-rounded inputs."""
    rows: Tensor = torch.arange(0, case.X.shape[0], RANDOM_ROW_STRIDE if case.name in RANDOM_CASES else ROW_STRIDE)
    values: Values = {}
    for dtype in dtypes:
        inp: OpInputs = _rows(make_inputs(G, case, dtype, "cpu"), rows)
        values |= {f"in/{DTYPE_NAMES[dtype]}/{k}": v for k, v in inp.named().items()}
        for device in REFERENCE_DEVICES:
            on_device: OpInputs = inp.map(lambda t, d=device: t.to(d))
            for op in ALL_OPS:
                values |= _numbered(f"out/{device}/{DTYPE_NAMES[dtype]}/{op}", call_op(lb, G, op, on_device))
                if truth and dtype == torch.float32:
                    values |= _numbered(f"truth/{device}/{op}", call_op(lb, G, op, on_device.map(lambda t: t.double())))
    return values


def _records(lietorch: ModuleType, lb: ModuleType) -> dict[str, Values]:
    records: dict[str, Values] = {}
    for name in GROUP_NAMES:
        G: type = getattr(lietorch, name)
        K: int = G.manifold_dim
        for case in op_cases(name):
            records[f"/ops/{name}/{case.name}"] = _op_record(lb, G, case, DTYPES, truth=name in F32_CPU_REFERENCE_GROUPS)
        if name in ("SE3", "SO3"):
            records[f"/ops_small_angle/{name}"] = _op_record(lb, G, ill_conditioned_case(name), (torch.float32,), truth=True)

        for shapes in BROADCAST_SHAPES:
            gen: torch.Generator = torch.Generator().manual_seed(0)
            xs, ys = shapes
            drawn: list[Tensor] = [
                random_group_data(name, xs, gen, F64, "cpu"),
                random_group_data(name, ys, gen, F64, "cpu"),
                torch.randn(*ys, K, generator=gen, dtype=F64),
                10 * torch.randn(*ys, 3, generator=gen, dtype=F64),
                torch.randn(*ys, 4, generator=gen, dtype=F64),
                torch.rand(*xs, generator=gen, dtype=F64),
            ]
            values: Values = {}
            for dtype in DTYPES:
                inputs: dict[str, Tensor] = {k: v.to(dtype) for k, v in zip(API_FORWARD_INPUTS, drawn, strict=True)}
                values |= {f"in/{DTYPE_NAMES[dtype]}/{k}": v for k, v in inputs.items()}
                for device in REFERENCE_DEVICES:
                    outs: dict[str, Tensor] = api_forward_outputs(lietorch, name, _on(inputs, device))
                    values |= {f"out/{device}/{DTYPE_NAMES[dtype]}/{k}": v for k, v in outs.items()}
            records[f"/api/forward/{name}/{shapes_id(shapes)}"] = values

        for suite in COMPOSITE_SUITES:
            for composite, (_, widths) in parity_composites(name).items():
                records[f"/api/composite/{suite.name}/{name}/{composite}"] = _composite_record(lietorch, suite, G, composite, widths)

        gen = torch.Generator().manual_seed(4)
        bcast: dict[str, Tensor] = {
            "x": random_group_data(name, (4, 1), gen, F64, "cpu"),
            "y": random_group_data(name, (1, 6), gen, F64, "cpu"),
            "a": torch.randn(1, 6, K, generator=gen, dtype=F64),
            "p": torch.randn(1, 6, 3, generator=gen, dtype=F64),
        }
        values = {f"in/{k}": v for k, v in bcast.items()}
        for device in REFERENCE_DEVICES:
            values |= _numbered(f"grad/{device}", broadcast_gradients(lietorch, name, _on(bcast, device)))
        records[f"/api/broadcast_gradients/{name}"] = values

        gen = torch.Generator().manual_seed(4)
        lgp: dict[str, Tensor] = {"x": _unit_group_data(name, (8,), gen), "pts": torch.randn(8, 3, generator=gen, dtype=F64)}
        values = {f"in/{k}": v for k, v in lgp.items()}
        for device in REFERENCE_DEVICES:
            values[f"grad/{device}"] = lie_group_parameter_gradient(lietorch, name, _on(lgp, device)).detach()
        records[f"/api/lie_group_parameter/{name}"] = values

        gen = torch.Generator().manual_seed(5)
        sgd: dict[str, Tensor] = {
            "target": random_group_data(name, (SGD_NODES,), gen, F64, "cpu"),
            "start": random_group_data(name, (SGD_NODES,), gen, F64, "cpu"),
            "ii": torch.randint(0, SGD_NODES, (40,), generator=gen),
            "jj": torch.randint(0, SGD_NODES, (40,), generator=gen),
        }
        values = {f"in/{k}": v for k, v in sgd.items()}
        for device in REFERENCE_DEVICES:
            grad0, final = sgd_trajectory(lietorch, name, _on(sgd, device))
            values |= {f"grad0/{device}": grad0, f"final/{device}": final.detach()}
        records[f"/api/sgd/{name}"] = values

        data: Tensor = _unit_group_data(name, (4,), torch.Generator().manual_seed(1))
        records[f"/api/repr/{name}"] = {"in/x": data, "repr": repr(G(data))}

    gen = torch.Generator().manual_seed(6)
    conv: dict[str, Tensor] = {k: random_group_data(g, (3,), gen, F64, "cpu") for k, g in (("se3", "SE3"), ("sim3", "Sim3"), ("so3", "SO3"))}
    records["/api/conversions"] = {f"in/{k}": v for k, v in conv.items()} | {f"out/{k}": fn(lietorch, conv) for k, fn in CONVERSIONS.items()}
    return records


def _composite_record(lietorch: ModuleType, suite: CompositeSuite, G: type, composite: str, widths: list[int | str]) -> Values:
    name: str = G.group_name
    gen: torch.Generator = torch.Generator().manual_seed(2)
    x: Tensor = SUITE_DATA[suite.name](name, (suite.batch,), gen)
    y: Tensor = SUITE_DATA[suite.name](name, (suite.batch,), gen)
    dims: dict[str, int] = {"K": G.manifold_dim, "N": G.embedded_dim}
    base: list[Tensor] = [
        suite.tangent_scale * torch.randn(suite.batch, dims[w] if isinstance(w, str) else w, generator=gen, dtype=F64) for w in widths
    ]
    values: Values = {}
    for dtype in DTYPES:
        inputs: dict[str, Tensor] = {"x": x.to(dtype), "y": y.to(dtype)} | {f"in{k}": b.to(dtype) for k, b in enumerate(base)}
        if composite == "scale_log":
            inputs["in1"] = inputs["in1"].abs() + 0.5
        weight: list[Tensor] = []

        def draw_weight(shape: torch.Size, dtype: torch.dtype = dtype, weight: list[Tensor] = weight) -> Tensor:
            w: Tensor = torch.randn(shape, generator=torch.Generator().manual_seed(3), dtype=F64).to(dtype)
            weight.append(w)
            return w

        for device in REFERENCE_DEVICES:
            grads: list[Tensor] = composite_gradients(lietorch, name, composite, _on(inputs, device), lambda s, d=device: draw_weight(s).to(d))
            values |= _numbered(f"grad/{device}/{DTYPE_NAMES[dtype]}", grads)
        assert torch.equal(weight[0], weight[1])
        values |= {f"in/{DTYPE_NAMES[dtype]}/{k}": v for k, v in inputs.items()} | {f"in/{DTYPE_NAMES[dtype]}/weight": weight[0]}
    return values


def _lietorch_build() -> dict[str, str]:
    """The installed lietorch conda package (version, build string, channel) from conda-meta."""
    metas: list[Path] = sorted(Path(sys.prefix, "conda-meta").glob("lietorch-*.json"))
    assert len(metas) == 1, metas
    meta: dict = json.loads(metas[0].read_text())
    return {
        "version": meta["version"],
        "build": meta["build"],
        "channel": meta.get("channel", ""),
        "commit": LIETORCH_COMMIT,
        "source": LIETORCH_SOURCE,
    }


def main(out: Path = REFERENCE_FILE) -> None:  # noqa: V103 (called by tools/freeze_lietorch_references.py)
    import lietorch
    import lietorch_backends

    assert torch.cuda.is_available(), "the references need lietorch's CUDA build too"
    records: dict[str, Values] = _records(lietorch, lietorch_backends)

    out.parent.mkdir(parents=True, exist_ok=True)
    rec: rr.RecordingStream = rr.RecordingStream("lietorch_mojo_references", recording_id="lietorch-references")
    rec.save(out)
    build: dict[str, str] = _lietorch_build()
    rec.send_property(
        "lietorch",
        rr.AnyValues(version=build["version"], build=build["build"], channel=build["channel"], commit=LIETORCH_COMMIT, source=LIETORCH_SOURCE),
    )
    rec.send_property(
        "freeze",
        rr.AnyValues(
            torch=torch.__version__,
            cuda_device=torch.cuda.get_device_name(),
            row_stride=str(ROW_STRIDE),
            random_row_stride=str(RANDOM_ROW_STRIDE),
            python=sys.version.split()[0],
        ),
    )
    for entity, values in records.items():
        rec.log(entity, rr.AnyValues(**encode(values)), static=True)
    rec.flush()
    rec.disconnect()

    # Bit-exact round trip: same dtype, shape and bits for every tensor, same text.
    back = load.__wrapped__(out)
    for entity, values in records.items():
        got = back[entity]
        for k, v in values.items():
            if isinstance(v, str):
                assert got.texts[k] == v, (entity, k)
            else:
                t: Tensor = got.tensors[k]
                assert t.dtype == v.dtype and t.shape == v.shape and torch.equal(t, v.detach().cpu()), (entity, k)
        assert len(got.tensors) + len(got.texts) == len(values), entity
    n_tensors: int = sum(len(r.tensors) for r in back.values())
    print(f"wrote {out} ({out.stat().st_size / 2**20:.2f} MiB): {len(records)} records, {n_tensors} tensors, round trip bit-exact")
