"""recipes/colmap-src/tools/caspar_rcs_helpers.py on a synthetic allocation layout (no COLMAP checkout needed)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT: Path = Path(__file__).resolve().parents[2] / "recipes/colmap-src/tools/caspar_rcs_helpers.py"
NODES: dict[str, int] = {"PinholePose": 6, "Point": 4, "SimpleRadialPose": 6}


def _helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("caspar_rcs_helpers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module: ModuleType = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _allocations() -> str:
    """The generated constructor's shape: per family one block per node, with end markers for p, z and step."""
    lines: list[str] = []
    for family in ("p", "z", "step", "w", "r_0", "r_k"):
        for node, width in NODES.items():
            lines.append(f"  nodes__{node}__{family}_ =\n      assign_and_increment<float>(origin_ptr_, offset, {width} * {node}_num_, 4);")
            if family in ("p", "z", "step"):
                lines.append(f"  nodes__{node}__{family}_end__ = assign_and_increment<float>(origin_ptr_, offset, 0 * 0, 4);")
    return "\n" + "\n".join(lines) + "\n"


def test_the_layout_gives_the_node_order_and_the_helpers_name_its_ends() -> None:
    helpers: ModuleType = _helpers()
    nodes: list[str] = helpers.layout(_allocations())
    assert nodes == list(NODES)
    block: str = helpers.render(nodes)
    assert "nodes__PinholePose__p_, nodes__Point__p_, nodes__SimpleRadialPose__p_, nodes__SimpleRadialPose__p_end__" in block


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "  nodes__SimpleRadialPose__p_ =",
            "  solver__scratch_ = assign_and_increment<float>(origin_ptr_, offset, 16, 4);\n  nodes__SimpleRadialPose__p_ =",
            "allocated inside the `p` blocks",
        ),
        ("offset, 4 * Point_num_, 4);\n  nodes__Point__z_end__", "offset, 3 * Point_num_, 4);\n  nodes__Point__z_end__", "holds 3 \\* Point_num_"),
        ("offset, 4 * Point_num_, 4);\n  nodes__Point__step_end__", "offset, 4 * Point_num_, 1);\n  nodes__Point__step_end__", "aligned to 1 elements"),
    ],
    ids=["scratch-between-blocks", "size-differs-from-r_k", "alignment-differs"],
)
def test_the_layout_refuses_what_the_span_copies_would_get_wrong(old: str, new: str, message: str) -> None:
    helpers: ModuleType = _helpers()
    text: str = _allocations()
    assert text.count(old) == 1
    with pytest.raises(SystemExit, match=message):
        helpers.layout(text.replace(old, new))
