"""The layout-derived part of `caspar-reduced-camera-system.patch`: the reduced camera system's span helpers in solver.cc.

usage: python caspar_rcs_helpers.py <colmap checkout with the recipe's patches applied>

The helpers address the camera side of a per-node vector as the blocks before and after the Point block, so their bodies
name the first and last node types of the generated layout. This script reads that layout from solver.cc's allocation
code, refuses a layout the span copies would get wrong, renders the helper block and checks that solver.cc holds it
verbatim. On a mismatch (a regenerated solver with another node set) it prints the rendered block: put it in place of
the old one and cut the patch again (`patches/PROVENANCE.md`). The rest of the patch is hand-written.

Follow-up: the helpers belong in Symforce-Caspar's `solver.cc.jinja`, which knows the layout without parsing it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SOLVER_CC = "src/thirdparty/Symforce-Caspar/generated/f32/solver.cc"
FAMILIES: tuple[str, ...] = ("r_k", "p", "z", "step", "r_0", "w")
"""The per-node vectors the helpers copy between or zero by span; `r_k` is the reference order and sizes."""
ALLOCATION = re.compile(r"\n  (\w+) =\s*assign_and_increment<\w+>\(\s*origin_ptr_,\s*offset,\s*([^,]+?),\s*(\d+)\);")


def layout(solver_cc: str) -> list[str]:
    """The node types in allocation order, after checking every family against the rule the span helpers rely on.

    Each family is one contiguous run of allocations: one block per node type in `r_k`'s order, each with `r_k`'s element
    count, optionally each followed by a zero-size end marker, all with one alignment, and nothing else in between. A
    span copy from one family into another then lands on the same blocks, and the Point block, strictly inside, splits
    the run into exactly two camera-side spans.
    """
    allocations: list[tuple[str, str, str]] = ALLOCATION.findall(solver_cc)
    if len(allocations) != solver_cc.count("assign_and_increment<"):
        raise SystemExit("generated layout changed: an allocation does not have the expected form")
    sizes: dict[str, str] = {}
    alignment: str = ""
    nodes: list[str] = []
    for family in FAMILIES:
        pattern = re.compile(rf"nodes__(\w+?)__{family}_(end__)?")
        members: list[int] = [index for index, (name, _, _) in enumerate(allocations) if pattern.fullmatch(name)]
        if not members:
            raise SystemExit(f"generated layout changed: no `{family}` blocks")
        run: list[tuple[str, str, str]] = allocations[members[0] : members[-1] + 1]
        blocks: list[str] = []
        for position, (name, count, align) in enumerate(run):
            match: re.Match[str] | None = pattern.fullmatch(name)
            if match is None:
                raise SystemExit(f"generated layout changed: `{name}` is allocated inside the `{family}` blocks")
            node, end = match.group(1), match.group(2) is not None
            if end and (position == 0 or run[position - 1][0] != f"nodes__{node}__{family}_" or count != "0 * 0"):
                raise SystemExit(f"generated layout changed: `{name}` is not a zero-size marker right after its block")
            if not end:
                blocks.append(node)
                if family == "r_k":
                    sizes[node] = count
                elif sizes.get(node) != count:
                    raise SystemExit(f"generated layout changed: `{name}` holds {count} elements, r_k's block {sizes.get(node)}")
            alignment = alignment or align
            if align != alignment:
                raise SystemExit(f"generated layout changed: `{name}` is aligned to {align} elements, the others to {alignment}")
        nodes = nodes or blocks
        if blocks != nodes or len(run) not in (len(nodes), 2 * len(nodes)):
            raise SystemExit(f"generated layout changed: `{family}` is not one block per node type in r_k's order")
        if "Point" not in nodes[1:-1]:
            raise SystemExit(f"generated layout changed: Point is not strictly inside `{family}`")
    return nodes


def render(nodes: list[str]) -> str:
    """The helper block for this node order (inserted in solver.cc before `GraphSolver::DoAlphaFirst`)."""
    first, last = nodes[0], nodes[-1]
    after: str = nodes[nodes.index("Point") + 1]

    def point_end(family: str) -> str:
        # The Point block ends where the next node type's block starts (`layout` refuses anything in between).
        return f"nodes__{after}__{family}_"

    def cameras(family: str) -> str:
        return f"nodes__{first}__{family}_, nodes__Point__{family}_, {point_end(family)}, nodes__{last}__{family}_end__"

    def points(family: str) -> str:
        return f"nodes__Point__{family}_, {point_end(family)}"

    return f'''
// ---------------------------------------------------------------------------------------------------------------
// Reduced camera system. The PCG above runs on the full normal equations, cameras and points together, so a long
// trajectory's low-frequency (drift) modes barely move in pcg_iter_max iterations. With reduced_camera_system the
// Point nodes are eliminated: the PCG runs on S = H_cc - H_cp H_pp^-1 H_pc (implicit, damped), the points are
// back-substituted after it, and the PCG stops on the Q criterion (Ceres / rootba) instead of the residual norm.
// Everything reuses the generated kernels: the matvec is DoJtjpDirect (it accumulates the coupling terms into w,
// which enters holding the block-diagonal M p), and PointNormalize is the exact damped H_pp^-1 per point. Keeping
// the Point part of r_k at zero keeps the Point parts of z, p, Mp and step at zero through the unchanged PCG updates.
// The camera side of a vector is every block before and after the Point block ({first} .. {last}, from the layout).
// ---------------------------------------------------------------------------------------------------------------
void GraphSolver::RcsPointSolve(float* in, float* out) {{
  PointNormalize(nodes__Point__precond_diag_, Point_num_, nodes__Point__precond_tril_, Point_num_, in, Point_num_,
                 solver__current_diag_, out, Point_num_, Point_num_);
}}

void GraphSolver::RcsCopyCameras(const float* src_first, const float* src_point, const float* src_point_end,
                                 const float* src_last_end, float* dst_first, float* dst_point_end) {{
  Copy(src_first, src_point, dst_first);
  Copy(src_point_end, src_last_end, dst_point_end);
}}

void GraphSolver::RcsZeroCameras(float* first, float* point, float* point_end, float* last_end) {{
  Zero(first, point);
  Zero(point_end, last_end);
}}

// r_k = r_0 reduced: g_c - H_cp H_pp^-1 g_p on the cameras, 0 on the points; Mp = r_k (M p_0 = r_0 in the PCG).
void GraphSolver::RcsReduceRhs() {{
  Copy(marker__r_0_start_, marker__r_0_end_, marker__r_k_start_);
  RcsPointSolve(nodes__Point__r_k_, nodes__Point__z_);
  Scale({points("z")}, -1.0f);
  RcsZeroCameras({cameras("p")});
  Copy({points("z")}, nodes__Point__p_);
  Copy(marker__r_k_start_, marker__r_k_end_, marker__w_start_);
  DoJtjpDirect();
  Copy(marker__w_start_, marker__w_end_, marker__r_k_start_);
  Zero({points("r_k")});
  Copy(marker__r_k_start_, marker__r_k_end_, marker__Mp_start_);
  Zero({points("p")});
  Zero({points("z")});
  Zero({points("w")});
}}

// w = S p on the cameras and 0 on the points; w enters holding M p, p has a zero Point part.
void GraphSolver::RcsMatvec() {{
  DoJtjpDirect();  // w_points = H_pc p
  RcsPointSolve(nodes__Point__w_, nodes__Point__z_);
  Scale({points("z")}, -1.0f);
  RcsCopyCameras({cameras("p")}, nodes__{first}__z_, {point_end("z")});
  RcsZeroCameras({cameras("p")});
  Copy({points("z")}, nodes__Point__p_);
  DoJtjpDirect();  // w_cameras += H_cp (-H_pp^-1 H_pc p)
  RcsCopyCameras({cameras("z")}, nodes__{first}__p_, {point_end("p")});
  Zero({points("p")});
  Zero({points("z")});
  Zero({points("w")});
}}

// step_points = H_pp^-1 (g_p - H_pc step_cameras).
void GraphSolver::RcsBackSubstitute() {{
  RcsCopyCameras({cameras("step")}, nodes__{first}__p_, {point_end("p")});
  Zero({points("p")});
  Zero(marker__w_start_, marker__w_end_);
  DoJtjpDirect();
  Subtract({points("r_0")}, nodes__Point__w_, nodes__Point__w_);
  RcsPointSolve(nodes__Point__w_, nodes__Point__step_);
  Zero({points("w")});
}}

'''


def main(root: Path) -> None:
    solver_cc: str = (root / SOLVER_CC).read_text()
    block: str = render(layout(solver_cc))
    if block + "void GraphSolver::DoAlphaFirst() {" not in solver_cc:
        print(block)
        raise SystemExit(f"{root / SOLVER_CC}: the helper block differs from the layout's (printed above)")
    print(f"{root / SOLVER_CC}: helper block matches the layout")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
