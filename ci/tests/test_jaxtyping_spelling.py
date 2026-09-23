"""Ratchet jaxtyping spelling in authored package code with per-file counts.

JT001 requires lowercase axis names. JT002 requires an explicit-width dtype on
numpy arrays: beartype checks dtype and rank per array, and a generic Float hides
the width (a float64 K under Float32 is the classic crash). Torch tensors are exempt,
because autocast and half precision legitimately vary their dtype. JT003 forbids
symbolic axes, which raise AnnotationError on every call without @jaxtyped. Genuine
numpy dtype polymorphism may carry the generic-dtype comment marker. Counts may
fall but must not exceed the baseline.
"""

import ast
import json
import re
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
BASELINE_PATH: Path = REPO_ROOT / "ci/jaxtyping_spelling_baseline.json"
# Shared path exclusions: generated directories, upstream files, and zipdepth's
# vendor boundary (its apis, catalog, and root __init__.py remain authored code).
SKIP_PATHS: re.Pattern[str] = re.compile(
    r"(?:^|/)(?:\.pixi|__pycache__|third_party|build|target|node_modules)(?:/|$)"
    r"|(?:^|/)upstream_[^/]*\.py$"
    r"|^packages/zipdepth/zipdepth/(?!(?:apis|catalog)(?:/|$)|__init__\.py$)"
)

DTYPES: set[str] = {
    "Float",
    "Float16",
    "Float32",
    "Float64",
    "BFloat16",
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Integer",
    "Num",
    "Inexact",
    "Real",
    "Complex",
    "Complex64",
    "Complex128",
    "Bool",
    "Shaped",
    "Key",
}
GENERIC_DTYPES: set[str] = {"Float", "Int", "UInt", "Integer", "Num", "Inexact", "Real", "Complex"}
# JT002 applies to numpy arrays only; torch dtypes vary under autocast and half precision.
NUMPY_ARRAY: str = "ndarray"


def _violations(source: str) -> list[tuple[int, str, str]]:
    """Return (line, rule, offending text) for each matching AST hint."""
    problems: list[tuple[int, str, str]] = []
    tree: ast.Module = ast.parse(source)
    lines: list[str] = source.splitlines()
    # Mask string literals so text inside a string cannot act as a comment marker.
    comment_lines: list[bytes] = [line.encode("utf-8") for line in lines]
    for literal in ast.walk(tree):
        if isinstance(literal, ast.JoinedStr) or (isinstance(literal, ast.Constant) and isinstance(literal.value, (str, bytes))):
            for index in range(literal.lineno - 1, literal.end_lineno):
                start: int = literal.col_offset if index == literal.lineno - 1 else 0
                end: int = literal.end_col_offset if index == literal.end_lineno - 1 else len(comment_lines[index])
                comment_lines[index] = comment_lines[index][:start] + b" " * (end - start) + comment_lines[index][end:]
    hints: list[ast.Subscript] = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)]
    for node in sorted(hints, key=lambda hint: (hint.lineno, hint.col_offset)):
        if not isinstance(node.value, (ast.Name, ast.Attribute)):
            continue
        dtype: str = node.value.id if isinstance(node.value, ast.Name) else node.value.attr
        if dtype not in DTYPES or not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
            continue
        array: ast.expr = node.slice.elts[0]
        is_numpy: bool = (isinstance(array, ast.Name) and array.id == NUMPY_ARRAY) or (
            isinstance(array, ast.Attribute) and array.attr == NUMPY_ARRAY
        )
        shape: ast.expr = node.slice.elts[1]
        if not isinstance(shape, ast.Constant) or not isinstance(shape.value, str):
            continue
        if dtype in GENERIC_DTYPES and is_numpy and not any(
            b"# jaxtyping: generic-dtype" in comment_lines[line - 1] for line in (node.lineno, shape.lineno, node.end_lineno)
        ):
            problems.append((node.lineno, "JT002", dtype))
        for token in shape.value.split():
            name: str = token.lstrip("*#_").split("=", maxsplit=1)[0]
            if name and name != "..." and not re.fullmatch(r"[0-9]+", name) and not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                problems.append((shape.lineno, "JT001", token))
            expression: str = token[1:] if token.startswith(("*", "#")) else token
            if re.search(r"[+\-/()*]", expression):
                problems.append((shape.lineno, "JT003", token))
    return sorted(problems, key=lambda problem: problem[0])


def _scan_packages() -> dict[str, list[tuple[int, str, str]]]:
    """Scan authored Python files, pruning excluded directories before descent."""
    files: dict[str, list[tuple[int, str, str]]] = {}
    for directory, subdirs, filenames in (REPO_ROOT / "packages").walk():
        subdirs[:] = [name for name in subdirs if not SKIP_PATHS.search((directory / name).relative_to(REPO_ROOT).as_posix())]
        for name in sorted(filenames):
            path: Path = directory / name
            relative: str = path.relative_to(REPO_ROOT).as_posix()
            if path.suffix == ".py" and not SKIP_PATHS.search(relative):
                problems: list[tuple[int, str, str]] = _violations(path.read_text(encoding="utf-8"))
                if problems:
                    files[relative] = problems
    return files


def _counts(files: dict[str, list[tuple[int, str, str]]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for path, problems in files.items():
        for _, rule, _ in problems:
            counts.setdefault(path, {})[rule] = counts.get(path, {}).get(rule, 0) + 1
    return counts


def _new_violations(files: dict[str, list[tuple[int, str, str]]], baseline: dict[str, dict[str, int]]) -> list[str]:
    """Report violations beyond each file/rule allowance in source order.

    Counts cannot identify which historical occurrence changed; the first N
    occurrences consume the baseline allowance and the remainder are reported.
    """
    problems: list[str] = []
    for path, violations in sorted(files.items()):
        seen: dict[str, int] = {}
        for line, rule, text in violations:
            seen[rule] = seen.get(rule, 0) + 1
            if seen[rule] > baseline.get(path, {}).get(rule, 0):
                problems.append(f"{path}:{line} {rule} {text}")
    return problems


def test_package_jaxtyping_spelling() -> None:
    baseline: dict[str, dict[str, int]] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    problems: list[str] = _new_violations(_scan_packages(), baseline)
    assert not problems, "\n".join(problems) + (
        "\nFix the spelling (lowercase axis names, a fixed-width dtype such as Float32, no symbolic axes),"
        " or mark genuine dtype polymorphism with `# jaxtyping: generic-dtype`."
        " Lowering counts is always welcome — regenerate with `pixi run -e ci --frozen ci-jaxtyping-baseline`."
    )


def test_lowercase_axes() -> None:
    assert _violations('x: Float32[Array, "H W 3 *B"]') == [(1, "JT001", "H"), (1, "JT001", "W"), (1, "JT001", "*B")]
    assert _violations('x: Float32[Array, "h w 3 n_joints=22 *batch #b _n _ *_ ..."]') == []


def test_explicit_width_and_escape_hatch() -> None:
    assert _violations('x: jt.Float[np.ndarray, "n"]') == [(1, "JT002", "Float")]
    assert _violations('x: Float32[ndarray, "n"]') == []
    assert _violations('x: Float[ndarray, "n"]  # jaxtyping: generic-dtype') == []
    assert _violations('x: Float[  # jaxtyping: generic-dtype\n ndarray, "n"\n]') == []
    assert _violations('x: Float[\n ndarray, "n"  # jaxtyping: generic-dtype\n]') == []
    assert _violations('note = "# jaxtyping: generic-dtype"; x: Float[ndarray, "n"]') == [(1, "JT002", "Float")]
    assert _violations('# jaxtyping: generic-dtype\nx: Float[ndarray, "n"]') == [(2, "JT002", "Float")]
    for dtype in ("Float", "Int", "UInt", "Integer", "Num", "Inexact", "Real", "Complex"):
        assert _violations(f'x: {dtype}[ndarray, "n"]') == [(1, "JT002", dtype)]
    # Torch (and other non-numpy) arrays are exempt: autocast varies their dtype.
    assert _violations('x: Float[Tensor, "b c"]') == []
    assert _violations('x: Float[torch.Tensor, "b c"]') == []
    for dtype in (
        "Float16",
        "Float32",
        "Float64",
        "BFloat16",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Complex64",
        "Complex128",
        "Bool",
        "Shaped",
        "Key",
    ):
        assert _violations(f'x: {dtype}[Array, "n"]') == []


def test_hints_are_found_in_all_ast_contexts() -> None:
    source: str = (
        'Alias = list[jt.Float32[Array, "B"] | None]\n'
        '@dataclass\nclass Record:\n    value: Float32[Array, "N"]\n'
        'def f(x: Float32[Array, "H"]) -> Float32[Array, "W"]:\n    pass\n'
    )
    assert _violations(source) == [(1, "JT001", "B"), (4, "JT001", "N"), (5, "JT001", "H"), (5, "JT001", "W")]
    assert _violations('x = Other[Array, "H"]; y = Float32[Array, shape]; z = Float32["H"]') == []


def test_symbolic_axes() -> None:
    problems: list[tuple[int, str, str]] = _violations('x: Float32[Array, "n+1 n-1 n/2 (n) n*m *n*m #n+1"]')
    assert [text for _, rule, text in problems if rule == "JT003"] == ["n+1", "n-1", "n/2", "(n)", "n*m", "*n*m", "#n+1"]
    assert _violations('x: Float32[Array, "n *batch #b 3"]') == []


def test_vendor_paths_are_skipped() -> None:
    for path in (
        "packages/demo/third_party/model.py",
        "packages/demo/.pixi/a.py",
        "packages/demo/__pycache__/a.py",
        "packages/demo/build/a.py",
        "packages/demo/target/a.py",
        "packages/demo/node_modules/a.py",
        "packages/demo/upstream_model.py",
        "packages/zipdepth/zipdepth/models/model.py",
    ):
        assert SKIP_PATHS.search(path), path
    for path in (
        "packages/demo/model.py",
        "packages/zipdepth/zipdepth/apis/main.py",
        "packages/zipdepth/zipdepth/catalog/query.py",
        "packages/zipdepth/zipdepth/__init__.py",
        "packages/zipdepth/tests/test_model.py",
    ):
        assert not SKIP_PATHS.search(path), path


def test_baseline_allows_improvements_but_rejects_growth() -> None:
    files: dict[str, list[tuple[int, str, str]]] = {"packages/demo/a.py": [(4, "JT001", "H"), (5, "JT001", "W")]}
    assert _new_violations(files, {"packages/demo/a.py": {"JT001": 3}}) == []
    assert _new_violations(files, {"packages/demo/a.py": {"JT001": 2}}) == []
    assert _new_violations(files, {"packages/demo/a.py": {"JT001": 1}}) == ["packages/demo/a.py:5 JT001 W"]
    assert _new_violations(files, {}) == ["packages/demo/a.py:4 JT001 H", "packages/demo/a.py:5 JT001 W"]
    assert _new_violations(files, {"packages/demo/a.py": {"JT002": 9}}) == ["packages/demo/a.py:4 JT001 H", "packages/demo/a.py:5 JT001 W"]


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--update-baseline"]:
        raise SystemExit("Usage: python ci/tests/test_jaxtyping_spelling.py --update-baseline")
    BASELINE_PATH.write_text(json.dumps(_counts(_scan_packages()), indent=2, sort_keys=True) + "\n", encoding="utf-8")
