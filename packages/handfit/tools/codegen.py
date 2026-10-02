"""Generate the f64 FK and camera kernels with SymForce 0.12.0.

Topology comes from the generic asset; numeric geometry remains an input. The
four-joint finger chains and 17 frames follow generic_hand_model_torch.py.
"""

import json
import re
import subprocess
import time
from pathlib import Path

import symforce  # pyrefly: ignore[missing-import]  # Only installed in handfit-codegen.
import sympy

symforce.set_epsilon_to_zero()
import symforce.symbolic as sf  # noqa: E402  # pyrefly: ignore[missing-import]
from symforce import codegen  # noqa: E402  # pyrefly: ignore[missing-import]
from symforce.codegen.backends.rust.rust_code_printer import RustCodePrinter, ScalarType  # noqa: E402  # pyrefly: ignore[missing-import]
from symforce.codegen.backends.rust.rust_config import RustConfig  # noqa: E402  # pyrefly: ignore[missing-import]


class CheckedRustPrinter(RustCodePrinter):
    """Fix 0.12's missing parentheses on composite power bases."""

    def _print_Pow(self, expr: sympy.Pow, rational: bool | None = None) -> str:
        del rational  # Required by the SymPy printer protocol.
        base: str = self._print(expr.base)
        if expr.exp.is_Integer:
            return f"({base}).powi({int(expr.exp)})"
        if expr.exp == sympy.Rational(1, 2):
            return f"({base}).sqrt()"
        return f"({base}).powf({self._print(expr.exp)})"


class CheckedRustConfig(RustConfig):
    def printer(self) -> CheckedRustPrinter:
        return CheckedRustPrinter(scalar_type=ScalarType.DOUBLE)


ROOT: Path = Path(__file__).resolve().parents[1]
ASSET: Path = ROOT.parent / "handtrack/handtrack/assets/generic_hand_model.json"
OUTPUT: Path = ROOT / "crates/handfit/src/generated"
# This generator reads only topology from the upstream asset, never model numbers.
TOPOLOGY: dict[str, list] = json.loads(ASSET.read_text())


class M20x3(sf.Matrix):
    SHAPE = (20, 3)


class M21x3(sf.Matrix):
    SHAPE = (21, 3)


class V20(sf.Matrix):
    SHAPE = (20, 1)


class V26(sf.Matrix):
    SHAPE = (26, 1)


class V63(sf.Matrix):
    SHAPE = (63, 1)


def hat(v: sf.V3) -> sf.M33:
    return sf.M33([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def landmarks(
    rotation: sf.M33, translation: sf.V3, angles: V20, axes: M20x3, pivots: M20x3, rest: M21x3, weights: M21x3, mirror: sf.Scalar, delta: V26
) -> V63:
    """World landmarks, metres; delta is the solver's first-order tangent step."""
    frames: list[sf.M44] = [sf.M44.eye(), sf.M44.eye()]
    for finger in range(5):
        chain: sf.M44 = sf.M44.eye()
        for k in range(4):
            j: int = 4 * finger + k
            vector: sf.V3 = axes[j, :].T * (angles[j] + delta[6 + j])
            angle: sf.Scalar = sf.sqrt(sf.Max(vector.dot(vector), 1e-4))
            cross: sf.M33 = hat(vector)
            rot: sf.M33 = sf.M33.eye() + sf.sin(angle) / angle * cross + (1 - sf.cos(angle)) / angle**2 * cross * cross
            pivot: sf.V3 = pivots[j, :].T
            local: sf.M44 = sf.M44.eye()
            local[:3, :3] = rot
            local[:3, 3] = pivot - rot * pivot
            chain = chain * local
            if k > 0:
                frames.append(chain)
    wrist: sf.M33 = rotation * (sf.M33.eye() + hat(delta[:3, 0]))
    output: V63 = V63.zero()
    for i in range(21):
        point: sf.V4 = sf.V4(rest[i, 0], rest[i, 1], rest[i, 2], 1)
        blended: sf.V4 = sf.V4.zero()
        # Repeated indices in the asset have zero weight in their extra slots.
        for slot, frame in enumerate(TOPOLOGY["landmark_rest_bone_indices"][i]):
            blended += weights[i, slot] * (frames[int(frame)] * point)
        local_point: sf.V3 = sf.V3(mirror * blended[0], blended[1], blended[2]) / 1000
        output[3 * i : 3 * i + 3, 0] = wrist * local_point + translation + delta[3:6, 0]
    return output


def project_fisheye62(point: sf.V3, focal: sf.V2, principal: sf.V2, distortion: sf.V8) -> sf.V2:
    radius: sf.Scalar = sf.sqrt(point[0] ** 2 + point[1] ** 2 + 1e-12)
    scale: sf.Scalar = sf.atan2(radius, point[2]) / radius
    x: sf.Scalar = point[0] * scale
    y: sf.Scalar = point[1] * scale
    radius_sq: sf.Scalar = sf.Min(x * x + y * y, sf.pi**2)
    radial: sf.Scalar = 1 + radius_sq * (
        distortion[0]
        + radius_sq
        * (distortion[1] + radius_sq * (distortion[2] + radius_sq * (distortion[3] + radius_sq * (distortion[4] + radius_sq * distortion[5]))))
    )
    u: sf.Scalar = x * radial
    v: sf.Scalar = y * radial
    uv_sq: sf.Scalar = u * u + v * v
    return sf.V2(
        focal[0] * (u + 2 * distortion[7] * u * v + distortion[6] * (uv_sq + 2 * u * u)) + principal[0],
        focal[1] * (v + 2 * distortion[6] * u * v + distortion[7] * (uv_sq + 2 * v * v)) + principal[1],
    )


def project_pinhole(point: sf.V3, focal: sf.V2, principal: sf.V2) -> sf.V2:
    z_safe: sf.Scalar = sf.Piecewise((1e-9, sf.Abs(point[2]) < 1e-9), (point[2], True))
    return sf.V2(focal[0] * point[0] / z_safe + principal[0], focal[1] * point[1] / z_safe + principal[1])


def main() -> None:
    start: float = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for fn, argument in [(project_pinhole, "point"), (project_fisheye62, "point"), (landmarks, "delta")]:
        name: str = fn.__name__ + "_with_jacobian"
        print(f"Generating {name}", flush=True)
        generator = codegen.Codegen.function(fn, config=CheckedRustConfig()).with_jacobians(which_args=[argument], include_results=True, name=name)
        generator.generate_function(output_dir=OUTPUT, skip_directory_nesting=True)
        generated: Path = OUTPUT / f"{name}.rs"
        source: str = generated.read_text()
        # SymForce 0.12's Rust template declares relational CSE temporaries as f64 (`let _tmp0: f64 = a < b;`). A relational
        # inside an `if` expression is a condition of an f64 value and stays.
        source = re.sub(r"(let _tmp\d+): f64 = (?![^;\n]*\bif\b)([^;\n]+ (?:<=|>=|==|!=|<|>) [^;\n]+);", r"\1: bool = \2;", source)
        # A repair that stops matching must fail here, not in rustc: every temporary tested as a condition is now a bool.
        for condition in sorted(set(re.findall(r"\((_tmp\d+) == true\)", source))):
            if f"let {condition}: bool = " not in source:
                raise RuntimeError(f"{name}: condition {condition} is not declared bool; SymForce's Rust output changed")
        # nalgebra only provides positional `new` for small fixed vectors.
        if name == "landmarks_with_jacobian":
            for old, new in [
                ("nalgebra::SVector::<f64, 63>::new(", "nalgebra::SVector::<f64, 63>::from_row_slice(&["),
                ("        )\n    }\n} // mod sym", "        ])\n    }\n} // mod sym"),
            ]:
                if source.count(old) != 1:
                    raise RuntimeError(f"{name}: the 63-vector repair expects {old!r} once, found it {source.count(old)} times")
                source = source.replace(old, new)
        generated.write_text(source)
        names.append(name)
    constants: list[str] = []
    for name in ["joint_parent", "joint_frame_index", "joint_first_child", "joint_next_sibling"]:
        values: list[int] = [int(v) for v in TOPOLOGY[name]]
        constants.append(f"pub const {name.upper()}: [i64; 22] = {values};")
    indices: list[list[int]] = TOPOLOGY["landmark_rest_bone_indices"]
    constants.append(f"pub const BONE_INDICES: [[i64; 3]; 21] = {indices};")
    (OUTPUT / "mod.rs").write_text(
        "// Generated by tools/codegen.py. Do not edit.\n#![allow(unused_parens, non_snake_case, clippy::all)]\n"
        + "\n".join(f"pub mod {name};" for name in names)
        + "\n"
        + "\n".join(constants)
        + "\n"
    )
    subprocess.run(["rustfmt", "--edition", "2021", str(OUTPUT / "mod.rs")], check=True)
    print(f"Generated {sum(p.stat().st_size for p in OUTPUT.glob('*.rs'))} bytes in {time.perf_counter() - start:.3f} s", flush=True)


if __name__ == "__main__":
    main()
