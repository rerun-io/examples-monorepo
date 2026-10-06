"""Guard the GPU revisions resolved by Cargo."""
from dataclasses import dataclass
from pathlib import Path

from serde import serde
from serde.toml import from_toml


@serde
@dataclass(frozen=True, slots=True)
class CargoPackage:
    """Fields used from Cargo's lock schema; Cargo owns its other fields."""

    name: str
    version: str
    source: str | None = None


@serde
@dataclass(frozen=True, slots=True)
class CargoLock:
    """Resolved package records from Cargo."""

    package: list[CargoPackage]


def test_rust_gpu_dependency_pins() -> None:
    """GPU crates resolve once; the approved observer adapter is the sole local Brush crate."""
    root: Path = Path(__file__).resolve().parents[1]
    lock: CargoLock = from_toml(CargoLock, (root / "Cargo.lock").read_text())
    pins: dict[str, str] = {
        "brush": "1388f74c6fe0236f68ee4915564bf00e9d2e3747",
        "burn": "faec398324e203fdf7998318e4889b987fc802fc",
        "cubecl": "467522442c7b68f02c7ff0c84b2e249e36d8da91",
    }
    seen: set[str] = set()
    for package in lock.package:
        family: str = package.name.split("-")[0]
        if family in (*pins, "wgpu", "naga") or package.name == "lpips":
            assert package.name not in seen, f"duplicate GPU-family dependency: {package.name}"
            seen.add(package.name)
            if family in pins or package.name == "lpips":
                if package.name == "brush-process":
                    assert package.source is None
                elif package.name != "cubecl-hip-sys":
                    revision: str = pins["brush" if package.name == "lpips" else family]
                    assert package.source is not None and package.source.startswith("git+")
                    assert package.source.endswith(f"#{revision}"), (package.name, package.source)
    assert {"brush-render", "burn", "cubecl", "wgpu", "naga"} <= seen
    assert next(p.version for p in lock.package if p.name == "wgpu") == "30.0.0"
