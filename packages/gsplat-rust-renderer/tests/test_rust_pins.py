"""Guard the GPU revisions and the installed Brush source family resolved by Cargo."""

import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest
from serde import serde
from serde.json import from_json
from serde.toml import from_toml


@serde
@dataclass(frozen=True, slots=True)
class CargoPackage:
    """Fields used from Cargo's schema; Cargo owns its other fields."""

    name: str
    version: str
    source: str | None = None
    manifest_path: Path | None = None


@serde
@dataclass(frozen=True, slots=True)
class CargoLock:
    """Resolved package records from Cargo."""

    package: list[CargoPackage]


@serde
@dataclass(frozen=True, slots=True)
class CargoMetadata:
    """Resolved package manifests from Cargo."""

    packages: list[CargoPackage]


def test_rust_gpu_dependency_pins() -> None:
    """GPU crates resolve once and Burn/CubeCL retain their reviewed revisions."""
    root: Path = Path(__file__).resolve().parents[1]
    lock: CargoLock = from_toml(CargoLock, (root / "Cargo.lock").read_text())
    pins: dict[str, str] = {
        "burn": "faec398324e203fdf7998318e4889b987fc802fc",
        "cubecl": "467522442c7b68f02c7ff0c84b2e249e36d8da91",
    }
    seen: set[str] = set()
    for package in lock.package:
        assert package.source is None or "github.com/ArthurBrussee/brush" not in package.source
        family: str = package.name.split("-")[0]
        if family in (*pins, "brush", "wgpu", "naga") or package.name == "lpips":
            assert package.name not in seen, f"duplicate GPU-family dependency: {package.name}"
            seen.add(package.name)
            if family == "brush" or package.name == "lpips":
                assert package.source is None
            elif family in pins and package.name != "cubecl-hip-sys":
                assert package.source is not None and package.source.startswith("git+")
                assert package.source.endswith(f"#{pins[family]}"), (package.name, package.source)
    assert {"brush-render", "burn", "cubecl", "wgpu", "naga"} <= seen
    assert next(p.version for p in lock.package if p.name == "wgpu") == "30.0.0"


@pytest.mark.integration
def test_brush_manifests_use_one_installed_tree() -> None:
    """Every Brush crate comes from the recipe-patched upstream source package."""
    root = Path(__file__).resolve().parents[1]
    tree = root / "target/brush-src"
    if not tree.is_dir():
        pytest.skip("missing installed Brush tree; activate a gsplat Pixi environment")
    metadata = from_json(
        CargoMetadata,
        subprocess.check_output(["cargo", "metadata", "--locked", "--offline", "--format-version=1"], cwd=root, text=True),
    )
    assert all(p.source is None or "github.com/ArthurBrussee/brush" not in p.source for p in metadata.packages)
    brush = [p for p in metadata.packages if p.name.startswith("brush-") or p.name in ("lpips", "colmap-reader", "rrfd")]
    assert len(brush) == 16
    for package in brush:
        assert package.source is None
        assert package.manifest_path is not None and package.manifest_path.resolve().is_relative_to(tree.resolve())


def test_brush_dependencies_use_recipe_tree() -> None:
    """Direct Brush dependencies share the installed recipe tree and workspace pins."""
    root: Path = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "Cargo.toml").read_text())
    dependencies = manifest["workspace"]["dependencies"]
    brush: set[str] = {"brush-render", "brush-serde", "brush-loss", "lpips", "brush-process", "brush-train", "brush-dataset", "colmap-reader"}
    for name in brush:
        assert dependencies[name] == {"path": f"target/brush-src/crates/{name}"}
    assert "https://github.com/ArthurBrussee/brush" not in manifest.get("patch", {})
    for crate in ("gsplat-core", "gsplat-viewer", "gsplat-cli"):
        crate_manifest = tomllib.loads((root / "crates" / crate / "Cargo.toml").read_text())
        assert crate_manifest["dependencies"]["half"]["workspace"] is True
