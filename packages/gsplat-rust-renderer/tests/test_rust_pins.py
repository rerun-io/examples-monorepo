"""Guard the resolved renderer stack against an accidental Cargo update."""
import tomllib
from pathlib import Path


def test_rust_gpu_dependency_pins() -> None:
    """All GPU-family crates resolve once, at the agreed source revision."""
    root: Path = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((root / "Cargo.lock").read_text())
    pins: dict[str, str] = {
        "brush": "1388f74c6fe0236f68ee4915564bf00e9d2e3747",
        "burn": "faec398324e203fdf7998318e4889b987fc802fc",
        "cubecl": "467522442c7b68f02c7ff0c84b2e249e36d8da91",
    }
    seen: set[str] = set()
    for package in lock["package"]:
        name: str = package["name"]
        family: str = name.split("-")[0]
        if family in (*pins, "wgpu", "naga") or name == "lpips":
            assert name not in seen, f"duplicate GPU-family dependency: {name}"
            seen.add(name)
            if (family in pins or name == "lpips") and package["source"].startswith("git+"):
                revision: str = pins["brush" if name == "lpips" else family]
                assert package["source"].endswith(f"#{revision}"), (name, package["source"])
        assert not (name.startswith("re_") or name == "rerun") or package["version"] != "0.36.3"
    assert {"brush-render", "burn", "cubecl", "wgpu", "naga"} <= seen
    wgpu_version: str = next(p["version"] for p in lock["package"] if p["name"] == "wgpu")
    assert wgpu_version == "30.0.0"
