"""re_renderer finds its embedded shaders when the compiler remaps a dependency's root to `/proc/self/cwd`
(patches/re_renderer-0.44.0-remapped-shader-path.patch).

kache, the fleet's rustc wrapper, passes `--remap-path-prefix <crate dir>=/proc/self/cwd` for crates outside the
workspace, such as the patched Rerun crates: `file!()` then starts with `/proc/self/cwd`, while `CARGO_MANIFEST_DIR`
keeps the real directory. The patched `include_file!` from the prepared checkout is compiled into a stub crate with
re_renderer's embedded-shader lookup, once as built and once remapped, and must resolve the same shader both times.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

FILE_SERVER: str = "crates/ui_tools/re_renderer/src/file_server.rs"
STUB_LIB: str = """use std::path::{Component, Path, PathBuf};

pub mod external {
    pub mod anyhow {
        pub trait Context<T> {
            fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, context: F) -> Result<T, String>;
        }
        impl<T> Context<T> for Result<T, String> {
            fn with_context<C: std::fmt::Display, F: FnOnce() -> C>(self, context: F) -> Result<T, String> {
                self.map_err(|err| format!("{}: {err}", context()))
            }
        }
    }
}

pub trait FileSystem {
    fn canonicalize(&self, path: &Path) -> Result<PathBuf, String>;
}

/// re_renderer's embedded filesystem: shaders under crate-relative paths.
pub struct Embedded;

impl FileSystem for Embedded {
    fn canonicalize(&self, path: &Path) -> Result<PathBuf, String> {
        let mut resolved = PathBuf::new();
        for component in path.components() {
            match component {
                Component::ParentDir => {
                    resolved.pop();
                }
                Component::Normal(part) => resolved.push(part),
                _ => return Err(format!("not crate-relative: {path:?}")),
            }
        }
        if resolved == Path::new("shader/probe.wgsl") { Ok(resolved) } else { Err(format!("no shader at {resolved:?}")) }
    }
}

pub fn get_filesystem() -> Embedded {
    Embedded
}

pub mod draw_phases;

#[macro_export]
"""


def include_file_macro(checkout: Path) -> str:
    """The embedded-shader `include_file!` (the `not(load_shaders_from_disk)` variant) of the prepared re_renderer."""
    source: str = (checkout / FILE_SERVER).read_text()
    match: re.Match[str] | None = re.search(r"#\[cfg\(not\(load_shaders_from_disk\)\)\]\n(macro_rules! include_file \{.*?\n\})\n", source, re.S)
    assert match is not None, f"include_file! not found in {FILE_SERVER}"
    return match.group(1)


def resolve_shader(root: Path, rustflags: str) -> subprocess.CompletedProcess[str]:
    """Build and run the probe app against the stub re_renderer at `root`; it prints the resolved shader path."""
    environment: dict[str, str] = {**os.environ, "RUSTFLAGS": rustflags, "CARGO_BUILD_RUSTC_WRAPPER": "", "RUSTC_WRAPPER": ""}
    target: str = str(root / ("target-remapped" if rustflags else "target-plain"))
    return subprocess.run(
        ["cargo", "run", "--quiet", "--offline", "--manifest-path", str(root / "app/Cargo.toml"), "--target-dir", target],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )


def test_embedded_shaders_resolve_with_a_remapped_crate_root(tmp_path: Path) -> None:
    from canonical_viewer.apis.prepare_rerun import PACKAGE_DIR, RERUN_TAG

    checkout: Path = PACKAGE_DIR / "target" / "patch" / f"rerun-{RERUN_TAG}"
    if not (checkout / FILE_SERVER).is_file():
        pytest.skip(f"{checkout} is missing: run `pixi run -e canonical-viewer canonical-viewer-build`")
    stub: Path = tmp_path / "re_renderer"
    (stub / "src" / "draw_phases").mkdir(parents=True)
    (stub / "Cargo.toml").write_text('[package]\nname = "re_renderer"\nversion = "0.0.0"\nedition = "2021"\n')
    (stub / "src" / "lib.rs").write_text(STUB_LIB + include_file_macro(checkout) + "\n")
    (stub / "src" / "draw_phases" / "mod.rs").write_text("pub mod probe;\n")
    (stub / "src" / "draw_phases" / "probe.rs").write_text(
        'pub fn shader() -> std::path::PathBuf {\n    crate::include_file!("../../shader/probe.wgsl")\n}\n'
    )
    app: Path = tmp_path / "app"
    (app / "src").mkdir(parents=True)
    (app / "Cargo.toml").write_text(
        '[package]\nname = "probe"\nversion = "0.0.0"\nedition = "2021"\n[dependencies]\nre_renderer = { path = "../re_renderer" }\n'
    )
    (app / "src" / "main.rs").write_text('fn main() {\n    println!("{}", re_renderer::draw_phases::probe::shader().display());\n}\n')
    for rustflags in ("", f"--remap-path-prefix={stub}=/proc/self/cwd"):
        result: subprocess.CompletedProcess[str] = resolve_shader(tmp_path, rustflags)
        assert result.returncode == 0, f"RUSTFLAGS={rustflags!r}\n{result.stderr[-3000:]}"
        assert result.stdout.strip() == "shader/probe.wgsl", f"RUSTFLAGS={rustflags!r}: {result.stdout}"
