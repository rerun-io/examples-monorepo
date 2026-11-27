"""Build the canonical viewer for the web, in the stock Rerun web viewer layout.

Steps: prepare the patched source trees (``prepare_rerun``), build ``rust/crates/web`` for wasm32 with the ``web``
profile, run wasm-bindgen (output named re_viewer, like the stock viewer) and wasm-opt, then write a viewer directory
with the stock layout: the app files (index.html, re_viewer.js, re_viewer_bg.wasm) and npm/ (index.js and the ES-module
re_viewer.js that @rerun-io/web-viewer imports). Runs in the canonical-viewer-web environment (wasm32 rust std,
wasm-bindgen-cli, binaryen, clang).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from canonical_viewer import PACKAGE_DIR
from canonical_viewer.apis import prepare_rerun

WASM: Path = PACKAGE_DIR / "target/wasm32-unknown-unknown/web/canonical_viewer_web.wasm"
"""What ``cargo build --profile web -p canonical-viewer-web --target wasm32-unknown-unknown`` writes."""


def npm_module(no_modules_js: str) -> str:
    """Turn wasm-bindgen's no-modules output into the ES module with a default factory that npm/index.js imports."""
    head: str = "let wasm_bindgen = (function(exports) {"
    tail: str = "return Object.assign(__wbg_init, { initSync }, exports);\n})({ __proto__: null });"
    if not no_modules_js.startswith(head) or tail not in no_modules_js:
        raise ValueError("unexpected wasm-bindgen no-modules layout")
    body: str = no_modules_js[len(head) : no_modules_js.rindex(tail)]
    # deinit() drops the module and every cached memory view, as the stock npm build does.
    resets: list[str] = ["wasmModule", "wasm", *sorted(set(re.findall(r"^\s*let (cached\w*Memory\w*)\b", body, flags=re.MULTILINE)))]
    return (
        "\nexport default function() {\nconst exports = { __proto__: null };\n"
        + body
        + "\nfunction deinit() {\n  __wbg_init.__wbindgen_wasm_module = null;\n"
        + "".join(f"  {name} = null;\n" for name in resets)
        + "}\n\nreturn Object.assign(__wbg_init, { initSync, deinit }, exports);\n}\n"
    )


@dataclass
class BuildWebConfig:
    """Build the canonical web viewer and write it in the stock /rerun-viewer/<version>/ layout."""

    stock_dir: Path
    """A stock Rerun web viewer directory of the same release (index.html, npm/index.js, ...) to copy the shell files from."""
    out_dir: Path
    """Output viewer directory, e.g. .../rerun-viewer/0.44.0-canonical-<sha7>; replaced if it exists."""
    cargo_home: Path = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
    """Cargo home with the crates.io archive cache and Cargo's git clones."""
    wasm_opt: bool = True
    """Run wasm-opt -O2 (strips debug info and names; about a minute)."""
    toolchain_prefix: Path | None = None
    """Prefix whose ``bin/clang`` and ``bin/ar`` compile the C deps for wasm32 (default ``$CONDA_PREFIX``)."""


def main(config: BuildWebConfig) -> None:
    """Prepare the patched trees, build, bind, optimise and lay out the web viewer."""
    prepare_rerun.main(prepare_rerun.PrepareRerunConfig(cargo_home=config.cargo_home))
    prefix: Path = config.toolchain_prefix or Path(os.environ["CONDA_PREFIX"])
    # Without CONDA_BUILD: re_build_tools (re_viewer's build script) treats a conda build as Rerun's own packaging and
    # runs `cargo metadata` on the patched crate, which fails outside Rerun's workspace.
    env: dict[str, str] = {name: value for name, value in os.environ.items() if name != "CONDA_BUILD"}
    env.update(
        CC_wasm32_unknown_unknown=f"{prefix}/bin/clang",
        AR_wasm32_unknown_unknown=f"{prefix}/bin/ar",
        CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS="--cfg=web_sys_unstable_apis",
    )
    start: float = time.perf_counter()
    subprocess.run(
        ["nice", "-n", "10", "cargo", "build", "--profile", "web", "--locked", "--manifest-path", "rust/Cargo.toml", "--target-dir", "target",
         "-p", "canonical-viewer-web", "--target", "wasm32-unknown-unknown"],
        cwd=PACKAGE_DIR, env=env, check=True,
    )  # fmt: skip
    print(f"cargo build: {time.perf_counter() - start:.0f} s")
    with tempfile.TemporaryDirectory(prefix="canonical-web-bindgen-") as temporary:
        bindgen: Path = Path(temporary)
        subprocess.run(
            ["wasm-bindgen", str(WASM), "--out-dir", str(bindgen), "--out-name", "re_viewer", "--target", "no-modules", "--no-typescript"],
            check=True,
        )
        bg: Path = bindgen / "re_viewer_bg.wasm"
        if config.wasm_opt:
            start = time.perf_counter()
            optimised: Path = bindgen / "re_viewer_bg.opt.wasm"
            subprocess.run(["wasm-opt", "-O2", "--enable-bulk-memory", "--enable-nontrapping-float-to-int", "--enable-sign-ext",
                            "--enable-mutable-globals", "--enable-reference-types", "--enable-multivalue", "--strip-debug",
                            str(bg), "-o", str(optimised)], check=True)  # fmt: skip
            print(f"wasm-opt: {time.perf_counter() - start:.0f} s")
            bg = optimised
        if config.out_dir.exists():
            shutil.rmtree(config.out_dir)
        # The stock build's own viewer files (and their pre-compressed copies) are replaced below.
        shutil.copytree(config.stock_dir, config.out_dir, ignore=shutil.ignore_patterns("SHA256SUMS", "re_viewer.js", "re_viewer_bg.wasm", "*.br"))
        js: str = (bindgen / "re_viewer.js").read_text()
        (config.out_dir / "re_viewer.js").write_text(js)
        shutil.copyfile(bg, config.out_dir / "re_viewer_bg.wasm")
        (config.out_dir / "npm/re_viewer.js").write_text(npm_module(js))
        shutil.copyfile(bg, config.out_dir / "npm/re_viewer_bg.wasm")
    sums: list[str] = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(config.out_dir)}" for p in sorted(config.out_dir.rglob("*")) if p.is_file()]
    (config.out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    for path in config.out_dir.rglob("*"):
        path.chmod(0o775 if path.is_dir() else 0o644)
    print(f"wrote {config.out_dir} (re_viewer_bg.wasm {(config.out_dir / 're_viewer_bg.wasm').stat().st_size / 1e6:.1f} MB)")
