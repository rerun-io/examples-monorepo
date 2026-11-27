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

# Rerun's own npm build (rerun_js/web-viewer/build-wasm.mjs at the release tag) patches these two wasm-bindgen 0.2.129
# blocks: after deinit() nulls `wasm`, finalizers and pending callbacks (requestAnimationFrame, setTimeout) may still
# run, and must not call into it.
CLOSURE_DTORS: tuple[str, str] = (
    """const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
        ? { register: () => {}, unregister: () => {} }
        : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));""",
    """const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
        ? { register: () => {}, unregister: () => {} }
        : new FinalizationRegistry(state => {
        if (wasm) wasm.__wbindgen_destroy_closure(state.a, state.b);
    });""",
)
MAKE_MUT_CLOSURE: tuple[str, str] = (
    """function makeMutClosure(arg0, arg1, f) {
        const state = { a: arg0, b: arg1, cnt: 1 };
        const real = (...args) => {

            // First up with a closure we increment the internal reference
            // count. This ensures that the Rust closure environment won't
            // be deallocated while we're invoking it.
            state.cnt++;
            const a = state.a;
            state.a = 0;
            try {
                return f(a, state.b, ...args);
            } finally {
                state.a = a;
                real._wbg_cb_unref();
            }
        };
        real._wbg_cb_unref = () => {
            if (--state.cnt === 0) {
                wasm.__wbindgen_destroy_closure(state.a, state.b);
                state.a = 0;
                CLOSURE_DTORS.unregister(state);
            }
        };
        CLOSURE_DTORS.register(real, state, state);
        return real;
    }""",
    """function makeMutClosure(arg0, arg1, f) {
        const state = { a: arg0, b: arg1, cnt: 1 };
        const real = (...args) => {
            state.cnt++;
            const a = state.a;
            state.a = 0;
            try {
                if (!wasm) return;
                return f(a, state.b, ...args);
            } finally {
                state.a = a;
                real._wbg_cb_unref();
            }
        };
        real._wbg_cb_unref = () => {
            if (--state.cnt === 0) {
                if (wasm) wasm.__wbindgen_destroy_closure(state.a, state.b);
                state.a = 0;
                CLOSURE_DTORS.unregister(state);
            }
        };
        CLOSURE_DTORS.register(real, state, state);
        return real;
    }""",
)


def npm_module(no_modules_js: str) -> str:
    """Turn wasm-bindgen's no-modules output into the ES module with a default factory that npm/index.js imports.

    The same transformation as Rerun's npm build: a factory with `deinit`, and the closure guards it needs.
    """
    head: str = "let wasm_bindgen = (function(exports) {"
    tail: str = "return Object.assign(__wbg_init, { initSync }, exports);\n})({ __proto__: null });"
    if not no_modules_js.startswith(head) or tail not in no_modules_js:
        raise ValueError("unexpected wasm-bindgen no-modules layout")
    body: str = no_modules_js[len(head) : no_modules_js.rindex(tail)]
    for original, patched in (CLOSURE_DTORS, MAKE_MUT_CLOSURE):
        if original not in body:
            raise ValueError("unexpected wasm-bindgen output: a block Rerun's npm build patches is missing")
        body = body.replace(original, patched)
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


def write_layout(out_dir: Path, stock_dir: Path, js: str, wasm: Path) -> None:
    """Write the viewer directory: the stock shell with our viewer files, checksums and read permissions."""
    # The stock build's own viewer files (and their pre-compressed copies) are replaced below.
    shutil.copytree(stock_dir, out_dir, ignore=shutil.ignore_patterns("SHA256SUMS", "re_viewer.js", "re_viewer_bg.wasm", "*.br"))
    (out_dir / "re_viewer.js").write_text(js)
    shutil.copyfile(wasm, out_dir / "re_viewer_bg.wasm")
    (out_dir / "npm/re_viewer.js").write_text(npm_module(js))
    shutil.copyfile(wasm, out_dir / "npm/re_viewer_bg.wasm")
    sums: list[str] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                sums.append(f"{hashlib.file_digest(stream, 'sha256').hexdigest()}  {path.relative_to(out_dir)}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    for path in out_dir.rglob("*"):
        path.chmod(0o775 if path.is_dir() else 0o644)
    out_dir.chmod(0o775)


def main(config: BuildWebConfig) -> None:
    """Prepare the patched trees, build, bind, optimise and lay out the web viewer; replace out_dir only when complete."""
    for shell in ("index.html", "npm/index.js"):
        if not (config.stock_dir / shell).is_file():
            raise FileNotFoundError(f"{config.stock_dir} is not a stock web viewer directory: {shell} is missing")
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
    config.out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{config.out_dir.name}-", dir=config.out_dir.parent) as temporary:
        staging: Path = Path(temporary)
        subprocess.run(
            [
                "wasm-bindgen",
                str(WASM),
                "--out-dir",
                str(staging / "bindgen"),
                "--out-name",
                "re_viewer",
                "--target",
                "no-modules",
                "--no-typescript",
            ],
            check=True,
        )
        bg: Path = staging / "bindgen/re_viewer_bg.wasm"
        if config.wasm_opt:
            start = time.perf_counter()
            optimised: Path = staging / "bindgen/re_viewer_bg.opt.wasm"
            subprocess.run(["wasm-opt", "-O2", "--enable-bulk-memory", "--enable-nontrapping-float-to-int", "--enable-sign-ext",
                            "--enable-mutable-globals", "--enable-reference-types", "--enable-multivalue", "--strip-debug",
                            str(bg), "-o", str(optimised)], check=True)  # fmt: skip
            print(f"wasm-opt: {time.perf_counter() - start:.0f} s")
            bg = optimised
        write_layout(staging / "viewer", config.stock_dir, (staging / "bindgen/re_viewer.js").read_text(), bg)
        # The previous directory moves into the temporary one (deleted with it) only once the new one is complete.
        if config.out_dir.exists():
            config.out_dir.rename(staging / "previous")
        (staging / "viewer").rename(config.out_dir)
    print(f"wrote {config.out_dir} (re_viewer_bg.wasm {(config.out_dir / 're_viewer_bg.wasm').stat().st_size / 1e6:.1f} MB)")
