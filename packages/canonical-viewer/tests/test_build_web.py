"""The web build's output: the npm module factory (as Rerun's own npm build makes it) and the viewer directory layout."""

import hashlib
from pathlib import Path

import pytest

from canonical_viewer.apis.build_web import CLOSURE_DTORS, MAKE_MUT_CLOSURE, npm_module, write_layout

HEAD: str = "let wasm_bindgen = (function(exports) {"
TAIL: str = "return Object.assign(__wbg_init, { initSync }, exports);\n})({ __proto__: null });"


def no_modules(*blocks: str) -> str:
    """A minimal wasm-bindgen no-modules file holding `blocks`."""
    body: str = (
        "\nlet wasmModule, wasm;\nlet cachedUint8ArrayMemory0 = null;\n" + "\n".join(blocks) + "\nfunction __wbg_init() {}\nfunction initSync() {}\n"
    )
    return HEAD + body + TAIL


def test_the_factory_has_deinit_and_rerun_s_closure_guards() -> None:
    module: str = npm_module(no_modules(CLOSURE_DTORS[0], MAKE_MUT_CLOSURE[0]))
    assert module.startswith("\nexport default function() {\nconst exports = { __proto__: null };\n")
    for name in ("wasmModule", "wasm", "cachedUint8ArrayMemory0"):
        assert f"  {name} = null;\n" in module
    for original, patched in (CLOSURE_DTORS, MAKE_MUT_CLOSURE):
        assert patched in module and original not in module
    assert module.endswith("return Object.assign(__wbg_init, { initSync, deinit }, exports);\n}\n")


def test_another_wasm_bindgen_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="no-modules layout"):
        npm_module("export function start() {}")
    with pytest.raises(ValueError, match="block Rerun's npm build patches"):
        npm_module(no_modules(CLOSURE_DTORS[0]))


def test_the_layout_replaces_the_stock_viewer_files_and_lists_every_file(tmp_path: Path) -> None:
    stock: Path = tmp_path / "stock"
    (stock / "npm").mkdir(parents=True)
    for name, text in {"index.html": "shell", "re_viewer.js": "stock js", "re_viewer_bg.wasm": "stock", "re_viewer_bg.wasm.br": "br",
                       "npm/index.js": "index", "npm/re_viewer_bg.wasm.br": "br"}.items():  # fmt: skip
        (stock / name).write_text(text)
    wasm: Path = tmp_path / "viewer.wasm"
    wasm.write_bytes(b"\0asm ours")
    out: Path = tmp_path / "out"
    write_layout(out, stock, no_modules(CLOSURE_DTORS[0], MAKE_MUT_CLOSURE[0]), wasm)
    files: set[str] = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    assert files == {"index.html", "re_viewer.js", "re_viewer_bg.wasm", "npm/index.js", "npm/re_viewer.js", "npm/re_viewer_bg.wasm", "SHA256SUMS"}
    assert (out / "re_viewer_bg.wasm").read_bytes() == b"\0asm ours"
    sums: dict[str, str] = {line.split("  ")[1]: line.split("  ")[0] for line in (out / "SHA256SUMS").read_text().splitlines()}
    assert sums["npm/re_viewer_bg.wasm"] == hashlib.sha256(b"\0asm ours").hexdigest()
    assert set(sums) == files - {"SHA256SUMS"}
