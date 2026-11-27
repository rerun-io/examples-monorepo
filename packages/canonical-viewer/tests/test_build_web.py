"""The web build's npm module: wasm-bindgen's no-modules output becomes the default factory @rerun-io/web-viewer imports."""

import pytest

from canonical_viewer.apis.build_web import npm_module

NO_MODULES: str = (
    "let wasm_bindgen = (function(exports) {"
    "\nlet wasmModule, wasm;\nlet cachedUint8ArrayMemory0 = null;\nfunction __wbg_init() {}\nfunction initSync() {}\n"
    "return Object.assign(__wbg_init, { initSync }, exports);\n})({ __proto__: null });"
)


def test_the_factory_wraps_the_body_and_deinit_drops_every_memory_view() -> None:
    module: str = npm_module(NO_MODULES)
    assert module.startswith("\nexport default function() {\nconst exports = { __proto__: null };\n")
    assert "let cachedUint8ArrayMemory0 = null;" in module
    for name in ("wasmModule", "wasm", "cachedUint8ArrayMemory0"):
        assert f"  {name} = null;\n" in module
    assert module.endswith("return Object.assign(__wbg_init, { initSync, deinit }, exports);\n}\n")


def test_another_wasm_bindgen_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="no-modules layout"):
        npm_module("export function start() {}")
