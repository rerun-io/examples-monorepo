#!/usr/bin/env bash
# 1. Our web viewer for wasm32 (gsplat-web-build's script, in the stock npm layout). 2. The gradio-rerun frontend
# against it (frozen pnpm lock from the patch). 3. The wheel, installed into the prefix.
set -euxo pipefail
GS="$SRC_DIR/gsplat-rust-renderer"
GRV="$SRC_DIR/gradio-rerun-viewer"
rm -rf "$SP_DIR"/gradio_rerun "$SP_DIR"/gradio_rerun-*.dist-info

mkdir -p "$SRC_DIR/stock"
cp -r "$SRC_DIR/web-viewer-npm" "$SRC_DIR/stock/npm"
export CARGO_HOME="$SRC_DIR/.cargo"
export CARGO_BUILD_JOBS="${CPU_COUNT:-8}"
# build_web_viewer.py takes clang and ar from CONDA_PREFIX: the build prefix here. Without CONDA_BUILD: re_build_tools
# (re_viewer's build script) treats a conda build as Rerun's own packaging and runs `cargo metadata` on the patched crate.
env -u CONDA_BUILD CONDA_PREFIX="$BUILD_PREFIX" PYTHONPATH="$GS" python -c "
from pathlib import Path
from gsplat_rust_renderer.apis.build_web_viewer import BuildWebViewerConfig, main
main(BuildWebViewerConfig(stock_dir=Path('$SRC_DIR/stock'), out_dir=Path('$SRC_DIR/web-out'), cargo_home=Path('$CARGO_HOME')))
"

rm -rf "$GRV/frontend/gsplat-web-viewer"
cp -r "$SRC_DIR/web-out/npm" "$GRV/frontend/gsplat-web-viewer"
cd "$GRV/frontend"
pnpm install --frozen-lockfile --store-dir "$SRC_DIR/.pnpm-store"
cd "$GRV"
# gradio cc build packages with "python -m build", which would fetch the build backend from PyPI into an isolated env:
# hand it a python that runs that step without isolation, on the host prefix's hatchling.
cat > "$SRC_DIR/python-no-isolation" <<'SH'
#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "build" ]; then shift 2; exec python -m build --no-isolation --wheel "$@"; fi
exec python "$@"
SH
chmod +x "$SRC_DIR/python-no-isolation"
PYTHONPATH="$GRV/backend" gradio cc build --no-generate-docs --python-path "$SRC_DIR/python-no-isolation"
python -m pip install dist/gradio_rerun-*.whl --no-deps --no-build-isolation -vv
# The bundle must carry our wasm, not the stock one.
python - "$GRV/backend/gradio_rerun/templates/component/index.js" "$SRC_DIR/web-out/npm/re_viewer_bg.wasm" <<'PY'
import base64, hashlib, re, sys
bundle = open(sys.argv[1], "rb").read()
inlined = base64.b64decode(re.search(rb"data:application/wasm;base64,([A-Za-z0-9+/=]+)", bundle).group(1))
ours = open(sys.argv[2], "rb").read()
assert hashlib.sha256(inlined).digest() == hashlib.sha256(ours).digest(), "the bundled wasm is not our gsplat web build"
print(f"bundled wasm = gsplat web build ({len(ours) / 1e6:.1f} MB)")
PY
