#!/usr/bin/env bash
# 1. The canonical web viewer for wasm32 (canonical-viewer-web-build's code, in the stock npm layout). 2. The
# gradio-rerun frontend against it (frozen pnpm lock from the patch). 3. The wheel, installed into the prefix.
set -euxo pipefail
CV="$SRC_DIR/canonical-viewer"
GRV="$SRC_DIR/gradio-rerun-viewer"
rm -rf "$SP_DIR"/gradio_rerun "$SP_DIR"/gradio_rerun-*.dist-info

mkdir -p "$SRC_DIR/stock"
cp -r "$SRC_DIR/web-viewer-npm" "$SRC_DIR/stock/npm"
export CARGO_HOME="$SRC_DIR/.cargo"
# Cargo still reads ~/.cargo/config.toml above the build directory, whose rustc wrapper (kache) remaps crate roots to
# /proc/self/cwd in file!(); the release build stays hermetic without it.
export RUSTC_WRAPPER=""
export CARGO_BUILD_JOBS="${CPU_COUNT:-8}"
# canonical-viewer-web-build's code, with clang and ar from the build prefix.
PYTHONPATH="$CV" python -c "
from pathlib import Path
from canonical_viewer.apis.build_web import BuildWebConfig, main
main(BuildWebConfig(stock_dir=Path('$SRC_DIR/stock'), out_dir=Path('$SRC_DIR/web-out'), cargo_home=Path('$CARGO_HOME'), toolchain_prefix=Path('$BUILD_PREFIX')))
"

rm -rf "$GRV/frontend/canonical-web-viewer"
cp -r "$SRC_DIR/web-out/npm" "$GRV/frontend/canonical-web-viewer"
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
assert hashlib.sha256(inlined).digest() == hashlib.sha256(ours).digest(), "the bundled wasm is not the canonical web build"
print(f"bundled wasm = canonical web build ({len(ours) / 1e6:.1f} MB)")
PY
