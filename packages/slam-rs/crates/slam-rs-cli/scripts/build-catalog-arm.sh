#!/usr/bin/env bash
# Run through the slam-rs-cli-catalog-build Pixi task from the slam-rs workspace.
set -euo pipefail
stdlib=$("${CXX_aarch64_unknown_linux_gnu}" -print-file-name=libstdc++.a)
[[ -f $stdlib ]] || { echo "C++ compiler did not resolve libstdc++.a" >&2; exit 1; }
export DAV1D_INCLUDE_DIR="$CONDA_PREFIX/include"
export CXXSTDLIB_aarch64_unknown_linux_gnu=static=stdc++
export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS:-} -L native=$(dirname "$stdlib")"
cargo build --locked --release --target aarch64-unknown-linux-gnu -p slam-rs-cli --features catalog,gpu-wgpu
