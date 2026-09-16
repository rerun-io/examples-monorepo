#!/usr/bin/env bash
# Run through the robocap-cross Pixi task. Device libraries are used only to link.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: robocap-direct-build /absolute/path/to/cap-runtime-sysroot" >&2
    exit 2
fi
cap_sysroot=$(realpath "$1")
for directory in usr/lib/pkgconfig usr/lib lib; do
    [[ -d "$cap_sysroot/$directory" ]] || {
        echo "Missing runtime sysroot directory: $cap_sysroot/$directory" >&2
        exit 2
    }
done
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PKG_CONFIG_ALLOW_CROSS=1
export PKG_CONFIG_SYSROOT_DIR="$cap_sysroot"
export PKG_CONFIG_LIBDIR="$cap_sysroot/usr/lib/pkgconfig"
# Encoded arguments preserve spaces in the sysroot and replace the old ARM
# opt-level override with the release profile verified on Cap A.
export CARGO_ENCODED_RUSTFLAGS
CARGO_ENCODED_RUSTFLAGS=$(printf '%s\037' \
    "-Lnative=$cap_sysroot/usr/lib" "-Lnative=$cap_sysroot/lib" \
    "-Clink-arg=-Wl,-rpath-link,$cap_sysroot/usr/lib" \
    "-Clink-arg=-Wl,-rpath-link,$cap_sysroot/lib")
CARGO_ENCODED_RUSTFLAGS=${CARGO_ENCODED_RUSTFLAGS%$'\037'}
exec cargo build --locked --release --target aarch64-unknown-linux-gnu \
    -p robocap-recorder --features gstreamer-capture,live-slam --bin robocap-direct
