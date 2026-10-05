#!/usr/bin/env bash
# Build robocap-live for this host (x86_64) and for the caps (aarch64-unknown-linux-gnu, release), then check the cap binary:
# glibc floor <= 2.34 and no GStreamer / RKNN / RGA link-time dependency (those are child processes or dlopened at run time).
#
# Usage: scripts/build-arm.sh [--host-only | --arm-only] [--test] [--examples]
# Runs inside the repo's `robocap-cross` pixi env (rust 1.98 + aarch64 std, aarch64-conda-linux-gnu-gcc, glibc 2.34 sysroot), from
# this checkout's pixi.toml (pixi installs the env on first use).
# Outputs: target/release/robocap-live (host) and target/aarch64-unknown-linux-gnu/release/robocap-live (cap).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo=$(cd "$here/../.." && pwd)

if [[ -z ${PIXI_ENVIRONMENT_NAME:-} || ${PIXI_ENVIRONMENT_NAME} != robocap-cross ]]; then
    exec pixi run --manifest-path "$repo/pixi.toml" -e robocap-cross --frozen bash "$here/scripts/build-arm.sh" "$@"
fi

host=1 arm=1 test=0 examples=()
for arg in "$@"; do
    case $arg in
        --host-only) arm=0 ;;
        --arm-only) host=0 ;;
        --test) test=1 ;;
        --examples) examples=(--examples) ;;
        *) echo "unknown argument $arg" >&2; exit 2 ;;
    esac
done
cd "$here"
[[ -d $repo/packages/slam-rs/target/patch/cubecl-common-0.11.0-pre.3 ]] || {
    echo "missing slam-rs patch tree: run 'pixi run -e slam-rs-dev --frozen slam-rs-patch-deps' once" >&2; exit 2; }

# The env's CC is the aarch64 compiler; the host build's C shim needs the host one.
export CC_x86_64_unknown_linux_gnu=${CC_x86_64_unknown_linux_gnu:-x86_64-conda-linux-gnu-cc}

if (( host )); then
    echo "== host (x86_64) release build"
    cargo build --release --bins "${examples[@]}"
    if (( test )); then cargo test --release; fi
    ls -la target/release/robocap-live
fi

if (( arm )); then
    echo "== cap (aarch64) release build"
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS:-} -C link-arg=-Wl,--enable-new-dtags,-rpath,\$ORIGIN/../lib"
    cargo build --release --target aarch64-unknown-linux-gnu --features robocap-live/gpu-wgpu --bins "${examples[@]}"
    binary=target/aarch64-unknown-linux-gnu/release/robocap-live
    readelf=aarch64-conda-linux-gnu-readelf
    objdump=aarch64-conda-linux-gnu-objdump
    runpath=$($readelf -d "$binary" | awk '/RUNPATH/ {print $NF}' | tr -d '[]')
    echo "RUNPATH: $runpath"
    case :$runpath: in
        *':$ORIGIN/../lib:'*) ;;
        *) echo 'FAIL: missing $ORIGIN/../lib RUNPATH' >&2; exit 1 ;;
    esac
    needed=$($readelf -d "$binary" | awk '/NEEDED/ {print $5}' | tr -d '[]' | tr '\n' ' ')
    echo "NEEDED: $needed"
    if grep -Eqi 'gst|rknn|rga|vulkan' <<<"$needed"; then echo "FAIL: links GStreamer/RKNN/RGA/Vulkan (must be dlopened)" >&2; exit 1; fi
    floor=$($objdump -T "$binary" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
    echo "glibc floor: $floor"
    if [[ $(printf '%s\nGLIBC_2.34\n' "$floor" | sort -V | tail -1) != GLIBC_2.34 ]]; then echo "FAIL: needs $floor > GLIBC_2.34" >&2; exit 1; fi
    ls -la "$binary"
    sha256sum "$binary"
fi
