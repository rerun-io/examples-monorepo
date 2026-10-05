#!/usr/bin/env bash
# Stage the Vulkan loader robocap-live ships beside its cap binary: libvulkan.so.1 from the conda-forge `libvulkan-loader` package that
# pixi.lock pins for the robocap-live environment on linux-aarch64 (pixi.toml: feature.robocap-live.target.linux-aarch64). The caps'
# firmware has the Mali ICD (/usr/share/vulkan/icd.d/mali.json) but no system loader, and wgpu dlopens libvulkan.so.1.
#
# Usage: stage-vulkan.sh <output-lib-dir>
# The package is downloaded once into ${XDG_CACHE_HOME:-~/.cache}/robocap-live/vulkan and checked against the lock's sha256.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lock=$(cd "$here/../.." && pwd)/pixi.lock
destination=${1:?output lib directory required}

# The robocap-live environment's linux-aarch64 libvulkan-loader URL, then that package's sha256 from the lock's package list.
url=$(awk '/^environments:/ {e = 1} /^packages:/ {exit} e && /^  [a-z0-9-]+:$/ {env = $1}
           env == "robocap-live:" && /\/linux-aarch64\/libvulkan-loader-/ {print $NF; exit}' "$lock")
[[ -n $url ]] || { echo "pixi.lock pins no linux-aarch64 libvulkan-loader for robocap-live" >&2; exit 1; }
sha=$(awk -v url="$url" '/^packages:/ {p = 1} p && $NF == url {f = 1} f && /^  sha256:/ {print $2; exit}' "$lock")
[[ $sha =~ ^[0-9a-f]{64}$ ]] || { echo "no sha256 for $url in pixi.lock" >&2; exit 1; }

cache=${XDG_CACHE_HOME:-$HOME/.cache}/robocap-live/vulkan
archive=$cache/$(basename "$url")
mkdir -p "$cache"
if ! echo "$sha  $archive" | sha256sum --check --status 2>/dev/null; then
    curl -fsSL -o "$archive.part" "$url"
    mv "$archive.part" "$archive"
fi
echo "$sha  $archive" | sha256sum --check --status || { echo "$archive does not match the sha256 pinned in pixi.lock" >&2; exit 1; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
unzip -p "$archive" 'pkg-*.tar.zst' | zstd -dq | tar -x -C "$work"
loader=$work/lib/libvulkan.so.1
[[ -f $loader ]] || { echo "$archive has no lib/libvulkan.so.1" >&2; exit 1; }
readelf=${READELF:-readelf}
floor=$($readelf --version-info "$loader" | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1)
[[ -n $floor && $(printf '%s\nGLIBC_2.34\n' "$floor" | sort -V | tail -1) == GLIBC_2.34 ]] || {
    echo "Vulkan loader needs $floor > GLIBC_2.34" >&2; exit 1;
}
while IFS= read -r library; do
    case $library in
        libc.so.6|libm.so.6|libdl.so.2|libpthread.so.0|librt.so.1|ld-linux-aarch64.so.1) ;;
        *) echo "Vulkan loader has an unbundled dependency: $library" >&2; exit 1 ;;
    esac
done < <($readelf -d "$loader" | awk '/NEEDED/ {print $5}' | tr -d '[]')

mkdir -p "$destination"
cp -L "$loader" "$destination/libvulkan.so.1"
printf 'package: %s\npackage sha256: %s\nglibc floor: %s\n' "$url" "$sha" "$floor" > "$destination/vulkan-source.txt"
cat "$destination/vulkan-source.txt"
