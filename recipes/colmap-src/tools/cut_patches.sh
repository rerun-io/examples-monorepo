#!/usr/bin/env bash
# Cut the three rig-refinement patches from a colmap checkout, re-hash SHA256SUMS, and verify on a fresh checkout of
# the pin that every patch applies in order and every hash matches.
#
# usage: cut_patches.sh <checkout> <rcs-ref> <diag-ref> [<head-ref>]
#   <checkout>  colmap clone whose history is the pin, one commit per earlier patch, then the rig commits
#   <rcs-ref>   commit with the five earlier patches applied (the last: caspar-reduced-camera-system)
#   <diag-ref>  commit that adds caspar-diag-exit-no-convergence on top of <rcs-ref>
#   <head-ref>  last rig-refinement commit (default HEAD)
# The checkout is only read: the verification runs in a temporary shared clone. The new patches and SHA256SUMS are
# written to a temporary directory and replace patches/ only after the check passes.
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  sed -n '5,10p' "$0" >&2
  exit 2
fi
C=$1
RCS=$2
DIAG=$3
HEAD_REF=${4:-HEAD}

RECIPE=$(cd "$(dirname "$0")/.." && pwd)
P=$RECIPE/patches
# The pin and the apply order; ai-demos recipes/colmap-caspar applies the patches in this order.
PIN=137f21680c79d3f2fffbb1d11deb367162aa6f83
PATCHES=(
  add-caspar-adapters.patch
  add-caspar-adapters-generated-f32.patch
  add-database-transaction-rollback.patch
  caspar-pcg-beta-recurrence.patch
  caspar-reduced-camera-system.patch
  caspar-diag-exit-no-convergence.patch
  caspar-rig-refine.patch
  caspar-rig-refine-generated-f32.patch
)

GEN=src/thirdparty/Symforce-Caspar/generated/f32
HAND=(
  src/colmap/estimators/bundle_adjustment_caspar.cc
  src/colmap/estimators/bundle_adjustment_caspar.h
  src/colmap/estimators/bundle_adjustment_caspar_test.cc
  src/colmap/estimators/caspar/caspar_model_adapter.h
  src/thirdparty/Symforce-Caspar/caspar_generate.py
)

V=$(mktemp -d /tmp/colmap-cut.XXXXXX)
trap 'rm -rf "$V"' EXIT
OUT=$V/out
mkdir "$OUT"
CUT=(caspar-diag-exit-no-convergence.patch caspar-rig-refine.patch caspar-rig-refine-generated-f32.patch)
git -C "$C" diff --full-index "$RCS" "$DIAG" > "$OUT/${CUT[0]}"
git -C "$C" diff --full-index "$DIAG" "$HEAD_REF" -- "${HAND[@]}" > "$OUT/${CUT[1]}"
git -C "$C" diff --full-index "$DIAG" "$HEAD_REF" -- "$GEN" > "$OUT/${CUT[2]}"

# Nothing outside the two rig cuts.
other=$(git -C "$C" diff --name-only "$DIAG" "$HEAD_REF" | grep -v "^$GEN/" | grep -vxF -f <(printf '%s\n' "${HAND[@]}") || true)
[ -z "$other" ] || { echo "uncut files: $other" >&2; exit 1; }

# Hash the committed tree of <head-ref>, not the work tree: a dirty checkout must not leak into SHA256SUMS.
git clone -q --shared --no-checkout "$C" "$V/head"
git -C "$V/head" checkout -q --detach "$(git -C "$C" rev-parse "$HEAD_REF")"
git -C "$C" diff --name-only "$PIN" "$HEAD_REF" | (cd "$V/head" && xargs sha256sum) > "$OUT/SHA256SUMS"

git clone -q --shared --no-checkout "$C" "$V/pin"
git -C "$V/pin" checkout -q --detach "$PIN"
for p in "${PATCHES[@]}"; do
  if [ -e "$OUT/$p" ]; then git -C "$V/pin" apply "$OUT/$p"; else git -C "$V/pin" apply "$P/$p"; fi
done
(cd "$V/pin" && sha256sum --quiet -c "$OUT/SHA256SUMS")
mv "$OUT"/* "$P/"
echo "SUMS OK: $(wc -l < "$P/SHA256SUMS") files, ${#PATCHES[@]} patches"
