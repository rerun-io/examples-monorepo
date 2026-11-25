#!/usr/bin/env bash
# Cut the three rig-refinement patches from a colmap checkout, re-hash SHA256SUMS, and verify on a fresh checkout of
# the pin that every patch listed in recipe.yaml applies in order and every hash matches.
#
# usage: cut_patches.sh <checkout> <rcs-ref> <diag-ref> [<head-ref>]
#   <checkout>  colmap clone whose history is the pin, one commit per earlier patch, then the rig commits
#   <rcs-ref>   commit with the five earlier patches applied (recipe: caspar-reduced-camera-system)
#   <diag-ref>  commit that adds caspar-diag-exit-no-convergence on top of <rcs-ref>
#   <head-ref>  last rig-refinement commit (default HEAD)
# The checkout is only read: the verification runs in a temporary shared clone.
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
PIN=$(sed -n 's/^ *colmap_rev: *\([0-9a-f]\{40\}\).*/\1/p' "$RECIPE/recipe.yaml")
[ -n "$PIN" ] || { echo "no colmap_rev in $RECIPE/recipe.yaml" >&2; exit 1; }
mapfile -t PATCHES < <(sed -n 's/^ *- patches\/\(.*\.patch\) *$/\1/p' "$RECIPE/recipe.yaml")
[ ${#PATCHES[@]} -gt 0 ] || { echo "no patches listed in $RECIPE/recipe.yaml" >&2; exit 1; }

GEN=src/thirdparty/Symforce-Caspar/generated/f32
HAND=(
  src/colmap/estimators/bundle_adjustment_caspar.cc
  src/colmap/estimators/bundle_adjustment_caspar.h
  src/colmap/estimators/bundle_adjustment_caspar_test.cc
  src/colmap/estimators/caspar/caspar_model_adapter.h
  src/thirdparty/Symforce-Caspar/caspar_generate.py
)

git -C "$C" diff --full-index "$RCS" "$DIAG" > "$P/caspar-diag-exit-no-convergence.patch"
git -C "$C" diff --full-index "$DIAG" "$HEAD_REF" -- "${HAND[@]}" > "$P/caspar-rig-refine.patch"
git -C "$C" diff --full-index "$DIAG" "$HEAD_REF" -- "$GEN" > "$P/caspar-rig-refine-generated-f32.patch"

# Nothing outside the two rig cuts.
other=$(git -C "$C" diff --name-only "$DIAG" "$HEAD_REF" | grep -v "^$GEN/" | grep -vxF -f <(printf '%s\n' "${HAND[@]}") || true)
[ -z "$other" ] || { echo "uncut files: $other" >&2; exit 1; }

# Hash the committed tree of <head-ref>, not the work tree: a dirty checkout must not leak into SHA256SUMS.
V=$(mktemp -d /tmp/colmap-cut.XXXXXX)
trap 'rm -rf "$V"' EXIT
git clone -q --shared --no-checkout "$C" "$V/head"
git -C "$V/head" checkout -q --detach "$(git -C "$C" rev-parse "$HEAD_REF")"
git -C "$C" diff --name-only "$PIN" "$HEAD_REF" | (cd "$V/head" && xargs sha256sum) > "$P/SHA256SUMS"

git clone -q --shared --no-checkout "$C" "$V/pin"
git -C "$V/pin" checkout -q --detach "$PIN"
for p in "${PATCHES[@]}"; do
  git -C "$V/pin" apply "$P/$p"
done
(cd "$V/pin" && sha256sum --quiet -c "$P/SHA256SUMS")
echo "SUMS OK: $(wc -l < "$P/SHA256SUMS") files, ${#PATCHES[@]} patches"
