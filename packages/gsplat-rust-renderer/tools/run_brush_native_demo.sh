#!/usr/bin/env bash
# Train a scene with brush and visualize it in our viewer with brush's *own*
# rich blueprint (loss/lr/psnr/ssim/splats/refine/memory + eval-view tabs) but
# with real GPU splats in the Scene view.
#
# This is the deterministic, no-discovery path: a fixed recording id is shared
# between brush (via BRUSH_RERUN_RECORDING_ID, honored by the local brush patch
# in crates/brush-rerun/src/visualize_tools.rs) and the --brush-native sidecar,
# so both stream into one rerun store under one blueprint.
#
# Prereqs: the custom viewer is already running headless on :9876
#   (gsplat-rust-renderer --headless), and brush-cli is built.
#
# Usage:
#   tools/run_brush_native_demo.sh DATA_DIR [TOTAL_ITERS] [EXPORT_DIR]
# Example:
#   tools/run_brush_native_demo.sh data/nerf-synthetic/lego-brush 30000
set -euo pipefail

DATA_DIR="${1:?usage: run_brush_native_demo.sh DATA_DIR [TOTAL_ITERS] [EXPORT_DIR]}"
TOTAL_ITERS="${2:-30000}"
EXPORT_DIR="${3:-/tmp/brush-runs/$(basename "$DATA_DIR")}"
EXPORT_EVERY="${EXPORT_EVERY:-200}"
EVAL_EVERY="${EVAL_EVERY:-500}"
# COLMAP captures (no transforms_val) need an eval split: brush holds out every
# Nth view and the sidecar must match it so the eval-tab count lines up. 0 = off.
EVAL_SPLIT_EVERY="${EVAL_SPLIT_EVERY:-0}"
BRUSH_CLI="${BRUSH_CLI:-brush-cli}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # package root
REPO_ROOT="$(cd "$HERE/../.." && pwd)"                    # monorepo root (pixi.toml)

# Optional eval-split flag, passed identically to brush and the sidecar.
SPLIT_FLAG=()
if [ "$EVAL_SPLIT_EVERY" -gt 0 ]; then
  SPLIT_FLAG=(--eval-split-every "$EVAL_SPLIT_EVERY")
fi

# A fixed, human-readable recording id shared by both processes.
RID="brush-native-$(basename "$DATA_DIR")-$$"

if ! ss -ltn 2>/dev/null | grep -q '127.0.0.1:9876'; then
  echo "WARNING: nothing is listening on :9876 — start the viewer first:" >&2
  echo "  $HERE/target/release/gsplat-rust-renderer --headless &" >&2
fi

mkdir -p "$EXPORT_DIR"
BRUSH_LOG="$EXPORT_DIR/brush.log"
echo "recording id : $RID"
echo "export dir   : $EXPORT_DIR"
echo "brush log    : $BRUSH_LOG"

# 1. brush: logs its full dashboard + sends its blueprint, pinned to RID, no
#    ellipsoid splat logging (we own world/splats), connects to the viewer.
BRUSH_RERUN_RECORDING_ID="$RID" RUST_LOG=brush_cli=info,brush_process=info \
  "$BRUSH_CLI" "$DATA_DIR" \
  --total-train-iters "$TOTAL_ITERS" \
  --export-every "$EXPORT_EVERY" \
  --export-path "$EXPORT_DIR" \
  --eval-every "$EVAL_EVERY" \
  "${SPLIT_FLAG[@]}" \
  --rerun-enabled \
  >"$BRUSH_LOG" 2>&1 &
BRUSH_PID=$!
trap 'kill "$BRUSH_PID" 2>/dev/null || true' EXIT
echo "brush pid    : $BRUSH_PID"

# Wait for brush to start training (so the recording exists before we join).
for _ in $(seq 1 30); do
  grep -q "Start training loop" "$BRUSH_LOG" 2>/dev/null && break
  kill -0 "$BRUSH_PID" 2>/dev/null || { echo "brush exited early — see $BRUSH_LOG" >&2; exit 1; }
  sleep 1
done

# 2. sidecar: join RID, overlay GaussianSplats3D on world/splats, send brush's
#    blueprint replica with the splat visualizer override. Follows to the end.
pixi run --frozen --manifest-path "$REPO_ROOT/pixi.toml" -e gsplat-rust-renderer-dev \
  python -u "$HERE/tools/visualize_brush_training.py" \
  --brush-native \
  --rr-config.connect \
  --rr-config.application-id Brush \
  --rr-config.recording-id "$RID" \
  --scene-dir "$DATA_DIR" \
  --export-dir "$EXPORT_DIR" \
  "${SPLIT_FLAG[@]}" \
  --total-iters "$TOTAL_ITERS"

wait "$BRUSH_PID" 2>/dev/null || true
echo "done."
