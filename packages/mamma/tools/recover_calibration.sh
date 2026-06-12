#!/usr/bin/env bash
# Calibration-free front end: recover a mamma rig from uncalibrated, time-synced
# videos, then (optionally) run the quality pipeline. ONLY for captures without a
# ground-truth calibration — a dir that already has meta/ is left untouched unless
# --force is passed. Chains the two solve-groups (monoprior VGGT, mamma BA).
#
# Usage:
#   tools/recover_calibration.sh <videos_dir> [--force] [--run OUT.rrd]
#
#   <videos_dir>  capture dir with per-camera mp4s (in videos_light/ or videos/)
#   --force       recompute even if meta/ already exists
#   --run OUT     after recovery, run the quality pipeline and save OUT.rrd
set -euo pipefail
DIR="${1:?usage: recover_calibration.sh <videos_dir> [--force] [--run OUT.rrd]}"; shift
FORCE=0; RUN_RRD=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --run) RUN_RRD="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # packages/mamma
ROOT="$(cd "$HERE/../.." && pwd)"                  # worktree root
MONO="$ROOT/packages/monoprior"
TRT="$HERE/.trt_cache/mammanet_b4_fp16_trt101339_sm120.plan"

if [ -d "$DIR/meta" ] && [ "$FORCE" -eq 0 ]; then
  echo "meta/ already present at $DIR — this is the calibrated path; nothing to recover."
  echo "(pass --force to re-derive calibration from the videos anyway.)"
else
  echo ">> stage A: VGGT + MoGe-v2 metric scale + ground-plane gravity (monoprior env)"
  ( cd "$MONO" && pixi run -e monoprior --frozen \
      python tools/demos/calibrate_synced_videos.py --videos-dir "$DIR" $([ "$FORCE" -eq 1 ] && echo --force) )
  echo ">> stage B: confidence-weighted keypoint bundle adjustment (mamma env)"
  ( cd "$HERE" && pixi run -e mamma --frozen \
      python tools/refine_calibration_ba.py --data-dir "$DIR" --trt-engine "$TRT" )
fi

if [ -n "$RUN_RRD" ]; then
  echo ">> running quality pipeline -> $RUN_RRD"
  ( cd "$HERE" && pixi run -e mamma --frozen \
      python tools/dump_artifacts.py --data-dir "$DIR" --preset quality \
      --trt-engine "$TRT" --out-dir "$(dirname "$RUN_RRD")/dump" --rr-config.save "$RUN_RRD" --rr-config.headless )
fi
echo "done."
