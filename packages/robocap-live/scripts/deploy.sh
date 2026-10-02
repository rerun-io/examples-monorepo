#!/bin/bash
# Copy robocap-live to a cap: the aarch64 binary, the RKNN models, the display asset and the handoff script go to
# /root/robocap-live/{bin,models,assets,scripts}, as one tar over ssh (cap-env.sh's cap_ssh), checked by sha256 on the cap before
# anything replaces the files in place. The previous binary is kept as bin/<name>.prev. Nothing else is touched.
#
# Usage: deploy.sh --cap a|b --models <dir> --display <asset.rrd> [--rig <rig.json>] [--root /root/robocap-live]
#                  [--binary <path>] [--extra <file>]... [--dry-run]
#   --models   required: *.rknn and MODELS.md from here (no *.rknn there is an error; models/MODELS.md says how they are made)
#   --display  required: the display asset (made by: packages/handtrack/tools/robocap_live_display.py --output <it> --root <RoboCap dataset root>)
#   --rig      the cap's calibration for the live source, to <root>/rig.json; a given rig must name the cap's device (for Cap A,
#              e.g. the s66 dump's rig.json: Cap A's factory calibration from the catalog). Without it the cap keeps the rig.json
#              it has (Cap B: its own factory calibration since 2026-10-01), and the deploy refuses if the cap has none
#   --binary   default: target/aarch64-unknown-linux-gnu/release/robocap-live (scripts/build-arm.sh)
#   --extra    more files for bin/ (e.g. the log_replay example)
set -euo pipefail
CAP=
root_override=
here=$(cd "$(dirname "$0")" && pwd)
pkg=$(dirname "$here")
binary=$pkg/target/aarch64-unknown-linux-gnu/release/robocap-live
extras=()
models=
display=
rig=
dry_run=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --cap) CAP=$2; shift ;;
        --root) root_override=$2; shift ;;
        --binary) binary=$2; shift ;;
        --extra) extras+=("$2"); shift ;;
        --models) models=$2; shift ;;
        --display) display=$2; shift ;;
        --rig) rig=$2; shift ;;
        --dry-run) dry_run=1 ;;
        -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0 ;;
        *) echo "deploy.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
# shellcheck source=cap-env.sh
source "$here/cap-env.sh"
CAP_ROOT=${root_override:-$CAP_ROOT}
log() { echo "[deploy $(date +%H:%M:%S)] $*" >&2; }
[[ -n $models ]] || { log "--models <dir with the RKNN models> is required (models/MODELS.md says how they are made)"; exit 2; }
[[ -n $display ]] || { log "--display <robocap-live-display.rrd> is required (packages/handtrack/tools/robocap_live_display.py makes it)"; exit 2; }

stage=$(mktemp -d /tmp/robocap-live-deploy.XXXXXX)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage"/{bin,models,assets,scripts}
for file in "$binary" "${extras[@]}"; do
    [[ -f $file ]] || { log "missing $file"; exit 1; }
    file -b "$file" | grep -q "ARM aarch64" || { log "$file is not an aarch64 executable: $(file -b "$file")"; exit 1; }
    cp "$file" "$stage/bin/"
done
compgen -G "$models/*.rknn" >/dev/null || { log "no *.rknn in $models: pass --models <dir with the RKNN models> (models/MODELS.md says how they are made)"; exit 1; }
cp "$models"/*.rknn "$stage/models/"
[[ -f $models/MODELS.md ]] && cp "$models/MODELS.md" "$stage/models/"
if [[ -f $display ]]; then cp "$display" "$stage/assets/robocap-live-display.rrd"; else log "warning: no display asset at $display"; fi
cp "$here/handoff-run.sh" "$stage/scripts/"
# Without --rig the cap keeps the rig.json it has; a real deploy checks that it has one.
[[ -z $rig ]] && log "no --rig: Cap $CAP keeps its own $CAP_ROOT/rig.json"
if [[ -n $rig ]]; then
    grep -q "\"device\": *\"cap_$CAP\"" "$rig" || { log "$rig is not a cap_$CAP rig"; exit 1; }
    cp "$rig" "$stage/rig.json"
fi
(cd "$stage" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
log "staged $(find "$stage" -type f | wc -l) files, $(du -sh "$stage" | cut -f1), for Cap $CAP ($CAP_HOSTNAME):"
sed 's/^/  /' "$stage/SHA256SUMS" >&2
if [[ $dry_run == 1 ]]; then log "dry run: nothing copied"; exit 0; fi

host=$(cap_ssh hostname)
[[ $host == "$CAP_HOSTNAME" ]] || { log "the cap answers as '$host', expected $CAP_HOSTNAME; refusing"; exit 1; }
if [[ -z $rig ]]; then
    cap_ssh "test -f $CAP_ROOT/rig.json" || { log "no --rig, and Cap $CAP has no $CAP_ROOT/rig.json: pass --rig <the cap's rig.json>; refusing"; exit 1; }
fi
# The caps' /root is small (~14 GB): keep at least 5 GB free after the copy (staged twice briefly).
need_kb=$(du -sk "$stage" | cut -f1)
free_kb=$(cap_ssh "df -k /root | tail -1" | awk '{print $4}')
floor_kb=$((5 * 1024 * 1024))
(( free_kb - 2 * need_kb > floor_kb )) || { log "/root has ${free_kb} KB free; copying ${need_kb} KB would leave less than 5 GB; refusing"; exit 1; }
log "/root on the cap: $((free_kb / 1024)) MB free before the copy"

start=$(date +%s.%N)
tar -C "$stage" -cf - . | cap_ssh "rm -rf $CAP_ROOT/.incoming && mkdir -p $CAP_ROOT/.incoming && tar -C $CAP_ROOT/.incoming -xf -"
log "copied ${need_kb} KB in $(awk "BEGIN {printf \"%.1f\", $(date +%s.%N) - $start}") s"
cap_ssh bash -s -- "$CAP_ROOT" <<'EOF'
set -euo pipefail
root=$1
cd "$root/.incoming"
sha256sum -c SHA256SUMS >/dev/null || { echo "sha256 mismatch on the cap; nothing replaced" >&2; sha256sum -c SHA256SUMS >&2 || true; exit 1; }
mkdir -p "$root"/{bin,models,assets,scripts,logs,run}
for file in $(awk '{print $2}' SHA256SUMS); do
    target=$root/${file#./}
    if [[ $file == ./bin/* && -f $target ]]; then cp -p "$target" "$target.prev"; fi
    mkdir -p "$(dirname "$target")"
    cp "$file" "$target.new" && mv -f "$target.new" "$target"
done
chmod +x "$root"/bin/* "$root"/scripts/*.sh
cd "$root" && sed 's#\./#'"$root"'/#' .incoming/SHA256SUMS | sha256sum -c - | sed 's/^/  verified /'
cp .incoming/SHA256SUMS "$root/run/SHA256SUMS.deployed"
rm -rf "$root/.incoming"
EOF
log "deployed to Cap $CAP:$CAP_ROOT"
