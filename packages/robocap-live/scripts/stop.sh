#!/bin/bash
# Stop the robocap-live run that start-live.sh started, cleanly: SIGINT to our program inside the run's session, so it
# flushes its logs and frees the devices; then the handoff script's EXIT trap resumes the vendor recorder. Waits for the whole session
# to end and prints the log tail (with the handoff's restore lines). Only processes under /root/robocap-live/bin in that session
# are signalled; never the vendor recorder or anything by name.
#
# Usage: stop.sh --cap a|b [--root /root/robocap-live] [--wait 90] [--viewer --viewer-host <ssh host>] [--port 9876] [--any] [--print]
#   --viewer   also stop our viewer job on the viewer Mac (viewer-mac.sh stop); --viewer-host names that Mac (its ssh host)
#   --any      with no start-live.sh session alive, stop any /root/robocap-live/bin program (a run started by hand); without it,
#              stop.sh signals nothing outside the recorded session (other test runs stay untouched)
#   --print    say what would be done; the cap is not contacted
set -euo pipefail
CAP=
root_override=
here=$(cd "$(dirname "$0")" && pwd)
wait_s=90
stop_viewer=0
any=0
print_only=0
port=9876
viewer_host=
while [[ $# -gt 0 ]]; do
    case $1 in
        --cap) CAP=$2; shift ;;
        --root) root_override=$2; shift ;;
        --wait) wait_s=$2; shift ;;
        --viewer) stop_viewer=1 ;;
        --viewer-host) viewer_host=$2; shift ;;
        --any) any=1 ;;
        --print) print_only=1 ;;
        --port) port=$2; shift ;;
        -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0 ;;
        *) echo "stop.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
# shellcheck source=cap-env.sh
source "$here/cap-env.sh"
CAP_ROOT=${root_override:-$CAP_ROOT}
if [[ $stop_viewer == 1 && -z $viewer_host && $(uname) != Darwin ]]; then
    echo "stop.sh: --viewer needs --viewer-host <the viewer Mac's ssh host>" >&2; exit 2
fi

if [[ $print_only == 1 ]]; then
    echo "[stop] --print: on Cap $CAP ($CAP_HOSTNAME) would SIGINT the $CAP_ROOT/bin programs in the session of $CAP_ROOT/run/live.pid,"
    echo "[stop]   SIGINT again after 20 s, SIGTERM after 40 s, wait up to ${wait_s} s for the session (handoff restore) to end, print the log tail"
    [[ $stop_viewer == 1 ]] && echo "[stop]   then: viewer-mac.sh stop --cap $CAP --viewer-host $viewer_host --port $port"
    exit 0
fi
cap_ssh bash -s -- "$CAP_ROOT" "$wait_s" "$any" <<'EOF'
set -uo pipefail
root=$1; wait_s=$2; any=$3
log() { echo "[stop $(date +%H:%M:%S)] $*" >&2; }
session=$(cat "$root/run/live.pid" 2>/dev/null || true)
logfile=$(cat "$root/run/live.log" 2>/dev/null || true)
# Our programs in the session (or, without a live session, any of ours: a run started by hand).
ours() {
    for stat in /proc/[0-9]*/stat; do
        pid=${stat#/proc/}; pid=${pid%/stat}
        exe=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || continue
        [[ $exe == "$root/bin/"* ]] || continue
        if [[ -n $session ]]; then
            sid=$(sed 's/.*) //' "$stat" 2>/dev/null | cut -d' ' -f4) || continue
            [[ $sid == "$session" ]] || continue
        fi
        echo "$pid"
    done
}
if [[ -n $session ]] && ! kill -0 "$session" 2>/dev/null; then log "session $session already ended"; session=; fi
if [[ -z $session && $any != 1 ]]; then
    log "no start-live.sh session is running; nothing signalled (--any stops a run started by hand)"
    [[ -n $logfile ]] && tail -5 "$logfile"
    exit 0
fi
pids=$(ours)
if [[ -z $pids && -z $session ]]; then log "nothing of ours is running"; [[ -n $logfile ]] && tail -5 "$logfile"; exit 0; fi
log "SIGINT to $(echo $pids) (session ${session:-none})"
[[ -n $pids ]] && kill -INT $pids 2>/dev/null
for ((i = 0; i < wait_s * 2; i++)); do
    alive=$(ours)
    if [[ -z $alive ]] && { [[ -z $session ]] || ! kill -0 "$session" 2>/dev/null; }; then break; fi
    if (( i == 40 )) && [[ -n $alive ]]; then log "still running after 20 s; SIGINT again"; kill -INT $alive 2>/dev/null; fi
    if (( i == 80 )) && [[ -n $alive ]]; then log "still running after 40 s; SIGTERM"; kill -TERM $alive 2>/dev/null; fi
    sleep 0.5
done
if [[ -n $(ours) ]] || { [[ -n $session ]] && kill -0 "$session" 2>/dev/null; }; then
    log "WARNING: the run has not ended after ${wait_s} s; check by hand (session $session)"; exit 1
fi
rm -f "$root/run/live.pid"
log "stopped; SoC $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) °C; vendor recorder pid(s): $(pgrep -x omni-specs.bin | tr '\n' ' ')"
[[ -n $logfile ]] && { log "log $logfile tail:"; tail -25 "$logfile"; }
EOF
if [[ $stop_viewer == 1 ]]; then "$here/viewer-mac.sh" stop --cap "$CAP" --viewer-host "$viewer_host" --port "$port"; fi
