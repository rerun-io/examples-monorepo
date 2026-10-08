#!/bin/bash
# The web panel on a cap (robocap-panel, port 8090): start/stop the live run, watch temperatures, CPU/NPU/GPU/DDR/VPU, power,
# Wi-Fi and the pipeline. It runs detached on the cap until stopped or the cap reboots (nothing is added to the cap's boot). The
# same binary supervises each run (`robocap-panel handoff`), so a run outlives a panel stop or restart.
#
# Usage: panel.sh [deploy|start|stop|status|url] --cap a|b [--cap-address <ip>] [--port 8090]
#   deploy  put the aarch64 binary (scripts/build-arm.sh builds it) in place with `deploy.sh --panel-only` (sha256-checked, by
#           rename; the old one becomes robocap-panel.prev), then restart the panel; a run that is going keeps its supervisor
#   start   start it (no-op if it already runs); stop: SIGTERM to its pid (run/panel.pid), never a kill by name
#   url     print http://<cap address>:<port>/
#   --cap-address  the cap's address on the network the browser is on, for the printed URL (required by deploy, start and url)
set -euo pipefail
action=${1:-status}; shift || true
CAP=
cap_address=
port=8090
while [[ $# -gt 0 ]]; do
    case $1 in
        --cap) CAP=$2; shift ;;
        --port) port=$2; shift ;;
        --cap-address) cap_address=$2; shift ;;
        *) echo "panel.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=cap-env.sh
source "$here/cap-env.sh"
if [[ -z $cap_address && $action =~ ^(deploy|start|url)$ ]]; then
    echo "panel.sh $action: --cap-address <the cap's address> is required (for the printed URL)" >&2; exit 2
fi

start_on_cap() {
    cap_ssh bash -s -- "$CAP_ROOT" "$port" <<'EOF'
root=$1; port=$2; pidfile=$root/run/panel.pid
mkdir -p "$root/run" "$root/logs"
if [[ -f $pidfile ]] && [[ $(head -z -n 1 "/proc/$(cat "$pidfile")/cmdline" 2>/dev/null | tr -d "\0") == "$root/bin/robocap-panel" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then echo "already running: pid $(cat "$pidfile")"; exit 0; fi
# On the A55 cores (0-3), off the A76s that SLAM and the hands use. A run it starts inherits the mask; robocap-live pins its own threads.
setsid nohup taskset -c 0-3 "$root/bin/robocap-panel" --port "$port" --root "$root" > "$root/logs/panel.log" 2>&1 < /dev/null &
echo $! > "$pidfile"; sleep 1
kill -0 "$(cat "$pidfile")" 2>/dev/null && echo "started: pid $(cat "$pidfile")" || { echo "it exited:"; cat "$root/logs/panel.log"; exit 1; }
EOF
}

case $action in
    deploy)
        "$here/deploy.sh" --cap "$CAP" --panel-only
        "$here/panel.sh" stop --cap "$CAP" --port "$port" || true
        start_on_cap
        echo "http://$cap_address:$port/"
        ;;
    start) start_on_cap; echo "http://$cap_address:$port/" ;;
    stop)
        cap_ssh bash -s -- "$CAP_ROOT" <<'EOF'
root=$1; pidfile=$root/run/panel.pid
if [[ -f $pidfile ]] && [[ $(head -z -n 1 "/proc/$(cat "$pidfile")/cmdline" 2>/dev/null | tr -d "\0") == "$root/bin/robocap-panel" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then kill "$(cat "$pidfile")" && echo "stopped pid $(cat "$pidfile")"; else echo "not running"; fi
rm -f "$pidfile"
EOF
        ;;
    status)
        cap_ssh bash -s -- "$CAP_ROOT" <<'EOF'
root=$1; pidfile=$root/run/panel.pid
if [[ -f $pidfile ]] && [[ $(head -z -n 1 "/proc/$(cat "$pidfile")/cmdline" 2>/dev/null | tr -d "\0") == "$root/bin/robocap-panel" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "running: pid $(cat "$pidfile")"
else
    echo "not running"
fi
EOF
        ;;
    url) echo "http://$cap_address:$port/" ;;
    *) echo "panel.sh: unknown action $action" >&2; exit 2 ;;
esac
