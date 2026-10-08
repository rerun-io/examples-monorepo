#!/bin/bash
# The web panel on a cap (robocap-panel, port 8090): start/stop the live run, watch temperatures, CPU/NPU/GPU/DDR/VPU, power,
# Wi-Fi and the pipeline. It runs detached on the cap until stopped or the cap reboots; install-boot makes the cap start it at boot,
# so a phone on the cap's hotspot finds it at http://192.168.11.1:8090/. The same binary supervises each run (`robocap-panel
# handoff`), so a run outlives a panel stop or restart. Nothing ever starts a run at boot.
#
# Usage: panel.sh [deploy|start|stop|status|url|install-boot|remove-boot] --cap a|b [--cap-address <ip>] [--port 8090]
#   deploy  put the aarch64 binary (scripts/build-arm.sh builds it) in place with `deploy.sh --panel-only` (sha256-checked, by
#           rename; the old one becomes robocap-panel.prev), then restart the panel; a run that is going keeps its supervisor
#   start   start it (no-op if it already runs); stop: SIGTERM to its pid (run/panel.pid, checked to be the panel), never by name
#   url     print http://<cap address>:<port>/
#   install-boot  link /etc/init.d/S87robocap-panel to the binary (no vendor file changes): rcS then runs `robocap-panel start` at
#           boot and rcK `robocap-panel stop` at shutdown; remove-boot deletes the link
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

# The lifecycle lives in the binary (robocap-panel start|stop|status|install-boot|remove-boot); this script only reaches the cap.
panel() { cap_ssh "$CAP_ROOT/bin/robocap-panel" "$@"; }

case $action in
    deploy)
        "$here/deploy.sh" --cap "$CAP" --panel-only
        panel stop
        panel start --port "$port"
        echo "http://$cap_address:$port/"
        ;;
    start) panel start --port "$port"; echo "http://$cap_address:$port/" ;;
    stop|status|install-boot|remove-boot) panel "$action" ;;
    url) echo "http://$cap_address:$port/" ;;
    *) echo "panel.sh: unknown action $action" >&2; exit 2 ;;
esac
