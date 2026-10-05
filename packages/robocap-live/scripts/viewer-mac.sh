#!/bin/bash
# The native Rerun viewer a cap streams to, on a Mac (--viewer-host) that is on the cap's network (--viewer-if).
#
# Runs on the control host (over ssh to the viewer Mac) or on that Mac itself. The viewer is ~/.pixi/bin/rerun 0.38.1, started as a
# launchd job in the logged-in user's GUI domain (so a visible window opens on that Mac's display), listening on gRPC :<port>,
# with ~/.pixi/bin (ffmpeg, for H.264) on its PATH. Idempotent: an already-listening viewer on the port is reported, not restarted.
# It touches only its own launchd job (label io.robocap-live.viewer-<port>); other viewers and MCP servers are left alone.
#
# Usage: viewer-mac.sh [start|stop|status|url] --cap a|b --viewer-host <ssh host> [--viewer-if <interface>] [--cap-address <ip>]
#                      [--headless] [--port N] [--memory-limit 4GB]
#   start   start (or confirm) the viewer and print the URL the cap must use
#   stop    stop our viewer job on that port (launchctl bootout; never a kill by name)
#   status  what listens on the port
#   url     print rerun+http://<viewer Mac's address on the cap's network>:<port>/proxy; no address there is an error
#   --viewer-host  the viewer Mac's ssh host (required unless this runs on that Mac)
#   --viewer-if    its interface on the cap's network, e.g. its Wi-Fi on Cap A's hotspot (required by start and url)
#   --cap-address  the cap's address on that network: the fallback takes any address of the Mac in its /24 (required by start and url)
set -euo pipefail

action=start
port=9876
headless=0
memory_limit=4GB
CAP=
mac_host=
wifi_if=
cap_address=
while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|status|url) action=$1 ;;
        --cap) CAP=$2; shift ;;
        --viewer-host) mac_host=$2; shift ;;
        --viewer-if) wifi_if=$2; shift ;;
        --cap-address) cap_address=$2; shift ;;
        --headless) headless=1 ;;
        --port) port=$2; shift ;;
        --memory-limit) memory_limit=$2; shift ;;
        -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0 ;;
        *) echo "viewer-mac.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
# shellcheck source=cap-env.sh
source "$(dirname "$0")/cap-env.sh"
[[ -n $mac_host || $(uname) == Darwin ]] || { echo "viewer-mac.sh: --viewer-host <the viewer Mac's ssh host> is required" >&2; exit 2; }
if [[ $action == start || $action == url ]] && [[ -z $wifi_if || -z $cap_address ]]; then
    echo "viewer-mac.sh $action: --viewer-if <the Mac's interface on the cap's network> and --cap-address <the cap's address> are required" >&2; exit 2
fi
label=io.robocap-live.viewer-$port

# Run a script on the viewer Mac (locally when this is a Mac).
on_mac() {
    if [[ $(uname) == Darwin ]]; then bash -s -- "$@"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$mac_host" bash -s -- "$@"; fi
}

mac_address() {
    # The address the cap reaches the viewer Mac at: its interface on the cap's network (DHCP, so looked up every time), else
    # any of its addresses on the cap's /24 (from --cap-address). No address is an error.
    local subnet=${cap_address%.*}. ip
    ip=$(on_mac "$wifi_if" "$subnet" <<'EOF'
ip=$(ipconfig getifaddr "$1" 2>/dev/null || true)
[[ -n $ip ]] || ip=$(ifconfig 2>/dev/null | awk -v subnet="$2" '$1 == "inet" && index($2, subnet) == 1 {print $2; exit}')
echo "$ip"
EOF
)
    [[ -n $ip ]] || { echo "viewer-mac.sh: $mac_host has no address on $wifi_if or in ${subnet}0/24 (not on Cap $CAP's network?)" >&2; return 1; }
    echo "$ip"
}

status() {
    on_mac "$port" "$label" <<'EOF'
port=$1; label=$2
pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u || true)
if [[ -z $pids ]]; then echo "port $port: nothing listening"; exit 1; fi
for pid in $pids; do echo "port $port: pid $pid $(ps -o command= -p "$pid" | cut -c1-160)"; done
if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then echo "launchd job $label: loaded"; else echo "launchd job $label: not ours"; fi
EOF
}

case $action in
    status)
        status
        ;;
    url)
        ip=$(mac_address) || exit 1
        echo "rerun+http://$ip:$port/proxy"
        ;;
    stop)
        on_mac "$label" "$port" <<'EOF'
label=$1; port=$2
if launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
    launchctl bootout "gui/$(id -u)/$label" && echo "stopped $label"
else
    echo "no job $label (nothing of ours to stop)"
fi
for _ in $(seq 1 20); do lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1 || exit 0; sleep 0.5; done
echo "warning: port $port still has a listener that is not our job:" >&2; lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
EOF
        ;;
    start)
        on_mac "$label" "$port" "$headless" "$memory_limit" <<'EOF' || exit 1
set -euo pipefail
label=$1; port=$2; headless=$3; memory_limit=$4
rerun=$HOME/.pixi/bin/rerun
version=$("$rerun" --version 2>/dev/null | sed -n 1p)
[[ $version == "rerun-cli 0.38.1 "* ]] || { echo "unexpected viewer version: $version (the SDK is 0.38.1)" >&2; exit 1; }
[[ -x $HOME/.pixi/bin/ffmpeg ]] || echo "warning: no ffmpeg in ~/.pixi/bin; H.264 panes will not decode" >&2
pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | sort -u || true)
if [[ -n $pids ]]; then
    for pid in $pids; do
        command=$(ps -o command= -p "$pid")
        if [[ $command == *rerun* ]]; then echo "already running: pid $pid ($command)"; exit 0; fi
        echo "port $port is taken by pid $pid ($command), not a Rerun viewer" >&2; exit 1
    done
fi
launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
plist=$HOME/Library/LaunchAgents/$label.plist
log=$HOME/Library/Logs/robocap-live-viewer-$port.log
mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs"
args="<string>$rerun</string><string>--port</string><string>$port</string><string>--memory-limit</string><string>$memory_limit</string><string>--expect-data-soon</string>"
[[ $headless == 1 ]] && args="$args<string>--headless</string>"
cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$label</string>
<key>ProgramArguments</key><array>$args</array>
<key>EnvironmentVariables</key><dict><key>PATH</key><string>$HOME/.pixi/bin:/usr/bin:/bin:/usr/sbin:/sbin</string></dict>
<key>StandardOutPath</key><string>$log</string>
<key>StandardErrorPath</key><string>$log</string>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><false/>
</dict></plist>
PLIST
# The GUI domain: the window opens on the logged-in user's display even though this runs over ssh.
launchctl bootstrap "gui/$(id -u)" "$plist"
# Not persistent across logins: the plist stays only for the bootout above to find on the next start.
for _ in $(seq 1 60); do
    pid=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
    if [[ -n $pid ]]; then echo "started: pid $pid, $version, log $log"; exit 0; fi
    sleep 0.5
done
echo "the viewer did not start listening on $port within 30 s; log tail:" >&2; tail -20 "$log" >&2; exit 1
EOF
        ip=$(mac_address) || exit 1
        # From the cap side: can the cap open a TCP connection to the viewer? (best effort; the cap may be off the network)
        if [[ $(uname) != Darwin ]] && cap_ssh "timeout 3 bash -c 'exec 3<>/dev/tcp/$ip/$port' 2>/dev/null || nc -z -w 3 $ip $port" >/dev/null 2>&1; then
            echo "Cap $CAP reaches the viewer at $ip:$port"
        else
            echo "note: could not confirm from Cap $CAP that $ip:$port is reachable (cap offline or no nc/bash there)" >&2
        fi
        echo "viewer URL for the cap: rerun+http://$ip:$port/proxy"
        echo "stop it with: $0 stop --cap $CAP${mac_host:+ --viewer-host $mac_host} --port $port"
        ;;
esac
