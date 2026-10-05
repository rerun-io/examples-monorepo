#!/bin/bash
# Start a live (or replay) run on a cap in one line. Every cap-side step goes through the power/liveness guard
# (robocap-guard, found on PATH: real charger input limit, VBAT >= 7.9 V, temperature, load, no copy or other heavy job;
# it records klog.txt, samples.jsonl and alarms.txt in a log dir of its own and prints where).
#   1. robocap-guard --check-only <cap> must pass (Cap B: --allow-low-input, and a loud warning: it runs on its small battery);
#   2. the viewer is started on the viewer Mac (--viewer-host) or, with --no-viewer-start, checked; with --viewer <url> the cap
#      streams to that URL and no Mac is contacted;
#   3. robocap-guard <cap> -- '<cap script>': the cap script starts robocap-live detached on the cap (it survives the session; log
#      in /root/robocap-live/logs/, session id in run/live.pid), through scripts/handoff-run.sh for a live run (it pauses the
#      vendor recorder and always restores it), and waits for it to end so the guard samples the whole run. The guard itself
#      runs detached on this host (log path printed).
# Stop it with scripts/stop.sh (SIGINT, so the handoff trap gives the cameras back).
# First-run defaults: --seconds 600 (longer needs --long), --video-cameras 0,1, no uclamp frequency hints (--uclamp opts in).
#
# Usage: start-live.sh --cap a|b --log-dir <dir>
#                      (--viewer <rerun+http://host:port/proxy> | --viewer-host <ssh host> --viewer-if <interface> --cap-address <ip>)
#                      [--seconds 600] [--long] [--replay <dump dir on the cap>] [--save] [--port 9876]
#                      [--headless-viewer] [--no-viewer-start] [--video-cameras 0,1] [--uclamp] [--print] [--force]
#                      [--slam-lane gpu|cpu] [--slam-lag auto|true|false]
#                      [-- <program> <args>...]
#   --log-dir      required: this host's log of the guard run goes to <dir>/start-live-<cap>-<time>.log
#   --viewer       the viewer URL the cap streams to, used as is (no viewer is started or looked up)
#   --viewer-host  the viewer Mac's ssh host, for viewer-mac.sh: the viewer is started there (--no-viewer-start: only looked
#                  up) and the cap streams to its address on --viewer-if
#   --viewer-if    that Mac's interface on the cap's network
#   --cap-address  the cap's address on that network (viewer-mac.sh's fallback: any address of the Mac in its /24)
#   --replay       run the replay source (no handoff) instead of the live cameras
#   --save         also save the stream to /root/robocap-live/recordings/<time>.rrd on the cap
#   --print        print the exact command chain (guard check, viewer, guard run, cap script); contacts no cap (with
#                  --viewer-host it still asks the viewer Mac for its address)
#   --video-cameras  cameras whose H.264 goes to the viewer (power budget) [0,1]
#   --uclamp       pass robocap-live uclamp.min 1024 frequency hints for SLAM and hands (more rate, more power)
#   --long         allow --seconds above 600
#   --force        start even if another /root/robocap-live/bin program (another test) is running
#   -- <program>   run this instead of the default robocap-live command line; {viewer}, {save} and {seconds} are substituted
# Examples:
#   scripts/start-live.sh --cap a --viewer-host <Mac> --viewer-if <its interface on the hotspot> --cap-address <Cap A's address> \
#       --log-dir <dir>                                     # Cap A, live, 10 min (600 s), viewer on that Mac's display
#   scripts/start-live.sh --cap b --viewer rerun+http://<viewer>:9876/proxy --log-dir <dir> --replay /root/robocap-live/dumps/s66-clip --seconds 120
set -euo pipefail
CAP=
root_override=
here=$(cd "$(dirname "$0")" && pwd)
seconds=600
long=0
video_cameras=0,1
uclamp=0
slam_lane=gpu
slam_lag=auto
replay=
save=0
port=9876
headless=()
start_viewer=1
print_only=0
force=0
custom=()
viewer=
viewer_host=
viewer_if=
cap_address=
log_dir=
while [[ $# -gt 0 ]]; do
    case $1 in
        --cap) CAP=$2; shift ;;
        --root) root_override=$2; shift ;;
        --seconds) seconds=$2; shift ;;
        --replay) replay=$2; shift ;;
        --save) save=1 ;;
        --port) port=$2; shift ;;
        --viewer) viewer=$2; shift ;;
        --viewer-host) viewer_host=$2; shift ;;
        --viewer-if) viewer_if=$2; shift ;;
        --cap-address) cap_address=$2; shift ;;
        --log-dir) log_dir=$2; shift ;;
        --headless-viewer) headless=(--headless) ;;
        --no-viewer-start) start_viewer=0 ;;
        --print) print_only=1 ;;
        --force) force=1 ;;
        --long) long=1 ;;
        --video-cameras) video_cameras=$2; shift ;;
        --uclamp) uclamp=1 ;;
        --slam-lane) slam_lane=$2; shift ;;
        --slam-lag) slam_lag=$2; shift ;;
        --) shift; custom=("$@"); break ;;
        -h|--help) sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0 ;;
        *) echo "start-live.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done
case $slam_lane in gpu|cpu) ;; *) echo '--slam-lane must be gpu or cpu' >&2; exit 2 ;; esac
case $slam_lag in auto|true|false) ;; *) echo '--slam-lag must be auto, true or false' >&2; exit 2 ;; esac
# shellcheck source=cap-env.sh
source "$here/cap-env.sh"
CAP_ROOT=${root_override:-$CAP_ROOT}
log() { echo "[start-live $(date +%H:%M:%S)] $*" >&2; }
stamp=$(date +%Y%m%d-%H%M%S)
(( seconds <= 600 || long == 1 )) || { log "--seconds $seconds: the first runs are capped at 600 s; add --long to go longer"; exit 2; }
[[ -n $log_dir ]] || { log "--log-dir <dir> is required (this host's log of the guard run)"; exit 2; }
if [[ -n $viewer ]]; then
    [[ -z $viewer_host$viewer_if$cap_address ]] || { log "--viewer is used as is: give it or --viewer-host/--viewer-if/--cap-address, not both"; exit 2; }
elif [[ -z $viewer_if || -z $cap_address || ( -z $viewer_host && $(uname) != Darwin ) ]]; then
    log "the viewer is required: --viewer <rerun+http://host:port/proxy>, or --viewer-host <ssh host> --viewer-if <interface> --cap-address <ip>"
    exit 2
fi

# 0. The power/liveness guard's preflight (read-only on the cap).
guard_flags=()
if [[ $CAP == b ]]; then
    guard_flags+=(--allow-low-input)
    log "!!!!!!!! WARNING: Cap B's charger input is limited to ~500 mA (~2.3 W): the full system runs on its small (~3.7 Wh)"
    log "!!!!!!!! battery and drains it (VBAT fell ~0.03 V/min under load). Keep runs short; the guard alarms below 7.6 V."
    log "Cap B uses its own factory calibration (rig.json device cap_b; SLAM picks it by device)."
fi
guard=$(command -v robocap-guard) || { log "robocap-guard is not on PATH (the power/liveness guard: robocap-guard [--check-only] a|b -- '<cmd>')"; exit 1; }
if [[ $print_only == 1 ]]; then
    log "--print: would run: $(printf '%q ' "$guard" --check-only "${guard_flags[@]}" "$CAP")"
else
    "$guard" --check-only "${guard_flags[@]}" "$CAP" || { log "the guard's preflight refused Cap $CAP (see above); nothing started"; exit 1; }
fi

# 1. The cap: reachable, the right one, cool, not already running ours, binary deployed; for live, the vendor recorder idle.
# --print is a dry run that does not touch the cap at all.
cap_checks() {
checks=$(cap_ssh bash -s -- "$CAP_ROOT" "$CAP_HOSTNAME" <<'EOF'
root=$1; expected=$2
echo "host=$(hostname)"
echo "temp_mc=$(cat /sys/class/thermal/thermal_zone0/temp)"
pid=$(cat "$root/run/live.pid" 2>/dev/null || true)
if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null; then echo "running=$pid"; else echo "running="; fi
echo "binary=$([[ -x $root/bin/robocap-live ]] && echo yes || echo no)"
others=
for cmdline in /proc/[0-9]*/cmdline; do
    cmd=$(tr '\0' ' ' < "$cmdline" 2>/dev/null) || continue
    # Any program under /root/robocap-live (not only this root's): one heavy job at a time on the cap.
    [[ $cmd == /root/robocap-live/* && $cmd != *cap-sampler* ]] && others="$others ${cmdline//[!0-9]/}"
done
echo "others=$others"
echo "recorder=$(pgrep -x omni-specs.bin | tr '\n' ' ')"
echo "load=$(cut -d' ' -f1-3 /proc/loadavg)"
EOF
) || { log "Cap $CAP does not answer"; exit 1; }
get() { sed -n "s/^$1=//p" <<<"$checks"; }
log "Cap $CAP: host $(get host), SoC $(( $(get temp_mc) / 1000 )) °C, load $(get load), vendor recorder pid(s) '$(get recorder)'"
[[ $(get host) == "$CAP_HOSTNAME" ]] || { log "expected $CAP_HOSTNAME; refusing"; exit 1; }
(( $(get temp_mc) < 70000 )) || { log "the SoC is above 70 °C; let it cool first"; exit 1; }
[[ -z $(get running) ]] || { log "robocap-live is already running (session $(get running)); stop it with scripts/stop.sh --cap $CAP"; exit 1; }
if [[ -n $(get others) && $force != 1 ]]; then
    log "other /root/robocap-live programs are running (pids$(get others)): one heavy job at a time; --force to start anyway"
    [[ $print_only == 1 ]] || exit 1
fi
if [[ ${#custom[@]} == 0 ]]; then [[ $(get binary) == yes ]] || { log "no $CAP_ROOT/bin/robocap-live; run scripts/deploy.sh --cap $CAP"; exit 1; }; fi
}
if [[ $print_only == 1 ]]; then
    log "--print: dry run; Cap $CAP is not contacted (no checks)"
else
    cap_checks
fi

# 2. The viewer: the URL given, or the viewer on the viewer Mac and the URL the cap reaches it at.
if [[ -z $viewer ]]; then
    viewer_mac=(--cap "$CAP" --viewer-host "$viewer_host" --viewer-if "$viewer_if" --cap-address "$cap_address" --port "$port")
    if [[ $start_viewer == 1 && $print_only == 0 ]]; then
        "$here/viewer-mac.sh" start "${viewer_mac[@]}" "${headless[@]}"
    fi
    viewer=$("$here/viewer-mac.sh" url "${viewer_mac[@]}")
fi
log "viewer URL: $viewer"

# 3. The command.
save_path=$CAP_ROOT/recordings/$stamp.rrd
if [[ ${#custom[@]} -gt 0 ]]; then
    command=()
    for word in "${custom[@]}"; do word=${word//\{viewer\}/$viewer}; word=${word//\{save\}/$save_path}; command+=("${word//\{seconds\}/$seconds}"); done
else
    command=("$CAP_ROOT/bin/robocap-live")
    if [[ -n $replay ]]; then command+=(--source replay "$replay" --realtime --loop --preload); else command+=(--source live --rig "$CAP_ROOT/rig.json"); fi
    command+=(--nets rknn "$CAP_ROOT/models" --hands on --viewer "$viewer" --video h264 --video-cameras "$video_cameras"
        --display "$CAP_ROOT/assets/robocap-live-display.rrd" --duration "$seconds" --slam-lane "$slam_lane")
    [[ $slam_lag == auto ]] || command+=(--slam-set "port.frontend_lag=$slam_lag")
    if [[ $uclamp == 1 ]]; then command+=(--slam-uclamp 1024 --hands-uclamp 1024); else command+=(--slam-uclamp none --hands-uclamp none); fi
    [[ $save == 1 ]] && command+=(--save "$save_path")
fi
# Only a run that takes the cameras needs the handoff; a replay (or a test program) just gets a time limit.
if [[ " ${command[*]} " == *" --source live "* ]]; then
    launch=("$CAP_ROOT/scripts/handoff-run.sh" "$((seconds + 30))" "${command[@]}")
else
    launch=(timeout -k 5 "$((seconds + 30))" "${command[@]}")
fi
quoted=$(printf '%q ' "${launch[@]}")
log "command on Cap $CAP: $quoted"

# 4. The cap script: start robocap-live detached on the cap (its own session outlives the ssh connection), with the log and the
# session id on the cap, then wait for it to end, so the guard's samples cover the whole run.
cap_script="set -euo pipefail
root=$CAP_ROOT; stamp=$stamp
mkdir -p \"\$root\"/{logs,run,recordings}
logfile=\$root/logs/live-\$stamp.log
cd \"\$root\"
setsid nohup bash -c $(printf '%q' "exec $quoted") > \"\$logfile\" 2>&1 < /dev/null &
echo \$! > run/live.pid
echo \"\$logfile\" > run/live.log
echo $(printf '%q' "$quoted") > run/live.cmd
sleep 5
if kill -0 \"\$(cat run/live.pid)\" 2>/dev/null; then echo \"running: session \$(cat run/live.pid), log \$logfile\"; else echo 'it exited at once; log:'; tail -15 \"\$logfile\"; exit 1; fi
while kill -0 \"\$(cat run/live.pid)\" 2>/dev/null; do sleep 2; done
echo 'ended; log tail:'; tail -25 \"\$logfile\""
guard_command=("$guard" "${guard_flags[@]}" --max-run-s "$((seconds + 120))" "$CAP" -- "echo $(printf %s "$cap_script" | base64 -w0) | base64 -d | bash")
local_log=$log_dir/start-live-$CAP-$stamp.log
if [[ $print_only == 1 ]]; then
    log "--print: would run on this host, detached, logging to $local_log:"
    log "  $(printf '%q ' "${guard_command[@]:0:${#guard_command[@]}-1}")'echo <base64 of the cap script> | base64 -d | bash'"
    log "--print: the cap script:"
    printf '%s\n' "$cap_script" | sed 's/^/    /' >&2
    log "--print: nothing started"
    exit 0
fi

mkdir -p "$(dirname "$local_log")"
setsid nohup "${guard_command[@]}" > "$local_log" 2>&1 < /dev/null &
guard_pid=$!
sleep 15
if kill -0 "$guard_pid" 2>/dev/null; then log "guard running (pid $guard_pid on this host); its log: $local_log"; else log "the guard ended already:"; fi
tail -20 "$local_log" >&2
root_flag=
[[ $CAP_ROOT == /root/robocap-live ]] || root_flag=" --root $CAP_ROOT"
log "stop it with: $here/stop.sh --cap $CAP$root_flag   (watch: tail -f $local_log; the guard prints its log dir, with klog/samples/alarms, there)"
