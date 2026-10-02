#!/bin/bash
# Run a command with the cameras, frame trigger, IIO and MPP taken over from the IDLE vendor recorder, then give them back.
# Runs ON a cap (Cap A or Cap B). The PR #270 recipe exactly (https://github.com/rerun-io/examples-monorepo/pull/270):
#   1. refuse unless this is a known cap (Cap A or Cap B), the vendor launcher loop and recorder exist, the recorder reports is_recording false, and the SoC is cool;
#   2. pause the launcher loop (SIGSTOP), kill the idle recorder (SIGKILL; SIGTERM hangs it);
#   3. run the command under a timeout, with a temperature watchdog that stops it above the limit;
#   4. ALWAYS resume the launcher loop (SIGCONT, in an EXIT trap) once the run is gone; it restarts the vendor recorder after ~5 s;
#   5. check the vendor recorder is back and idle, IIO buffers are off, the kernel is untainted; print dmesg's tail.
# Never: SIGTERM the recorder, unbind/rebind drivers, write MAG sampling frequency, reboot, touch /frodobots_disk or init scripts.
# Usage: handoff-run.sh <max seconds> <command...>
set -euo pipefail
max_seconds=$1; shift
temp_limit_mc=85000   # the robocap-panel STOP_TEMP_C, in millidegrees
log() { echo "[handoff $(date +%H:%M:%S)] $*" >&2; }
temp() { cat /sys/class/thermal/thermal_zone0/temp; }

case $(hostname) in
    robocap_f403b0) device=f403b0 ;;   # Cap A
    robocap_fe62fa) device=fe62fa ;;   # Cap B
    *) log "unknown host $(hostname); refusing"; exit 2 ;;
esac
launcher=$(pgrep -x frodobots.sh || true); recorder=$(pgrep -x omni-specs.bin || true)
[[ $launcher =~ ^[0-9]+$ && $recorder =~ ^[0-9]+$ ]] || { log "launcher '$launcher' / recorder '$recorder' not found exactly once; refusing"; exit 2; }
props=$(mosquitto_sub -h 127.0.0.1 -t robocap/$device/prop/all -C 1 -W 15 || true)
grep -Eq '"is_recording"[[:space:]]*:[[:space:]]*false' <<<"$props" || { log "vendor recorder not reporting is_recording=false; refusing"; exit 2; }
grep -Eq '"is_recording_key_on"[[:space:]]*:[[:space:]]*false' <<<"$props" || { log "recording key is on; refusing"; exit 2; }
(( $(temp) < temp_limit_mc - 10000 )) || { log "SoC $(temp) m°C is too warm to start; refusing"; exit 2; }
tainted_before=$(cat /proc/sys/kernel/tainted)

# The run is `timeout` and what it runs. The caps' timeout (the coreutils kind, the command's parent) runs the command in its own
# process group (pgid = its pid) and at its -k deadline SIGKILLs that group, itself included, so it can be gone a moment before the
# command is. BusyBox's timeout execs the command in place: there the pid is the command's and there is no such group.
child=
stop_sent=0
run_alive() { kill -0 "$child" 2>/dev/null || kill -0 -- "-$child" 2>/dev/null; }
# The one stop: SIGINT to timeout, once (to what is left of its group if timeout is gone). Coreutils' timeout passes it on to the
# command and starts its -k deadline (SIGKILL 5 s later); with BusyBox's the SIGINT reaches the command itself and -k only follows
# the time limit. Nothing here kills.
stop_run() { (( stop_sent )) || { stop_sent=1; kill -INT "$child" 2>/dev/null || kill -INT -- "-$child" 2>/dev/null || true; }; }

restore() {
    local status=$?
    # The vendor gets the devices back only once the run is gone; a run that this script leaves early (a signal, an error) is stopped.
    if [[ -n $child ]] && run_alive; then
        (( stop_sent )) || { log "the run is still going; stopping it"; stop_run; }
        log "waiting for the run to end before resuming the launcher"
        while run_alive; do sleep 0.5; done
    fi
    kill -CONT "$launcher" 2>/dev/null || true
    log "launcher resumed (exit status of run: $status); waiting for the vendor recorder"
    for _ in $(seq 1 30); do
        sleep 1
        if pgrep -x omni-specs.bin | grep -qv "^$recorder$"; then break; fi
    done
    sleep 3
    local now; now=$(mosquitto_sub -h 127.0.0.1 -t robocap/$device/prop/all -C 1 -W 15 || true)
    if grep -Eq '"is_recording"[[:space:]]*:[[:space:]]*false' <<<"$now"; then log "vendor recorder back and idle"; else log "WARNING: vendor recorder state unknown"; fi
    for enable in /sys/bus/iio/devices/iio:device*/buffer/enable; do
        [[ $(cat "$enable" 2>/dev/null || echo 0) == 0 ]] || log "WARNING: $enable still 1"
    done
    [[ $(cat /proc/sys/kernel/tainted) == "$tainted_before" ]] || log "WARNING: kernel taint changed $tainted_before -> $(cat /proc/sys/kernel/tainted)"
    log "SoC $(temp) m°C; dmesg tail:"; dmesg 2>/dev/null | tail -5 >&2 || true
}
trap restore EXIT

log "pausing launcher $launcher, killing idle recorder $recorder"
kill -STOP "$launcher"
kill -KILL "$recorder"
sleep 1

timeout -k 5 "$max_seconds" "$@" &
child=$!
while kill -0 "$child" 2>/dev/null; do
    if (( ! stop_sent && $(temp) >= temp_limit_mc )); then log "SoC $(temp) m°C >= limit; stopping the run"; stop_run; fi
    sleep 2
done
wait "$child"
