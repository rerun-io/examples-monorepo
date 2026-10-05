//! The run supervisor, `robocap-panel handoff <max seconds> <command...>`: it takes the cameras, frame trigger, IIO and MPP over
//! from the IDLE vendor recorder, runs the command, and always gives them back. The panel starts it as the leader of the run's own
//! session, so a run outlives a panel restart. The PR #270 recipe (https://github.com/rerun-io/examples-monorepo/pull/270):
//! 1. refuse unless the host is a cap (`robocap_<device>`), the vendor launcher loop and recorder each run once, the recorder
//!    reports `is_recording` and `is_recording_key_on` false over MQTT, and the SoC is below [`MAX_START_TEMP_C`];
//! 2. pause the launcher loop (SIGSTOP) and kill the idle recorder (SIGKILL; SIGTERM hangs it);
//! 3. run the command in its own process group; stop it (SIGINT, then SIGKILL to its group after a grace period) at the time
//!    limit, at [`STOP_TEMP_C`], or when the supervisor itself gets SIGINT, SIGTERM or SIGHUP (the panel's Stop is a SIGTERM);
//! 4. once all other processes in the session are gone, resume the launcher loop (SIGCONT); it restarts the vendor recorder after ~5 s;
//! 5. check that the vendor recorder is back and idle, the IIO buffers are off and the kernel taint is unchanged.
//!
//! Never: SIGTERM the recorder, unbind or rebind drivers, write the MAG sampling frequency, reboot, touch /frodobots_disk or the
//! init scripts. Every log line starts with `[handoff`; the panel reads [`RUN_ENDED`] as the end of the run.

use std::fs;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::Value;

use crate::{hottest_c, processes, read_trim, stdout_of};

/// The supervisor stops a run at this temperature...
pub const STOP_TEMP_C: f64 = 85.0;
/// ...and starts one only below this.
pub const MAX_START_TEMP_C: f64 = STOP_TEMP_C - 10.0;
/// The vendor's launcher loop and recorder, by process name.
pub const LAUNCHER: &str = "frodobots.sh";
pub const RECORDER: &str = "omni-specs.bin";
/// The supervisor's log says this once the run is gone and it gives the cameras back.
pub const RUN_ENDED: &str = "resuming the vendor launcher";
/// From SIGINT to SIGKILL when the supervisor stops a run.
const GRACE: Duration = Duration::from_secs(20);

/// What the supervisor takes from the vendor and gives back.
pub struct Held {
    /// The cap's device id, from its hostname (`robocap_<device>`): the vendor's MQTT topics carry it.
    pub device: String,
    pub launcher: u32,
    pub recorder: u32,
    /// /proc/sys/kernel/tainted before the run.
    pub tainted: String,
}

/// The cap's vendor side, as the supervisor uses it.
pub trait Vendor {
    /// The launcher loop and the recorder, if the cap may be taken now; else the refusal.
    fn check_idle(&self) -> Result<Held, String>;
    /// Pause the launcher loop and kill the idle recorder.
    fn take(&self, held: &Held) -> Result<(), String>;
    /// Resume the launcher loop (it restarts the recorder) and report whether the vendor has everything back, as log lines; the
    /// last one sums it up (the panel shows it).
    fn give_back(&self, held: &Held) -> Vec<String>;
    fn hottest_c(&self) -> f64;
}

pub struct Limits {
    /// The time limit of the run.
    pub max_run: Duration,
    /// From SIGINT to SIGKILL.
    pub grace: Duration,
    pub poll: Duration,
}

/// `[handoff HH:MM:SS]` (UTC) lines on stderr, which is the run log. A failed write is dropped (`eprintln!` would panic): a full
/// log disk must not end the supervisor before it gives the cameras back.
fn log(text: &str) {
    let seconds = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) % 86_400;
    let _ = writeln!(std::io::stderr(), "[handoff {:02}:{:02}:{:02}] {text}", seconds / 3600, seconds / 60 % 60, seconds % 60);
}

/// kill(2): `pid` < 0 signals that process group.
pub fn signal(pid: i32, signal: libc::c_int) -> Result<(), String> {
    // SAFETY: kill(2) takes plain integers and touches no memory of ours.
    if unsafe { libc::kill(pid, signal) } == 0 { Ok(()) }
    else { Err(format!("signal {signal} to pid {pid}: {}", std::io::Error::last_os_error())) }
}

/// One lease for the capture devices. Dropping the file releases the lock.
pub fn lock(root: &Path) -> Result<fs::File, String> {
    fs::create_dir_all(root.join("run")).map_err(|e| e.to_string())?;
    let file = fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("run/device.lock")).map_err(|e| e.to_string())?;
    // SAFETY: the descriptor belongs to file and stays open until give_back finishes.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(format!("capture devices are held by another supervisor: {}", std::io::Error::last_os_error()));
    }
    Ok(file)
}

/// Take the cap from the vendor, run `command`, give the cap back once the run is gone. The run's exit status, or why nothing
/// ran (then nothing was taken, except when the command itself could not start).
pub fn supervise(vendor: &impl Vendor, command: &[String], limits: &Limits, stop: &AtomicBool) -> Result<ExitStatus, String> {
    let (program, args) = command.split_first().ok_or("no command")?;
    let held = vendor.check_idle()?;
    let hottest = vendor.hottest_c();
    if hottest >= MAX_START_TEMP_C {
        return Err(format!("SoC {hottest:.1} °C: a run starts below {MAX_START_TEMP_C} °C"));
    }
    if stop.load(Ordering::SeqCst) {
        return Err("asked to stop before the run started".into());
    }
    log(&format!("pausing launcher {}, killing idle recorder {}", held.launcher, held.recorder));
    vendor.take(&held)?;
    let mut command = Command::new(program);
    // Tests run several supervisors in one process; isolate each synthetic run's session.
    #[cfg(test)]
    unsafe { command.pre_exec(|| {
        if libc::setsid() < 0 { return Err(std::io::Error::last_os_error()); }
        Ok(())
    }); }
    #[cfg(not(test))]
    command.process_group(0);
    let ended = command
        .args(args)
        .stdin(Stdio::null())
        .spawn()
        .map_err(|error| format!("start {program}: {error}"))
        .and_then(|child| watch(child, vendor, limits, stop));
    let run = ended.as_ref().map_or_else(Clone::clone, ToString::to_string);
    log(&format!("run: {run}; {RUN_ENDED} and waiting for the vendor recorder"));
    for line in vendor.give_back(&held) {
        log(&line);
    }
    ended
}

/// Wait for the run to end: robocap-live and every other process left in its session (its encoders can outlive it for a moment). At the
/// time limit, at [`STOP_TEMP_C`] or on `stop`: SIGINT to robocap-live. After that SIGINT, or once robocap-live has ended, the
/// group gets the grace period, then SIGKILL. The run's exit status is robocap-live's.
fn watch(mut child: Child, vendor: &impl Vendor, limits: &Limits, stop: &AtomicBool) -> Result<ExitStatus, String> {
    let pid = i32::try_from(child.id()).map_err(|e| e.to_string())?;
    let session = crate::stat(child.id()).ok_or("cannot read run session")?.session;
    let supervisor = std::process::id();
    let started = Instant::now();
    let (mut status, mut stopping, mut killed) = (None, None::<Instant>, false);
    loop {
        if status.is_none() {
            status = child.try_wait().map_err(|e| format!("waiting for the run: {e}"))?;
        }
        // Encoders have their own process groups but remain in the supervisor's session.
        let remaining: Vec<u32> = fs::read_dir("/proc").map_err(|e| e.to_string())?.flatten()
            .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
            .filter(|&pid| pid != supervisor && crate::stat(pid).is_some_and(|s| s.session == session && s.state != 'Z'))
            .collect();
        if let Some(done) = status
            && remaining.is_empty()
        {
            return Ok(done);
        }
        match stopping {
            Some(at) if !killed && at.elapsed() >= limits.grace => {
                log(&format!("the run's session still has processes {} s on: SIGKILL to them", limits.grace.as_secs()));
                for pid in remaining {
                    if let Err(error) = signal(pid as i32, libc::SIGKILL) { log(&error); }
                }
                killed = true;
            }
            Some(_) => {}
            None if status.is_some() => stopping = Some(Instant::now()),
            None => {
                let hottest = vendor.hottest_c();
                let reason = if stop.load(Ordering::SeqCst) {
                    Some("the supervisor was asked to stop".to_string())
                } else if started.elapsed() >= limits.max_run {
                    Some(format!("time limit {} s", limits.max_run.as_secs()))
                } else {
                    (hottest >= STOP_TEMP_C).then(|| format!("SoC {hottest:.1} °C >= {STOP_TEMP_C} °C"))
                };
                if let Some(reason) = reason {
                    log(&format!("{reason}: stopping the run (SIGINT)"));
                    if let Err(error) = signal(pid, libc::SIGINT) { log(&error); }
                    stopping = Some(Instant::now());
                }
            }
        }
        std::thread::sleep(limits.poll);
    }
}

/// The vendor recorder's state (its MQTT `prop/all` message) says it may be taken: not recording, recording key off.
fn recorder_idle(props: &str) -> Result<(), String> {
    let props: Value = serde_json::from_str(props).map_err(|_| "no vendor recorder state over MQTT".to_string())?;
    if props.get("is_recording") != Some(&Value::Bool(false)) {
        return Err("the vendor recorder does not report is_recording=false".into());
    }
    if props.get("is_recording_key_on") != Some(&Value::Bool(false)) {
        return Err("the recording key is on".into());
    }
    Ok(())
}

/// The cap itself: /proc, sysfs, and the vendor's MQTT broker on 127.0.0.1.
pub struct CapVendor;

impl CapVendor {
    /// The one process called `name`.
    fn only(name: &str) -> Result<u32, String> {
        let pids: Vec<u32> = processes().into_iter().filter(|p| p.name == name).map(|p| p.pid).collect();
        match pids.as_slice() {
            [pid] => Ok(*pid),
            _ => Err(format!("{name}: {} processes, expected one", pids.len())),
        }
    }

    /// One `prop/all` message from the vendor's broker (15 s at most); empty if none came.
    fn props(device: &str) -> String {
        stdout_of("mosquitto_sub", &["-h", "127.0.0.1", "-t", &format!("robocap/{device}/prop/all"), "-C", "1", "-W", "15"])
    }
}

const TAINTED: &str = "/proc/sys/kernel/tainted";

impl Vendor for CapVendor {
    fn check_idle(&self) -> Result<Held, String> {
        let host = read_trim("/proc/sys/kernel/hostname").unwrap_or_default();
        let device = host.strip_prefix("robocap_").ok_or_else(|| format!("host {host:?} is not a RoboCap (robocap_<device>)"))?.to_string();
        recorder_idle(&Self::props(&device))?;
        let (launcher, recorder) = (Self::only(LAUNCHER)?, Self::only(RECORDER)?);
        Ok(Held { device, launcher, recorder, tainted: read_trim(TAINTED).unwrap_or_default() })
    }

    fn take(&self, held: &Held) -> Result<(), String> {
        if Self::only(LAUNCHER)? != held.launcher || Self::only(RECORDER)? != held.recorder {
            return Err("vendor processes changed after the idle check".into());
        }
        if crate::stat(held.launcher).is_none_or(|s| s.state == 'T') {
            return Err("vendor launcher is absent or already stopped".into());
        }
        signal(held.launcher as i32, libc::SIGSTOP)?;
        let taken = (|| {
            let stopped = (0..100).any(|_| {
                if crate::stat(held.launcher).is_some_and(|s| s.state == 'T') { return true; }
                std::thread::sleep(Duration::from_millis(10));
                false
            });
            if !stopped { return Err("vendor launcher did not stop".into()); }
            signal(held.recorder as i32, libc::SIGKILL)?;
            let gone = (0..100).any(|_| {
                if !crate::run::alive(held.recorder) { return true; }
                std::thread::sleep(Duration::from_millis(10));
                false
            });
            if !gone { return Err("vendor recorder did not exit".into()); }
            Ok(())
        })();
        if taken.is_err() && crate::stat(held.launcher).is_some_and(|s| s.name == LAUNCHER) {
            signal(held.launcher as i32, libc::SIGCONT)?;
        }
        taken
    }

    fn give_back(&self, held: &Held) -> Vec<String> {
        if let Err(error) = signal(held.launcher as i32, libc::SIGCONT) { return vec![error]; }
        let dmesg = stdout_of("dmesg", &[]);
        let tail: Vec<&str> = dmesg.lines().rev().take(5).collect();
        let mut lines = vec!["dmesg tail:".to_string()];
        lines.extend(tail.into_iter().rev().map(|line| format!("  {line}")));
        let restarted = (0..30).any(|_| {
            std::thread::sleep(Duration::from_secs(1));
            processes().iter().any(|p| p.name == RECORDER && p.pid != held.recorder)
        });
        if !restarted {
            lines.push("WARNING: no new vendor recorder after 30 s".to_string());
        }
        std::thread::sleep(Duration::from_secs(3));
        for entry in fs::read_dir("/sys/bus/iio/devices").into_iter().flatten().flatten() {
            let enable = entry.path().join("buffer/enable");
            if read_trim(&enable).is_some_and(|value| value != "0") {
                lines.push(format!("WARNING: {} is still on", enable.display()));
            }
        }
        let tainted = read_trim(TAINTED).unwrap_or_default();
        if tainted != held.tainted {
            lines.push(format!("WARNING: kernel taint changed {} -> {tainted}", held.tainted));
        }
        lines.push(match recorder_idle(&Self::props(&held.device)) {
            Ok(()) => format!("vendor recorder back and idle; SoC {:.1} °C", self.hottest_c()),
            Err(error) => format!("WARNING: vendor recorder state unknown ({error}); SoC {:.1} °C", self.hottest_c()),
        });
        lines
    }

    fn hottest_c(&self) -> f64 {
        hottest_c()
    }
}

/// Set by SIGINT, SIGTERM and SIGHUP: stop the run, then give the cap back.
static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

/// `robocap-panel handoff <max seconds> <command...>`: the exit code is the run's (128 + its signal if a signal ended it), or 2
/// when nothing ran.
pub fn main(args: &[String]) -> i32 {
    let Some((max_seconds, command)) = args.split_first().and_then(|(max, command)| Some((max.parse::<u64>().ok()?, command))) else {
        eprintln!("usage: robocap-panel handoff <max seconds> <command...>");
        return 2;
    };
    for signum in [libc::SIGINT, libc::SIGTERM, libc::SIGHUP] {
        // SAFETY: the handler only stores to an atomic, which is async-signal-safe.
        unsafe { libc::signal(signum, on_signal as extern "C" fn(libc::c_int) as libc::sighandler_t) };
    }
    let limits = Limits { max_run: Duration::from_secs(max_seconds), grace: GRACE, poll: Duration::from_millis(500) };
    let root = match std::env::current_dir() { Ok(root) => root, Err(error) => { log(&error.to_string()); return 2; } };
    let _lease = match lock(&root) { Ok(lease) => lease, Err(error) => { log(&error); return 2; } };
    match supervise(&CapVendor, command, &limits, &STOP) {
        Ok(status) => status.code().unwrap_or_else(|| 128 + status.signal().unwrap_or(0)),
        Err(error) => {
            log(&error);
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::path::PathBuf;
    use std::sync::Mutex;

    use super::*;

    /// The cap's vendor side, faked: it notes what the supervisor does to it, and whether any process of the run was still alive
    /// when the launcher got resumed (the run writes its pids to `run_pid_file`, one per line).
    struct FakeVendor {
        idle: Result<(), String>,
        hottest_c: Mutex<f64>,
        events: Mutex<Vec<String>>,
        run_pid_file: PathBuf,
    }

    impl FakeVendor {
        fn new(name: &str) -> Self {
            let run_pid_file = std::env::temp_dir().join(format!("robocap-handoff-{}-{name}.pid", std::process::id()));
            let _ = std::fs::remove_file(&run_pid_file);
            Self { idle: Ok(()), hottest_c: Mutex::new(45.0), events: Mutex::default(), run_pid_file }
        }

        /// A run that notes its pid, then does `then` (shell).
        fn run(&self, then: &str) -> Vec<String> {
            vec!["sh".into(), "-c".into(), format!("echo $$ > {}; {then}", self.run_pid_file.display())]
        }

        fn set_hottest_c(&self, celsius: f64) {
            if let Ok(mut hottest) = self.hottest_c.lock() {
                *hottest = celsius;
            }
        }

        fn events(&self) -> Vec<String> {
            self.events.lock().map(|e| e.clone()).unwrap_or_default()
        }
    }

    impl Vendor for FakeVendor {
        fn check_idle(&self) -> Result<Held, String> {
            self.idle.clone().map(|()| Held { device: "fe62fa".into(), launcher: 11, recorder: 22, tainted: "0".into() })
        }

        fn take(&self, held: &Held) -> Result<(), String> {
            self.events.lock().map(|mut e| e.push(format!("take {} {}", held.launcher, held.recorder))).ok();
            Ok(())
        }

        fn give_back(&self, held: &Held) -> Vec<String> {
            let pids = std::fs::read_to_string(&self.run_pid_file).unwrap_or_default();
            let run_alive = pids.lines().filter_map(|line| line.trim().parse().ok()).any(crate::run::alive);
            self.events.lock().map(|mut e| e.push(format!("resume {} (run alive: {run_alive})", held.launcher))).ok();
            Vec::new()
        }

        fn hottest_c(&self) -> f64 {
            self.hottest_c.lock().map_or(0.0, |t| *t)
        }
    }

    fn quick() -> Limits {
        Limits { max_run: Duration::from_secs(20), grace: Duration::from_millis(500), poll: Duration::from_millis(20) }
    }

    #[test]
    fn the_device_lock_refuses_a_second_owner_until_release() -> Result<(), String> {
        let root = std::env::temp_dir().join(format!("robocap-lock-{}", std::process::id()));
        let first = lock(&root)?;
        assert!(lock(&root).is_err());
        drop(first);
        let second = lock(&root)?;
        drop(second);
        fs::remove_dir_all(root).map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn a_recording_cap_is_refused_and_nothing_is_taken() {
        let vendor = FakeVendor { idle: Err("vendor recorder not reporting is_recording=false".into()), ..FakeVendor::new("recording") };
        let result = supervise(&vendor, &vendor.run("exit 0"), &quick(), &AtomicBool::new(false));
        assert!(result.is_err_and(|e| e.contains("is_recording")));
        assert!(vendor.events().is_empty());
        assert!(!vendor.run_pid_file.exists(), "the run never started");
    }

    #[test]
    fn a_warm_cap_is_refused_and_nothing_is_taken() {
        let vendor = FakeVendor::new("warm");
        vendor.set_hottest_c(MAX_START_TEMP_C);
        assert!(supervise(&vendor, &vendor.run("exit 0"), &quick(), &AtomicBool::new(false)).is_err());
        assert!(vendor.events().is_empty());
    }

    #[test]
    fn the_launcher_is_resumed_once_the_run_has_ended() -> Result<(), String> {
        let vendor = FakeVendor::new("normal");
        let status = supervise(&vendor, &vendor.run("sleep 0.3; exit 3"), &quick(), &AtomicBool::new(false))?;
        assert_eq!(status.code(), Some(3));
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn a_hot_soc_stops_the_run_then_the_launcher_is_resumed() -> Result<(), String> {
        let vendor = FakeVendor::new("hot");
        let status = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                vendor.set_hottest_c(STOP_TEMP_C);
            });
            supervise(&vendor, &vendor.run("exec sleep 30"), &quick(), &AtomicBool::new(false))
        })?;
        assert_eq!(status.signal(), Some(libc::SIGINT));
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn a_run_that_ignores_sigint_at_its_time_limit_is_killed_with_its_group() -> Result<(), String> {
        let vendor = FakeVendor::new("deaf");
        let limits = Limits { max_run: Duration::from_millis(300), ..quick() };
        let started = Instant::now();
        // The shell and its child both ignore SIGINT; only the group's SIGKILL ends them.
        let pids = vendor.run_pid_file.display().to_string();
        let run = vendor.run(&format!("trap '' INT; sleep 30 & echo $! >> {pids}; wait"));
        let status = supervise(&vendor, &run, &limits, &AtomicBool::new(false))?;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        assert!(started.elapsed() >= limits.max_run + limits.grace, "SIGKILL only after the grace period");
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn a_signal_to_the_supervisor_stops_the_run_then_the_launcher_is_resumed() -> Result<(), String> {
        let vendor = FakeVendor::new("signalled");
        let stop = AtomicBool::new(false);
        let status = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(300));
                stop.store(true, Ordering::SeqCst);
            });
            supervise(&vendor, &vendor.run("exec sleep 30"), &quick(), &stop)
        })?;
        assert_eq!(status.signal(), Some(libc::SIGINT));
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn a_process_the_run_leaves_behind_holds_the_cap_until_its_group_is_gone() -> Result<(), String> {
        let vendor = FakeVendor::new("leftover");
        // The run ends at once, but leaves a child in its process group (robocap-live's encoders can outlive it the same way).
        let pids = vendor.run_pid_file.display().to_string();
        let started = Instant::now();
        let status = supervise(&vendor, &vendor.run(&format!("sleep 30 & echo $! >> {pids}; exit 0")), &quick(), &AtomicBool::new(false))?;
        assert_eq!(status.code(), Some(0), "the run's own exit status");
        assert!(started.elapsed() >= quick().grace, "the leftover got the grace period, then SIGKILL");
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn an_encoder_in_a_separate_group_holds_the_cap_until_it_is_gone() -> Result<(), String> {
        let vendor = FakeVendor::new("encoder-group");
        let pids = vendor.run_pid_file.display().to_string();
        // Python is only a test child: setpgid models the encoder's process_group(0).
        let run = vendor.run(&format!("python3 -c 'import os,time; os.setpgid(0,0); open(\"{pids}\",\"a\").write(str(os.getpid())+\"\\n\"); time.sleep(2)' & sleep 0.1; exit 0"));
        supervise(&vendor, &run, &quick(), &AtomicBool::new(false))?;
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
        Ok(())
    }

    #[test]
    fn a_command_that_cannot_start_still_gives_the_cap_back() {
        let vendor = FakeVendor::new("missing");
        let result = supervise(&vendor, &["/nonexistent/robocap-live".to_string()], &quick(), &AtomicBool::new(false));
        assert!(result.is_err_and(|e| e.contains("/nonexistent/robocap-live")));
        assert_eq!(vendor.events(), ["take 11 22", "resume 11 (run alive: false)"]);
    }

    #[test]
    fn the_recorder_reports_idle_only_when_not_recording_and_the_key_is_off() {
        let props = |recording: bool, key: bool| format!(r#"{{"device_id":"fe62fa","is_recording":{recording},"is_recording_key_on":{key}}}"#);
        assert!(recorder_idle(&props(false, false)).is_ok());
        assert!(recorder_idle(&props(true, false)).is_err_and(|e| e.contains("is_recording")));
        assert!(recorder_idle(&props(false, true)).is_err_and(|e| e.contains("key")));
        assert!(recorder_idle("").is_err(), "no answer from the broker");
    }
}
