//! The run this panel starts and stops. The start form ([`StartRequest`]) builds the command line; one record ([`RunRecord`], under
//! [`Run`]'s lock) owns the run's lifecycle: a start reserves it before its checks and, holding it, spawns the run supervisor
//! (`robocap-panel handoff`, see [`crate::handoff`]) and publishes its session and the run files; a stop reads the session from
//! it; only that session's end frees it.

use std::fs;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::handoff::signal;
use crate::{Process, read_trim};

/// What the run is doing, for the page's state and buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    /// The panel runs its checks, or the supervisor checks and pauses the vendor recorder; robocap-live is not running yet.
    Starting,
    Streaming,
    /// Stop was asked for, or the run ended: robocap-live drains, then the supervisor restores the vendor recorder (~15 s).
    Stopping,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Starting => "starting",
            Self::Streaming => "streaming",
            Self::Stopping => "stopping",
        }
    }
}

/// The phase from whether the run's session is alive, whether its robocap-live runs, and whether it is ending (stop asked for,
/// or the log shows the run ended).
pub fn phase(session_alive: bool, live_running: bool, ending: bool) -> Phase {
    match (session_alive, live_running, ending) {
        (false, _, _) => Phase::Idle,
        (true, _, true) => Phase::Stopping,
        (true, true, false) => Phase::Streaming,
        (true, false, false) => Phase::Starting,
    }
}

/// Who holds the run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Owner {
    /// Not this panel: no run, or one that an earlier panel started; run/live.pid names it.
    #[default]
    Nobody,
    /// A start, from before its checks until the supervisor is spawned.
    Starting { cancel: bool },
    /// The supervisor this panel spawned: its pid, which is its session's id.
    Session { pid: u32, stopping: bool },
}

/// The run's lifecycle: one value under [`Run`]'s lock.
#[derive(Clone, Debug, Default)]
pub struct RunRecord {
    pub owner: Owner,
    /// The exit status of the last run this panel started, once it ended.
    pub last_exit: Option<String>,
    /// The form of the last run this panel started ([`StartRequest::to_json`]); a new browser fills its form from it.
    pub last_request: Option<Value>,
}

impl RunRecord {
    pub fn stopping(&self) -> bool {
        matches!(self.owner, Owner::Starting { cancel: true } | Owner::Session { stopping: true, .. })
    }

    /// The run's session: this panel's (also before run/live.pid is written), else the one in run/live.pid; `None` while a start
    /// runs its checks.
    pub fn session(&self, root: &Path) -> Option<u32> {
        match self.owner {
            Owner::Nobody => run_pid(root),
            Owner::Starting { .. } => None,
            Owner::Session { pid, .. } => Some(pid),
        }
    }
}

/// The one owner of this panel's run: [`start`], [`stop`] and the run's end all go through its record.
#[derive(Default)]
pub struct Run {
    record: Mutex<RunRecord>,
}

impl Run {
    /// The record, also after a panic in another request (every update leaves it whole).
    fn record(&self) -> MutexGuard<'_, RunRecord> {
        self.record.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A copy of the record, for `/api/status`.
    pub fn snapshot(&self) -> RunRecord {
        self.record().clone()
    }

    /// The end of `session`: it frees the record and keeps `status` only while `session` is the run the record holds, so a late end
    /// of an older session changes nothing.
    fn finish(&self, session: u32, status: String) {
        let mut record = self.record();
        if matches!(record.owner, Owner::Session { pid, .. } if pid == session) {
            record.owner = Owner::Nobody;
            record.last_exit = Some(status);
        }
    }
}

/// Start's checks on the cap: any refusal stops the start; warnings go with the started run to the page.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Checks {
    pub refusals: Vec<String>,
    pub warnings: Vec<String>,
}

/// The run supervisor's session in run/live.pid (written at every start, so a restarted panel finds the run), while that pid is
/// still a supervisor: the leader of its own session, started as `<program> handoff ...`. After a reboot or a pid reuse the file
/// names some other program, which is no run and must never be signalled.
pub fn run_pid(root: &Path) -> Option<u32> {
    let pid: u32 = read_trim(root.join("run/live.pid"))?.parse().ok()?;
    let session = crate::stat(pid)?.session;
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let second = String::from_utf8_lossy(cmdline.split(|&b| b == 0).nth(1)?).to_string();
    (session == pid && second.rsplit('/').next() == Some("handoff")).then_some(pid)
}

/// Whether `pid` runs (a zombie does not).
pub fn alive(pid: u32) -> bool {
    crate::stat(pid).is_some_and(|stat| stat.state != 'Z')
}

/// The run's robocap-live processes: its session's programs under `bin/`, except the session's leader (the supervisor).
pub fn live_processes(processes: &[Process], root: &Path, session: u32) -> Vec<u32> {
    let bin = root.join("bin/").display().to_string();
    processes.iter().filter(|p| p.session == session && p.pid != session && p.program.starts_with(&bin)).map(|p| p.pid).collect()
}

/// The start form: what the live run streams, and for how long.
#[derive(Debug, PartialEq)]
pub struct StartRequest {
    /// `rerun+http://<host>:<port>/proxy`; empty = no viewer.
    viewer: String,
    /// Cameras with an H.264 pane (0-5); empty = no video.
    video_cameras: Vec<u8>,
    hands: bool,
    /// `--hand-overlays`: fit, debug or verbose (each level costs more of the Wi-Fi link).
    hand_overlays: &'static str,
    slam_hz: u32,
    slam_lane: &'static str,
    slam_lag: &'static str,
    /// uclamp.min 1024 for SLAM and hands (faster, hotter); off = the kernel's default.
    uclamp: bool,
    duration_s: u32,
    /// Opt-in capture metadata CSV beside the summary.
    frame_csv: bool,
}

impl StartRequest {
    /// Parse `application/x-www-form-urlencoded` fields; everything is checked, nothing is passed through a shell.
    pub fn parse(body: &str) -> Result<Self, String> {
        let field = |name: &str| -> Option<String> {
            body.split('&').find_map(|pair| {
                let (key, value) = pair.split_once('=')?;
                (key == name).then(|| decode(value))
            })
        };
        let viewer = field("viewer").unwrap_or_default();
        if !viewer.is_empty() {
            let host_port = viewer.strip_prefix("rerun+http://").and_then(|rest| rest.strip_suffix("/proxy")).ok_or("viewer: expected rerun+http://<host>:<port>/proxy")?;
            let (host, port) = host_port.rsplit_once(':').ok_or("viewer: no port")?;
            let host_ok = !host.is_empty() && host.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
            if !host_ok || port.parse::<u16>().is_err() {
                return Err("viewer: bad host or port".into());
            }
        }
        let cameras_text = field("video_cameras").unwrap_or_default();
        let mut video_cameras: Vec<u8> = Vec::new();
        for item in cameras_text.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let camera: u8 = item.parse().map_err(|_| format!("video_cameras: {item:?} is not 0-5"))?;
            if camera > 5 || video_cameras.contains(&camera) {
                return Err(format!("video_cameras: {item:?} is not a new camera 0-5"));
            }
            video_cameras.push(camera);
        }
        let number = |name: &str, default: u32, range: std::ops::RangeInclusive<u32>| -> Result<u32, String> {
            let value = field(name).filter(|v| !v.is_empty()).map_or(Ok(default), |v| v.parse::<u32>().map_err(|_| format!("{name}: not a number")))?;
            if range.contains(&value) { Ok(value) } else { Err(format!("{name}: {value} is outside {}-{}", range.start(), range.end())) }
        };
        let flag = |name: &str, default: bool| field(name).map_or(default, |v| v == "on" || v == "1" || v == "true");
        let choice = |name: &str, default: &'static str, allowed: &'static [&'static str]| {
            let value = field(name).filter(|v| !v.is_empty());
            let value = value.as_deref().unwrap_or(default);
            allowed.iter().copied().find(|&v| v == value).ok_or_else(|| format!("{name}: {value:?} is not one of {}", allowed.join(", ")))
        };
        Ok(Self {
            viewer,
            video_cameras,
            hands: flag("hands", true),
            hand_overlays: choice("hand_overlays", "fit", &["fit", "debug", "verbose"])?,
            slam_hz: number("slam_hz", 15, 1..=30)?,
            slam_lane: choice("slam_lane", "gpu", &["gpu", "cpu"])?,
            slam_lag: choice("slam_lag", "auto", &["auto", "on", "off"])?,
            uclamp: flag("uclamp", false),
            // 0 (the default) is no time limit: the run goes until Stop or the supervisor's temperature stop.
            duration_s: match number("duration_s", 0, 0..=7200)? {
                1..=9 => return Err("duration_s: 0 (no limit) or 10-7200".into()),
                seconds => seconds,
            },
            frame_csv: flag("frame_csv", false),
        })
    }

    /// The request in the form's field names (the values [`Self::parse`] read).
    fn to_json(&self) -> Value {
        json!({
            "viewer": self.viewer,
            "video_cameras": self.video_cameras,
            "hands": self.hands,
            "hand_overlays": self.hand_overlays,
            "slam_hz": self.slam_hz,
            "slam_lane": self.slam_lane,
            "slam_lag": self.slam_lag,
            "uclamp": self.uclamp,
            "duration_s": self.duration_s,
            "frame_csv": self.frame_csv,
        })
    }

    /// The robocap-live command line for this request (the program first).
    fn command(&self, root: &Path, stamp: u64) -> Vec<String> {
        let at = |rel: &str| root.join(rel).display().to_string();
        let mut command = vec![at("bin/robocap-live"), "--source".into(), "live".into(), "--rig".into(), at("rig.json")];
        command.extend(["--nets".into(), "rknn".into(), at("models"), "--hands".into(), if self.hands { "on" } else { "off" }.into()]);
        command.extend(["--hand-overlays".into(), self.hand_overlays.into()]);
        if !self.viewer.is_empty() {
            command.extend(["--viewer".into(), self.viewer.clone()]);
        }
        if self.video_cameras.is_empty() {
            command.extend(["--video".into(), "off".into()]);
        } else {
            let list: Vec<String> = self.video_cameras.iter().map(u8::to_string).collect();
            command.extend(["--video".into(), "h264".into(), "--video-cameras".into(), list.join(",")]);
        }
        command.extend(["--display".into(), at("assets/robocap-live-display.rrd"), "--slam-hz".into(), self.slam_hz.to_string()]);
        command.extend(["--slam-lane".into(), self.slam_lane.into()]);
        if self.slam_lag != "auto" {
            command.extend(["--slam-set".into(), format!("port.frontend_lag={}", self.slam_lag == "on")]);
        }
        let uclamp = if self.uclamp { "1024" } else { "none" };
        command.extend(["--slam-uclamp".into(), uclamp.into(), "--hands-uclamp".into(), uclamp.into()]);
        if self.duration_s > 0 {
            command.extend(["--duration".into(), self.duration_s.to_string()]);
        }
        command.extend(["--summary-json".into(), at(&format!("logs/rt-{stamp}.json"))]);
        if self.frame_csv { command.extend(["--frame-csv".into(), at(&format!("logs/rt-{stamp}.frames.csv"))]); }
        command
    }
}

/// `%XX` and `+` decoding of one form value.
fn decode(value: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let escaped = (bytes[i] == b'%' && i + 2 < bytes.len()).then(|| Some(hex(bytes[i + 1])? * 16 + hex(bytes[i + 2])?)).flatten();
        match (bytes[i], escaped) {
            (_, Some(byte)) => {
                out.push(byte);
                i += 2;
            }
            (b'+', None) => out.push(b' '),
            (other, None) => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

// Allow robocap-live's duration stop to drain capture and flush its encoders.
const SUPERVISOR_MARGIN_S: u32 = 40;

/// Start a run: reserve the record, run `preflight`, then, holding the record, spawn `supervisor` (the program and its first
/// arguments: `robocap-panel handoff`) with the time limit and the robocap-live command line, and publish its session and the run
/// files. The reply carries the preflight's warnings.
///
/// # Errors
///
/// The text the page shows: a start or run of this panel's holds the record, a refusal, a stop that came during the checks
/// (nothing starts), a failed spawn, or a run file that could not be written (the run is then stopped as [`stop`] stops it, and
/// the supervisor restores the vendor recorder).
pub fn start(
    run: &Arc<Run>,
    root: &Path,
    supervisor: &[String],
    request: &StartRequest,
    preflight: impl FnOnce() -> Checks,
) -> Result<Value, String> {
    {
        let mut record = run.record();
        if record.owner != Owner::Nobody {
            return Err("a run is already starting or going (stop it first)".into());
        }
        record.owner = Owner::Starting { cancel: false };
    }
    let Checks { refusals, warnings } = preflight();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // The supervisor, its time limit (0: none), then robocap-live: what is spawned and what run/live.cmd records.
    let mut argv = supervisor.to_vec();
    argv.push(if request.duration_s == 0 { 0 } else { request.duration_s + SUPERVISOR_MARGIN_S }.to_string());
    argv.extend(request.command(root, stamp));
    let log_path = root.join(format!("logs/rt-{stamp}.log"));
    let mut record = run.record();
    let spawned = if record.stopping() {
        Err("stop was asked for while the run was starting; nothing started".to_string())
    } else if !refusals.is_empty() {
        Err(refusals.join("; "))
    } else {
        spawn_supervisor(root, &argv, &log_path)
    };
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            record.owner = Owner::Nobody;
            return Err(error);
        }
    };
    let pid = child.id();
    let command_line = argv.join(" ");
    record.owner = Owner::Session { pid, stopping: false };
    record.last_exit = None;
    record.last_request = Some(request.to_json());
    let run_files =
        [("run/live.pid", format!("{pid}\n")), ("run/live.log", format!("{}\n", log_path.display())), ("run/live.cmd", format!("{command_line}\n"))];
    let written = run_files.iter().try_for_each(|(name, text)| fs::write(root.join(name), text).map_err(|e| format!("{name}: {e}")));
    drop(record);
    let waiter = run.clone();
    std::thread::spawn(move || {
        let status = child.wait().map(|s| s.to_string()).unwrap_or_else(|e| e.to_string());
        waiter.finish(pid, status);
    });
    if let Err(error) = written {
        let stopping = if stop(run, root).is_ok() { "it is being stopped" } else { "it has ended" };
        return Err(format!("{error}: the run started without its run files; {stopping}"));
    }
    Ok(json!({"pid": pid, "log": log_path.display().to_string(), "cmd": command_line, "warnings": warnings}))
}

/// The supervisor becomes its own session leader before exec, with the same pid;
/// its output goes to `log_path`.
fn spawn_supervisor(root: &Path, argv: &[String], log_path: &Path) -> Result<Child, String> {
    fs::create_dir_all(root.join("logs")).and_then(|()| fs::create_dir_all(root.join("run"))).map_err(|e| format!("mkdir: {e}"))?;
    let log = fs::File::create(log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let log_err = log.try_clone().map_err(|e| e.to_string())?;
    let mut command = Command::new(&argv[0]);
    // SAFETY: setsid is async-signal-safe and the closure touches no shared state.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    command.args(&argv[1..]).current_dir(root).stdin(Stdio::null()).stdout(log).stderr(log_err).spawn().map_err(|e| format!("start {}: {e}", argv[0]))
}

/// Stop the run: SIGTERM to its supervisor, which stops robocap-live (SIGINT, then SIGKILL to its group 20 s later) and gives the
/// cameras back; before it has taken them, it takes nothing. The run is this panel's (its session from the record, whether or not
/// run/live.pid was written), else the one in run/live.pid. A stop during a start's checks is noted, and that start launches
/// nothing.
pub fn stop(run: &Run, root: &Path) -> Result<Value, String> {
    let mut record = run.record();
    if matches!(record.owner, Owner::Starting { .. }) {
        record.owner = Owner::Starting { cancel: true };
        if let Some(pid) = run_pid(root).filter(|&pid| alive(pid)) {
            signal(pid as i32, libc::SIGTERM)?;
        }
        return Ok(json!({"pending": true}));
    }
    let session = record.session(root).filter(|&pid| alive(pid)).ok_or("no run is going")?;
    record.owner = Owner::Session { pid: session, stopping: true };
    signal(i32::try_from(session).map_err(|e| e.to_string())?, libc::SIGTERM)?;
    Ok(json!({"session": session}))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::Barrier;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    use crate::processes;

    use super::*;

    /// The run supervisor, faked as `robocap-panel handoff` behaves: it notes its launch once its SIGTERM handler is set (a SIGTERM
    /// before that kills it, as it kills the real one before it takes anything) and "checks the vendor recorder" for 1 s (a SIGTERM
    /// then makes it exit 2 and launch nothing), then notes and runs robocap-live as its child (a sleep named
    /// `<root>/bin/robocap-live`, with SIGINT at its default even when the test runner ignores it), passes a SIGTERM on as SIGINT,
    /// and exits with the child's status (128 + the signal).
    const FAKE_SUPERVISOR: &str = r#"#!/bin/bash
here=$(dirname "$0"); stop=0; child=
trap 'stop=1; [ -n "$child" ] && kill -INT "$child"' TERM
echo $$ >> "$here/launches"
sleep 1 & wait $!
[ $stop = 1 ] && exit 2
echo $$ >> "$here/lives"
env --default-signal=INT bash -c 'exec -a "$0" sleep 30' "$2" & child=$!
[ $stop = 1 ] && kill -INT "$child"
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done
exit $status
"#;

    /// A root with the fake supervisor at `<root>/handoff` (its command line then reads as a supervisor's); `name` keeps the tests
    /// that run at once apart.
    fn fake_root(name: &str) -> std::io::Result<(PathBuf, Vec<String>)> {
        let root = std::env::temp_dir().join(format!("robocap-panel-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root)?;
        let supervisor = root.join("handoff");
        fs::write(&supervisor, FAKE_SUPERVISOR)?;
        fs::set_permissions(&supervisor, fs::Permissions::from_mode(0o755))?;
        Ok((root, vec![supervisor.display().to_string()]))
    }

    /// Whether the run ended because robocap-live got SIGINT (the supervisor exits with 128 + 2).
    fn interrupted(record: &RunRecord) -> bool {
        record.last_exit.as_deref().is_some_and(|e| e.contains("exit status: 130"))
    }

    /// How many supervisors ran (`launches`), or how many of them launched robocap-live (`lives`).
    fn count(root: &Path, name: &str) -> usize {
        fs::read_to_string(root.join(name)).map_or(0, |text| text.lines().count())
    }

    /// Wait (5 s at most) until the fake supervisor has noted its launch: its SIGTERM handler is set.
    fn supervisor_up(root: &Path) -> bool {
        (0..100).any(|_| {
            std::thread::sleep(Duration::from_millis(50));
            count(root, "launches") > 0
        })
    }

    /// The record once no run holds it (within 10 s).
    fn ended(run: &Run) -> Result<RunRecord, String> {
        for _ in 0..200 {
            let record = run.snapshot();
            if record.owner == Owner::Nobody {
                return Ok(record);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Err(format!("the run still holds the record after 10 s: {:?}", run.snapshot()))
    }

    #[test]
    fn frame_csv_is_opt_in_and_uses_this_runs_log_path() -> Result<(), String> {
        let root = Path::new("/root/robocap-live");
        assert!(!StartRequest::parse("")?.command(root, 7).iter().any(|arg| arg == "--frame-csv"));
        let command = StartRequest::parse("frame_csv=on")?.command(root, 7);
        assert!(command.windows(2).any(|args| args == ["--frame-csv", "/root/robocap-live/logs/rt-7.frames.csv"]));
        Ok(())
    }

    #[test]
    fn the_start_form_builds_the_command_line_and_refuses_anything_else() -> Result<(), String> {
        let request = StartRequest::parse("viewer=rerun%2Bhttp%3A%2F%2F198.51.100.7%3A9876%2Fproxy&video_cameras=0%2C1%2C5&slam_hz=30&duration_s=600&hands=on")?;
        assert_eq!(request.viewer, "rerun+http://198.51.100.7:9876/proxy");
        assert_eq!((request.video_cameras.clone(), request.slam_hz, request.duration_s, request.hands, request.uclamp), (vec![0, 1, 5], 30, 600, true, false));
        let form = json!({"viewer": "rerun+http://198.51.100.7:9876/proxy", "video_cameras": [0, 1, 5], "hands": true, "hand_overlays": "fit",
            "slam_hz": 30, "slam_lane": "gpu", "slam_lag": "auto", "uclamp": false, "duration_s": 600, "frame_csv": false});
        assert_eq!(request.to_json(), form, "the page refills its form from these names");
        let command = request.command(Path::new("/root/robocap-live"), 7).join(" ");
        assert_eq!(
            command,
            "/root/robocap-live/bin/robocap-live --source live --rig /root/robocap-live/rig.json --nets rknn /root/robocap-live/models --hands on \
             --hand-overlays fit --viewer rerun+http://198.51.100.7:9876/proxy --video h264 --video-cameras 0,1,5 --display /root/robocap-live/assets/robocap-live-display.rrd \
             --slam-hz 30 --slam-lane gpu --slam-uclamp none --hands-uclamp none --duration 600 --summary-json /root/robocap-live/logs/rt-7.json"
        );
        assert!(StartRequest::parse("viewer=rerun%2Bhttp%3A%2F%2Fx%3B%20rm%20-rf%20%2F%3A9876%2Fproxy").is_err(), "a shell in the host");
        assert!(StartRequest::parse("video_cameras=0,6").is_err());
        assert!(StartRequest::parse("video_cameras=1,1").is_err());
        assert!(StartRequest::parse("duration_s=99999").is_err());
        assert!(StartRequest::parse("slam_hz=60").is_err());
        let defaults = StartRequest::parse("")?;
        assert_eq!((defaults.viewer.as_str(), defaults.video_cameras.is_empty(), defaults.slam_hz, defaults.duration_s), ("", true, 15, 0));
        assert!(!defaults.command(Path::new("/r"), 1).iter().any(|a| a == "--duration"), "the default run has no time limit");
        assert!(StartRequest::parse("duration_s=5").is_err());
        assert!(defaults.command(Path::new("/r"), 1).join(" ").contains("--video off"));
        Ok(())
    }

    #[test]
    fn the_start_form_picks_the_hand_overlays() -> Result<(), String> {
        let debug = StartRequest::parse("hand_overlays=debug")?.command(Path::new("/r"), 1).join(" ");
        assert!(debug.contains(" --hand-overlays debug "), "{debug}");
        let default = StartRequest::parse("")?.command(Path::new("/r"), 1).join(" ");
        assert!(default.contains(" --hand-overlays fit "), "the live default: {default}");
        assert!(StartRequest::parse("hand_overlays=all").is_err());
        assert!(StartRequest::parse("hand_overlays=fit%3B%20reboot").is_err());
        Ok(())
    }

    #[test]
    fn the_start_form_selects_the_slam_lane_and_lag() -> Result<(), String> {
        let default = StartRequest::parse("")?.command(Path::new("/r"), 1).join(" ");
        assert!(default.contains(" --slam-lane gpu "));
        assert!(!default.contains("--slam-set"), "auto lets the actual lane choose, including CPU fallback");
        let cpu = StartRequest::parse("slam_lane=cpu&slam_lag=on")?.command(Path::new("/r"), 1).join(" ");
        assert!(cpu.contains(" --slam-lane cpu --slam-set port.frontend_lag=true "));
        let sync = StartRequest::parse("slam_lane=gpu&slam_lag=off")?.command(Path::new("/r"), 1).join(" ");
        assert!(sync.contains(" --slam-set port.frontend_lag=false "));
        assert!(StartRequest::parse("slam_lane=other").is_err());
        assert!(StartRequest::parse("slam_lag=maybe").is_err());
        let empty = StartRequest::parse("hand_overlays=&slam_lane=&slam_lag=")?;
        assert_eq!((empty.hand_overlays, empty.slam_lane, empty.slam_lag), ("fit", "gpu", "auto"));
        Ok(())
    }

    #[test]
    fn the_phase_says_what_the_run_is_doing() {
        // (session alive, robocap-live running, stop requested) -> phase.
        assert_eq!(phase(false, false, false), Phase::Idle);
        assert_eq!(phase(true, false, false), Phase::Starting, "the supervisor checks and pauses the vendor recorder first");
        assert_eq!(phase(true, true, false), Phase::Streaming);
        assert_eq!(phase(true, true, true), Phase::Stopping);
        assert_eq!(phase(true, false, true), Phase::Stopping, "robocap-live ended; the handoff restores the vendor recorder");
        assert_eq!(phase(false, false, true), Phase::Idle);
    }

    #[test]
    fn two_starts_at_once_launch_one_run() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("two-starts")?;
        let run = Arc::new(Run::default());
        let request = StartRequest::parse("")?;
        let both = Barrier::new(2);
        // Checks that take a while: two starts that only checked run/live.pid would both pass them.
        let slow_checks = || {
            std::thread::sleep(Duration::from_millis(100));
            Checks::default()
        };
        let results: Vec<Result<Value, String>> = std::thread::scope(|scope| {
            let starts: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        both.wait();
                        start(&run, &root, &supervisor, &request, slow_checks)
                    })
                })
                .collect();
            starts.into_iter().map(|s| s.join().unwrap_or_else(|_| Err("a start panicked".into()))).collect()
        });
        let refused: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        assert_eq!(refused.len(), 1, "{results:?}");
        assert!(refused[0].contains("already starting or going"), "{refused:?}");
        assert!(supervisor_up(&root));
        stop(&run, &root)?;
        let record = ended(&run)?;
        assert_eq!(count(&root, "launches"), 1, "exactly one supervisor ran");
        assert!(record.last_exit.is_some(), "{record:?}");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stop_during_the_start_checks_launches_nothing() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("stop-in-checks")?;
        let run = Arc::new(Run::default());
        let request = StartRequest::parse("")?;
        let (checking, in_checks) = channel();
        let (go, wait_for_go) = channel::<()>();
        let started = std::thread::scope(|scope| {
            let starting = scope.spawn(|| {
                start(&run, &root, &supervisor, &request, move || {
                    let _ = checking.send(());
                    let _ = wait_for_go.recv();
                    Checks::default()
                })
            });
            let noted = in_checks.recv().map_err(|e| e.to_string()).and_then(|()| stop(&run, &root));
            let _ = go.send(());
            (noted, starting.join().unwrap_or_else(|_| Err("the start panicked".into())))
        });
        let (noted, started) = started;
        assert_eq!(noted?["pending"], json!(true));
        assert!(started.as_ref().is_err_and(|e| e.contains("nothing started")), "{started:?}");
        let record = run.snapshot();
        assert_eq!((record.owner, count(&root, "launches")), (Owner::Nobody, 0));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stop_during_checks_also_stops_an_inherited_run() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("inherited-stop-in-checks")?;
        let old = Arc::new(Run::default());
        start(&old, &root, &supervisor, &StartRequest::parse("")?, Checks::default)?;
        assert!(supervisor_up(&root));
        let run = Arc::new(Run::default());
        let result = start(&run, &root, &supervisor, &StartRequest::parse("")?, || {
            assert!(stop(&run, &root).is_ok());
            Checks { refusals: vec!["a run is already going".into()], warnings: Vec::new() }
        });
        assert!(result.is_err());
        let finished = ended(&old);
        if finished.is_err() {
            stop(&old, &root)?;
            ended(&old)?;
        }
        assert!(finished.is_ok(), "the inherited supervisor must receive the stop");
        fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn a_stop_before_the_supervisor_takes_the_cap_launches_no_robocap_live() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("stop-in-supervisor-checks")?;
        let run = Arc::new(Run::default());
        let started = start(&run, &root, &supervisor, &StartRequest::parse("")?, Checks::default)?;
        assert!(supervisor_up(&root));
        let stopped = stop(&run, &root)?;
        assert_eq!(stopped["session"], started["pid"], "the stop went to the supervisor");
        let record = ended(&run)?;
        assert_eq!(record.last_exit.as_deref(), Some("exit status: 2"), "the supervisor refused: {record:?}");
        assert_eq!(count(&root, "lives"), 0, "robocap-live never started");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stop_during_the_run_interrupts_robocap_live() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("stop-in-run")?;
        let run = Arc::new(Run::default());
        let started = start(&run, &root, &supervisor, &StartRequest::parse("")?, Checks::default)?;
        let session = started["pid"].as_u64().and_then(|pid| u32::try_from(pid).ok()).ok_or("no session")?;
        let up = (0..100).any(|_| {
            std::thread::sleep(Duration::from_millis(50));
            !live_processes(&processes(), &root, session).is_empty()
        });
        assert!(up, "robocap-live came up within 5 s");
        stop(&run, &root)?;
        let record = ended(&run)?;
        assert!(interrupted(&record), "robocap-live got the SIGINT: {record:?}");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_run_file_that_cannot_be_written_stops_the_run() -> Result<(), Box<dyn std::error::Error>> {
        let (root, supervisor) = fake_root("no-run-files")?;
        fs::create_dir_all(root.join("run/live.pid"))?;
        let run = Arc::new(Run::default());
        let started = start(&run, &root, &supervisor, &StartRequest::parse("")?, Checks::default);
        assert!(started.as_ref().is_err_and(|e| e.contains("run/live.pid") && e.contains("being stopped")), "{started:?}");
        // The stop found the run through the record alone: run/live.pid never named it.
        let record = ended(&run)?;
        assert!(record.last_exit.is_some(), "{record:?}");
        assert_eq!(count(&root, "lives"), 0, "the stop came before robocap-live");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stale_run_file_names_no_run() -> Result<(), Box<dyn std::error::Error>> {
        let (root, _) = fake_root("stale-pid")?;
        // After a reboot or a pid reuse, run/live.pid names some other program: it is not a run, and Stop must not signal it.
        let mut other = Command::new("sleep").arg("30").spawn()?;
        fs::create_dir_all(root.join("run"))?;
        fs::write(root.join("run/live.pid"), format!("{}\n", other.id()))?;
        let run = Run::default();
        assert_eq!(run.snapshot().session(&root), None);
        assert!(stop(&run, &root).is_err_and(|e| e.contains("no run")));
        assert!(other.try_wait()?.is_none(), "the other program still runs");
        other.kill()?;
        other.wait()?;
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_late_end_of_an_older_session_changes_nothing() {
        let run = Run::default();
        *run.record() = RunRecord { owner: Owner::Session { pid: 200, stopping: true }, ..RunRecord::default() };
        run.finish(100, "exit status: 0".into());
        let record = run.snapshot();
        assert_eq!((record.owner, record.stopping(), record.last_exit), (Owner::Session { pid: 200, stopping: true }, true, None));
        run.finish(200, "exit status: 1".into());
        let record = run.snapshot();
        assert_eq!((record.owner, record.stopping(), record.last_exit.as_deref()), (Owner::Nobody, false, Some("exit status: 1")));
    }
}
