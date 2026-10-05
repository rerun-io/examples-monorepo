//! The run this panel starts and stops. The start form ([`StartRequest`]) builds the command line; one record ([`RunRecord`], under
//! [`Run`]'s lock) owns the run's lifecycle: a start reserves it before its checks and, holding it, spawns the handoff and publishes
//! its session and the run files; a stop reads the session from it; only that session's end frees it.

use std::fs;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::{processes, read_trim};

/// What the run is doing, for the page's state and buttons.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Idle,
    /// The panel runs its checks, or handoff-run.sh checks and pauses the vendor recorder; robocap-live is not running yet.
    Starting,
    Streaming,
    /// Stop was asked for, or the run ended: robocap-live drains, then the handoff restores the vendor recorder (~15 s).
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
    /// Not this panel: no run, or one that start-live.sh (or an earlier panel) started; run/live.pid names it.
    #[default]
    Nobody,
    /// A start, from before its checks until the handoff is spawned.
    Starting,
    /// The handoff this panel spawned: its pid, which is its session's id.
    Session(u32),
}

/// The run's lifecycle: one value under [`Run`]'s lock.
#[derive(Clone, Debug, Default)]
pub struct RunRecord {
    pub owner: Owner,
    /// Stop was asked for: of the starting run (it then launches nothing) or of the run that is going; cleared by the next start
    /// and by the end of this panel's run.
    pub stop_requested: bool,
    /// The exit status of the last run this panel started, once it ended.
    pub last_exit: Option<String>,
    /// The form of the last run this panel started ([`StartRequest::to_json`]); a new browser fills its form from it.
    pub last_request: Option<Value>,
}

impl RunRecord {
    /// The run's session: this panel's (also before run/live.pid is written), else the one in run/live.pid; `None` while a start
    /// runs its checks.
    pub fn session(&self, root: &Path) -> Option<u32> {
        match self.owner {
            Owner::Nobody => run_pid(root),
            Owner::Starting => None,
            Owner::Session(session) => Some(session),
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
        if record.owner == Owner::Session(session) {
            record.owner = Owner::Nobody;
            record.stop_requested = false;
            record.last_exit = Some(status);
        }
    }
}

/// The session id in run/live.pid (the panel and start-live.sh both write it).
pub fn run_pid(root: &Path) -> Option<u32> {
    read_trim(root.join("run/live.pid"))?.parse().ok()
}

/// Whether `pid` runs (a zombie does not).
pub fn alive(pid: u32) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat")).ok().and_then(|stat| stat.rsplit(')').next()?.split_whitespace().next().map(str::to_owned))
        .is_some_and(|state| state != "Z")
}

/// The run's robocap-live processes: its session's programs under `bin/`.
pub fn live_processes(root: &Path, session: u32) -> Vec<u32> {
    let bin = root.join("bin/").display().to_string();
    processes().into_iter().filter(|p| p.session == session && p.program.starts_with(&bin)).map(|p| p.pid).collect()
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
            duration_s: number("duration_s", 1800, 10..=7200)?,
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
        command.extend(["--duration".into(), self.duration_s.to_string(), "--summary-json".into(), at(&format!("logs/rt-{stamp}.json"))]);
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

/// Start a run: reserve the record, run `preflight` (refusals as text, empty = go), then, holding the record, spawn the handoff
/// and publish its session and the run files. The run files are written as scripts/start-live.sh writes them, so scripts/stop.sh
/// works on the run too.
///
/// # Errors
///
/// The text the page shows: a start or run of this panel's holds the record, a refusal, a stop that came during the checks
/// (nothing starts), a failed spawn, or a run file that could not be written (the run is then stopped as [`stop`] stops it, and
/// the handoff restores the vendor recorder).
pub fn start(run: &Arc<Run>, root: &Path, request: &StartRequest, preflight: impl FnOnce() -> Vec<String>) -> Result<Value, String> {
    {
        let mut record = run.record();
        if record.owner != Owner::Nobody {
            return Err("a run is already starting or going (stop it first)".into());
        }
        record.owner = Owner::Starting;
        record.stop_requested = false;
    }
    let refusals = preflight();
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // The handoff script, its time limit, then robocap-live: what is spawned and what run/live.cmd records.
    let mut argv = vec![root.join("scripts/handoff-run.sh").display().to_string(), (request.duration_s + 40).to_string()];
    argv.extend(request.command(root, stamp));
    let log_path = root.join(format!("logs/rt-{stamp}.log"));
    let mut record = run.record();
    let spawned = if record.stop_requested {
        Err("stop was asked for while the run was starting; nothing started".to_string())
    } else if !refusals.is_empty() {
        Err(refusals.join("; "))
    } else {
        spawn_handoff(root, &argv, &log_path)
    };
    let mut child = match spawned {
        Ok(child) => child,
        Err(error) => {
            record.owner = Owner::Nobody;
            return Err(error);
        }
    };
    let pid = child.id();
    let quoted = argv.join(" ");
    record.owner = Owner::Session(pid);
    record.last_exit = None;
    record.last_request = Some(request.to_json());
    let run_files = [("run/live.pid", format!("{pid}\n")), ("run/live.log", format!("{}\n", log_path.display())), ("run/live.cmd", format!("{quoted}\n"))];
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
    Ok(json!({"pid": pid, "log": log_path.display().to_string(), "cmd": quoted}))
}

/// The handoff, detached: setsid (not a group leader, so it does not fork) makes it its own session's leader, with this pid; its
/// output goes to `log_path`.
fn spawn_handoff(root: &Path, argv: &[String], log_path: &Path) -> Result<Child, String> {
    fs::create_dir_all(root.join("logs")).and_then(|()| fs::create_dir_all(root.join("run"))).map_err(|e| format!("mkdir: {e}"))?;
    let log = fs::File::create(log_path).map_err(|e| format!("{}: {e}", log_path.display()))?;
    let log_err = log.try_clone().map_err(|e| e.to_string())?;
    Command::new("setsid")
        .args(argv)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .spawn()
        .map_err(|e| format!("start {}: {e}", argv[0]))
}

/// SIGINT to the run's robocap-live (only processes of its session under `bin/`), as scripts/stop.sh does. The run is this
/// panel's (its session from the record, whether or not run/live.pid was written), else the one in run/live.pid. A stop during a
/// start's checks is noted, and that start launches nothing; one that comes while the handoff is still starting waits (up to
/// 60 s) for robocap-live to appear, then signals it the same way.
pub fn stop(run: &Run, root: &Path) -> Result<Value, String> {
    let session = {
        let mut record = run.record();
        if record.owner == Owner::Starting {
            record.stop_requested = true;
            return Ok(json!({"signalled": [], "pending": true}));
        }
        let session = record.session(root).filter(|&pid| alive(pid)).ok_or("no run is going")?;
        record.stop_requested = true;
        session
    };
    let signal = |targets: &[u32]| {
        for pid in targets {
            let _ = Command::new("kill").args(["-INT", &pid.to_string()]).status();
        }
    };
    let targets = live_processes(root, session);
    if !targets.is_empty() {
        signal(&targets);
        return Ok(json!({"signalled": targets, "session": session}));
    }
    let root = root.to_path_buf();
    std::thread::spawn(move || {
        for _ in 0..300 {
            std::thread::sleep(Duration::from_millis(200));
            if !alive(session) {
                return;
            }
            let targets = live_processes(&root, session);
            if !targets.is_empty() {
                signal(&targets);
                return;
            }
        }
    });
    Ok(json!({"signalled": [], "pending": true, "session": session}))
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use std::sync::Barrier;
    use std::sync::mpsc::channel;

    use super::*;

    /// handoff-run.sh, faked: it notes the launch, "pauses the vendor recorder" for 1 s, then becomes robocap-live (a sleep named
    /// `<root>/bin/robocap-live`, with SIGINT at its default even when the test runner ignores it).
    const FAKE_HANDOFF: &str = "#!/bin/bash\necho $$ >> \"${0%/scripts/*}/launches\"\nsleep 1\nexec env --default-signal=INT bash -c 'exec -a \"$0\" sleep 30' \"$2\"\n";

    /// A root with the fake handoff in scripts/; `name` keeps the tests that run at once apart.
    fn fake_root(name: &str) -> std::io::Result<PathBuf> {
        let root = std::env::temp_dir().join(format!("robocap-panel-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("scripts"))?;
        let handoff = root.join("scripts/handoff-run.sh");
        fs::write(&handoff, FAKE_HANDOFF)?;
        fs::set_permissions(&handoff, fs::Permissions::from_mode(0o755))?;
        Ok(root)
    }

    fn launches(root: &Path) -> usize {
        fs::read_to_string(root.join("launches")).map_or(0, |text| text.lines().count())
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
    fn the_start_form_builds_the_command_line_and_refuses_anything_else() -> Result<(), String> {
        let request = StartRequest::parse("viewer=rerun%2Bhttp%3A%2F%2F198.51.100.7%3A9876%2Fproxy&video_cameras=0%2C1%2C5&slam_hz=30&duration_s=600&hands=on")?;
        assert_eq!(request.viewer, "rerun+http://198.51.100.7:9876/proxy");
        assert_eq!((request.video_cameras.clone(), request.slam_hz, request.duration_s, request.hands, request.uclamp), (vec![0, 1, 5], 30, 600, true, false));
        let form = json!({"viewer": "rerun+http://198.51.100.7:9876/proxy", "video_cameras": [0, 1, 5], "hands": true, "hand_overlays": "fit",
            "slam_hz": 30, "slam_lane": "gpu", "slam_lag": "auto", "uclamp": false, "duration_s": 600});
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
        assert_eq!((defaults.viewer.as_str(), defaults.video_cameras.is_empty(), defaults.slam_hz, defaults.duration_s), ("", true, 15, 1800));
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
        assert_eq!(phase(true, false, false), Phase::Starting, "handoff-run.sh checks and pauses the vendor recorder first");
        assert_eq!(phase(true, true, false), Phase::Streaming);
        assert_eq!(phase(true, true, true), Phase::Stopping);
        assert_eq!(phase(true, false, true), Phase::Stopping, "robocap-live ended; the handoff restores the vendor recorder");
        assert_eq!(phase(false, false, true), Phase::Idle);
    }

    #[test]
    fn two_starts_at_once_launch_one_run() -> Result<(), Box<dyn std::error::Error>> {
        let root = fake_root("two-starts")?;
        let run = Arc::new(Run::default());
        let request = StartRequest::parse("")?;
        let both = Barrier::new(2);
        // Checks that take a while: two starts that only checked run/live.pid would both pass them.
        let slow_checks = || {
            std::thread::sleep(Duration::from_millis(100));
            Vec::new()
        };
        let results: Vec<Result<Value, String>> = std::thread::scope(|scope| {
            let starts: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        both.wait();
                        start(&run, &root, &request, slow_checks)
                    })
                })
                .collect();
            starts.into_iter().map(|s| s.join().unwrap_or_else(|_| Err("a start panicked".into()))).collect()
        });
        let refused: Vec<&String> = results.iter().filter_map(|r| r.as_ref().err()).collect();
        assert_eq!(refused.len(), 1, "{results:?}");
        assert!(refused[0].contains("already starting or going"), "{refused:?}");
        stop(&run, &root)?;
        let record = ended(&run)?;
        assert_eq!(launches(&root), 1, "exactly one handoff ran");
        assert!(record.last_exit.as_deref().is_some_and(|e| e.contains("signal: 2")), "{record:?}");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stop_during_the_start_checks_launches_nothing() -> Result<(), Box<dyn std::error::Error>> {
        let root = fake_root("stop-in-checks")?;
        let run = Arc::new(Run::default());
        let request = StartRequest::parse("")?;
        let (checking, in_checks) = channel();
        let (go, wait_for_go) = channel::<()>();
        let started = std::thread::scope(|scope| {
            let starting = scope.spawn(|| {
                start(&run, &root, &request, move || {
                    let _ = checking.send(());
                    let _ = wait_for_go.recv();
                    Vec::new()
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
        assert_eq!((record.owner, launches(&root)), (Owner::Nobody, 0));
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_stop_while_the_handoff_starts_signals_robocap_live_once_it_is_up() -> Result<(), Box<dyn std::error::Error>> {
        let root = fake_root("stop-in-handoff")?;
        let run = Arc::new(Run::default());
        let started = start(&run, &root, &StartRequest::parse("")?, Vec::new)?;
        let stopped = stop(&run, &root)?;
        assert_eq!((stopped["pending"].clone(), stopped["session"].clone()), (json!(true), started["pid"].clone()), "the handoff is still in its pause");
        let record = ended(&run)?;
        assert!(record.last_exit.as_deref().is_some_and(|e| e.contains("signal: 2")), "robocap-live got the SIGINT: {record:?}");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_run_file_that_cannot_be_written_stops_the_run() -> Result<(), Box<dyn std::error::Error>> {
        let root = fake_root("no-run-files")?;
        fs::create_dir_all(root.join("run/live.pid"))?;
        let run = Arc::new(Run::default());
        let started = start(&run, &root, &StartRequest::parse("")?, Vec::new);
        assert!(started.as_ref().is_err_and(|e| e.contains("run/live.pid") && e.contains("being stopped")), "{started:?}");
        // The stop found the run through the record alone: run/live.pid never named it.
        let record = ended(&run)?;
        assert_eq!(launches(&root), 1);
        assert!(record.last_exit.as_deref().is_some_and(|e| e.contains("signal: 2")), "{record:?}");
        fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn a_late_end_of_an_older_session_changes_nothing() {
        let run = Run::default();
        *run.record() = RunRecord { owner: Owner::Session(200), stop_requested: true, ..RunRecord::default() };
        run.finish(100, "exit status: 0".into());
        let record = run.snapshot();
        assert_eq!((record.owner, record.stop_requested, record.last_exit), (Owner::Session(200), true, None));
        run.finish(200, "exit status: 1".into());
        let record = run.snapshot();
        assert_eq!((record.owner, record.stop_requested, record.last_exit.as_deref()), (Owner::Nobody, false, Some("exit status: 1")));
    }
}
