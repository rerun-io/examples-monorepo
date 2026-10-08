//! The panel as a service on the cap, with no shell in between: `robocap-panel start | stop | status | install-boot |
//! remove-boot`. `install-boot` makes `/etc/init.d/S87robocap-panel` a symlink to this binary, so the cap's rcS runs
//! `robocap-panel start` at boot and rcK runs `robocap-panel stop` at shutdown. Nothing here starts a run.
//!
//! The panel lives at `<root>/bin/robocap-panel` (root: `/root/robocap-live`); `start` runs it in its own session on the A55
//! cores (0-3), off the A76s that SLAM and the hands use (a run it starts inherits the mask; robocap-live pins its own
//! threads), logs to `<root>/logs/panel.log` and writes `<root>/run/panel.pid`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::run::{alive, session_pid, spawn_session};

/// rcS runs every `/etc/init.d/S??*` in order with `start` (one that is not a `.sh` file is executed, not sourced).
const INIT_LINK: &str = "/etc/init.d/S87robocap-panel";

/// `robocap-panel <action> [--port 8090]`: the exit code, or None when `args` names no service action.
pub fn main(args: &[String]) -> Option<i32> {
    let (action, rest) = args.split_first()?;
    let action: fn(&Path, &Path, u16) -> Result<String, String> = match action.as_str() {
        "start" => start,
        "stop" => |_, root, _| stop(root),
        "status" => |_, root, _| Ok(running(root).map_or("not running".into(), |pid| format!("running: pid {pid}"))),
        "install-boot" => |exe, _, _| install_boot(exe),
        "remove-boot" => |exe, _, _| remove_boot(exe),
        _ => return None,
    };
    let context = || -> Result<(PathBuf, PathBuf, u16), String> {
        let exe = std::env::current_exe().map_err(|e| format!("cannot find this binary: {e}"))?;
        // <root>/bin/robocap-panel
        let root = exe.parent().and_then(Path::parent).ok_or("this binary is not in <root>/bin")?.to_path_buf();
        let port = match rest {
            [] => 8090,
            [flag, port] if flag == "--port" => port.parse().map_err(|_| format!("bad port {port}"))?,
            _ => return Err("usage: robocap-panel start|stop|status|install-boot|remove-boot [--port 8090]".into()),
        };
        Ok((exe, root, port))
    };
    Some(match context().and_then(|(exe, root, port)| action(&exe, &root, port)) {
        Ok(text) => {
            println!("{text}");
            0
        }
        Err(error) => {
            eprintln!("robocap-panel: {error}");
            1
        }
    })
}

/// The panel's pid from `run/panel.pid` while that pid is still the panel (a stale file after a power cut may name another
/// process, even this `robocap-panel start`).
fn running(root: &Path) -> Option<u32> {
    session_pid(&root.join("run/panel.pid"), "--port")
}

fn start(exe: &Path, root: &Path, port: u16) -> Result<String, String> {
    if let Some(pid) = running(root) {
        return Ok(format!("already running: pid {pid}"));
    }
    // The mask is set on this process before the spawn, and the panel inherits it.
    // SAFETY: a zeroed cpu_set_t is a valid empty set, and the calls only read or write that local set.
    let pinned = unsafe {
        let mut cpus: libc::cpu_set_t = std::mem::zeroed();
        (0..4).for_each(|cpu| libc::CPU_SET(cpu, &mut cpus));
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpus) == 0
    };
    if !pinned {
        eprintln!("robocap-panel: the panel is not pinned to cores 0-3: {}", std::io::Error::last_os_error());
    }
    let log = root.join("logs/panel.log");
    let argv = [exe.display().to_string(), "--port".into(), port.to_string(), "--root".into(), root.display().to_string()];
    let mut child = spawn_session(root, &argv, &log)?;
    // A panel that cannot bind its port exits at once: give it a second before calling it started.
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(format!("the panel exited ({status}): {}", fs::read_to_string(&log).unwrap_or_default().trim()));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    fs::write(root.join("run/panel.pid"), child.id().to_string()).map_err(|e| format!("panel.pid: {e}"))?;
    Ok(format!("started: pid {} on port {port}", child.id()))
}

/// SIGTERM to the panel (never by name), then up to 3 s for it to go. A run it started keeps going: its supervisor has its own
/// session.
fn stop(root: &Path) -> Result<String, String> {
    let Some(pid) = running(root) else { return Ok("not running".into()) };
    crate::handoff::signal(pid as i32, libc::SIGTERM)?;
    let deadline = Instant::now() + Duration::from_secs(3);
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    if alive(pid) {
        return Err(format!("pid {pid} is still running 3 s after SIGTERM"));
    }
    let _ = fs::remove_file(root.join("run/panel.pid"));
    Ok(format!("stopped pid {pid}"))
}

fn install_boot(exe: &Path) -> Result<String, String> {
    match fs::symlink_metadata(INIT_LINK) {
        Ok(meta) if !meta.file_type().is_symlink() => return Err(format!("{INIT_LINK} exists and is not our symlink; left alone")),
        Ok(_) => fs::remove_file(INIT_LINK).map_err(|e| format!("{INIT_LINK}: {e}"))?,
        Err(_) => {}
    }
    std::os::unix::fs::symlink(exe, INIT_LINK).map_err(|e| format!("{INIT_LINK}: {e}"))?;
    Ok(format!("installed {INIT_LINK} -> {}", exe.display()))
}

fn remove_boot(exe: &Path) -> Result<String, String> {
    match fs::read_link(INIT_LINK) {
        Ok(target) if target == exe => fs::remove_file(INIT_LINK).map(|()| format!("removed {INIT_LINK}")).map_err(|e| format!("{INIT_LINK}: {e}")),
        Ok(target) => Err(format!("{INIT_LINK} points to {}, not this binary; left alone", target.display())),
        Err(_) => Ok(format!("{INIT_LINK} is not installed")),
    }
}
