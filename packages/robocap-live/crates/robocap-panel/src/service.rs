//! The panel as a service on the cap, with no shell in between: `robocap-panel start | stop | status | install-boot |
//! remove-boot`. `install-boot` makes `/etc/init.d/S87robocap-panel` a symlink to this binary, so the cap's rcS runs
//! `robocap-panel start` at boot and rcK runs `robocap-panel stop` at shutdown. Nothing here starts a run.
//!
//! The panel lives at `<root>/bin/robocap-panel` (root: `/root/robocap-live`); `start` runs it in its own session on the A55
//! cores (0-3), off the A76s that SLAM and the hands use (a run it starts inherits the mask; robocap-live pins its own
//! threads), logs to `<root>/logs/panel.log` and writes `<root>/run/panel.pid`.

use std::fs;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// rcS runs every `/etc/init.d/S??*` in order with `start` (one that is not a `.sh` file is executed, not sourced).
const INIT_LINK: &str = "/etc/init.d/S87robocap-panel";

/// `robocap-panel <action> [--port 8090]`: the exit code.
pub fn main(action: &str, args: &[String]) -> i32 {
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => return fail(&format!("cannot find this binary: {error}")),
    };
    // <root>/bin/robocap-panel
    let Some(root) = exe.parent().and_then(Path::parent).map(Path::to_path_buf) else { return fail("this binary is not in <root>/bin") };
    let port = match args {
        [] => 8090,
        [flag, port] if flag == "--port" => match port.parse::<u16>() {
            Ok(port) => port,
            Err(_) => return fail(&format!("bad port {port}")),
        },
        _ => return fail("usage: robocap-panel start|stop|status|install-boot|remove-boot [--port 8090]"),
    };
    let result = match action {
        "start" => start(&exe, &root, port),
        "stop" => stop(&root),
        "status" => Ok(match running(&root, &exe) {
            Some(pid) => format!("running: pid {pid}"),
            None => "not running".into(),
        }),
        "install-boot" => install_boot(&exe),
        "remove-boot" => remove_boot(&exe),
        _ => Err(format!("unknown action {action}")),
    };
    match result {
        Ok(text) => {
            println!("{text}");
            0
        }
        Err(error) => fail(&error),
    }
}

fn fail(error: &str) -> i32 {
    eprintln!("robocap-panel: {error}");
    1
}

/// The panel's pid from `run/panel.pid`, if that process is this binary (a stale pid file may name another process).
fn running(root: &Path, exe: &Path) -> Option<u32> {
    let pid: u32 = fs::read_to_string(root.join("run/panel.pid")).ok()?.trim().parse().ok()?;
    let target = fs::read_link(format!("/proc/{pid}/exe")).ok()?;
    // A binary replaced by a deploy shows as "<path> (deleted)".
    (target == exe || target.to_string_lossy().strip_suffix(" (deleted)") == Some(&*exe.to_string_lossy())).then_some(pid)
}

fn start(exe: &Path, root: &Path, port: u16) -> Result<String, String> {
    if let Some(pid) = running(root, exe) {
        return Ok(format!("already running: pid {pid}"));
    }
    fs::create_dir_all(root.join("run")).and_then(|()| fs::create_dir_all(root.join("logs"))).map_err(|e| format!("mkdir: {e}"))?;
    let log = fs::File::create(root.join("logs/panel.log")).map_err(|e| format!("panel.log: {e}"))?;
    let log_err = log.try_clone().map_err(|e| e.to_string())?;
    let mut command = Command::new(exe);
    // SAFETY: setsid and sched_setaffinity are async-signal-safe and the closure touches no shared state.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut cpus: libc::cpu_set_t = std::mem::zeroed();
            for cpu in 0..4 {
                libc::CPU_SET(cpu, &mut cpus);
            }
            // A failed pin is not worth a missing panel.
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpus);
            Ok(())
        });
    }
    let child = command
        .args(["--port", &port.to_string(), "--root", &root.display().to_string()])
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(log)
        .stderr(log_err)
        .spawn()
        .map_err(|e| format!("start {}: {e}", exe.display()))?;
    fs::write(root.join("run/panel.pid"), child.id().to_string()).map_err(|e| format!("panel.pid: {e}"))?;
    Ok(format!("started: pid {} on port {port}", child.id()))
}

/// SIGTERM to the panel (never by name). A run it started keeps going: its supervisor has its own session.
fn stop(root: &Path) -> Result<String, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let Some(pid) = running(root, &exe) else { return Ok("not running".into()) };
    crate::handoff::signal(pid as i32, libc::SIGTERM)?;
    let _ = fs::remove_file(root.join("run/panel.pid"));
    Ok(format!("stopped pid {pid}"))
}

fn install_boot(exe: &Path) -> Result<String, String> {
    let link = PathBuf::from(INIT_LINK);
    match fs::symlink_metadata(&link) {
        Ok(meta) if !meta.file_type().is_symlink() => return Err(format!("{INIT_LINK} exists and is not our symlink; left alone")),
        Ok(_) => fs::remove_file(&link).map_err(|e| format!("{INIT_LINK}: {e}"))?,
        Err(_) => {}
    }
    std::os::unix::fs::symlink(exe, &link).map_err(|e| format!("{INIT_LINK}: {e}"))?;
    Ok(format!("installed {INIT_LINK} -> {}", exe.display()))
}

fn remove_boot(exe: &Path) -> Result<String, String> {
    match fs::read_link(INIT_LINK) {
        Ok(target) if target == exe => fs::remove_file(INIT_LINK).map(|()| format!("removed {INIT_LINK}")).map_err(|e| format!("{INIT_LINK}: {e}")),
        Ok(target) => Err(format!("{INIT_LINK} points to {}, not this binary; left alone", target.display())),
        Err(_) => Ok(format!("{INIT_LINK} is not installed")),
    }
}
