use anyhow::{Result, ensure};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

static STOP: AtomicBool = AtomicBool::new(false);
extern "C" fn interrupted(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

fn main() -> Result<()> {
    #[cfg(feature = "live-slam")]
    if std::env::args().nth(1).as_deref() == Some("--slam-worker") {
        return robocap_recorder::slam_worker();
    }
    // SAFETY: handler only sets a lock-free atomic flag; capture cleanup runs on
    // the main thread. No non-signal-safe operation is called by the handler.
    unsafe {
        libc::signal(libc::SIGINT, interrupted as *const () as libc::sighandler_t);
        libc::signal(
            libc::SIGTERM,
            interrupted as *const () as libc::sighandler_t,
        );
    }
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure!(
        args.len() == 2,
        "usage: robocap-direct NEW_SESSION_DIRECTORY DURATION_SECONDS"
    );
    robocap_recorder::session::run(PathBuf::from(&args[0]), args[1].parse()?, &STOP)
}
