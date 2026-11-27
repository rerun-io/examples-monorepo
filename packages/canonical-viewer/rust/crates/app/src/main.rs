//! `rerun-canonical`: the stock `rerun` command line (viewer, `--headless`, `mcp`, `rrd`, SDK spawn), with every
//! extension of the canonical viewer in each viewer it opens.

use re_memory::AccountingAllocator;
use std::process::ExitCode;

// Rerun's own allocator choice for the `rerun` binary.
#[global_allocator]
static GLOBAL: AccountingAllocator<mimalloc::MiMalloc> =
    AccountingAllocator::new(mimalloc::MiMalloc);

fn main() -> ExitCode {
    let main_thread_token = rerun::MainThreadToken::i_promise_i_am_on_the_main_thread();
    re_log::setup_logging();
    canonical_viewer::install_hooks();
    match rerun::run(
        main_thread_token,
        build_info(),
        rerun::CallSource::Cli,
        std::env::args(),
    ) {
        Ok(exit_code) => ExitCode::from(exit_code),
        Err(err) => {
            let _err = re_error::try_eprintln!("Error: {}", re_error::format(err));
            ExitCode::FAILURE
        }
    }
}

/// `re_build_info` versions in lockstep with the Rerun crates, so `CrateVersion::LOCAL` is the Rerun release: what the
/// SDK checks when it spawns or connects to this viewer.
fn build_info() -> re_build_info::BuildInfo {
    re_build_info::BuildInfo {
        version: re_build_info::CrateVersion::LOCAL,
        ..re_build_info::cargo_build_info!()
    }
}
