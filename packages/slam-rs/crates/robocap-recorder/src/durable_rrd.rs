use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use re_log_encoding::{Encoder, EncodingOptions};
use re_log_types::LogMsg;
use rerun::sink::{LogSink, SinkFlushError};

struct State {
    encoder: Option<Encoder<File>>,
    sync_file: File,
    failure: Option<String>,
}

/// RRD sink with explicit storage synchronization and completion publication.
///
/// The SDK owns its dispatch thread; this sink adds no second message backlog.
/// SDK `send` cannot return errors, so the first write error is retained and
/// returned by every subsequent flush or finish. Flush the RecordingStream
/// before calling `finish`, so queued SDK messages precede the footer.
/// Storage synchronization is not proof of power-loss behavior of the device.
#[derive(Clone)]
pub struct DurableRrdSink {
    state: Arc<Mutex<State>>,
    partial: PathBuf,
    target: PathBuf,
    directory: Arc<File>,
}

impl DurableRrdSink {
    /// Exclusively create an incomplete file beside its intended final path.
    pub fn create(target: &Path) -> Result<Self> {
        ensure!(
            target
                .extension()
                .is_some_and(|extension| extension == "rrd"),
            "recording must use .rrd extension"
        );
        ensure!(
            !target.try_exists()?,
            "recording already exists: {}",
            target.display()
        );
        let partial = target.with_extension("rrd.partial");
        let directory = File::open(
            target
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or(Path::new(".")),
        )?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)
            .with_context(|| format!("create incomplete recording {}", partial.display()))?;
        let sync_file = file.try_clone()?;
        let encoder = Encoder::new_eager(
            re_build_info::CrateVersion::LOCAL,
            EncodingOptions::PROTOBUF_COMPRESSED,
            file,
        )?;
        directory.sync_all()?;
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                encoder: Some(encoder),
                sync_file,
                failure: None,
            })),
            partial,
            target: target.to_path_buf(),
            directory: Arc::new(directory),
        })
    }

    /// Write the footer, sync its contents, then publish without replacing data.
    pub fn finish(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("RRD writer lock poisoned"))?;
        if let Some(failure) = &state.failure {
            bail!("RRD writer failed: {failure}");
        }
        let Some(mut encoder) = state.encoder.take() else {
            bail!("RRD writer is already closed");
        };
        let result = (|| -> Result<()> {
            encoder.finish()?;
            encoder.flush_blocking()?;
            state.sync_file.sync_all()?;
            publish_without_replacing(&self.partial, &self.target)?;
            self.directory.sync_all()?;
            Ok(())
        })();
        if let Err(error) = &result {
            state.failure = Some(error.to_string());
        }
        result
    }
}

impl LogSink for DurableRrdSink {
    fn send(&self, message: LogMsg) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state.failure.is_some() {
            return;
        }
        let Some(encoder) = &mut state.encoder else {
            state.failure = Some("message received after recording closed".to_owned());
            return;
        };
        if let Err(error) = encoder.append(&message) {
            state.failure = Some(error.to_string());
        }
    }

    fn flush_blocking(&self, _timeout: Duration) -> Result<(), SinkFlushError> {
        // File I/O and sync_all are synchronous OS operations. The caller's
        // RecordingStream supplies the dispatch wait timeout; a blocked storage
        // syscall itself cannot be cancelled by this sink.
        let mut state = self
            .state
            .lock()
            .map_err(|_| SinkFlushError::failed("RRD writer lock poisoned"))?;
        if let Some(failure) = &state.failure {
            return Err(SinkFlushError::failed(failure));
        }
        let result = (|| -> Result<()> {
            if let Some(encoder) = &mut state.encoder {
                encoder.flush_blocking()?;
            }
            state.sync_file.sync_all()?;
            Ok(())
        })();
        result.map_err(|error| {
            let message = error.to_string();
            state.failure = Some(message.clone());
            SinkFlushError::failed(message)
        })
    }

    fn defers_finalization_to_shutdown(&self) -> bool {
        true
    }
}

#[cfg(target_os = "linux")]
fn publish_without_replacing(source: &Path, target: &Path) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let source = CString::new(source.as_os_str().as_bytes())?;
    let target = CString::new(target.as_os_str().as_bytes())?;
    // SAFETY: both strings are NUL-terminated and live for the syscall. The
    // RENAME_NOREPLACE flag atomically rejects an existing destination.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn publish_without_replacing(source: &Path, target: &Path) -> Result<()> {
    std::fs::hard_link(source, target)?;
    std::fs::remove_file(source)?;
    Ok(())
}
