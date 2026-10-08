//! Opt-in capture diagnostic CSV contract; only numeric metadata crosses the bounded writer channel.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use kornia_staging_io::v4l::mplane::DequeueMeta;

const COLUMNS: [&str; 26] = [
    "kind",
    "camera",
    "sequence",
    "driver_ns",
    "flags",
    "dequeue_ns",
    "buffer_index",
    "storage",
    "held",
    "unrequeued",
    "bytesused",
    "admitted",
    "planes",
    "frameset",
    "complete",
    "members",
    "anchor_ns",
    "member_ns",
    "wait_ns",
    "member_sequences",
    "dropped_rows",
    "mono_ns",
    "tolerance_ns",
    "max_wait_ns",
    "capture_sync",
    "resyncs",
];

fn column(name: &str) -> usize { COLUMNS.iter().position(|&c| c == name).expect("known CSV column") }


/// Buffer disposition recorded without retaining pixels.
#[derive(Clone, Copy, Debug)]
pub enum Storage {
    /// Published using a driver-buffer lease.
    Leased,
    /// Published as an owned copy.
    Copied,
    /// Driver buffer rejected before publication.
    Rejected,
    /// Frame deliberately discarded during resynchronization.
    Discarded,
}

impl Storage {
    fn as_str(self) -> &'static str {
        match self { Self::Leased => "leased", Self::Copied => "copied", Self::Rejected => "rejected", Self::Discarded => "discarded" }
    }
}

// Fixed-size records contain no image/lease and require no formatting on capture threads.
#[derive(Debug)]
enum Row {
    Discard { camera: usize, sequence: u64, timestamp: i64 },
    Sync {
        mono: i64,
        state: crate::source::CaptureSync,
        resyncs: u32,
    },
    Frame {
        camera: usize,
        meta: DequeueMeta,
        storage: Storage,
        held: u32,
        unrequeued: Option<u32>,
        admitted: bool,
    },
    Emit {
        index: u64,
        anchor: i64,
        stamps: [Option<i64>; 6],
        sequences: [Option<u64>; 6],
        wait: u64,
    },
    Start {
        mono: i64,
        tolerance: i64,
        max_wait: i64,
    },
}

/// A nonblocking sender; full/disconnected channels count the rejected row.
#[derive(Clone, Debug)]
pub struct DiagnosticSink {
    tx: mpsc::SyncSender<Row>,
    dropped: Arc<AtomicU64>,
    closed: Arc<AtomicBool>,
}

impl DiagnosticSink {
    /// Record a sync transition or trigger recovery attempt on the capture clock.
    pub fn sync(&self, mono: i64, state: crate::source::CaptureSync, resyncs: u32) {
        self.send(Row::Sync {
            mono,
            state,
            resyncs,
        });
    }
    fn send(&self, row: Row) {
        if self.closed.load(Ordering::Acquire) || self.tx.try_send(row).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of diagnostic rows lost so far, unrelated to capture frame drops.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Record one dequeue, including a rejected buffer before its error is returned.
    pub fn frame(
        &self,
        camera: usize,
        meta: DequeueMeta,
        storage: Storage,
        held: u32,
        unrequeued: Option<u32>,
        admitted: bool,
    ) {
        self.send(Row::Frame {
            camera,
            meta,
            storage,
            held,
            unrequeued,
            admitted,
        });
    }

    /// Mark a previously dequeued source-held frame as discarded during recovery.
    pub fn discard(&self, camera: usize, sequence: u64, timestamp: i64) {
        self.send(Row::Discard { camera, sequence, timestamp });
    }

    /// Record one actual matcher emission; fixed arrays never retain pixel buffers.
    pub fn emit(
        &self,
        index: u64,
        anchor: i64,
        stamps: [Option<i64>; 6],
        sequences: [Option<u64>; 6],
        wait: u64,
    ) {
        self.send(Row::Emit {
            index,
            anchor,
            stamps,
            sequences,
            wait,
        });
    }

    /// Record the monotonic stream-start anchor and matcher settings.
    pub fn start(&self, mono: i64, tolerance: i64, max_wait: i64) {
        self.send(Row::Start {
            mono,
            tolerance,
            max_wait,
        });
    }
}

/// Final CSV accounting, also included in the CLI summary JSON.
#[derive(Debug, serde::Serialize)]
pub struct DiagnosticStats {
    /// Data rows written, excluding header and final accounting row.
    pub written: u64,
    /// Rows rejected by the bounded channel.
    pub dropped: u64,
}

/// Background CSV writer. Explicit finish also works when a capture sender is stuck.
pub struct DiagnosticLog {
    sink: Option<DiagnosticSink>,
    worker: Option<JoinHandle<io::Result<DiagnosticStats>>>,
}

impl DiagnosticLog {
    /// Create a new log (never overwrite an earlier run) and start its bounded writer.
    /// # Errors
    /// File creation, initial header write or thread creation failed.
    pub fn open(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
        writeln!(file, "{}", COLUMNS.join(","))?;
        file.flush()?;
        let (tx, rx) = mpsc::sync_channel(8192);
        let dropped = Arc::new(AtomicU64::new(0));
        let count = dropped.clone();
        let closed = Arc::new(AtomicBool::new(false));
        let closing = closed.clone();
        let worker = std::thread::Builder::new()
            .name("rl-frame-log".into())
            .spawn(move || {
                let mut written = 0;
                let mut flushed = Instant::now();
                loop {
                    let row = if closing.load(Ordering::Acquire) {
                        match rx.try_recv() {
                            Ok(row) => Ok(row),
                            Err(_) => break,
                        }
                    } else {
                        rx.recv_timeout(Duration::from_millis(250))
                    };
                    match row {
                        Ok(row) => {
                            let mut cells: [String; 26] = std::array::from_fn(|_| String::new());
                            match row {
                                Row::Discard { camera, sequence, timestamp } => {
                                    cells[column("kind")] = "discard".into();
                                    cells[column("camera")] = camera.to_string();
                                    cells[column("sequence")] = sequence.to_string();
                                    cells[column("driver_ns")] = timestamp.to_string();
                                    cells[column("storage")] = Storage::Discarded.as_str().into();
                                }
                                Row::Sync {
                                    mono,
                                    state,
                                    resyncs,
                                } => {
                                    cells[column("kind")] = "sync".into();
                                    cells[column("mono_ns")] = mono.to_string();
                                    cells[column("capture_sync")] = state.as_str().into();
                                    cells[column("resyncs")] = resyncs.to_string();
                                }
                                Row::Frame {
                                    camera,
                                    meta,
                                    storage,
                                    held,
                                    unrequeued,
                                    admitted,
                                } => {
                                    cells[column("kind")] = "frame".into();
                                    cells[column("camera")] = camera.to_string();
                                    cells[column("sequence")] = meta.sequence.to_string();
                                    cells[column("driver_ns")] = meta.timestamp_ns.to_string();
                                    cells[column("flags")] = meta.flags.to_string();
                                    cells[column("dequeue_ns")] = meta.dequeue_ns.to_string();
                                    cells[column("buffer_index")] = meta.index.to_string();
                                    cells[column("storage")] = storage.as_str().into();
                                    cells[column("held")] = held.to_string();
                                    cells[column("unrequeued")] =
                                        unrequeued.map(|x| x.to_string()).unwrap_or_default();
                                    cells[column("bytesused")] = meta.bytesused[..meta.planes as usize]
                                        .iter()
                                        .map(u32::to_string)
                                        .collect::<Vec<_>>()
                                        .join(";");
                                    cells[column("admitted")] = admitted.to_string();
                                    cells[column("planes")] = meta.planes.to_string();
                                }
                                Row::Emit {
                                    index,
                                    anchor,
                                    stamps,
                                    sequences,
                                    wait,
                                } => {
                                    cells[column("kind")] = "emit".into();
                                    cells[column("frameset")] = index.to_string();
                                    cells[column("complete")] = stamps.iter().all(Option::is_some).to_string();
                                    cells[column("members")] = (0..6)
                                        .filter(|&c| stamps[c].is_some())
                                        .map(|c| c.to_string())
                                        .collect::<Vec<_>>()
                                        .join(";");
                                    cells[column("anchor_ns")] = anchor.to_string();
                                    cells[column("member_ns")] = stamps
                                        .iter()
                                        .flatten()
                                        .map(i64::to_string)
                                        .collect::<Vec<_>>()
                                        .join(";");
                                    cells[column("wait_ns")] = wait.to_string();
                                    cells[column("member_sequences")] = sequences
                                        .iter()
                                        .flatten()
                                        .map(u64::to_string)
                                        .collect::<Vec<_>>()
                                        .join(";");
                                }
                                Row::Start {
                                    mono,
                                    tolerance,
                                    max_wait,
                                } => {
                                    cells[column("kind")] = "start".into();
                                    cells[column("mono_ns")] = mono.to_string();
                                    cells[column("tolerance_ns")] = tolerance.to_string();
                                    cells[column("max_wait_ns")] = max_wait.to_string();
                                }
                            }
                            writeln!(file, "{}", cells.join(","))?;
                            written += 1;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    if flushed.elapsed() >= Duration::from_millis(250) {
                        file.flush()?;
                        flushed = Instant::now();
                    }
                }
                let stats = DiagnosticStats {
                    written,
                    dropped: count.load(Ordering::Relaxed),
                };
                let mut cells: [String; 26] = std::array::from_fn(|_| String::new());
                cells[column("kind")] = "end".into();
                cells[column("dropped_rows")] = stats.dropped.to_string();
                writeln!(file, "{}", cells.join(","))?;
                file.flush()?;
                Ok(stats)
            })?;
        eprintln!(
            "robocap-live: frame diagnostics {} (bounded 8192 rows)",
            path.display()
        );
        Ok(Self {
            sink: Some(DiagnosticSink {
                tx,
                dropped,
                closed,
            }),
            worker: Some(worker),
        })
    }

    /// Sender to attach to capture and matcher threads.
    pub fn sink(&self) -> DiagnosticSink {
        self.sink.as_ref().expect("active log").clone()
    }

    /// Drain accepted rows, write the loss count, and report any writer failure.
    /// # Errors
    /// CSV write/flush failed or writer panicked.
    pub fn finish(mut self) -> io::Result<DiagnosticStats> {
        self.join()
    }

    fn join(&mut self) -> io::Result<DiagnosticStats> {
        if let Some(sink) = self.sink.take() {
            sink.closed.store(true, Ordering::Release);
        }
        let stats = self
            .worker
            .take()
            .expect("active writer")
            .join()
            .map_err(|_| io::Error::other("diagnostic writer panicked"))??;
        eprintln!(
            "robocap-live: frame diagnostics written {} dropped {}",
            stats.written, stats.dropped
        );
        Ok(stats)
    }
}

impl Drop for DiagnosticLog {
    fn drop(&mut self) {
        if self.worker.is_some()
            && let Err(error) = self.join()
        {
            eprintln!("robocap-live: frame diagnostics FAILED: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finish_does_not_wait_for_a_stuck_capture_sender() -> Result<(), Box<dyn std::error::Error>> {
        let path =
            std::env::temp_dir().join(format!("split-fix-stuck-sender-{}.csv", std::process::id()));
        let log = DiagnosticLog::open(&path)?;
        let sink = log.sink();
        sink.start(1, 3, 100);
        let (done, wait) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            let _ = done.send(log.finish());
        });
        let result = wait.recv_timeout(Duration::from_secs(1));
        drop(sink);
        worker.join().unwrap();
        std::fs::remove_file(path)?;
        assert_eq!(result??.written, 1);
        Ok(())
    }

    #[test]
    fn a_full_or_closed_diagnostic_channel_counts_loss_without_waiting() {
        let (tx, rx) = mpsc::sync_channel(1);
        let sink = DiagnosticSink {
            tx,
            dropped: Arc::new(AtomicU64::new(0)),
            closed: Arc::new(AtomicBool::new(false)),
        };
        sink.start(1, 3, 100);
        sink.start(2, 3, 100);
        assert_eq!(sink.dropped(), 1);
        drop(rx);
        sink.start(3, 3, 100);
        assert_eq!(sink.dropped(), 2);
    }

    #[test]
    fn csv_preserves_error_flags_and_matcher_members_and_finishes_with_loss_count()
    -> std::io::Result<()> {
        let path = std::env::temp_dir().join(format!("split-diag-{}.csv", std::process::id()));
        let log = DiagnosticLog::open(&path)?;
        let sink = log.sink();
        sink.frame(
            2,
            DequeueMeta {
                index: 7,
                sequence: 41,
                flags: 0x2040,
                timestamp_ns: 100,
                dequeue_ns: 300,
                planes: 1,
                bytesused: [123, 0, 0, 0, 0, 0, 0, 0],
            },
            Storage::Rejected,
            4,
            None,
            false,
        );
        sink.emit(
            9,
            100,
            [Some(100), None, Some(102), None, None, None],
            [Some(40), None, Some(41), None, None, None],
            10,
        );
        drop(sink);
        assert_eq!(log.finish()?.dropped, 0);
        let csv = std::fs::read_to_string(&path)?;
        let rows = csv
            .lines()
            .map(|line| line.split(',').collect::<Vec<_>>())
            .collect::<Vec<_>>();
        assert!(rows.iter().all(|row| row.len() == rows[0].len()));
        assert_eq!(
            &rows[1][..10],
            &[
                "frame", "2", "41", "100", "8256", "300", "7", "rejected", "4", ""
            ]
        );
        assert_eq!(
            &rows[2][13..19],
            &["9", "false", "0;2", "100", "100;102", "10"]
        );
        assert_eq!(rows.last().unwrap()[0], "end");
        std::fs::remove_file(path)?;
        Ok(())
    }
}
