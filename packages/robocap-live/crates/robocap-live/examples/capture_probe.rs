//! Probe: is the rkisp mainpath capture buffer cached memory for CPU readers, which buffer modes does the driver offer, and does
//! the stream keep 30 fps when buffers stay out of the queue for a while?
//!
//! `capture_probe --mode mmap|mmap-noncoherent|dmabuf:/dev/dma_heap/<heap> [--seconds 40] [--hold N] [--buffers 8] [--json out]`
//!
//! Runs only on a cap, under `robocap-panel handoff` (the crate's device checks refuse anywhere else). It opens the six
//! mainpath cameras with the crate's [`Camera`] in the chosen mode, prints the REQBUFS capabilities and whether VIDIOC_EXPBUF and
//! a mapping of the exported dma-buf work, starts the crate's [`FrameTrigger`] at 30 fps and then:
//! - `--hold 0` (timing): camera 0's thread runs pinned to an A55 and camera 1's to an A76. Each of their frames times one
//!   pattern, in rotation: (0) a sequential checksum of the mapped luma, cold and again warm; (1) a sparse read shaped like the
//!   hands' remap (4 crops of 96x96, 4 bilinear taps per pixel), cold and warm; (2) today's memcpy, then both reads on the copy;
//!   (3) the sparse read, cold, through a mapping of the EXPBUF dma-buf. The other cameras do today's work (memcpy, requeue).
//! - `--hold N` (lifetime): every camera keeps its last N frames out of the queue, so a buffer goes back N frames after it came out
//!   (kornia-io's `MmapStream` rule with a reader N frames behind). Reports fps and sequence gaps.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::Parser;
use robocap_live::capture::CAMERA_DEVICES;
use robocap_live::capture::camera::{BufferInfo, Camera, CaptureMemory, LUMA_BYTES, LentBuffer};
use robocap_live::capture::device::{FrameTrigger, monotonic_ns, require_cap, require_vendor_recorder_stopped};
use robocap_live::sched::{detect_big_little, pin_current_thread};
use serde_json::{Value, json};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Parser)]
struct Args {
    /// mmap | mmap-noncoherent | dmabuf:/dev/dma_heap/<heap>
    #[arg(long, default_value = "mmap")]
    mode: String,
    /// Streaming time after the trigger starts.
    #[arg(long, default_value_t = 40)]
    seconds: u64,
    /// 0 = timing run; N > 0 = lifetime run, every camera holds its last N frames.
    #[arg(long, default_value_t = 0)]
    hold: usize,
    /// Buffers per camera.
    #[arg(long, default_value_t = 8)]
    buffers: u32,
    /// Write the results here as JSON.
    #[arg(long)]
    json: Option<PathBuf>,
}

const WIDTH: usize = 1920;
const CROP: usize = 96;
/// Crop centres over the 1080p plane; each crop samples a ~210 px square, rotated, as a hand crop does.
const CROP_CENTRES: [(f32, f32); 4] = [(420.0, 330.0), (1500.0, 360.0), (600.0, 760.0), (1320.0, 700.0)];
const PATTERNS: usize = 4;

static STOP: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

fn checksum(luma: &[u8]) -> u64 {
    let words = luma.chunks_exact(8);
    let rest = words.remainder();
    let sum = words.fold(0u64, |acc, word| acc.wrapping_add(u64::from_le_bytes(<[u8; 8]>::try_from(word).unwrap_or_default())));
    rest.iter().fold(sum, |acc, &b| acc.wrapping_add(u64::from(b)))
}

/// The hands' remap shape: 4 crops of 96x96, each output pixel a bilinear sample (4 taps) of a rotated, 2.2x scaled grid.
fn sparse_read(luma: &[u8], out: &mut [u8]) {
    let (cos, sin) = (0.35f32.cos() * 2.2, 0.35f32.sin() * 2.2);
    for (crop, &(cx, cy)) in CROP_CENTRES.iter().enumerate() {
        for v in 0..CROP {
            for u in 0..CROP {
                let (du, dv) = (u as f32 - 47.5, v as f32 - 47.5);
                let (x, y) = (cx + du * cos - dv * sin, cy + du * sin + dv * cos);
                let (x0, y0) = (x.floor(), y.floor());
                let (fx, fy) = (x - x0, y - y0);
                let i = y0 as usize * WIDTH + x0 as usize;
                let (a, b, c, d) = (f32::from(luma[i]), f32::from(luma[i + 1]), f32::from(luma[i + WIDTH]), f32::from(luma[i + WIDTH + 1]));
                let top = a + (b - a) * fx;
                let bottom = c + (d - c) * fx;
                out[crop * CROP * CROP + v * CROP + u] = (top + (bottom - top) * fy) as u8;
            }
        }
    }
}

fn micros(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e6
}

/// A read-only mapping of an exported dma-buf.
struct ExportMap {
    address: *mut libc::c_void,
    length: usize,
    _fd: std::os::fd::OwnedFd,
}

impl ExportMap {
    fn new(camera: &Camera, index: u32, length: usize) -> ProbeResult<Self> {
        use std::os::fd::AsRawFd;
        let fd = camera.export(index)?;
        // SAFETY: a fresh shared read-only mapping of a dma-buf we own; checked against MAP_FAILED.
        let address = unsafe { libc::mmap(std::ptr::null_mut(), length, libc::PROT_READ, libc::MAP_SHARED, fd.as_raw_fd(), 0) };
        if address == libc::MAP_FAILED {
            return Err(format!("mmap of the exported buffer {index}: {}", std::io::Error::last_os_error()).into());
        }
        Ok(Self { address, length, _fd: fd })
    }

    fn luma(&self) -> &[u8] {
        // SAFETY: the mapping is `length` bytes and lives as long as `self`. The probe reads it only while the camera has the
        // buffer dequeued, so the device does not write it meanwhile. (It reads from offset 0, not the plane's data offset: the
        // timing is what counts.)
        unsafe { std::slice::from_raw_parts(self.address.cast::<u8>(), LUMA_BYTES.min(self.length)) }
    }
}

impl Drop for ExportMap {
    fn drop(&mut self) {
        // SAFETY: our own mapping, unmapped once.
        unsafe { libc::munmap(self.address, self.length) };
    }
}

#[derive(Default)]
struct CameraStats {
    frames: u64,
    sequence_gaps: u64,
    max_gap: u64,
    first_ns: i64,
    last_ns: i64,
    latency_ms: Vec<f64>,
    min_queued: u32,
    /// (pattern name, microseconds per sample).
    timings: Vec<(&'static str, Vec<f64>)>,
    notes: Vec<String>,
}

impl CameraStats {
    fn time(&mut self, name: &'static str, us: f64) {
        match self.timings.iter_mut().find(|(n, _)| *n == name) {
            Some((_, values)) => values.push(us),
            None => self.timings.push((name, vec![us])),
        }
    }
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

struct Role {
    measure: bool,
    pin: Option<usize>,
    hold: usize,
    sync: bool,
}

fn camera_thread(device: Camera, role: Role, info: BufferInfo) -> ProbeResult<CameraStats> {
    if let Some(cpu) = role.pin {
        pin_current_thread(&[cpu])?;
    }
    let mut stats = CameraStats { min_queued: info.count, ..CameraStats::default() };
    let exports: Vec<ExportMap> = if role.measure && !role.sync {
        match (0..info.count).map(|index| ExportMap::new(&device, index, LUMA_BYTES * 3 / 2)).collect::<ProbeResult<Vec<_>>>() {
            Ok(maps) => maps,
            Err(error) => {
                stats.notes.push(format!("EXPBUF mapping: {error}"));
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    let mut copy = vec![0u8; LUMA_BYTES];
    let mut crops = vec![0u8; 4 * CROP * CROP];
    let mut held: VecDeque<LentBuffer> = VecDeque::new();
    let mut previous: Option<u32> = None;
    let mut sink = 0u64;
    while !STOP.load(Ordering::Relaxed) {
        let Some(buffer) = device.dequeue(200)? else { continue };
        let now = monotonic_ns()?;
        if let Some(last) = previous {
            let gap = u64::from(buffer.sequence.wrapping_sub(last).wrapping_sub(1));
            stats.sequence_gaps += gap;
            stats.max_gap = stats.max_gap.max(gap);
        }
        previous = Some(buffer.sequence);
        if stats.frames == 0 {
            stats.first_ns = buffer.timestamp_ns;
        }
        stats.last_ns = buffer.timestamp_ns;
        stats.latency_ms.push((now - buffer.timestamp_ns) as f64 / 1e6);
        stats.frames += 1;
        stats.min_queued = stats.min_queued.min(info.count.saturating_sub(held.len() as u32 + 1));
        if role.hold > 0 {
            held.push_back(buffer);
            while held.len() > role.hold {
                if let Some(oldest) = held.pop_front() {
                    oldest.queue()?;
                }
            }
            continue;
        }
        let t = Instant::now();
        buffer.sync_read(true)?;
        if role.sync {
            stats.time("sync_start", micros(t));
        }
        let luma = buffer.luma();
        let pattern = if role.measure { stats.frames as usize % PATTERNS } else { 2 };
        match pattern {
            0 => {
                let t = Instant::now();
                sink = sink.wrapping_add(checksum(luma));
                stats.time("mapped_seq_cold", micros(t));
                let t = Instant::now();
                sink = sink.wrapping_add(checksum(luma));
                stats.time("mapped_seq_warm", micros(t));
            }
            1 => {
                let t = Instant::now();
                sparse_read(luma, &mut crops);
                stats.time("mapped_sparse_cold", micros(t));
                let t = Instant::now();
                sparse_read(luma, &mut crops);
                stats.time("mapped_sparse_warm", micros(t));
            }
            2 => {
                let t = Instant::now();
                copy.copy_from_slice(luma);
                if role.measure {
                    stats.time("memcpy", micros(t));
                    let t = Instant::now();
                    sink = sink.wrapping_add(checksum(&copy));
                    stats.time("copy_seq", micros(t));
                    let t = Instant::now();
                    sparse_read(&copy, &mut crops);
                    stats.time("copy_sparse", micros(t));
                }
            }
            _ => {
                if let Some(map) = exports.get(buffer.index() as usize) {
                    let t = Instant::now();
                    sparse_read(map.luma(), &mut crops);
                    stats.time("expbuf_sparse_cold", micros(t));
                }
            }
        }
        sink = sink.wrapping_add(u64::from(crops[0]));
        let t = Instant::now();
        buffer.sync_read(false)?;
        if role.sync {
            stats.time("sync_end", micros(t));
        }
        buffer.queue()?;
    }
    for buffer in held.drain(..) {
        buffer.queue()?;
    }
    drop(exports);
    std::hint::black_box(sink);
    drop(device);
    Ok(stats)
}

fn caps_names(caps: u32) -> String {
    const NAMES: [&str; 7] = ["MMAP", "USERPTR", "DMABUF", "REQUESTS", "ORPHANED_BUFS", "M2M_HOLD_CAPTURE_BUF", "MMAP_CACHE_HINTS"];
    let names: Vec<&str> = NAMES.iter().enumerate().filter(|(bit, _)| caps & (1 << bit) != 0).map(|(_, name)| *name).collect();
    format!("{caps:#x} [{}]", names.join(" "))
}

fn main() -> ProbeResult<()> {
    let args = Args::parse();
    let memory = match args.mode.as_str() {
        "mmap" => CaptureMemory::Mmap,
        "mmap-noncoherent" => CaptureMemory::MmapNonCoherent,
        other => match other.strip_prefix("dmabuf:") {
            Some(heap) => CaptureMemory::DmaBufHeap(heap.to_owned()),
            None => return Err(format!("unknown --mode {other}").into()),
        },
    };
    let sync = matches!(memory, CaptureMemory::DmaBufHeap(_));
    for signal in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: the handler only stores an atomic (async-signal-safe); the sigaction struct is zeroed, then the handler set.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_signal as *const () as libc::sighandler_t;
            libc::sigemptyset(&mut action.sa_mask);
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
    let cap = require_cap()?;
    require_vendor_recorder_stopped()?;
    let layout = detect_big_little().ok_or("not a big.LITTLE machine")?;
    let (little, big) = (*layout.little.last().ok_or("no A55")?, *layout.big.last().ok_or("no A76")?);
    let trigger = FrameTrigger::stopped()?;
    let mut cameras = Vec::new();
    for path in CAMERA_DEVICES {
        cameras.push(Camera::open_with(path, &memory, args.buffers)?);
    }
    let infos: Vec<BufferInfo> = cameras.iter().map(Camera::buffer_info).collect();
    for (path, info) in CAMERA_DEVICES.iter().zip(&infos) {
        println!(
            "{path}: count {} x {} bytes, memory_flags {:#x} | REQBUFS MMAP caps {} | DMABUF caps {}",
            info.count,
            info.buffer_bytes,
            info.memory_flags,
            caps_names(info.mmap_capabilities),
            caps_names(info.dmabuf_capabilities)
        );
    }
    let expbuf = if sync { "n/a (DMABUF mode)".to_owned() } else { cameras[0].export(0).map_or_else(|e| format!("FAILED: {e}"), |_| "ok".to_owned()) };
    println!("VIDIOC_EXPBUF on {} buffer 0: {expbuf}", CAMERA_DEVICES[0]);
    println!("cap {cap:?}; mode {memory:?}; {} buffers; hold {}; measuring cores: A55 cpu{little}, A76 cpu{big}", args.buffers, args.hold);
    let mut handles = Vec::new();
    for (camera, device) in cameras.into_iter().enumerate() {
        let measure = args.hold == 0 && camera < 2;
        let role = Role { measure, pin: measure.then_some(if camera == 0 { little } else { big }), hold: args.hold, sync };
        let info = infos[camera];
        handles.push(std::thread::Builder::new().name(format!("probe-cam-{camera}")).spawn(move || camera_thread(device, role, info))?);
    }
    trigger.start()?;
    let started = Instant::now();
    while !STOP.load(Ordering::Relaxed) && started.elapsed() < Duration::from_secs(args.seconds) && handles.iter().all(|h| !h.is_finished()) {
        std::thread::sleep(Duration::from_millis(100));
    }
    STOP.store(true, Ordering::SeqCst);
    let streamed = started.elapsed().as_secs_f64();
    drop(trigger);
    let mut cameras_json = Vec::new();
    let mut failures = Vec::new();
    for (camera, handle) in handles.into_iter().enumerate() {
        let mut stats = match handle.join() {
            Ok(Ok(stats)) => stats,
            Ok(Err(error)) => {
                failures.push(format!("camera {camera}: {error}"));
                continue;
            }
            Err(_) => {
                failures.push(format!("camera {camera}: thread panicked"));
                continue;
            }
        };
        let fps = if stats.frames > 1 { (stats.frames - 1) as f64 / ((stats.last_ns - stats.first_ns) as f64 / 1e9) } else { 0.0 };
        let latency = (percentile(&mut stats.latency_ms, 0.5), percentile(&mut stats.latency_ms, 0.9));
        println!(
            "camera {camera} ({}): {} frames, {fps:.2} fps, seq_gaps {} (max {}), dequeue latency p50 {:.2} ms p90 {:.2} ms, min queued {}",
            CAMERA_DEVICES[camera], stats.frames, stats.sequence_gaps, stats.max_gap, latency.0, latency.1, stats.min_queued
        );
        let mut timings = serde_json::Map::new();
        for (name, values) in &mut stats.timings {
            let n = values.len();
            let (p10, p50, p90, max) = (percentile(values, 0.1), percentile(values, 0.5), percentile(values, 0.9), percentile(values, 1.0));
            println!("  {name:20} n {n:4}  p10 {p10:8.1} us  p50 {p50:8.1} us  p90 {p90:8.1} us  max {max:8.1} us");
            timings.insert((*name).to_owned(), json!({"n": n, "p10_us": p10, "p50_us": p50, "p90_us": p90, "max_us": max}));
        }
        for note in &stats.notes {
            println!("  note: {note}");
        }
        cameras_json.push(json!({
            "camera": camera, "device": CAMERA_DEVICES[camera], "frames": stats.frames, "fps": fps, "sequence_gaps": stats.sequence_gaps,
            "max_gap": stats.max_gap, "latency_p50_ms": latency.0, "latency_p90_ms": latency.1, "min_queued": stats.min_queued,
            "timings": Value::Object(timings), "notes": stats.notes,
        }));
    }
    for failure in &failures {
        println!("FAILURE {failure}");
    }
    if let Some(path) = &args.json {
        let infos_json: Vec<Value> = infos
            .iter()
            .map(|i| json!({"count": i.count, "buffer_bytes": i.buffer_bytes, "memory_flags": i.memory_flags, "mmap_caps": i.mmap_capabilities, "dmabuf_caps": i.dmabuf_capabilities}))
            .collect();
        let out = json!({
            "mode": args.mode, "buffers": args.buffers, "hold": args.hold, "seconds": streamed, "cap": format!("{cap:?}"),
            "a55_cpu": little, "a76_cpu": big, "expbuf": expbuf, "buffer_info": infos_json, "cameras": cameras_json, "failures": failures,
        });
        std::fs::write(path, serde_json::to_string_pretty(&out)?)?;
    }
    if failures.is_empty() { Ok(()) } else { Err(format!("{} camera thread(s) failed", failures.len()).into()) }
}
