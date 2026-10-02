//! robocap-panel: a small web panel on the cap (port 8090). It starts and stops the live run and shows the device and the
//! pipeline: temperatures, CPU per core and per cluster, NPU/GPU/DDR/VPU load and clocks, battery and charger, Wi-Fi and its
//! throughput, memory, disk, the vendor recorder, and robocap-live's 1 Hz status line.
//!
//! One embedded page and a JSON API, std + serde_json only, so the binary stays small on the cap's 14 GB root:
//! - `GET /` the page; `GET /api/status` one JSON snapshot (cumulative counters; the page turns them into rates);
//! - `GET /api/log` the run log's last lines; `POST /api/start` (form fields, see [`run::StartRequest`]); `POST /api/stop`.
//!
//! Start follows the same handoff + `run/live.{pid,log,cmd}` protocol as `scripts/start-live.sh`: `handoff-run.sh` (pauses the
//! vendor recorder, always restores it, stops at 85 °C) around `bin/robocap-live`, detached in its own session, with the run
//! files written the same way so `scripts/stop.sh` works on it too. The robocap-live command line is the panel's own, built
//! from the form ([`run::StartRequest::command`]); it is not the one start-live.sh builds. Before it starts, it applies robocap-guard's
//! default power limits and busy checks on the cap itself. Stop sends SIGINT to the run's robocap-live (never to the vendor
//! recorder, never by name). Start, stop and the run's end share one record ([`run::Run`]). Nothing starts at boot.
//!
//! Usage: robocap-panel [--port 8090] [--bind 0.0.0.0] [--root /root/robocap-live]

use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

mod run;

use run::{Owner, Phase, Run, StartRequest, alive, live_processes, phase, run_pid};

const PAGE: &str = include_str!("panel.html");
/// The regmap file of the bq25790 charger. Only registers 0x19-0x21 are read (as robocap-guard does): 0x22-0x27 clear on read.
const CHARGER_REGMAP: &str = "/sys/kernel/debug/regmap/6-006b/registers";
/// Start refuses below these (robocap-guard's defaults). `/api/status` hands every limit to the page, which colours from them.
const MIN_INPUT_LIMIT_MA: u32 = 2000;
const MIN_VBAT_V: f64 = 7.9;
/// handoff-run.sh stops a run at this temperature (its `temp_limit_mc`)...
const STOP_TEMP_C: f64 = 85.0;
/// ...and refuses to start above it minus 10 °C; say so before trying.
const MAX_START_TEMP_C: f64 = STOP_TEMP_C - 10.0;
const MIN_FREE_BYTES: u64 = 1 << 30;

struct Panel {
    root: PathBuf,
    /// The run this panel starts and stops.
    run: Arc<Run>,
    /// `df -k /` for the page, re-read every 30 s (start's own check reads it fresh).
    disk: Cached<(Option<u64>, Option<u64>)>,
    /// `iw dev wlan0 link`, re-read every 5 s.
    wifi_link: Cached<String>,
}

/// A fact that costs a child process, re-read at most every `max_age` (the page polls every second, from every open tab).
struct Cached<T> {
    max_age: Duration,
    slot: Mutex<Option<(Instant, T)>>,
}

impl<T: Clone> Cached<T> {
    fn new(max_age: Duration) -> Self {
        Self { max_age, slot: Mutex::new(None) }
    }

    /// The cached value while it is younger than `max_age`, else a new one from `read`.
    fn get(&self, read: impl FnOnce() -> T) -> T {
        let Ok(mut slot) = self.slot.lock() else { return read() };
        match slot.as_ref() {
            Some((at, value)) if at.elapsed() < self.max_age => value.clone(),
            _ => {
                let value = read();
                *slot = Some((Instant::now(), value.clone()));
                value
            }
        }
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut port, mut bind, mut root) = (8090u16, "0.0.0.0".to_string(), PathBuf::from("/root/robocap-live"));
    while let Some(arg) = args.next() {
        let value = args.next().unwrap_or_default();
        match arg.as_str() {
            "--port" => port = value.parse().unwrap_or(port),
            "--bind" => bind = value,
            "--root" => root = PathBuf::from(value),
            _ => {
                eprintln!("usage: robocap-panel [--port 8090] [--bind 0.0.0.0] [--root /root/robocap-live]");
                std::process::exit(2);
            }
        }
    }
    let listener = match TcpListener::bind((bind.as_str(), port)) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("robocap-panel: bind {bind}:{port}: {error}");
            std::process::exit(1);
        }
    };
    eprintln!("robocap-panel: http://{bind}:{port}/ (root {})", root.display());
    let panel = Arc::new(Panel {
        root,
        run: Arc::default(),
        disk: Cached::new(Duration::from_secs(30)),
        wifi_link: Cached::new(Duration::from_secs(5)),
    });
    for stream in listener.incoming().flatten() {
        let panel = panel.clone();
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
            if let Err(error) = handle(stream, &panel) {
                eprintln!("robocap-panel: {error}");
            }
        });
    }
}

/// One HTTP/1.1 request (Connection: close).
fn handle(mut stream: TcpStream, panel: &Arc<Panel>) -> std::io::Result<()> {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            return Ok(());
        }
        buffer.extend_from_slice(&chunk[..n]);
        if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
        if buffer.len() > 16 * 1024 {
            return respond(&mut stream, 431, "text/plain", "request head too large");
        }
    };
    let head = String::from_utf8_lossy(&buffer[..head_end]).to_string();
    let mut lines = head.lines();
    let mut request = lines.next().unwrap_or_default().split_whitespace();
    let (method, path) = (request.next().unwrap_or_default().to_string(), request.next().unwrap_or_default().to_string());
    let header = |name: &str| {
        head.lines().skip(1).find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.trim().eq_ignore_ascii_case(name).then(|| value.trim().to_string())
        })
    };
    let length: usize = header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0).min(64 * 1024);
    while buffer.len() < head_end + length {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8_lossy(&buffer[head_end..(head_end + length).min(buffer.len())]).to_string();
    if method == "POST" && !same_origin(header("origin").as_deref(), header("host").as_deref()) {
        return respond(&mut stream, 403, "text/plain", "cross-origin request refused");
    }
    match (method.as_str(), path.split('?').next().unwrap_or_default()) {
        ("GET", "/") => respond(&mut stream, 200, "text/html; charset=utf-8", PAGE),
        ("GET", "/api/status") => respond(&mut stream, 200, "application/json", &status(panel).to_string()),
        ("GET", "/api/log") => respond(&mut stream, 200, "text/plain; charset=utf-8", &log_tail(panel, 60)),
        ("POST", "/api/start") => {
            let result = StartRequest::parse(&body).and_then(|request| run::start(&panel.run, &panel.root, &request, || preflight(&panel.root)));
            respond_result(&mut stream, result)
        }
        ("POST", "/api/stop") => respond_result(&mut stream, run::stop(&panel.run, &panel.root)),
        _ => respond(&mut stream, 404, "text/plain", "not found"),
    }
}

/// A browser's POST must come from this page (a fetch from another site carries its own Origin); curl sends no Origin.
fn same_origin(origin: Option<&str>, host: Option<&str>) -> bool {
    match (origin, host) {
        (None, _) => true,
        (Some(origin), Some(host)) => origin.split_once("://").is_some_and(|(_, rest)| rest == host),
        (Some(_), None) => false,
    }
}

fn respond_result(stream: &mut TcpStream, result: Result<Value, String>) -> std::io::Result<()> {
    match result {
        Ok(value) => respond(stream, 200, "application/json", &json!({"ok": true, "result": value}).to_string()),
        Err(error) => respond(stream, 409, "application/json", &json!({"ok": false, "error": error}).to_string()),
    }
}

fn respond(stream: &mut TcpStream, code: u16, kind: &str, body: &str) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        403 => "Forbidden",
        404 => "Not Found",
        409 => "Conflict",
        _ => "Error",
    };
    write!(stream, "HTTP/1.1 {code} {reason}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len())?;
    stream.write_all(body.as_bytes())
}

fn read_trim(path: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(path).ok().map(|text| text.trim().to_string())
}

fn read_number(path: impl AsRef<Path>) -> Option<f64> {
    read_trim(path)?.parse().ok()
}

/// devfreq `load`: "37@300000000Hz" -> (37 %, 300 MHz).
fn parse_devfreq_load(text: &str) -> Option<(u32, f64)> {
    let (load, hz) = text.trim().split_once('@')?;
    Some((load.trim().parse().ok()?, hz.trim().trim_end_matches("Hz").parse::<f64>().ok()? / 1e6))
}

/// rknpu `load`: "NPU load:  Core0:  9%, Core1: 22%, Core2: 21%," -> [9, 22, 21].
fn parse_npu_load(text: &str) -> Vec<u32> {
    text.split("Core").skip(1).filter_map(|part| part.split(':').nth(1)?.trim().split('%').next()?.trim().parse().ok()).collect()
}

/// bq25790 registers 0x19-0x1b as the regmap file prints them ("19: 01" ...): (ICO input limit mA, IINDPM active).
fn parse_charger(text: &str) -> (Option<u32>, Option<bool>) {
    let register = |name: &str| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == name).then(|| u32::from_str_radix(value.trim(), 16).ok()).flatten()
        })
    };
    let limit = register("19").zip(register("1a")).map(|(high, low)| ((high % 2) * 256 + low) * 10);
    (limit, register("1b").map(|status| status & 0x80 != 0))
}

/// Registers 0x19-0x21 only: lines of 7 bytes ("19: 01\n"), as robocap-guard's `dd bs=7 skip=25 count=9`.
fn read_charger() -> (Option<u32>, Option<bool>) {
    let mut text = String::new();
    let read = fs::File::open(CHARGER_REGMAP)
        .and_then(|mut file| file.seek(SeekFrom::Start(25 * 7)).and_then(|_| file.take(9 * 7).read_to_string(&mut text)));
    if read.is_err() {
        return (None, None);
    }
    parse_charger(&text)
}

/// One process: pid, session, state, the first word of its command line, and the whole line.
struct Process {
    pid: u32,
    session: u32,
    state: char,
    program: String,
    args: String,
}

fn processes() -> Vec<Process> {
    let Ok(entries) = fs::read_dir("/proc") else { return Vec::new() };
    entries
        .flatten()
        .filter_map(|entry| {
            let pid: u32 = entry.file_name().to_str()?.parse().ok()?;
            let stat = fs::read_to_string(entry.path().join("stat")).ok()?;
            // "pid (comm) state ppid pgrp session ..."; comm may hold spaces and parentheses.
            let mut rest = stat.get(stat.rfind(')')? + 1..)?.split_whitespace();
            let state = rest.next()?.chars().next()?;
            let session: u32 = rest.nth(2)?.parse().ok()?;
            let raw = fs::read(entry.path().join("cmdline")).ok()?;
            let words: Vec<String> = raw.split(|&b| b == 0).filter(|w| !w.is_empty()).map(|w| String::from_utf8_lossy(w).to_string()).collect();
            let program = words.first()?.clone();
            Some(Process { pid, session, state, program, args: words.join(" ") })
        })
        .collect()
}

/// robocap-guard's notion of "busy": transfers, fstrim, or another heavy job.
fn busy(processes: &[Process]) -> Vec<String> {
    const TRANSFERS: [&str; 10] = ["tar", "gzip", "gunzip", "zcat", "pigz", "unpigz", "scp", "sftp-server", "rsync", "dd"];
    const HEAVY: [&str; 9] = ["nets_bench", "robocap-live", "rknn", "gst-launch", "stress", "fitbench", "cold_bench", "npu-bench", "slam_bench"];
    processes
        .iter()
        .filter_map(|p| {
            let name = p.program.rsplit('/').next().unwrap_or_default();
            let kind = if TRANSFERS.contains(&name) {
                "transfer"
            } else if name == "fstrim" {
                "fstrim"
            } else if HEAVY.iter().any(|h| p.args.contains(h)) && !p.args.contains("robocap-panel") {
                "heavy"
            } else {
                return None;
            };
            Some(format!("{kind} {}: {}", p.pid, p.args.chars().take(120).collect::<String>()))
        })
        .collect()
}

/// The last `max_bytes` of the current run log.
fn log_text(panel: &Panel, max_bytes: u64) -> String {
    let Some(path) = read_trim(panel.root.join("run/live.log")) else { return String::new() };
    let Ok(mut file) = fs::File::open(path) else { return String::new() };
    let length = file.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = file.seek(SeekFrom::Start(length.saturating_sub(max_bytes)));
    let mut bytes = Vec::new();
    let _ = file.read_to_end(&mut bytes);
    String::from_utf8_lossy(&bytes).to_string()
}

fn log_tail(panel: &Panel, lines: usize) -> String {
    let text = log_text(panel, 64 * 1024);
    let all: Vec<&str> = text.lines().collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

/// The newest log line that starts with `prefix` (after leading spaces).
fn last_line<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    text.lines().rev().find(|line| line.trim_start().starts_with(prefix))
}

fn status(panel: &Panel) -> Value {
    let thermal: Vec<Value> = (0..16)
        .map_while(|zone| {
            let base = format!("/sys/class/thermal/thermal_zone{zone}");
            Some(json!({"name": read_trim(format!("{base}/type"))?, "c": read_number(format!("{base}/temp"))? / 1000.0}))
        })
        .collect();
    let cpu: Vec<Vec<u64>> = fs::read_to_string("/proc/stat")
        .unwrap_or_default()
        .lines()
        .filter(|line| line.starts_with("cpu") && line.as_bytes().get(3).is_some_and(u8::is_ascii_digit))
        .map(|line| line.split_whitespace().skip(1).take(8).filter_map(|v| v.parse().ok()).collect())
        .collect();
    let policies: Vec<Value> = ["policy0", "policy4", "policy6"]
        .iter()
        .filter_map(|p| {
            let base = format!("/sys/devices/system/cpu/cpufreq/{p}");
            Some(json!({"name": p, "cpus": read_trim(format!("{base}/related_cpus"))?, "mhz": read_number(format!("{base}/scaling_cur_freq"))? / 1e3}))
        })
        .collect();
    let devfreq = |name: &str| -> Value {
        let base = format!("/sys/class/devfreq/{name}");
        match read_trim(format!("{base}/load")).as_deref().and_then(parse_devfreq_load) {
            Some((load, mhz)) => json!({"load": load, "mhz": mhz}),
            None => json!({"load": null, "mhz": read_number(format!("{base}/cur_freq")).map(|hz| hz / 1e6)}),
        }
    };
    let supply = |name: &str, field: &str| read_number(format!("/sys/class/power_supply/{name}/{field}"));
    let (input_limit_ma, iindpm) = read_charger();
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mem_kb = |key: &str| meminfo.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok());
    let (disk_total, disk_free) = panel.disk.get(|| disk_bytes("/"));
    let iw = panel.wifi_link.get(|| {
        Command::new("iw").args(["dev", "wlan0", "link"]).stderr(Stdio::null()).output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default()
    });
    let iw_field = |key: &str| iw.lines().find_map(|l| l.trim().strip_prefix(key).map(|v| v.trim().to_string()));
    let procs = processes();
    let recorder = procs.iter().find(|p| p.program.ends_with("omni-specs.bin"));
    let launcher = procs.iter().find(|p| p.args.contains("frodobots.sh"));
    let record = panel.run.snapshot();
    let pid = record.session(&panel.root);
    let running = pid.is_some_and(alive);
    let log = log_text(panel, 32 * 1024);
    let live_running = running && pid.is_some_and(|session| !live_processes(&panel.root, session).is_empty());
    let ended = log.contains("robocap-live: stopping") || log.contains("robocap-live: capture stopped") || log.contains("launcher resumed");
    let run_phase = if record.owner == Owner::Starting { Phase::Starting } else { phase(running, live_running, record.stop_requested || ended) };
    let uptime = read_trim("/proc/uptime").and_then(|u| u.split_whitespace().next()?.parse::<f64>().ok());
    json!({
        "time": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0),
        "uptime_s": uptime,
        "host": read_trim("/proc/sys/kernel/hostname"),
        "device": fs::read_to_string(panel.root.join("rig.json")).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|r| r.get("device").cloned()),
        "thermal": thermal,
        "cpu": cpu,
        "cpufreq": policies,
        "npu": {"load": parse_npu_load(&read_trim("/sys/kernel/debug/rknpu/load").unwrap_or_default()), "mhz": read_number("/sys/kernel/debug/rknpu/freq").map(|hz| hz / 1e6), "mv": read_number("/sys/kernel/debug/rknpu/volt").map(|uv| uv / 1e3)},
        "gpu": devfreq("fb000000.gpu"),
        "ddr": devfreq("dmc"),
        "vpu": [devfreq("fdbd0000.rkvenc-core"), devfreq("fdbe0000.rkvenc-core")],
        "power": {
            "battery_v": supply("bq25790-battery", "voltage_now").map(|uv| uv / 1e6),
            "battery_a": supply("bq25790-battery", "current_now").map(|ua| ua / 1e6),
            "input_v": supply("bq25790-charger", "voltage_now").map(|uv| uv / 1e6),
            "input_a": supply("bq25790-charger", "current_now").map(|ua| ua / 1e6),
            "status": read_trim("/sys/class/power_supply/bq25790-charger/status"),
            "input_limit_ma": input_limit_ma,
            "iindpm": iindpm,
        },
        "mem_kb": {"total": mem_kb("MemTotal:"), "available": mem_kb("MemAvailable:")},
        "disk": {"total": disk_total, "free": disk_free},
        "net": {
            "tx_bytes": read_number("/sys/class/net/wlan0/statistics/tx_bytes"),
            "rx_bytes": read_number("/sys/class/net/wlan0/statistics/rx_bytes"),
            "ssid": iw_field("SSID:"), "freq": iw_field("freq:"), "signal": iw_field("signal:"), "tx_bitrate": iw_field("tx bitrate:"),
        },
        "vendor": {
            "recorder_pid": recorder.map(|p| p.pid),
            "launcher_state": launcher.map(|p| if p.state == 'T' { "paused (our run)" } else { "running" }),
        },
        "limits": {
            "min_free_bytes": MIN_FREE_BYTES,
            "min_input_limit_ma": MIN_INPUT_LIMIT_MA,
            "min_vbat_v": MIN_VBAT_V,
            "max_start_temp_c": MAX_START_TEMP_C,
            "stop_temp_c": STOP_TEMP_C,
        },
        "busy": busy(&procs).into_iter().filter(|b| !running || !b.contains("robocap-live")).collect::<Vec<_>>(),
        "run": {
            "running": running,
            "phase": run_phase.as_str(),
            "pid": pid,
            "cmd": read_trim(panel.root.join("run/live.cmd")),
            "log": read_trim(panel.root.join("run/live.log")),
            "status": last_line(&log, "[ ").or_else(|| last_line(&log, "[")).filter(|l| l.contains(" src ")),
            "power": last_line(&log, "power:"),
            "threads": last_line(&log, "threads:"),
            "live": last_line(&log, "robocap-live: camera fps").or_else(|| last_line(&log, "robocap-live: capture stopped")),
            "handoff": last_line(&log, "[handoff"),
            "error": last_line(&log, "Error:"),
            "last_exit": record.last_exit,
            "request": record.last_request,
        },
    })
}

/// `df -k <path>`: (total, free) bytes.
fn disk_bytes(path: &str) -> (Option<u64>, Option<u64>) {
    let output = Command::new("df").args(["-k", path]).stderr(Stdio::null()).output().map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let fields: Vec<u64> = output.lines().nth(1).map(|l| l.split_whitespace().skip(1).take(3).filter_map(|v| v.parse().ok()).collect()).unwrap_or_default();
    (fields.first().map(|k| k * 1024), fields.get(2).map(|k| k * 1024))
}

/// robocap-guard's preflight, read on the cap: refusals as text (empty = go).
fn preflight(root: &Path) -> Vec<String> {
    let mut refusals = Vec::new();
    if run_pid(root).is_some_and(alive) {
        refusals.push("a run is already going (stop it first)".to_string());
    }
    refusals.extend(busy(&processes()).into_iter().map(|b| format!("busy: {b}")));
    let (limit, iindpm) = read_charger();
    match limit {
        Some(ma) if ma < MIN_INPUT_LIMIT_MA => refusals.push(format!("charger input limit {ma} mA is below {MIN_INPUT_LIMIT_MA} mA (USB replug: reboot restores 2.3 A)")),
        None => refusals.push("cannot read the charger's input limit".to_string()),
        _ => {}
    }
    if iindpm == Some(true) {
        refusals.push("the charger is in input current regulation (the battery carries part of the load)".to_string());
    }
    if let Some(volts) = read_number("/sys/class/power_supply/bq25790-battery/voltage_now").map(|uv| uv / 1e6)
        && volts < MIN_VBAT_V
    {
        refusals.push(format!("battery {volts:.2} V is below {MIN_VBAT_V} V"));
    }
    let hottest = (0..16).map_while(|z| read_number(format!("/sys/class/thermal/thermal_zone{z}/temp"))).fold(f64::MIN, f64::max) / 1000.0;
    if hottest >= MAX_START_TEMP_C {
        refusals.push(format!("SoC {hottest:.1} °C: handoff-run.sh starts below {MAX_START_TEMP_C} °C"));
    }
    if disk_bytes("/").1.is_some_and(|free| free < MIN_FREE_BYTES) {
        refusals.push(format!("less than {} GiB free on /", MIN_FREE_BYTES >> 30));
    }
    refusals
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_files_parse_as_the_cap_writes_them() {
        assert_eq!(parse_devfreq_load("37@300000000Hz"), Some((37, 300.0)));
        assert_eq!(parse_devfreq_load("garbage"), None);
        assert_eq!(parse_npu_load("NPU load:  Core0:  9%, Core1: 22%, Core2: 21%,"), vec![9, 22, 21]);
        // Cap B, 2026-10-01 16:5x: robocap-guard read 2610 mA, IINDPM 0 from these registers.
        assert_eq!(parse_charger("19: 01\n1a: 05\n1b: 0f\n1c: e7\n"), (Some(2610), Some(false)));
        assert_eq!(parse_charger("19: 00\n1a: 32\n1b: 8f\n"), (Some(500), Some(true)));
    }

    #[test]
    fn only_this_page_may_post() {
        assert!(same_origin(None, Some("192.0.2.10:8090")), "curl");
        assert!(same_origin(Some("http://192.0.2.10:8090"), Some("192.0.2.10:8090")));
        assert!(!same_origin(Some("https://evil.example"), Some("192.0.2.10:8090")));
    }

    #[test]
    fn a_cached_fact_is_read_again_only_once_it_is_old() {
        let reads = std::cell::Cell::new(0);
        let read = || {
            reads.set(reads.get() + 1);
            reads.get()
        };
        let fresh = Cached::new(Duration::from_secs(60));
        assert_eq!((fresh.get(read), fresh.get(read)), (1, 1));
        let stale = Cached::new(Duration::ZERO);
        assert_eq!((stale.get(read), stale.get(read)), (2, 3));
    }

    #[test]
    fn the_status_line_is_the_newest_one() {
        let log = "[   1.0 s] src 30/s a\n           power: x\n[   2.0 s] src 30/s b\nrobocap-live: camera fps [30] live: 5.0 s\n";
        assert_eq!(last_line(log, "[ ").map(|l| l.contains(" b")), Some(true));
        assert_eq!(last_line(log, "power:").map(str::trim), Some("power: x"));
    }
}
