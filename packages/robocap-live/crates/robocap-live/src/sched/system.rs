//! The machine under the pipeline: CPU lists, the big.LITTLE layout, thread affinity and frequency hints, cpufreq caps, and the
//! once-per-second load samples (per-thread and per-core CPU, power supplies, NPU load, temperatures).

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use super::{SchedError, stage_error};

/// Parse a CPU list: `4-7`, `0,2-3`, or `none`/empty (no pinning).
///
/// # Errors
///
/// [`SchedError::CpuList`] when it does not parse.
pub fn parse_cpu_list(text: &str) -> Result<Option<Vec<usize>>, SchedError> {
    let text = text.trim();
    if text.is_empty() || text == "none" || text == "off" {
        return Ok(None);
    }
    let bad = || SchedError::CpuList(text.to_owned());
    let mut cpus = Vec::new();
    for part in text.split(',') {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?);
                if b < a {
                    return Err(bad());
                }
                cpus.extend(a..=b);
            }
            None => cpus.push(part.trim().parse().map_err(|_| bad())?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Ok(Some(cpus))
}

/// A big.LITTLE machine's cores, by `cpu_capacity` and cpufreq policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreLayout {
    /// One cpufreq policy of top-capacity cores, for SLAM: the last one (cpus 6-7 on the RK3588, on Cap B the best-binned pair at
    /// 1024 vs 1003 for 4-5; on Cap A all four A76s report 1024). One policy, because schedutil clocks a policy by its busiest
    /// core: spread over two policies, SLAM's half-busy cores sit at 408 MHz.
    pub fastest: Vec<usize>,
    /// Every core at least half the top capacity (the four A76s).
    pub big: Vec<usize>,
    /// The rest (the A55s at ~410).
    pub little: Vec<usize>,
}

/// The cores by `/sys/devices/system/cpu/cpu*/cpu_capacity` and `cpufreq/related_cpus`; `None` when the machine is not big.LITTLE.
pub fn detect_big_little() -> Option<CoreLayout> {
    let mut capacities = Vec::new();
    for cpu in 0..256 {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cpu_capacity");
        match std::fs::read_to_string(&path) {
            Ok(text) => capacities.push((cpu, text.trim().parse::<u32>().ok()?)),
            Err(_) if cpu > 0 && !Path::new(&format!("/sys/devices/system/cpu/cpu{cpu}")).exists() => break,
            Err(_) => return None,
        }
    }
    let policy_of = |cpu: usize| {
        let text = std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/related_cpus")).ok()?;
        text.split_whitespace().map(|c| c.parse().ok()).collect::<Option<Vec<usize>>>()
    };
    split_big_little(&capacities, &policy_of)
}

/// [`detect_big_little`] on `(cpu, capacity)` pairs; `policy_of(cpu)` = the cores sharing `cpu`'s cpufreq policy, `None` when
/// unknown (then `fastest` is every top-capacity core).
fn split_big_little(capacities: &[(usize, u32)], policy_of: &dyn Fn(usize) -> Option<Vec<usize>>) -> Option<CoreLayout> {
    let max = capacities.iter().map(|(_, c)| *c).max()?;
    let top: Vec<usize> = capacities.iter().filter(|(_, c)| *c == max).map(|(cpu, _)| *cpu).collect();
    let fastest = match top.last().and_then(|&cpu| policy_of(cpu)) {
        Some(policy) => top.iter().copied().filter(|cpu| policy.contains(cpu)).collect(),
        None => top,
    };
    let big: Vec<usize> = capacities.iter().filter(|(_, c)| 2 * *c >= max).map(|(cpu, _)| *cpu).collect();
    let little: Vec<usize> = capacities.iter().filter(|(_, c)| 2 * *c < max).map(|(cpu, _)| *cpu).collect();
    (!little.is_empty()).then_some(CoreLayout { fastest, big, little })
}

/// Pin the calling thread to `cpus` (threads it spawns afterwards inherit the mask).
///
/// # Errors
///
/// [`SchedError::Stage`] when `sched_setaffinity` refuses.
pub fn pin_current_thread(cpus: &[usize]) -> Result<(), SchedError> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: a zeroed cpu_set_t is the empty set; CPU_SET writes inside it for cpu < CPU_SETSIZE (checked); pid 0 = the
        // calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut set);
            for &cpu in cpus {
                if cpu >= libc::CPU_SETSIZE as usize {
                    return Err(stage_error("affinity", format!("cpu {cpu} out of range")));
                }
                libc::CPU_SET(cpu, &mut set);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                return Err(stage_error("affinity", format!("sched_setaffinity {cpus:?}: {}", std::io::Error::last_os_error())));
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpus;
    Ok(())
}

/// Set the calling thread's `uclamp.min` (0..=1024; threads it spawns afterwards inherit it): schedutil then runs the core at
/// no less than about that share of its top frequency while the thread is runnable. SLAM is mostly serial, so its two cores sit
/// ~50 % busy and schedutil leaves them at ~1.4 of 2.35 GHz without it (Cap B). Needs `CONFIG_UCLAMP_TASK` (the cap kernels
/// have it); the clamp ends with the thread, nothing system-wide changes.
///
/// # Errors
///
/// [`SchedError::Stage`] when `sched_setattr` refuses (no uclamp support, value out of range).
pub fn set_uclamp_min(min: u32) -> Result<(), SchedError> {
    #[cfg(target_os = "linux")]
    {
        /// `struct sched_attr` (SCHED_ATTR_SIZE_VER1 = 56 bytes).
        #[repr(C)]
        struct SchedAttr {
            size: u32,
            sched_policy: u32,
            sched_flags: u64,
            sched_nice: i32,
            sched_priority: u32,
            sched_runtime: u64,
            sched_deadline: u64,
            sched_period: u64,
            sched_util_min: u32,
            sched_util_max: u32,
        }
        const SCHED_FLAG_KEEP_POLICY: u64 = 0x08;
        const SCHED_FLAG_KEEP_PARAMS: u64 = 0x10;
        const SCHED_FLAG_UTIL_CLAMP_MIN: u64 = 0x20;
        let attr = SchedAttr {
            size: std::mem::size_of::<SchedAttr>() as u32,
            sched_policy: 0,
            sched_flags: SCHED_FLAG_KEEP_POLICY | SCHED_FLAG_KEEP_PARAMS | SCHED_FLAG_UTIL_CLAMP_MIN,
            sched_nice: 0,
            sched_priority: 0,
            sched_runtime: 0,
            sched_deadline: 0,
            sched_period: 0,
            sched_util_min: min.min(1024),
            sched_util_max: 1024,
        };
        // SAFETY: sched_setattr(pid 0 = this thread, a fully initialised sched_attr of the size it declares, flags 0); the kernel
        // only reads the struct.
        let result = unsafe { libc::syscall(libc::SYS_sched_setattr, 0, &attr as *const SchedAttr, 0u32) };
        if result != 0 {
            return Err(stage_error("uclamp", format!("sched_setattr uclamp.min {min}: {}", std::io::Error::last_os_error())));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = min;
    Ok(())
}

/// Per-thread-name CPU use from `/proc/self/task/*/stat` (Linux), as % of one core.
#[derive(Default)]
pub(super) struct ThreadCpu {
    last: std::collections::HashMap<u32, (String, u64)>,
    last_at: Option<Instant>,
}

impl ThreadCpu {
    pub(super) fn sample(&mut self) -> Vec<(String, f64)> {
        let mut now: std::collections::HashMap<u32, (String, u64)> = std::collections::HashMap::new();
        if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
            for entry in entries.flatten() {
                let Ok(tid) = entry.file_name().to_string_lossy().parse::<u32>() else { continue };
                let Ok(stat) = std::fs::read_to_string(entry.path().join("stat")) else { continue };
                let (Some(open), Some(close)) = (stat.find('('), stat.rfind(')')) else { continue };
                let name = stat[open + 1..close].to_owned();
                let fields: Vec<&str> = stat[close + 1..].split_whitespace().collect();
                let ticks = fields.get(11).and_then(|u| u.parse::<u64>().ok()).unwrap_or(0) + fields.get(12).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
                now.insert(tid, (name, ticks));
            }
        }
        let elapsed = self.last_at.map(|at| at.elapsed().as_secs_f64());
        let mut by_name: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        if let Some(elapsed) = elapsed.filter(|e| *e > 0.0) {
            for (tid, (name, ticks)) in &now {
                let before = self.last.get(tid).map_or(0, |(_, t)| *t);
                let group = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '-').to_owned();
                *by_name.entry(group).or_default() += ticks.saturating_sub(before) as f64 / 100.0 / elapsed * 100.0;
            }
        }
        self.last = now;
        self.last_at = Some(Instant::now());
        let mut out: Vec<(String, f64)> = by_name.into_iter().filter(|(_, pct)| *pct >= 0.5).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1));
        out
    }
}

/// Aggregate CPU use per core group from `/proc/stat`.
#[derive(Default)]
pub(super) struct CoreCpu {
    last: Vec<(u64, u64)>,
}

impl CoreCpu {
    /// Busy % per CPU since the last call.
    pub(super) fn sample(&mut self) -> Vec<f64> {
        let Ok(text) = std::fs::read_to_string("/proc/stat") else { return Vec::new() };
        let now: Vec<(u64, u64)> = text
            .lines()
            .filter(|line| line.starts_with("cpu") && line.as_bytes().get(3).is_some_and(u8::is_ascii_digit))
            .map(|line| {
                let values: Vec<u64> = line.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
                let idle = values.get(3).copied().unwrap_or(0) + values.get(4).copied().unwrap_or(0);
                (values.iter().sum::<u64>(), idle)
            })
            .collect();
        let busy = now
            .iter()
            .zip(self.last.iter())
            .map(|(&(total, idle), &(total0, idle0))| {
                let dt = total.saturating_sub(total0) as f64;
                if dt > 0.0 { 100.0 * (1.0 - idle.saturating_sub(idle0) as f64 / dt) } else { 0.0 }
            })
            .collect();
        self.last = now;
        busy
    }
}

/// The SoC temperature (`thermal_zone0`), degrees Celsius; `None` where the zone does not exist.
pub fn soc_temperature_c() -> Option<f64> {
    std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?.trim().parse::<f64>().ok().map(|mc| mc / 1000.0)
}

/// One second of power-relevant load (the caps run from 5 V USB without PD, ~11 W: Cap A lost power under NPU + CPU + Wi-Fi).
#[derive(Clone, Debug, Default, Serialize)]
pub struct PowerSample {
    /// Seconds since the run started.
    pub t_s: f64,
    /// `scaling_cur_freq` per cpufreq policy (`policy0` = A55, `policy4`/`policy6` = A76 pairs on the RK3588), MHz.
    pub cpu_mhz: Vec<(String, u32)>,
    /// `/sys/kernel/debug/rknpu/load` per NPU core, %.
    pub npu_load_pct: Vec<u32>,
    /// `/sys/class/power_supply/*`: name, status, volts, amps.
    pub supplies: Vec<(String, String, f64, f64)>,
    /// `thermal_zone0`, Celsius.
    pub soc_c: Option<f64>,
    /// Hottest thermal zone, Celsius.
    pub max_zone_c: Option<f64>,
}

impl PowerSample {
    /// Read everything that exists on this machine (missing nodes are skipped).
    pub fn read(t_s: f64) -> Self {
        let read = |path: &Path| std::fs::read_to_string(path).ok().map(|text| text.trim().to_owned());
        let mut cpu_mhz = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu/cpufreq") {
            let mut policies: Vec<_> = entries.flatten().map(|e| e.path()).filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("policy"))).collect();
            policies.sort();
            for policy in policies {
                if let Some(khz) = read(&policy.join("scaling_cur_freq")).and_then(|t| t.parse::<u32>().ok()) {
                    cpu_mhz.push((policy.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(), khz / 1000));
                }
            }
        }
        let npu_load_pct = read(Path::new("/sys/kernel/debug/rknpu/load"))
            .map(|text| text.split('%').filter_map(|part| part.rsplit([' ', ':']).next()?.trim().parse::<u32>().ok()).collect())
            .unwrap_or_default();
        let mut supplies = Vec::new();
        if let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") {
            let mut nodes: Vec<_> = entries.flatten().map(|e| e.path()).collect();
            nodes.sort();
            for node in nodes {
                let micro = |name: &str| read(&node.join(name)).and_then(|t| t.parse::<f64>().ok()).map_or(0.0, |v| v / 1e6);
                let name = node.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                supplies.push((name, read(&node.join("status")).unwrap_or_default(), micro("voltage_now"), micro("current_now")));
            }
        }
        let mut max_zone_c: Option<f64> = None;
        for zone in 0..32 {
            match read(Path::new(&format!("/sys/class/thermal/thermal_zone{zone}/temp"))).and_then(|t| t.parse::<f64>().ok()) {
                Some(mc) => max_zone_c = Some(max_zone_c.map_or(mc / 1e3, |m| m.max(mc / 1e3))),
                None => break,
            }
        }
        Self { t_s, cpu_mhz, npu_load_pct, supplies, soc_c: soc_temperature_c(), max_zone_c }
    }

    /// One log line.
    pub fn line(&self) -> String {
        let cpus: Vec<String> = self.cpu_mhz.iter().map(|(policy, mhz)| format!("{} {mhz}", policy.trim_start_matches("policy"))).collect();
        let npu: Vec<String> = self.npu_load_pct.iter().map(|pct| format!("{pct}%")).collect();
        let supplies: Vec<String> = self
            .supplies
            .iter()
            .filter(|(_, _, v, a)| *v > 0.0 || *a > 0.0)
            .map(|(name, status, v, a)| format!("{name} {status} {v:.2} V {a:.2} A"))
            .collect();
        format!(
            "           power: cpu MHz [{}] | npu [{}] | {} | soc {} max zone {}",
            cpus.join(", "),
            npu.join(" "),
            supplies.join(", "),
            self.soc_c.map_or("-".into(), |c| format!("{c:.1} C")),
            self.max_zone_c.map_or("-".into(), |c| format!("{c:.1} C")),
        )
    }
}

/// Caps `scaling_max_freq` of every cpufreq policy that covers one of `cpus`, and restores the previous values on drop
/// (normal exit, SIGINT/SIGTERM, panic; not SIGKILL: then the cap lasts until reboot or the next run).
pub struct CpuFreqCap {
    saved: Vec<(std::path::PathBuf, String)>,
}

impl CpuFreqCap {
    /// Apply `khz` to the policies of `cpus`.
    ///
    /// # Errors
    ///
    /// [`SchedError::Stage`] when a policy cannot be read or written (the ones already written are restored).
    pub fn apply(cpus: &[usize], khz: u32) -> Result<Self, SchedError> {
        let mut cap = Self { saved: Vec::new() };
        let entries = std::fs::read_dir("/sys/devices/system/cpu/cpufreq").map_err(|e| stage_error("cpufreq", e))?;
        for policy in entries.flatten().map(|e| e.path()) {
            let Ok(related) = std::fs::read_to_string(policy.join("related_cpus")) else { continue };
            if !related.split_whitespace().filter_map(|c| c.parse::<usize>().ok()).any(|c| cpus.contains(&c)) {
                continue;
            }
            let path = policy.join("scaling_max_freq");
            let previous = std::fs::read_to_string(&path).map_err(|e| stage_error("cpufreq", format!("{}: {e}", path.display())))?;
            std::fs::write(&path, khz.to_string()).map_err(|e| stage_error("cpufreq", format!("{}: {e}", path.display())))?;
            eprintln!("robocap-live: {} {} -> {khz} kHz (restored on exit)", path.display(), previous.trim());
            cap.saved.push((path, previous.trim().to_owned()));
        }
        Ok(cap)
    }
}

impl Drop for CpuFreqCap {
    fn drop(&mut self) {
        for (path, previous) in self.saved.iter().rev() {
            if let Err(error) = std::fs::write(path, previous) {
                eprintln!("robocap-live: restoring {} to {previous} failed: {error}", path.display());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_parse() -> Result<(), SchedError> {
        assert_eq!(parse_cpu_list("4-7")?, Some(vec![4, 5, 6, 7]));
        assert_eq!(parse_cpu_list("0,2-3,2")?, Some(vec![0, 2, 3]));
        assert_eq!(parse_cpu_list("none")?, None);
        assert!(parse_cpu_list("7-4").is_err() && parse_cpu_list("a").is_err());
        Ok(())
    }

    /// SLAM gets one cpufreq policy of the top-capacity cores, also when all four A76s report the same capacity (Cap A).
    #[test]
    fn the_fastest_cores_are_one_cpufreq_policy() {
        let policy = |cpu: usize| Some(match cpu {
            0..=3 => vec![0, 1, 2, 3],
            4 | 5 => vec![4, 5],
            _ => vec![6, 7],
        });
        let cap_a: Vec<(usize, u32)> = (0..8).map(|cpu| (cpu, if cpu < 4 { 414 } else { 1024 })).collect();
        let cap_b: Vec<(usize, u32)> = (0..8).map(|cpu| (cpu, [414, 414, 414, 414, 1003, 1003, 1024, 1024][cpu])).collect();
        for capacities in [&cap_a, &cap_b] {
            let layout = split_big_little(capacities, &policy);
            assert_eq!(layout, Some(CoreLayout { fastest: vec![6, 7], big: vec![4, 5, 6, 7], little: vec![0, 1, 2, 3] }));
        }
        let unknown_policy = split_big_little(&cap_a, &|_| None);
        assert_eq!(unknown_policy.map(|layout| layout.fastest), Some(vec![4, 5, 6, 7]), "without cpufreq: every top-capacity core");
        assert_eq!(split_big_little(&[(0, 1024), (1, 1024)], &policy), None, "not big.LITTLE");
    }
}
