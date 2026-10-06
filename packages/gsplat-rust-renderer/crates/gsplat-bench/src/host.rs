//! Quiet-host sampling with an explicit, recorded 90-minute loaded-host fallback.
use anyhow::Result;
#[cfg(not(target_os = "macos"))]
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
#[cfg(not(target_os = "macos"))]
use std::fs;
use std::{
    process::Command,
    time::{Duration, Instant},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostSample {
    pub load_average_1m: Option<f64>,
    pub logical_cores: Option<usize>,
    pub context_switches_per_second: Option<f64>,
    pub gpu_percent: Option<u32>,
    pub gpu_memory_mib: Option<u64>,
    pub sm_clock_mhz: Option<u32>,
    pub memory_clock_mhz: Option<u32>,
    pub pstate: Option<String>,
    pub power_watts: Option<f64>,
}
#[cfg(not(target_os = "macos"))]
fn context_switches() -> Result<u64> {
    fs::read_to_string("/proc/stat")?
        .lines()
        .find_map(|l| l.strip_prefix("ctxt "))
        .context("missing /proc/stat ctxt")?
        .trim()
        .parse()
        .context("invalid ctxt count")
}
/// Sample over one second so the context-switch rate has a defined interval.
#[cfg(not(target_os = "macos"))]
pub fn sample() -> Result<HostSample> {
    let before = context_switches()?;
    let start = Instant::now();
    std::thread::sleep(Duration::from_secs(1));
    let rate = (context_switches()?.saturating_sub(before)) as f64 / start.elapsed().as_secs_f64();
    let load = fs::read_to_string("/proc/loadavg")?
        .split_whitespace()
        .next()
        .context("empty loadavg")?
        .parse()?;
    let logical_cores = fs::read_to_string("/proc/cpuinfo")?
        .lines()
        .filter(|l| l.starts_with("processor\t"))
        .count();
    ensure!(logical_cores > 0, "cannot determine logical core count");
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,clocks.sm,clocks.mem,pstate,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    ensure!(
        result.status.success(),
        "nvidia-smi failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8(result.stdout)?;
    let rows: Vec<_> = text.lines().collect();
    ensure!(
        rows.len() == 1,
        "timing requires one unambiguous NVIDIA GPU"
    );
    let fields: Vec<_> = rows[0].split(',').map(str::trim).collect();
    ensure!(fields.len() == 6, "invalid nvidia-smi sample");
    Ok(HostSample {
        load_average_1m: Some(load),
        logical_cores: Some(logical_cores),
        context_switches_per_second: Some(rate),
        gpu_percent: Some(fields[0].parse()?),
        gpu_memory_mib: Some(fields[1].parse()?),
        sm_clock_mhz: Some(fields[2].parse()?),
        memory_clock_mhz: Some(fields[3].parse()?),
        pstate: Some(fields[4].into()),
        power_watts: Some(fields[5].parse()?),
    })
}
#[cfg(not(target_os = "macos"))]
pub fn cpuset() -> Result<String> {
    Ok(fs::read_to_string("/proc/self/status")?
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
        .context("missing process cpuset")?
        .trim()
        .into())
}
impl HostSample {
    // Missing macOS telemetry is not evidence of a busy or an idle host.
    fn admissible(&self) -> bool {
        self.gpu_percent.is_none_or(|gpu| gpu < 5)
            && self
                .load_average_1m
                .zip(self.logical_cores)
                .is_none_or(|(load, cores)| load < cores as f64 * 0.25)
    }
    pub fn quiet_verified(&self) -> bool {
        self.gpu_percent.is_some()
            && self.load_average_1m.is_some()
            && self.logical_cores.is_some()
            && self.admissible()
    }
}

#[cfg(any(target_os = "macos", test))]
fn macos_sample(load: Option<&str>, cores: Option<&str>, accelerator: Option<&str>) -> HostSample {
    let load_average_1m = load
        .and_then(|text| {
            text.trim()
                .trim_start_matches('{')
                .split_whitespace()
                .next()?
                .parse::<f64>()
                .ok()
        })
        .filter(|value| value.is_finite() && *value >= 0.0);
    let logical_cores = cores
        .and_then(|text| text.trim().parse().ok())
        .filter(|value| *value > 0);
    let gpu_percent = accelerator.and_then(|text| {
        text.split("\"Device Utilization %\"")
            .skip(1)
            .filter_map(|tail| {
                tail.trim_start()
                    .strip_prefix('=')?
                    .trim_start()
                    .split(|c: char| !c.is_ascii_digit())
                    .next()?
                    .parse::<u32>()
                    .ok()
                    .filter(|value| *value <= 100)
            })
            .max()
    });
    HostSample {
        load_average_1m,
        logical_cores,
        gpu_percent,
        context_switches_per_second: None,
        gpu_memory_mib: None,
        sm_clock_mhz: None,
        memory_clock_mhz: None,
        pstate: None,
        power_watts: None,
    }
}

#[cfg(target_os = "macos")]
pub fn sample() -> Result<HostSample> {
    std::thread::sleep(Duration::from_secs(1));
    let output = |name: &str, args: &[&str]| {
        let result = Command::new(name).args(args).output().ok()?;
        result
            .status
            .success()
            .then(|| String::from_utf8(result.stdout).ok())
            .flatten()
    };
    Ok(macos_sample(
        output("sysctl", &["-n", "vm.loadavg"]).as_deref(),
        output("sysctl", &["-n", "hw.ncpu"]).as_deref(),
        output("ioreg", &["-r", "-d", "1", "-c", "IOAccelerator"]).as_deref(),
    ))
}

#[cfg(target_os = "macos")]
pub fn cpuset() -> Result<String> {
    Ok("unpinned".into())
}

fn admission_complete(quiet_samples: usize, samples: usize, deadline: Instant) -> Option<bool> {
    if quiet_samples == 10 {
        Some(false)
    } else if samples == 10 && Instant::now() >= deadline {
        Some(true)
    } else {
        None
    }
}

/// All repeats in a case share the same deadline; retain ten samples even after it expires.
pub fn wait_quiet(deadline: Instant) -> Result<(Vec<HostSample>, bool)> {
    let start = Instant::now();
    let mut samples = std::collections::VecDeque::new();
    let mut quiet_samples = 0;
    let mut last_notice = Instant::now();
    loop {
        let s = sample()?;
        let quiet = s.admissible();
        if last_notice.elapsed().as_secs() >= 30 {
            eprintln!(
                "quiet-host: load {:?}/{:?} cores, GPU {:?}%, elapsed {}s",
                s.load_average_1m,
                s.logical_cores,
                s.gpu_percent,
                start.elapsed().as_secs()
            );
            last_notice = Instant::now();
        }
        quiet_samples = if quiet { quiet_samples + 1 } else { 0 };
        samples.push_back(s);
        if samples.len() > 10 {
            samples.pop_front();
        }
        if let Some(loaded_host) = admission_complete(quiet_samples, samples.len(), deadline) {
            if loaded_host {
                #[cfg(not(target_os = "macos"))]
                ensure!(
                    cpuset()? == "8-15,24-31",
                    "loaded-host fallback requires CCD1 cpuset 8-15,24-31"
                );
                eprintln!("quiet-host: case deadline reached; proceeding with loaded_host=true");
            }
            return Ok((samples.into_iter().collect(), loaded_host));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expired_case_deadline_keeps_ten_samples_without_restarting_the_wait() {
        let deadline = Instant::now() - Duration::from_secs(1);
        for _repeat in 0..2 {
            for samples in 1..10 {
                assert_eq!(admission_complete(0, samples, deadline), None);
            }
            assert_eq!(admission_complete(0, 10, deadline), Some(true));
        }
        assert_eq!(admission_complete(10, 10, deadline), Some(false));
        assert_eq!(
            admission_complete(0, 10, Instant::now() + Duration::from_secs(60)),
            None
        );
    }
    #[test]
    fn unavailable_mac_samplers_record_null_and_allow_admission() {
        let sample = macos_sample(None, None, None);
        assert!(sample.admissible());
        assert!(!sample.quiet_verified());
        let json = serde_json::to_value(sample).unwrap();
        assert!(
            json.as_object()
                .unwrap()
                .values()
                .all(serde_json::Value::is_null)
        );
    }
    #[test]
    fn mac_samples_parse_load_cores_and_accelerator_utilization() {
        let sample = macos_sample(
            Some("{ 1.25 2.50 3.75 }"),
            Some("10\n"),
            Some(
                r#""PerformanceStatistics" = {"Device Utilization %"=7,"Renderer Utilization %"=0}"#,
            ),
        );
        assert_eq!(sample.load_average_1m, Some(1.25));
        assert_eq!(sample.logical_cores, Some(10));
        assert_eq!(sample.gpu_percent, Some(7));
        assert!(!sample.admissible());
        assert!(!sample.quiet_verified());
    }
}
