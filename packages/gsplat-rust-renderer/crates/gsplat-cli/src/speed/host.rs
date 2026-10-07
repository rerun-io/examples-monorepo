//! Quiet-host sampling with an explicit, recorded 90-minute loaded-host fallback.
#[cfg(not(target_os = "macos"))]
use anyhow::Context;
use anyhow::Result;
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
    pub gpu_percent: Option<u32>,
    pub gpu_memory_mib: Option<u64>,
}
/// Record unknown GPU utilization when no NVIDIA sampler is available.
#[cfg(not(target_os = "macos"))]
pub fn sample() -> Result<HostSample> {
    std::thread::sleep(Duration::from_secs(1));
    let load = fs::read_to_string("/proc/loadavg")?
        .split_whitespace()
        .next()
        .context("empty loadavg")?
        .parse()?;
    let logical_cores = fs::read_to_string("/proc/cpuinfo")?
        .lines()
        .filter(|l| l.starts_with("processor\t"))
        .count();
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()
        .filter(|r| r.status.success());
    let text = result.and_then(|r| String::from_utf8(r.stdout).ok());
    let (gpu_percent, gpu_memory_mib) = text.as_deref().map_or((None, None), nvidia_sample);
    Ok(HostSample {
        load_average_1m: Some(load),
        logical_cores: Some(logical_cores),
        gpu_percent,
        gpu_memory_mib,
    })
}
#[cfg(any(not(target_os = "macos"), test))]
fn nvidia_sample(text: &str) -> (Option<u32>, Option<u64>) {
    let rows: Vec<_> = text.lines().collect();
    // The busiest GPU controls admission on machines with multiple adapters.
    let samples: Option<Vec<(u32, u64)>> = rows
        .iter()
        .map(|row| {
            let (gpu, memory) = row.split_once(',')?;
            Some((gpu.trim().parse().ok()?, memory.trim().parse().ok()?))
        })
        .collect();
    samples.filter(|s| !s.is_empty()).map_or((None, None), |s| {
        (
            s.iter().map(|x| x.0).max(),
            Some(s.iter().map(|x| x.1).sum()),
        )
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
    // macOS admission includes its measured desktop GPU baseline.
    fn admissible(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.gpu_percent.is_some_and(|gpu| gpu <= 20)
                && self.load_average_1m.is_some_and(|load| load < 4.0)
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.gpu_percent.is_none_or(|gpu| gpu < 5)
                && self
                    .load_average_1m
                    .zip(self.logical_cores)
                    .is_none_or(|(load, cores)| load < cores as f64 * 0.25)
        }
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
        gpu_memory_mib: None,
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
    fn nvidia_samples_record_multiple_adapters_and_missing_fields() {
        assert_eq!(nvidia_sample("1, 120\n87, 340\n"), (Some(87), Some(460)));
        assert_eq!(nvidia_sample("N/A, 0"), (None, None));
        assert_eq!(nvidia_sample(""), (None, None));
    }
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
    fn unavailable_mac_samplers_record_null_without_claiming_quiet() {
        let sample = macos_sample(None, None, None);
        assert_eq!(sample.admissible(), !cfg!(target_os = "macos"));
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
        assert_eq!(sample.admissible(), cfg!(target_os = "macos"));
        assert_eq!(sample.quiet_verified(), cfg!(target_os = "macos"));
    }
    #[cfg(target_os = "macos")]
    #[test]
    fn mac_desktop_admission_requires_measured_gpu_and_load_below_thresholds() {
        let mut sample = macos_sample(Some("{ 3.99 2.0 1.0 }"), Some("10"), None);
        sample.gpu_percent = Some(15);
        assert!(sample.quiet_verified());
        sample.gpu_percent = Some(20);
        assert!(sample.quiet_verified());
        sample.gpu_percent = Some(21);
        assert!(!sample.quiet_verified());
        sample.gpu_percent = Some(20);
        sample.load_average_1m = Some(4.0);
        assert!(!sample.quiet_verified());
        sample.load_average_1m = None;
        assert!(!sample.admissible());
        sample.load_average_1m = Some(1.0);
        sample.gpu_percent = None;
        assert!(!sample.admissible());
    }
}
