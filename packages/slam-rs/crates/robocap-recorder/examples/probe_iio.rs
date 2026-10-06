use anyhow::{Result, ensure};
use robocap_recorder::{monotonic_ns, start_imu};
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let trigger = if std::env::args().any(|arg| arg == "--trigger") {
        let trigger = robocap_recorder::FrameTrigger::stopped()?;
        trigger.start()?;
        Some(trigger)
    } else {
        None
    };
    let mut devices = (1..=7).map(start_imu).collect::<Result<Vec<_>>>()?;
    let mut counts = [0; 7];
    let mut previous = [None; 7];
    let start = Instant::now();
    let mut scans = Vec::new();
    while start.elapsed() < Duration::from_secs(10) {
        for (index, device) in devices.iter_mut().enumerate() {
            scans.clear();
            device.read_scans(&mut scans)?;
            for sample in &scans {
                let age = monotonic_ns()? - sample.timestamp_ns;
                ensure!(
                    (0..500_000_000).contains(&age),
                    "IIO{} clock mismatch: age={age}",
                    index + 1
                );
                if let Some(last) = previous[index] {
                    ensure!(sample.timestamp_ns > last, "IIO time regressed");
                } else {
                    eprintln!("IIO{} first={sample:?} age_ns={age}", index + 1);
                }
                previous[index] = Some(sample.timestamp_ns);
                counts[index] += 1;
            }
        }
    }
    eprintln!("counts={counts:?} elapsed={:?}", start.elapsed());
    ensure!(
        counts[..6].iter().all(|&count| count >= 1800),
        "IMU did not reach 200 Hz"
    );
    ensure!(counts[6] >= 900, "MAG did not reach 100 Hz");
    drop(trigger);
    Ok(())
}
