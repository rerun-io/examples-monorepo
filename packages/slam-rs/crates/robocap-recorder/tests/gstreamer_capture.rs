#![cfg(feature = "gstreamer-capture")]

use robocap_recorder::SamplePipeline;
use std::time::Duration;

#[test]
fn pipeline_delivers_complete_frames_with_capture_timing_and_releases_resources()
-> anyhow::Result<()> {
    for _ in 0..2 {
        let mut pipeline = SamplePipeline::start(
            "videotestsrc num-buffers=3 pattern=black ! video/x-raw,format=GRAY8,width=640,height=360,framerate=30/1 ! appsink name=frames sync=false max-buffers=4 drop=false",
            "frames",
        )?;
        let mut timestamps = Vec::new();
        while let Some(frame) = pipeline.next(Duration::from_secs(5))? {
            assert_eq!(frame.bytes.len(), 640 * 360);
            timestamps.push(frame.pts_ns);
        }
        assert_eq!(timestamps, [0, 33_333_333, 66_666_666]);
    }
    Ok(())
}

/// Device integration check: generated images exercise the real MPP encoder.
/// It does not acquire a camera, trigger, or sensor, and is not a SLAM test.
#[test]
#[ignore = "requires an idle Cap B with the installed rockchipmpp plugin"]
fn cap_b_encodes_six_generated_streams_without_losing_frames() -> anyhow::Result<()> {
    let started = std::time::Instant::now();
    let mut pipelines = (0..6)
        .map(|_| {
            SamplePipeline::start(
                "videotestsrc num-buffers=300 is-live=true pattern=ball ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! mpph264enc bps=4000000 gop=30 header-mode=1 max-pending=2 ! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au ! appsink name=frames sync=false max-buffers=4 drop=false",
                "frames",
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut counts = [0u64; 6];
    let mut previous = [None; 6];
    let mut keyframes = [0u64; 6];
    let mut complete = [false; 6];
    while !complete.iter().all(|value| *value) {
        for (camera, pipeline) in pipelines.iter_mut().enumerate() {
            if complete[camera] {
                continue;
            }
            let Some(frame) = pipeline.next(Duration::from_secs(5))? else {
                complete[camera] = true;
                continue;
            };
            anyhow::ensure!(!frame.bytes.is_empty(), "empty encoded access unit");
            if let Some(timestamp) = previous[camera] {
                let interval = frame
                    .pts_ns
                    .checked_sub(timestamp)
                    .ok_or_else(|| anyhow::anyhow!("encoder time regressed"))?;
                anyhow::ensure!(
                    (33_333_333..=33_333_334).contains(&interval),
                    "camera {camera} timestamp gap {interval}"
                );
            } else {
                anyhow::ensure!(frame.keyframe, "first encoded frame must be a keyframe");
            }
            previous[camera] = Some(frame.pts_ns);
            counts[camera] += 1;
            keyframes[camera] += u64::from(frame.keyframe);
        }
    }
    eprintln!(
        "generated_stream_frames={counts:?} keyframes={keyframes:?} elapsed_seconds={:.3}",
        started.elapsed().as_secs_f64()
    );
    assert_eq!(counts, [300; 6]);
    anyhow::ensure!(
        keyframes.iter().all(|count| *count >= 10),
        "missing periodic IDRs"
    );
    anyhow::ensure!(
        started.elapsed() < Duration::from_secs(13),
        "encoder transport did not keep up with realtime"
    );
    Ok(())
}
