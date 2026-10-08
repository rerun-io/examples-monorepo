//! Encoder presets, Rerun payload ownership, and Linux process telemetry.
use kornia_image::ImageSize;
use kornia_staging_io::video::{EncoderConfig, EncoderStats, VideoError};
use std::str::FromStr;

/// One encoded camera frame, shared by the save stream and preview.
#[derive(Clone, Debug)]
pub struct VideoSample {
    /// Camera index.
    pub camera: usize,
    /// Input timestamp in nanoseconds.
    pub t_ns: i64,
    /// Shared Annex-B bytes; cloning does not copy pixels.
    pub data: rerun::datatypes::Blob,
    /// Whether this access unit contains an IDR slice.
    pub keyframe: bool,
}

/// Encoder counters and the application's last CPU observation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EncoderReport {
    /// Counters from the subprocess encoder.
    pub stats: EncoderStats,
    /// Child CPU seconds, user plus system.
    pub cpu_seconds: f64,
}

/// The H.264 encoders [`encoder_config`] knows, by their command-line names (`mpp`, `x264`, `openh264`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    /// The cap's hardware encoder ([`mpp`]).
    Mpp,
    /// ffmpeg's libx264 ([`x264`]).
    X264,
    /// GStreamer's openh264 ([`openh264`]).
    Openh264,
}

impl FromStr for EncoderKind {
    type Err = super::LogError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "mpp" => Ok(Self::Mpp),
            "x264" => Ok(Self::X264),
            "openh264" => Ok(Self::Openh264),
            other => Err(super::LogError::Encoder(other.to_owned())),
        }
    }
}

/// The command of one encoder kind.
///
/// # Arguments
///
/// * `kind` - Which encoder.
/// * `size`, `fps`, `bps`, `gop` - As for [`mpp`].
///
/// # Returns
///
/// [`mpp`], [`x264`] or [`openh264`] with these settings.
/// # Errors
/// Rejects zero/odd/overflowing geometry or a zero frame rate.
pub fn encoder_config(
    kind: EncoderKind,
    size: ImageSize,
    fps: u32,
    bps: u32,
    gop: u32,
) -> Result<EncoderConfig, VideoError> {
    match kind {
        EncoderKind::Mpp => mpp(size, fps, bps, gop),
        EncoderKind::X264 => x264(size, fps, bps, gop),
        EncoderKind::Openh264 => openh264(size, fps, bps, gop),
    }
}

/// Rockchip MPP (`mpph264enc`, the RK3588's VPU), constant bitrate, SPS/PPS with every IDR so a viewer can join mid-stream.
///
/// # Arguments
///
/// * `size` - Frame size (640x360 on the cap).
/// * `fps` - Nominal frame rate (rate control only; samples carry their own timestamps).
/// * `bps` - Target bitrate, bits per second.
/// * `gop` - Frames per group of pictures (the longest a dropped preview frame breaks a pane).
/// # Errors
/// Rejects zero/odd/overflowing geometry or a zero frame rate.
pub fn mpp(size: ImageSize, fps: u32, bps: u32, gop: u32) -> Result<EncoderConfig, VideoError> {
    let bps = format!("bps={bps}");
    let gop = format!("gop={gop}");
    EncoderConfig::gst(
        size,
        fps,
        &[
            "mpph264enc",
            &bps,
            &gop,
            "header-mode=each-idr",
            "rc-mode=cbr",
        ],
    )
}

/// Cisco's software `openh264enc` (host tests and the host fallback; GStreamer's bad plugins; it takes I420, hence the
/// `videoconvert`).
/// # Errors
/// Rejects zero/odd/overflowing geometry or a zero frame rate.
pub fn openh264(
    size: ImageSize,
    fps: u32,
    bps: u32,
    gop: u32,
) -> Result<EncoderConfig, VideoError> {
    let bps = format!("bitrate={bps}");
    let gop = format!("gop-size={gop}");
    EncoderConfig::gst(
        size,
        fps,
        &[
            "videoconvert",
            "!",
            "openh264enc",
            &bps,
            &gop,
            "complexity=low",
        ],
    )
}

/// `ffmpeg` with libx264 (zerolatency, no B-frames): a host encoder where GStreamer lacks one.
/// # Errors
/// Rejects zero/odd/overflowing geometry or a zero frame rate.
pub fn x264(size: ImageSize, fps: u32, bps: u32, gop: u32) -> Result<EncoderConfig, VideoError> {
    if fps == 0 {
        return Err(VideoError::Rate { fps });
    }
    let args = [
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "nv12",
        "-s",
        &format!("{}x{}", size.width, size.height),
        "-r",
        &fps.to_string(),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-bf",
        "0",
        "-g",
        &gop.to_string(),
        "-b:v",
        &bps.to_string(),
        "-x264-params",
        "repeat-headers=1",
        "-flush_packets",
        "1",
        "-f",
        "h264",
        "pipe:1",
    ];
    EncoderConfig::new(
        "ffmpeg".into(),
        args.iter().map(|s| s.to_string()).collect(),
        size,
    )
}
/// User + system CPU seconds of a process, from `/proc/<pid>/stat` (fields 14 and 15, in 100 Hz ticks).
///
/// # Arguments
///
/// * `pid` - The process (`std::process::id()` for this one).
///
/// # Returns
///
/// The seconds, or `None` when the file cannot be read or parsed (the process is gone, or not Linux).
pub fn process_cpu_seconds(pid: u32) -> Option<f64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    // The command name (field 2) may contain spaces; the fields after its closing parenthesis are space separated.
    let rest = &stat[stat.rfind(')')? + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some((utime + stime) / 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::Image;
    use kornia_staging_io::video::{H264Encoder, has_nal};
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    #[test]
    fn gst_arguments_frame_nv12_from_stdin_and_write_annex_b_to_stdout() {
        let config = mpp(
            ImageSize {
                width: 640,
                height: 360,
            },
            30,
            1_000_000,
            30,
        )
        .unwrap();
        let line = config.args.join(" ");
        assert!(line.starts_with(
            "-q filesrc location=/dev/stdin blocksize=345600 ! video/x-raw,format=NV12,width=640,height=360,framerate=30/1 ! queue max-size-buffers=3 ! mpph264enc bps=1000000 gop=30"
        ));
        assert!(line.ends_with(
            "! video/x-h264,stream-format=byte-stream,alignment=au ! fdsink fd=1 sync=false"
        ));
    }

    #[test]
    fn encoder_kinds_parse_by_their_command_line_names() -> Result<(), super::super::LogError> {
        let size = ImageSize {
            width: 640,
            height: 360,
        };
        assert_eq!(
            encoder_config("mpp".parse()?, size, 30, 1_000_000, 30)?,
            mpp(size, 30, 1_000_000, 30)?
        );
        assert_eq!(
            encoder_config("x264".parse()?, size, 30, 1_000_000, 30)?,
            x264(size, 30, 1_000_000, 30)?
        );
        assert_eq!(
            encoder_config("openh264".parse()?, size, 30, 1_000_000, 30)?,
            openh264(size, 30, 1_000_000, 30)?
        );
        assert!(matches!(
            "h265".parse::<EncoderKind>(),
            Err(super::super::LogError::Encoder(_))
        ));
        Ok(())
    }

    /// End to end through a real software encoder, when the host has one (GStreamer's openh264enc, or ffmpeg's libx264).
    #[test]
    fn a_host_encoder_returns_one_access_unit_per_frame_with_the_input_timestamps()
    -> Result<(), VideoError> {
        let size = ImageSize {
            width: 640,
            height: 360,
        };
        let has = |program: &str, args: &[&str]| {
            Command::new(program)
                .args(args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        };
        let config = if has("gst-inspect-1.0", &["openh264enc"]) {
            openh264(size, 30, 1_000_000, 10)?
        } else if has("ffmpeg", &["-hide_banner", "-h", "encoder=libx264"]) {
            x264(size, 30, 1_000_000, 10)?
        } else {
            eprintln!("skipped: no host H.264 encoder (openh264enc or ffmpeg libx264)");
            return Ok(());
        };
        let samples = Arc::new(Mutex::new(Vec::new()));
        let collected = samples.clone();
        let mut encoder = H264Encoder::spawn(&config, move |sample| {
            if let Ok(mut s) = collected.lock() {
                s.push(sample);
            }
        })?;
        for frame in 0..25u8 {
            let data: Vec<u8> = (0..size.width * size.height)
                .map(|i| ((i % size.width) as u8).wrapping_add(frame * 4))
                .collect();
            let image = Image::<u8, 1>::new(size, data).map_err(|e| VideoError::Io {
                source: std::io::Error::other(e.to_string()),
                stderr: String::new(),
            })?;
            encoder.push(1_000 + i64::from(frame) * 33_333_333, &image)?;
        }
        let stats = encoder.finish(Instant::now() + Duration::from_secs(10), |_| {})?;
        let samples = samples.lock().map(|s| s.clone()).unwrap_or_default();
        assert_eq!((stats.frames_in, stats.samples_out), (25, 25));
        assert_eq!(samples.len(), 25);
        assert!(samples[0].unit.keyframe && has_nal(&samples[0].unit.data, 7));
        assert!(
            samples
                .iter()
                .enumerate()
                .all(|(i, s)| s.timestamp_ns == 1_000 + i as i64 * 33_333_333)
        );
        assert!(
            samples.iter().filter(|s| s.unit.keyframe).count() >= 2,
            "gop 10 over 25 frames"
        );
        Ok(())
    }
    #[test]
    fn cpu_seconds_of_this_process_are_readable() {
        assert!(super::process_cpu_seconds(std::process::id()).is_some_and(|s| s >= 0.0));
    }
}
