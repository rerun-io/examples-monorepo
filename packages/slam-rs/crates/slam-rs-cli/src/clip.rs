//! Load the dump_clip.py clip format before replay starts.

use slam_rs::calib::Calibration;
use std::path::Path;

/// The fields of `clip.json` the replay reads.
#[derive(Debug, serde::Deserialize)]
pub(super) struct Clip {
    pub(super) segment_id: String,
    pub(super) num_cameras: usize,
    pub(super) framesets: usize,
    pub(super) frame_t_ns: Vec<i64>,
    pub(super) resolution_wh: Vec<(usize, usize)>,
}

/// One gray8 raster of a PGM.
struct Raster {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

fn read_pgm(path: &Path) -> Result<Raster, String> {
    let bytes: Vec<u8> =
        std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    // "P5\n<w> <h>\n255\n" then the raster: dump_clip.py writes exactly that.
    let mut fields: Vec<usize> = Vec::with_capacity(3);
    let mut cursor: usize = 2;
    while fields.len() < 3 {
        while bytes.get(cursor).is_some_and(u8::is_ascii_whitespace) {
            cursor += 1;
        }
        let start: usize = cursor;
        while bytes
            .get(cursor)
            .is_some_and(|byte| !byte.is_ascii_whitespace())
        {
            cursor += 1;
        }
        let text = std::str::from_utf8(&bytes[start..cursor])
            .map_err(|_| format!("{}: bad PGM header", path.display()))?;
        fields.push(
            text.parse()
                .map_err(|_| format!("{}: bad PGM header", path.display()))?,
        );
    }
    let (width, height) = (fields[0], fields[1]);
    let pixels: Vec<u8> = bytes[cursor + 1..].to_vec();
    if pixels.len() != width * height {
        return Err(format!(
            "{}: {} bytes of raster for {width}x{height}",
            path.display(),
            pixels.len()
        ));
    }
    Ok(Raster {
        width,
        height,
        pixels,
    })
}

/// One sample of `imu.csv`.
use slam_rs::catalog_timing::ImuRow;

pub(super) fn read_imu(path: &Path) -> Result<Vec<ImuRow>, String> {
    let text: String =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    text.lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let value = |index: usize| -> Result<f64, String> {
                fields
                    .get(index)
                    .and_then(|field| field.parse().ok())
                    .ok_or_else(|| format!("imu.csv: bad line {line:?}"))
            };
            Ok(ImuRow {
                t_ns: fields[0]
                    .parse()
                    .map_err(|_| format!("imu.csv: bad line {line:?}"))?,
                gyro: [value(1)?, value(2)?, value(3)?],
                accel: [value(4)?, value(5)?, value(6)?],
            })
        })
        .collect()
}

/// Fully prepared inputs; neither loader does any work inside the replay clock.
pub(super) struct ReplayInput {
    pub(super) clip: Clip,
    pub(super) calibration: Calibration<f64>,
    pub(super) imu: Vec<ImuRow>,
    pub(super) pixels: Vec<u8>,
}

pub(super) fn load_clip(
    clip_dir: &Path,
    max_framesets: Option<usize>,
    frames: Option<&Path>,
) -> Result<ReplayInput, String> {
    let mut clip: Clip = serde_json::from_str(
        &std::fs::read_to_string(clip_dir.join("clip.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("clip.json: {error}"))?;
    let calibration: Calibration<f64> = Calibration::from_json_str(
        &std::fs::read_to_string(clip_dir.join("calib.json")).map_err(|error| error.to_string())?,
    )
    .map_err(|error| format!("calib.json: {error:?}"))?;
    let imu: Vec<ImuRow> = read_imu(&clip_dir.join("imu.csv"))?;
    let framesets: usize = max_framesets.map_or(clip.framesets, |limit| limit.min(clip.framesets));
    clip.framesets = framesets;
    clip.frame_t_ns.truncate(framesets);

    // Every frame in one buffer, frameset-major then camera, before the clock starts.
    let sizes: Vec<usize> = clip
        .resolution_wh
        .iter()
        .map(|&(width, height)| width * height)
        .collect();
    if sizes.len() != clip.num_cameras {
        return Err(format!(
            "clip.json: {} resolutions for {} cameras",
            sizes.len(),
            clip.num_cameras
        ));
    }
    let frameset_bytes: usize = sizes.iter().sum();
    let mut pixels: Vec<u8> = vec![0; framesets * frameset_bytes];
    match frames {
        Some(path) => {
            use std::io::Read as _;
            let mut reader: Box<dyn std::io::Read> = if path == Path::new("-") {
                Box::new(std::io::stdin().lock())
            } else {
                Box::new(
                    std::fs::File::open(path)
                        .map_err(|error| format!("{}: {error}", path.display()))?,
                )
            };
            reader
                .read_exact(&mut pixels)
                .map_err(|error| format!("{}: {error}", path.display()))?;
        }
        None => {
            for frame in 0..framesets {
                let mut offset: usize = frame * frameset_bytes;
                for (camera, &size) in sizes.iter().enumerate() {
                    let raster: Raster =
                        read_pgm(&clip_dir.join(format!("frame_{frame:03}_cam{camera}.pgm")))?;
                    if (raster.width, raster.height) != clip.resolution_wh[camera] {
                        return Err(format!(
                            "frame {frame} cam {camera}: {}x{} is not the clip's resolution",
                            raster.width, raster.height
                        ));
                    }
                    pixels[offset..offset + size].copy_from_slice(&raster.pixels);
                    offset += size;
                }
            }
        }
    }
    Ok(ReplayInput {
        clip,
        calibration,
        imu,
        pixels,
    })
}
