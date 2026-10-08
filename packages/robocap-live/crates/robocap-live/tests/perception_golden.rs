//! Golden tests of the hand perception port against handtrack @ 54eaf309 (`tools/golden_perception.py`, data in
//! `tests/data/perception/`, < 2 MB).
//!
//! Python computes in float32, the port in float64 (camera geometry) and float32 (decoding), so geometry is compared with
//! tolerances a few float32 ulps wide at the magnitudes involved (pixel coordinates up to ~2000 have a float32 ulp of 1.2e-4):
//! - projections 2e-3 px (relative 2e-6 for the far extrapolated behind-camera points), unprojected rays 2e-5 rad;
//! - crop maps 2e-3 px, crop pixels 2e-3 (in [0, 1]; a 1e-4 px position difference times the steepest image gradients;
//!   the crop file itself is u16-quantised, < 8e-6), crop rotations 1e-5, focals 1e-5 relative;
//! - keypoint inputs 1e-5, decoded crop points and distances 1e-4 px / 1e-3 mm (identical f32 formulas), final net-frame
//!   keypoints 2e-3 px.
//!
//! The DetNet letterbox from the runtime's 3x3 area-mean small image equals Python's area variant exactly; against the
//! antialiased-bilinear small image the Python runs used it differs by 1.8 levels on average, 39 at most, on the hand edges (the
//! two downscale kernels differ; stated, not a parity target).
//!
//! Unprojection is compared where handtrack's own solve converges: in the outermost image corners (r > ~976 px, theta > ~85 deg;
//! 0 to 4 % of a camera's pixels) its 10 Newton steps on (a, b, 1) do not reproject onto the pixel, while kornia's KB4 inverse is
//! exact wherever an inverse exists (past r ~1080 px the polynomial has none). Acquisitions there differ; nowhere else.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kornia_image::{Image, ImageSize};
use kornia_staging_3d::camera::virtual_camera::maps_from_virtual_pinhole_f32;
use kornia_staging_imgproc::resize::resize_area_u8;
use nalgebra::{Matrix3, Vector2, Vector3};
use robocap_live::frame::isometry_from_matrix;
use robocap_live::frame::{CameraFrame, FULL_SIZE, FrameMeta, Luma, NUM_CAMERAS, Rig, SMALL_SIZE};
use robocap_live::hands::CropSource;
use robocap_live::hands::camera::{Lens, rig_models};
use robocap_live::hands::detect::{decode_detections, detect};
use robocap_live::hands::estimator::{PerspectiveKeyNet, ViewRequest};
use robocap_live::hands::heatmaps::{decode_distance, decode_heatmaps};
use robocap_live::hands::letterbox::{BarLetterbox, NET_SIZE};
use robocap_live::hands::perspective::{CropCamera, CropMaps};
use robocap_live::nets::golden::f32_values;
use robocap_live::nets::{
    CROP_LEN, DISTANCE_LEN, DetNetRaw, HEATMAP_LEN, HandNets, KeyNetRaw, NUM_LANDMARKS, NetFrame,
    NetsError,
};
use serde_json::Value;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// The golden data (`PERCEPTION_GOLDEN_DIR` overrides, e.g. to run the cross-built test binary on the cap).
fn data_dir() -> PathBuf {
    std::env::var_os("PERCEPTION_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/perception"))
}

fn manifest() -> Result<Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(
        data_dir().join("manifest.json"),
    )?)?)
}

/// A JSON number, with null read as NaN.
fn num(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

fn vec1(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .map(|items| items.iter().map(num).collect())
        .unwrap_or_default()
}

fn vec2(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .map(|items| items.iter().map(vec1).collect())
        .unwrap_or_default()
}

fn vec3(value: &Value) -> Vec<Vec<Vec<f64>>> {
    value
        .as_array()
        .map(|items| items.iter().map(vec2).collect())
        .unwrap_or_default()
}

fn f32_file(name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(f32_values(&std::fs::read(data_dir().join(name))?)
        .ok_or_else(|| format!("{name}: not whole f32 values"))?)
}

/// NaN-aware closeness.
fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() <= tolerance
}

fn rig() -> Result<Rig, Box<dyn std::error::Error>> {
    Ok(Rig::load(&data_dir().join("rig.json"))?)
}

/// The masked 1920x1080 frames of `frame` (zeros outside the stored crop footprints), per camera.
fn masked_frames(manifest: &Value, frame: u64) -> Result<Vec<Luma>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(data_dir().join("frame_pixels_u8.bin"))?;
    let mut frames: Vec<Vec<u8>> = vec![vec![0u8; FULL_SIZE.width * FULL_SIZE.height]; NUM_CAMERAS];
    for entry in manifest["frame_pixels"]["entries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if entry["frame"].as_u64() != Some(frame) {
            continue;
        }
        let camera = entry["camera"].as_u64().ok_or("camera")? as usize;
        let mut mask = vec![false; FULL_SIZE.width * FULL_SIZE.height];
        for rect in vec2(&entry["rects"]) {
            let [x0, y0, x1, y1] = [
                rect[0] as usize,
                rect[1] as usize,
                rect[2] as usize,
                rect[3] as usize,
            ];
            for y in y0..y1 {
                mask[y * FULL_SIZE.width + x0..y * FULL_SIZE.width + x1].fill(true);
            }
        }
        let mut source = bytes[entry["offset"].as_u64().ok_or("offset")? as usize..].iter();
        for (pixel, inside) in frames[camera].iter_mut().zip(&mask) {
            if *inside {
                *pixel = *source.next().ok_or("short frame data")?;
            }
        }
    }
    frames
        .into_iter()
        .map(|data| Ok(Arc::new(Image::new(FULL_SIZE, data)?)))
        .collect()
}

#[test]
fn camera_projection_and_unprojection_match_handtrack() -> TestResult {
    let manifest = manifest()?;
    let models = rig_models(&rig()?)?;
    let points = vec2(&manifest["camera"]["points_cam"]);
    let pixels = vec2(&manifest["camera"]["pixels"]);
    let (mut worst_px, mut worst_rad) = (0.0f64, 0.0f64);
    for (camera, model) in models.iter().enumerate() {
        for (point, expected) in points
            .iter()
            .zip(vec2(&manifest["camera"]["project"][camera]))
        {
            let projected = model
                .project(&Vector3::new(point[0], point[1], point[2]))
                .ok_or("no projection")?;
            let error = (projected - Vector2::new(expected[0], expected[1])).amax();
            let scale = expected[0].abs().max(expected[1].abs()).max(1000.0) / 1000.0;
            assert!(
                error <= 2e-3 * scale,
                "camera {camera} point {point:?}: {projected} vs {expected:?}"
            );
            worst_px = worst_px.max(error / scale);
        }
        for (pixel, expected) in pixels
            .iter()
            .zip(vec2(&manifest["camera"]["unproject"][camera]))
        {
            let px = Vector2::new(pixel[0], pixel[1]);
            let ray = model.unproject(&px);
            let back = model.project(&ray).ok_or("no projection")?;
            let python = Vector3::new(expected[0], expected[1], expected[2]);
            let python_back = model.project(&python).ok_or("no projection")?;
            if (back - px).amax() > 1e-6 || (python_back - px).amax() > 0.5 {
                // The outermost image corners (r > ~976 px from the centre, theta > ~85 deg): handtrack's 10 Newton steps on (a, b, 1)
                // from the pinhole guess do not converge there (and past r ~1080 px the KB4 polynomial has no inverse at all), so its
                // ray does not reproject onto the pixel. kornia's solve is exact wherever an inverse exists. Nothing to compare.
                println!(
                    "camera {camera} pixel {pixel:?}: handtrack's ray reprojects to {python_back:?}, kornia's to {back:?}; skipped"
                );
                continue;
            }
            let angle = ray.angle(&Vector3::new(expected[0], expected[1], expected[2]));
            assert!(
                angle <= 2e-5,
                "camera {camera} pixel {pixel:?}: {ray} vs {expected:?} ({angle} rad)"
            );
            worst_rad = worst_rad.max(angle);
        }
    }
    println!(
        "camera golden: worst projection {worst_px:.2e} px (per 1000 px), worst unprojection {worst_rad:.2e} rad"
    );
    Ok(())
}

#[test]
fn letterbox_maps_match_handtrack() -> TestResult {
    let manifest = manifest()?;
    let letterbox = BarLetterbox::robocap();
    let full = vec2(&manifest["letterbox_points"]["full"]);
    for ((uv, net), back) in full
        .iter()
        .zip(vec2(&manifest["letterbox_points"]["to_net"]))
        .zip(vec2(&manifest["letterbox_points"]["from_net_of_full"]))
    {
        let to_net = letterbox.to_net(&Vector2::new(uv[0], uv[1]));
        let from_net = letterbox.from_net(&Vector2::new(uv[0], uv[1]));
        assert!(
            (to_net - Vector2::new(net[0], net[1])).amax() < 1e-4,
            "{to_net} vs {net:?}"
        );
        assert!(
            (from_net - Vector2::new(back[0], back[1])).amax() < 1e-3,
            "{from_net} vs {back:?}"
        );
        assert!((letterbox.from_net(&to_net) - Vector2::new(uv[0], uv[1])).amax() < 1e-9);
    }
    Ok(())
}

#[test]
fn the_detnet_frame_from_the_area_small_image_matches_handtrack() -> TestResult {
    let manifest = manifest()?;
    let bytes = std::fs::read(data_dir().join("letterbox_u8.bin"))?;
    let letterbox = BarLetterbox::robocap();
    for entry in manifest["letterbox"]["entries"]
        .as_array()
        .into_iter()
        .flatten()
    {
        let frame = entry["frame"].as_u64().ok_or("frame")?;
        let camera = entry["camera"].as_u64().ok_or("camera")? as usize;
        let frames = masked_frames(&manifest, frame)?;
        let mut small = Image::from_size_val(SMALL_SIZE, 0u8)?;
        resize_area_u8(&frames[camera], &mut small)?;
        let net_frame = letterbox.net_frame(&small)?;
        let mut net = vec![0u8; NET_SIZE.width * NET_SIZE.height];
        net[net_frame.top * NET_SIZE.width..][..net_frame.pixels.len()]
            .copy_from_slice(net_frame.pixels);
        let b = vec1(&entry["box"])
            .iter()
            .map(|v| *v as usize)
            .collect::<Vec<_>>();
        let (x0, y0, x1, y1) = (b[0], b[1], b[2], b[3]);
        let offset = entry["offset"].as_u64().ok_or("offset")? as usize;
        let expected = &bytes[offset..offset + (x1 - x0) * (y1 - y0)];
        let (mut max, mut sum, mut outside) = (0i32, 0i64, 0usize);
        for (index, value) in net.iter().enumerate() {
            let (x, y) = (index % 640, index / 640);
            if x >= x0 && x < x1 && y >= y0 && y < y1 {
                let difference =
                    (i32::from(*value) - i32::from(expected[(y - y0) * (x1 - x0) + x - x0])).abs();
                max = max.max(difference);
                sum += i64::from(difference);
            } else if *value != 0 {
                outside += 1;
            }
        }
        let mean = sum as f64 / ((x1 - x0) * (y1 - y0)) as f64;
        println!(
            "letterbox f{frame} c{camera} vs Python {}: max |diff| {max}, mean {mean:.3}, nonzero outside {outside}",
            entry["variant"]
        );
        assert_eq!(outside, 0);
        match entry["variant"].as_str() {
            Some("area") => assert_eq!(max, 0, "the area letterbox must match exactly"),
            // Antialiased bilinear (kernel [1 2 3 2 1]/9 per axis) against the box mean: a few levels on edges.
            // Measured on s66: mean 1.8 levels, max 39 on the hand edges.
            _ => assert!(
                mean < 3.0 && max <= 64,
                "bilinear variant: mean {mean} max {max}"
            ),
        }
    }
    Ok(())
}

#[test]
fn detnet_decode_matches_handtrack() -> TestResult {
    let manifest = manifest()?;
    let d = &manifest["detnet"];
    let (center, radius, logit) = (
        vec3(&d["center"]),
        vec2(&d["radius"]),
        vec2(&d["presence_logit"]),
    );
    let (circle, probability, boxes) =
        (vec3(&d["circle"]), vec2(&d["probability"]), vec3(&d["box"]));
    let present = d["present_0_8"].as_array().ok_or("present")?;
    for i in 0..center.len() {
        let raw = DetNetRaw {
            center: [
                [center[i][0][0] as f32, center[i][0][1] as f32],
                [center[i][1][0] as f32, center[i][1][1] as f32],
            ],
            radius: [radius[i][0] as f32, radius[i][1] as f32],
            presence_logit: [logit[i][0] as f32, logit[i][1] as f32],
        };
        let decoded = decode_detections(&raw);
        for side in 0..2 {
            for (k, (ours, theirs)) in decoded.circle_net[side]
                .iter()
                .zip(&circle[i][side])
                .enumerate()
            {
                assert!(close(f64::from(*ours), *theirs, 1e-4), "{i} {side} {k}");
            }
            assert!(
                close(
                    f64::from(decoded.probability[side]),
                    probability[i][side],
                    1e-7
                ),
                "{i} {side}: {} vs {}",
                decoded.probability[side],
                probability[i][side]
            );
            assert_eq!(
                decoded.present(side, 0.8),
                present[i][side].as_bool().unwrap_or(false)
            );
            let b = decoded.box_net(side).ok_or("box")?;
            for k in 0..4 {
                assert!(close(f64::from(b[k]), boxes[i][side][k], 1e-4));
            }
        }
    }
    Ok(())
}

#[test]
fn heatmap_and_distance_decoding_match_handtrack() -> TestResult {
    let manifest = manifest()?;
    let heatmaps = f32_file("heatmaps_f32.bin")?;
    let distance = f32_file("distance_f32.bin")?;
    let points = vec3(&manifest["heatmaps"]["points_crop"]);
    let confidence = vec2(&manifest["heatmaps"]["confidence"]);
    let d_rel = vec2(&manifest["heatmaps"]["d_rel_mm"]);
    for crop in 0..points.len() {
        let (decoded, peaks) =
            decode_heatmaps(&heatmaps[crop * HEATMAP_LEN..(crop + 1) * HEATMAP_LEN])
                .ok_or("heatmaps")?;
        let mm = decode_distance(&distance[crop * DISTANCE_LEN..(crop + 1) * DISTANCE_LEN])
            .ok_or("distance")?;
        for k in 0..NUM_LANDMARKS {
            for axis in 0..2 {
                assert!(
                    close(f64::from(decoded[k][axis]), points[crop][k][axis], 1e-4),
                    "crop {crop} landmark {k}: {:?} vs {:?}",
                    decoded[k],
                    points[crop][k]
                );
            }
            assert_eq!(f64::from(peaks[k]), confidence[crop][k]);
            assert!(
                close(f64::from(mm[k]), d_rel[crop][k], 1e-3),
                "crop {crop} landmark {k}: {} vs {}",
                mm[k],
                d_rel[crop][k]
            );
        }
    }
    Ok(())
}

#[test]
fn crop_maps_match_handtrack() -> TestResult {
    let manifest = manifest()?;
    let models = rig_models(&rig()?)?;
    let mut maps = CropMaps::new()?;
    for entry in manifest["crop_maps"].as_array().into_iter().flatten() {
        let rotation = vec2(&entry["rotation"]);
        let crop = CropCamera {
            rotation: Matrix3::from_fn(|r, c| rotation[r][c]),
            focal: num(&entry["focal"]),
            mirror: entry["mirror"].as_bool().unwrap_or(false),
        };
        let camera = entry["camera"].as_u64().ok_or("camera")? as usize;
        match models[camera].lens() {
            Lens::Pinhole(lens) => maps_from_virtual_pinhole_f32(
                lens,
                &crop.virtual_pinhole(),
                1e-6,
                &mut maps.map_x,
                &mut maps.map_y,
            )?,
            Lens::Fisheye(lens) => maps_from_virtual_pinhole_f32(
                lens,
                &crop.virtual_pinhole(),
                1e-6,
                &mut maps.map_x,
                &mut maps.map_y,
            )?,
        }
        let mut worst = 0.0f64;
        for ((index, source), valid) in vec1(&entry["pixel_index"])
            .iter()
            .zip(vec2(&entry["source"]))
            .zip(entry["valid"].as_array().ok_or("valid")?)
        {
            let i = *index as usize;
            let (x, y) = (
                f64::from(maps.map_x.as_slice()[i]),
                f64::from(maps.map_y.as_slice()[i]),
            );
            if valid.as_bool().unwrap_or(false) {
                let scale = source[0].abs().max(source[1].abs()).max(1000.0) / 1000.0;
                let error = (x - source[0]).abs().max((y - source[1]).abs()) / scale;
                assert!(
                    error <= 2e-3,
                    "{} pixel {i}: ({x}, {y}) vs {source:?}",
                    entry["name"]
                );
                worst = worst.max(error);
            } else {
                assert!(
                    x.is_nan() && y.is_nan(),
                    "{} pixel {i} should sample nothing",
                    entry["name"]
                );
            }
        }
        println!(
            "crop map {}: worst {worst:.2e} px (per 1000 px)",
            entry["name"]
        );
    }
    Ok(())
}

/// A KeyNet stand-in: returns Python's raw outputs and records how far the crops and keypoint inputs it got are from Python's.
struct ReplayNets {
    crops: Vec<Vec<f32>>,
    features: Vec<Vec<f64>>,
    raw: Vec<KeyNetRaw>,
    worst_crop: f32,
    mean_crop: f64,
    worst_feature: f64,
}

impl HandNets for ReplayNets {
    fn detnet(&mut self, _frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Err(NetsError::Run {
            net: "detnet",
            message: "replay has no DetNet".into(),
        })
    }

    fn keynet(
        &mut self,
        crops: &[&[f32]],
        keypoints: &[[f32; 3 * NUM_LANDMARKS]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        if crops.len() != self.raw.len() || keypoints.len() != self.raw.len() {
            return Err(NetsError::Input {
                net: "keynet",
                message: format!("{} crops, expected {}", crops.len(), self.raw.len()),
            });
        }
        let mut total = 0.0f64;
        for (i, crop) in crops.iter().enumerate() {
            for (a, b) in crop.iter().zip(&self.crops[i]) {
                self.worst_crop = self.worst_crop.max((a - b).abs());
                total += f64::from((a - b).abs());
            }
            for (a, b) in keypoints[i].iter().zip(&self.features[i]) {
                self.worst_feature = self.worst_feature.max((f64::from(*a) - b).abs());
            }
        }
        self.mean_crop = total / (crops.len() * CROP_LEN).max(1) as f64;
        Ok(self.raw.clone())
    }

    fn describe(&self) -> String {
        "replay".into()
    }
}

#[test]
fn perspective_keynet_matches_handtrack_end_to_end() -> TestResult {
    let manifest = manifest()?;
    let keynet = &manifest["keynet"];
    let crop_bytes = std::fs::read(data_dir().join("crops_u16.bin"))?;
    let crops_all: Vec<f32> = crop_bytes.as_chunks::<2>().0.iter().map(|c| f32::from(u16::from_le_bytes([c[0], c[1]])) / 65535.0).collect();
    let raw_all = f32_file("keynet_raw_f32.bin")?;
    let raw_width = HEATMAP_LEN + DISTANCE_LEN + 2;
    let mut estimator = PerspectiveKeyNet::new(&rig()?, num(&manifest["phi"]))?;
    let mut first = 0usize;
    for call in keynet["calls"].as_array().ok_or("calls")? {
        let name = call["name"].as_str().unwrap_or("?");
        let frames = masked_frames(&manifest, call["frame"].as_u64().ok_or("frame")?)?;
        let frames: Vec<CameraFrame> = frames
            .into_iter()
            .map(|full| CameraFrame {
                meta: FrameMeta::default(),
                full,
            })
            .collect();
        let full: [Option<&CameraFrame>; NUM_CAMERAS] =
            std::array::from_fn(|camera| Some(&frames[camera]));
        let m = vec2(&call["world_from_rig"]);
        let world_from_rig = isometry_from_matrix(&std::array::from_fn(|i| m[i / 4][i % 4]))
            .ok_or("a non-finite headset pose")?;
        let views = call["views"].as_array().ok_or("views")?;
        let requests: Vec<ViewRequest> = views
            .iter()
            .map(|view| {
                let circle = vec1(&view["circle_net"]);
                let landmarks = view["landmarks_world"].as_array().map(|_| {
                    let points = vec2(&view["landmarks_world"]);
                    std::array::from_fn(|i| [points[i][0], points[i][1], points[i][2]])
                });
                ViewRequest {
                    camera: view["camera"].as_u64().unwrap_or(0) as usize,
                    side: view["side"].as_u64().unwrap_or(0) as usize,
                    planning_pose_landmarks_world: landmarks,
                    circle_net: Some([circle[0] as f32, circle[1] as f32, circle[2] as f32]),
                    source: if landmarks.is_some() {
                        CropSource::Pose
                    } else {
                        CropSource::DetNet
                    },
                }
            })
            .collect();
        let n = requests.len();
        let raw: Vec<KeyNetRaw> = (0..n)
            .map(|i| {
                let row = &raw_all[(first + i) * raw_width..(first + i + 1) * raw_width];
                KeyNetRaw {
                    heatmaps: row[..HEATMAP_LEN].to_vec(),
                    distance: row[HEATMAP_LEN..HEATMAP_LEN + DISTANCE_LEN].to_vec(),
                    presence_logit: row[raw_width - 2],
                    pinch_logit: Some(row[raw_width - 1]).filter(|v| v.is_finite()),
                }
            })
            .collect();
        let mut nets = ReplayNets {
            crops: (0..n)
                .map(|i| crops_all[(first + i) * CROP_LEN..(first + i + 1) * CROP_LEN].to_vec())
                .collect(),
            features: vec2(&call["keypoint_input"]),
            raw,
            worst_crop: 0.0,
            mean_crop: 0.0,
            worst_feature: 0.0,
        };
        // The crop cameras.
        let rotations = vec3(&call["crop_rotation"]);
        let focals = vec1(&call["crop_focal"]);
        for (i, request) in requests.iter().enumerate() {
            let plan = estimator.plan(&world_from_rig, request)?;
            let rotation_error = (0..9)
                .map(|k| (plan.crop.rotation[(k / 3, k % 3)] - rotations[i][k / 3][k % 3]).abs())
                .fold(0.0, f64::max);
            assert!(
                rotation_error <= 1e-5 || (focals[i].is_nan() && plan.crop.focal.is_nan()),
                "{name} view {i}: rotation off by {rotation_error}"
            );
            assert!(
                close(plan.crop.focal, focals[i], 1e-5 * focals[i].abs()),
                "{name} view {i}: focal {} vs {}",
                plan.crop.focal,
                focals[i]
            );
            assert_eq!(
                plan.crop.mirror,
                call["crop_mirror"][i].as_bool().unwrap_or(false)
            );
        }
        let (estimates, _) = estimator.estimate(&mut nets, &full, &world_from_rig, &requests)?;
        println!(
            "{name}: crops worst |diff| {:.2e} (mean {:.2e}), keypoint input worst {:.2e}",
            nets.worst_crop, nets.mean_crop, nets.worst_feature
        );
        assert!(
            nets.worst_crop <= 2e-3,
            "{name}: crop pixels off by {}",
            nets.worst_crop
        );
        assert!(
            nets.worst_feature <= 1e-5,
            "{name}: keypoint input off by {}",
            nets.worst_feature
        );
        let (points, d_rel, presence, confidence) = (
            vec3(&call["points_net"]),
            vec2(&call["d_rel_mm"]),
            vec1(&call["presence"]),
            vec2(&call["confidence"]),
        );
        let pinch = vec1(&call["pinch"]);
        let mut worst_net = 0.0f64;
        for (i, estimate) in estimates.iter().enumerate() {
            assert!(
                close(f64::from(estimate.presence), presence[i], 1e-6),
                "{name} view {i}: presence {} vs {}",
                estimate.presence,
                presence[i]
            );
            assert!(
                close(
                    f64::from(estimate.pinch.unwrap_or(f32::NAN)),
                    pinch[i],
                    1e-6
                ),
                "{name} view {i}: pinch"
            );
            for k in 0..NUM_LANDMARKS {
                let error = (f64::from(estimate.points_net[k][0]) - points[i][k][0])
                    .abs()
                    .max((f64::from(estimate.points_net[k][1]) - points[i][k][1]).abs());
                assert!(
                    error <= 2e-3,
                    "{name} view {i} landmark {k}: {:?} vs {:?}",
                    estimate.points_net[k],
                    points[i][k]
                );
                worst_net = worst_net.max(error);
                assert!(
                    close(f64::from(estimate.d_rel_mm[k]), d_rel[i][k], 1e-3),
                    "{name} view {i} landmark {k}: d_rel"
                );
                assert_eq!(
                    f64::from(estimate.confidence[k]),
                    confidence[i][k],
                    "{name} view {i} landmark {k}: confidence"
                );
                let back = estimator.letterbox().from_net(&Vector2::new(
                    f64::from(estimate.points_net[k][0]),
                    f64::from(estimate.points_net[k][1]),
                ));
                assert!(
                    !estimate.usable
                        || (back
                            - Vector2::new(
                                f64::from(estimate.points_px[k][0]),
                                f64::from(estimate.points_px[k][1])
                            ))
                        .amax()
                            < 2e-3
                );
            }
        }
        println!("{name}: final net-frame keypoints worst |diff| {worst_net:.2e} px");
        // Optional evidence: the port's crops and keypoints, for a side-by-side Rerun view against Python's.
        if let Some(dir) = std::env::var_os("PERCEPTION_DUMP_DIR").map(PathBuf::from) {
            std::fs::create_dir_all(&dir)?;
            let crops: Vec<u8> = estimator
                .crops()
                .iter()
                .flat_map(|c| c.as_slice().iter().flat_map(|v| v.to_le_bytes()))
                .collect();
            std::fs::write(dir.join(format!("{name}_crops_f32.bin")), crops)?;
            let points: Vec<Value> = estimates.iter().map(|e| serde_json::json!({"points_net": e.points_net, "points_px": e.points_px, "presence": e.presence})).collect();
            std::fs::write(
                dir.join(format!("{name}_estimates.json")),
                serde_json::to_string(&points)?,
            )?;
        }
        first += n;
    }
    Ok(())
}

#[test]
fn detect_letterboxes_runs_and_decodes() -> TestResult {
    struct Fixed(Vec<(Vec<u8>, usize)>);
    impl HandNets for Fixed {
        fn detnet(&mut self, frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
            self.0 = frames.iter().map(|f| (f.pixels.to_vec(), f.top)).collect();
            Ok(vec![
                DetNetRaw {
                    center: [[0.5, 0.25], [0.1, 0.9]],
                    radius: [0.05, 0.1],
                    presence_logit: [3.0, -3.0]
                };
                frames.len()
            ])
        }
        fn keynet(
            &mut self,
            _crops: &[&[f32]],
            _keypoints: &[[f32; 3 * NUM_LANDMARKS]],
        ) -> Result<Vec<KeyNetRaw>, NetsError> {
            Ok(Vec::new())
        }
        fn describe(&self) -> String {
            "fixed".into()
        }
    }
    let small = Image::new(
        SMALL_SIZE,
        (0..SMALL_SIZE.width * SMALL_SIZE.height)
            .map(|i| (i % 251) as u8 + 1)
            .collect(),
    )?;
    let mut nets = Fixed(Vec::new());
    let stale = Image::<u8, 1>::from_size_val(SMALL_SIZE, 200)?;
    detect(
        &mut nets,
        &BarLetterbox::robocap(),
        &[&stale, &stale, &stale],
    )?;
    let detections = detect(&mut nets, &BarLetterbox::robocap(), &[&small, &small])?;
    assert_eq!(
        nets.0.len(),
        2,
        "the second call hands the net its own batch only"
    );
    assert_eq!(detections.len(), 2);
    assert_eq!(detections[0].circle_net[0], [320.0, 120.0, 32.0]);
    assert!(detections[0].present(0, 0.8) && !detections[0].present(1, 0.8));
    // The net gets the small image's rows as they are, at net row 60 (the bars are implicit).
    let (pixels, top) = &nets.0[0];
    assert_eq!((pixels.as_slice(), *top), (small.as_slice(), 60));
    let wrong = Image::<u8, 1>::from_size_val(
        ImageSize {
            width: 640,
            height: 480,
        },
        0,
    )?;
    assert!(detect(&mut nets, &BarLetterbox::robocap(), &[&wrong]).is_err());
    Ok(())
}
