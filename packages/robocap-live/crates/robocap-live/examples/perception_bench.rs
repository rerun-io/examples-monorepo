//! Perception timings: 4 perspective KeyNet crops sampled from 1080p frames (crop cameras, maps, bilinear remap),
//! their decode (heatmaps, distance, back-projection through the lens), and one camera's DetNet input (the shipped fp16 feed):
//! the one-pass pool of the small image's rows against the old chain (pad to 640x480, pool to f32, convert to fp16).
//!
//! Inputs are the golden test data (s66 rig, frame 448's four tracked views and Python's KeyNet outputs for them), embedded at
//! build time so the binary runs alone on the cap:
//!     taskset -c 0 ./perception_bench 300     # an A55 core
//!     taskset -c 4 ./perception_bench 300     # an A76 core

use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use kornia_imgproc::padding::{Padding2D, PaddingMode, spatial_padding};
use kornia_staging_3d::camera::virtual_camera::maps_from_virtual_pinhole_kb4_f32;
use robocap_live::kornia_ext::remap::remap_f32_from_u8 as remap_f32_from_u8_zero_border;
use robocap_live::kornia_ext::pool::pool4_mean_f32;
use robocap_live::frame::isometry_from_matrix;
use robocap_live::frame::{CameraFrame, FULL_SIZE, FrameMeta, Luma, NUM_CAMERAS, Rig, SMALL_SIZE};
use robocap_live::hands::CropSource;
use robocap_live::hands::camera::Lens;
use robocap_live::hands::estimator::{PerspectiveKeyNet, ViewRequest};
use robocap_live::hands::letterbox::{BarLetterbox, NET_SIZE};
use robocap_live::hands::perspective::{CROP_IMAGE_SIZE, CropMaps, MIN_RAY_Z};
use robocap_live::nets::golden::{f32_values, keynet_from_rows};
use robocap_live::nets::rknn::{
    ImageFeed, InputData, POOLED_HEIGHT, POOLED_WIDTH, PooledInput, f16_bytes_from_f32,
};
use robocap_live::nets::{
    DETNET_HEIGHT, DETNET_WIDTH, DetNetRaw, HandNets, KeyNetRaw, NUM_LANDMARKS, NetFrame, NetsError,
};
use serde_json::Value;

const MANIFEST: &str = include_str!("../tests/data/perception/manifest.json");
const RIG: &str = include_str!("../tests/data/perception/rig.json");
const RAW: &[u8] = include_bytes!("../tests/data/perception/keynet_raw_f32.bin");

struct Replay(Vec<KeyNetRaw>);

impl HandNets for Replay {
    fn detnet(&mut self, _frames: &[NetFrame<'_>]) -> Result<Vec<DetNetRaw>, NetsError> {
        Ok(Vec::new())
    }
    fn keynet(
        &mut self,
        crops: &[&[f32]],
        _keypoints: &[[f32; 3 * NUM_LANDMARKS]],
    ) -> Result<Vec<KeyNetRaw>, NetsError> {
        Ok(self.0.iter().take(crops.len()).cloned().collect())
    }
    fn describe(&self) -> String {
        "replay".into()
    }
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
        .unwrap_or_default()
}

fn stats(name: &str, mut samples: Vec<f64>) {
    samples.sort_by(f64::total_cmp);
    let pick = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    println!(
        "{name:<34} median {:7.3} ms   p90 {:7.3} ms   min {:7.3} ms",
        pick(0.5),
        pick(0.9),
        pick(0.0)
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let iterations: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200);
    let manifest: Value = serde_json::from_str(MANIFEST)?;
    let rig: Rig = serde_json::from_str(RIG)?;
    let call = &manifest["keynet"]["calls"][0];
    let m: Vec<Vec<f64>> = call["world_from_rig"]
        .as_array()
        .ok_or("pose")?
        .iter()
        .map(floats)
        .collect();
    let world_from_rig = isometry_from_matrix(&std::array::from_fn(|i| m[i / 4][i % 4]))
        .ok_or("a non-finite headset pose")?;
    let requests: Vec<ViewRequest> = call["views"]
        .as_array()
        .ok_or("views")?
        .iter()
        .map(|view| {
            let points: Vec<Vec<f64>> = view["landmarks_world"]
                .as_array()
                .map(|a| a.iter().map(floats).collect())
                .unwrap_or_default();
            let circle = floats(&view["circle_net"]);
            let planning_pose_landmarks_world = (points.len() == NUM_LANDMARKS)
                .then(|| std::array::from_fn(|i| [points[i][0], points[i][1], points[i][2]]));
            ViewRequest {
                camera: view["camera"].as_u64().unwrap_or(0) as usize,
                side: view["side"].as_u64().unwrap_or(0) as usize,
                source: if planning_pose_landmarks_world.is_some() {
                    CropSource::Pose
                } else {
                    CropSource::DetNet
                },
                planning_pose_landmarks_world,
                circle_net: Some([circle[0] as f32, circle[1] as f32, circle[2] as f32]),
            }
        })
        .collect();
    let raw_f32: Vec<f32> = f32_values(RAW).ok_or("keynet_raw_f32.bin: not whole f32 values")?;
    let raw: Vec<KeyNetRaw> = keynet_from_rows(&raw_f32)
        .into_iter()
        .take(requests.len())
        .collect();
    // A textured synthetic 1080p frame per camera (sampling cost does not depend on content).
    let frames: Vec<Luma> = (0..NUM_CAMERAS)
        .map(|c| {
            Image::new(
                FULL_SIZE,
                (0..FULL_SIZE.width * FULL_SIZE.height)
                    .map(|i| ((i * 7 + c * 13 + i / 1920 * 3) % 251) as u8)
                    .collect(),
            )
            .map(Arc::new)
        })
        .collect::<Result<_, _>>()?;
    let camera_frames: Vec<CameraFrame> = frames
        .iter()
        .map(|full| CameraFrame {
            meta: FrameMeta::default(),
            full: full.clone(),
        })
        .collect();
    let full: [Option<&CameraFrame>; NUM_CAMERAS] =
        std::array::from_fn(|c| Some(&camera_frames[c]));
    let mut estimator = PerspectiveKeyNet::new(&rig, manifest["phi"].as_f64().unwrap_or(1.0))?;
    let mut nets = Replay(raw);
    let letterbox = BarLetterbox::robocap();
    let small = Image::new(
        SMALL_SIZE,
        frames[0].as_slice()[..SMALL_SIZE.width * SMALL_SIZE.height].to_vec(),
    )?;
    let mut net_frame = Image::from_size_val(NET_SIZE, 0u8)?;
    let mut pooled_f32: Vec<f32> = vec![0.0; POOLED_WIDTH * POOLED_HEIGHT];
    let mut pooled_f16: Vec<u8> = vec![0; 2 * POOLED_WIDTH * POOLED_HEIGHT];
    let mut pooled = PooledInput::new(ImageFeed::F16Native);

    let (mut plan, mut crops, mut after, mut total) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut chain_ms, mut one_pass_ms) = (Vec::new(), Vec::new());
    for iteration in 0..iterations + 10 {
        let start = Instant::now();
        std::hint::black_box(
            requests
                .iter()
                .map(|r| estimator.plan(&world_from_rig, r))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let plan_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let (estimates, crops_ms) =
            estimator.estimate(&mut nets, &full, &world_from_rig, &requests)?;
        let elapsed = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        spatial_padding(
            &small,
            &mut net_frame,
            Padding2D {
                top: 60,
                bottom: 60,
                left: 0,
                right: 0,
            },
            PaddingMode::Constant,
            [0u8],
        )?;
        pool4_mean_f32(
            net_frame.as_slice(),
            DETNET_WIDTH,
            DETNET_HEIGHT,
            &mut pooled_f32,
        )?;
        f16_bytes_from_f32(&pooled_f32, 1.0 / 255.0, &mut pooled_f16)?;
        let chain = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let InputData::Native(one_pass) = pooled.fill(&letterbox.net_frame(&small)?)? else {
            return Err("not an fp16 input".into());
        };
        let one = start.elapsed().as_secs_f64() * 1e3;
        if one_pass != pooled_f16.as_slice() {
            return Err("the one-pass DetNet input differs from the old chain".into());
        }
        if estimates.len() != requests.len() {
            return Err("estimate count".into());
        }
        if iteration >= 10 {
            plan.push(plan_ms);
            crops.push(crops_ms);
            after.push(elapsed - crops_ms);
            total.push(elapsed);
            chain_ms.push(chain);
            one_pass_ms.push(one);
        }
    }
    // The two halves of crop sampling, for 4 crops.
    let plans = requests
        .iter()
        .map(|r| estimator.plan(&world_from_rig, r))
        .collect::<Result<Vec<_>, _>>()?;
    let mut maps = CropMaps::new()?;
    let mut crop = Image::from_size_val(CROP_IMAGE_SIZE, 0f32)?;
    let (mut maps_ms, mut remap_ms) = (Vec::new(), Vec::new());
    for iteration in 0..iterations + 10 {
        let (mut m_ms, mut r_ms) = (0.0, 0.0);
        for (request, plan) in requests.iter().zip(&plans) {
            let Lens::Fisheye(lens) = estimator.models()[request.camera].lens() else {
                return Err("not a fisheye".into());
            };
            let start = Instant::now();
            maps_from_virtual_pinhole_kb4_f32(
                lens,
                &plan.crop.virtual_pinhole(),
                MIN_RAY_Z,
                &mut maps.map_x,
                &mut maps.map_y,
            )?;
            m_ms += start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            remap_f32_from_u8_zero_border(
                &frames[request.camera],
                &mut crop,
                &maps.map_x,
                &maps.map_y,
                1.0 / 255.0,
            )?;
            r_ms += start.elapsed().as_secs_f64() * 1e3;
        }
        if iteration >= 10 {
            maps_ms.push(m_ms);
            remap_ms.push(r_ms);
        }
    }
    println!(
        "perception bench: {} views, {iterations} iterations",
        requests.len()
    );
    stats("  of which crop maps (4)", maps_ms);
    stats("  of which bilinear remap (4)", remap_ms);
    stats("plan (crop cameras + inputs)", plan);
    stats("plan + sample 4 crops (maps + remap)", crops);
    stats("net replay + decode (heatmaps, distance, lens)", after);
    stats("estimate total (net = replay)", total);
    stats("detnet input, old chain (1 camera)", chain_ms);
    stats("detnet input, one pass (1 camera)", one_pass_ms);
    Ok(())
}
