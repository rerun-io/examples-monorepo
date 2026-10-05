//! Read-only, two-rig catalog adapter. No DataFusion, files, or replay work here.
use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, ListArray, RecordBatch,
    StringArray, UInt8Array,
};
use arrow::compute::{cast, concat_batches, sort_to_indices, take_record_batch};
use arrow::datatypes::DataType;
use re_log_encoding::ToApplication;
use re_protos::{cloud::v1alpha1::*, common::v1alpha1::*, headers::RerunHeadersInjectorExt};
use slam_rs::calib::{
    Calibration, CameraParts, ImuParts,
    catalog::{CameraStatics, ImuStatics},
};
pub(super) use slam_rs::catalog_timing::RigProfile;
use slam_rs::catalog_timing::{Frameset, MSD_G2, ROBOCAP, match_framesets, pair_imu};

use crate::{Clip, ImuRow, ReplayInput};

const RIG: &str = "/world/rig_00";
const IMU: &str = "/world/rig_00/imu_00";
type Client = rerun_cloud_service_client::RerunCloudServiceClient<tonic::transport::Channel>;
type Statics = BTreeMap<String, ArrayRef>;

pub(super) struct Packet {
    pub t: i64,
    pub data: arrow::buffer::ScalarBuffer<u8>,
}

pub(super) fn load(url: &str, segment: &str, limit: Option<usize>) -> Result<ReplayInput> {
    ensure!(limit != Some(0), "--max-framesets must be positive");
    let dataset = segment
        .split_once("__")
        .context("segment must begin with robocap__ or msd-g2__")?
        .0;
    ensure!(
        matches!(dataset, "robocap" | "msd-g2"),
        "unsupported dataset {dataset:?}; expected robocap or msd-g2"
    );
    let endpoint = url
        .strip_prefix("rerun+")
        .filter(|s| s.starts_with("http://"))
        .context("expected rerun+http://host:port")?;
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(load_async(endpoint, dataset, segment, limit))
}

fn request<T>(body: T, dataset: &str) -> Result<tonic::Request<T>> {
    Ok(tonic::Request::new(body).with_entry_name(re_protos::EntryName::new(dataset)?))
}

async fn query(
    client: &mut Client,
    dataset: &str,
    segment: &str,
    mut q: QueryDatasetRequest,
) -> Result<Vec<RecordBatch>> {
    q.segment_ids = vec![SegmentId {
        id: Some(segment.into()),
    }];
    let mut stream = client
        .query_dataset(request(q, dataset)?)
        .await?
        .into_inner();
    let mut batches = Vec::new();
    while let Some(response) = stream.message().await? {
        if let Some(part) = response.data {
            let batch: RecordBatch = (&part).try_into()?;
            if batch.num_rows() > 0 {
                batches.push(batch);
            }
        }
    }
    ensure!(!batches.is_empty(), "{segment}: no matching catalog chunks");
    Ok(batches)
}

async fn fetch(
    client: &mut Client,
    dataset: &str,
    batches: &[RecordBatch],
    mut consume: impl FnMut(re_chunk::Chunk) -> Result<()>,
) -> Result<()> {
    let mut stream = client
        .fetch_chunks(request(
            FetchChunksRequest {
                chunk_infos: batches.iter().map(DataframePart::from).collect(),
            },
            dataset,
        )?)
        .await?
        .into_inner();
    while let Some(response) = stream.message().await? {
        for message in response.chunks {
            let message = message.to_application(())?;
            consume(re_chunk::Chunk::from_chunk_record_batch(&message.batch)?)?;
        }
    }
    Ok(())
}

fn paths(paths: impl IntoIterator<Item = String>) -> Vec<EntityPath> {
    paths.into_iter().map(|path| EntityPath { path }).collect()
}

async fn load_async(
    endpoint: &str,
    dataset: &str,
    segment: &str,
    limit: Option<usize>,
) -> Result<ReplayInput> {
    let profile = if dataset == "robocap" {
        &ROBOCAP
    } else {
        &MSD_G2
    };
    let mut client = Client::new(
        tonic::transport::Endpoint::from_shared(endpoint.to_owned())?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(120))
            .connect()
            .await?,
    )
    .max_decoding_message_size(256 * 1024 * 1024);
    let (calibration, video_paths) = load_statics(&mut client, dataset, segment, profile).await?;
    let (videos, sets) =
        load_videos(&mut client, dataset, segment, &video_paths, limit, profile).await?;
    let imu = load_imu(&mut client, dataset, segment, &sets, &calibration, profile).await?;
    let resolution_wh: Vec<_> = calibration
        .resolution
        .iter()
        .map(|&[w, h]| (w as usize, h as usize))
        .collect();
    let pixels = decode_videos(videos, &sets, &video_paths, &resolution_wh, profile)?;
    eprintln!(
        "catalog: {segment}: {} framesets, {} IMU samples, {} MiB gray8 ready",
        sets.len(),
        imu.len(),
        pixels.len() / (1024 * 1024)
    );
    let frame_t_ns = sets
        .iter()
        .map(|s| {
            s.0.checked_add(calibration.cam_time_offset_ns)
                .context("camera timestamp overflow")
        })
        .collect::<Result<Vec<_>>>()?;
    let clip = Clip {
        segment_id: segment.into(),
        num_cameras: 4,
        framesets: sets.len(),
        frame_t_ns,
        resolution_wh,
    };
    Ok(ReplayInput {
        clip,
        calibration,
        imu,
        pixels,
    })
}

async fn load_statics(
    client: &mut Client,
    dataset: &str,
    segment: &str,
    profile: &RigProfile,
) -> Result<(Calibration<f64>, Vec<String>)> {
    // Statics are small; use their camera names to select the current SLAM profile.
    let batches = query(
        client,
        dataset,
        segment,
        QueryDatasetRequest {
            select_all_entity_paths: true,
            exclude_temporal_data: true,
            ..Default::default()
        },
    )
    .await?;
    let mut statics = Statics::new();
    fetch(client, dataset, &batches, |chunk| {
        for column in chunk.components().values() {
            let key = format!("{}:{}", chunk.entity_path(), column.descriptor.component);
            ensure!(
                column.list_array.len() == 1 && column.list_array.is_valid(0),
                "invalid static {key}"
            );
            let value = column.list_array.value(0);
            if let Some(previous) = statics.insert(key.clone(), value.clone()) {
                ensure!(previous == value, "conflicting static {key}");
            }
        }
        Ok(())
    })
    .await?;
    ensure!(
        string(&statics, &format!("{RIG}:reference"))? == "imu_00",
        "rig reference must be imu_00"
    );
    let count = numbers::<1>(&statics, &format!("{RIG}:num_cameras"))?[0] as usize;
    ensure!(
        count == profile.rig_cameras,
        "unexpected camera count {count}"
    );
    let mut cameras = Vec::new();
    for name in profile.camera_names {
        let mut matches = Vec::new();
        for i in 0..count {
            let entity = format!("{RIG}/cam_{i:02}");
            if string(&statics, &format!("{entity}:name"))?.replace('-', "_") == name {
                matches.push(entity);
            }
        }
        ensure!(matches.len() == 1, "expected one camera named {name}");
        cameras.push(matches.remove(0));
    }
    let calibration = calibration(&statics, &cameras, profile)?;
    let video_paths: Vec<String> = cameras
        .iter()
        .map(|p| format!("{p}/pinhole/video"))
        .collect();
    for path in &video_paths {
        let codec = numbers::<1>(&statics, &format!("{path}:VideoStream:codec"))?[0] as u32;
        ensure!(
            codec == u32::from_be_bytes(profile.codec),
            "unexpected video codec at {path}"
        );
    }
    Ok((calibration, video_paths))
}

async fn load_videos(
    client: &mut Client,
    dataset: &str,
    segment: &str,
    video_paths: &[String],
    limit: Option<usize>,
    profile: &RigProfile,
) -> Result<([Vec<Packet>; 4], Vec<Frameset>)> {
    let batches = query(
        client,
        dataset,
        segment,
        QueryDatasetRequest {
            entity_paths: paths(video_paths.iter().cloned()),
            exclude_static_data: true,
            fuzzy_descriptors: vec!["VideoStream:sample".into()],
            ..Default::default()
        },
    )
    .await?;
    // Fetch chunks in start order and stop once the prefix is final. No sample precedes its
    // chunk's start, and the last requested frameset depends only on samples within two
    // tolerances of its time, so a chunk starting later cannot change the prefix even when
    // chunks overlap. A short prefix of a long session is never fetched whole.
    let index = concat_batches(&batches[0].schema(), &batches)?;
    let starts = index
        .column_by_name("video_time:start")
        .context("missing video_time chunk index")?;
    let index = take_record_batch(&index, &sort_to_indices(starts, None, None)?)?;
    let starts = cast(
        index
            .column_by_name("video_time:start")
            .context("missing video_time chunk index")?,
        &DataType::Int64,
    )?;
    let starts = starts
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("video_time chunk starts must be integers")?;
    let mut videos: [Vec<Packet>; 4] = std::array::from_fn(|_| Vec::new());
    let mut sets = Vec::new();
    for start in (0..index.num_rows()).step_by(64) {
        let end = (start + 64).min(index.num_rows());
        let batch = index.slice(start, end - start);
        fetch(client, dataset, &[batch], |chunk| {
            let path = chunk.entity_path().to_string();
            let camera = video_paths
                .iter()
                .position(|p| *p == path)
                .context("unexpected video entity")?;
            for (t, samples) in rows(&chunk, "VideoStream:sample")? {
                let samples = samples
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .context("video samples must be a list")?;
                ensure!(samples.len() == 1, "expected one video sample per row");
                let bytes = samples.value(0);
                let bytes = bytes
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .context("video sample must contain bytes")?;
                videos[camera].push(Packet {
                    t,
                    data: bytes.values().clone(),
                });
            }
            Ok(())
        })
        .await?;
        if let Some(n) = limit {
            sets = sorted_framesets(&mut videos, profile.tolerance_ns)?;
            let horizon = sets.get(n - 1).map(|set| set.0 + 2 * profile.tolerance_ns);
            let last = end == index.num_rows();
            if horizon.is_some_and(|horizon| last || starts.value(end) > horizon) {
                break;
            }
        }
    }
    if limit.is_none() {
        sets = sorted_framesets(&mut videos, profile.tolerance_ns)?;
    }
    if let Some(n) = limit {
        sets.truncate(n);
    }
    ensure!(!sets.is_empty(), "no complete framesets");
    Ok((videos, sets))
}

async fn load_imu(
    client: &mut Client,
    dataset: &str,
    segment: &str,
    sets: &[Frameset],
    calibration: &Calibration<f64>,
    profile: &RigProfile,
) -> Result<Vec<ImuRow>> {
    let first = sets[0].0;
    let last = sets[sets.len() - 1].0;
    // The Python feed reads two frame periods and two IMU periods either side.
    let fps = if sets.len() > 1 {
        (((sets.len() - 1) as f64 * 1e9 / (last - first) as f64).round() as i64).max(1)
    } else {
        profile.nominal_fps
    };
    let margin =
        (2 * (1_000_000_000 / fps) + 2 * (1e9 / calibration.imu_update_rate) as i64).max(2_000_000);
    let batches = query(
        client,
        dataset,
        segment,
        QueryDatasetRequest {
            entity_paths: paths([format!("{IMU}/gyro"), format!("{IMU}/accel")]),
            exclude_static_data: true,
            query: Some(Query {
                range: Some(QueryRange {
                    index: Some(IndexColumnSelector {
                        timeline: Some(Timeline {
                            name: "video_time".into(),
                        }),
                    }),
                    index_range: Some(TimeRange {
                        start: first - margin,
                        end: last + margin,
                    }),
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .await?;
    let mut gyro = Vec::new();
    let mut accel = Vec::new();
    fetch(client, dataset, &batches, |chunk| {
        let target = if chunk.entity_path().to_string() == format!("{IMU}/gyro") {
            &mut gyro
        } else {
            &mut accel
        };
        for (t, values) in rows(&chunk, "Scalars:scalars")? {
            if t < first - margin || t > last + margin {
                continue;
            }
            let values = values
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("IMU must be float64")?;
            ensure!(
                values.len() == 3 && values.values().iter().all(|x| x.is_finite()),
                "invalid IMU vector"
            );
            target.push((t, [values.value(0), values.value(1), values.value(2)]));
        }
        Ok(())
    })
    .await?;
    let offset = calibration.cam_time_offset_ns;
    let mut imu = pair_imu(gyro, accel, profile.interpolate_accel)?;
    ensure!(!imu.is_empty(), "no IMU samples");
    for row in &mut imu {
        row.t_ns = row
            .t_ns
            .checked_add(offset)
            .context("IMU timestamp overflow")?;
    }
    Ok(imu)
}

fn decode_videos(
    videos: [Vec<Packet>; 4],
    sets: &[Frameset],
    video_paths: &[String],
    resolution_wh: &[(usize, usize)],
    profile: &RigProfile,
) -> Result<Vec<u8>> {
    let frame_bytes: usize = resolution_wh.iter().map(|&(w, h)| w * h).sum();
    let mut pixels = vec![
        0;
        frame_bytes
            .checked_mul(sets.len())
            .context("clip size overflow")?
    ];
    let mut camera_offset = 0;
    for (camera, mut video) in videos.into_iter().enumerate() {
        // Always decode from the first packet/keyframe; trim only the tail.
        video.truncate(sets[sets.len() - 1].1[camera] + 1);
        let (width, height) = resolution_wh[camera];
        let mut next = 0;
        let mut destinations = pixels
            .chunks_exact_mut(frame_bytes)
            .map(|frame| &mut frame[camera_offset..camera_offset + width * height]);
        crate::decode::decode(&video, profile, width, height, |index| {
            if next < sets.len() && sets[next].1[camera] == index {
                next += 1;
                destinations.next()
            } else {
                None
            }
        })
        .with_context(|| format!("decode {}", video_paths[camera]))?;
        ensure!(
            next == sets.len(),
            "camera {camera}: decoded {next}/{} selected images",
            sets.len()
        );
        camera_offset += width * height;
    }
    Ok(pixels)
}

fn rows<'a>(
    chunk: &'a re_chunk::Chunk,
    component: &'a str,
) -> Result<impl Iterator<Item = (i64, ArrayRef)> + 'a> {
    let times = chunk
        .timelines()
        .iter()
        .find(|(name, _)| name.as_str() == "video_time")
        .context("chunk missing video_time")?
        .1
        .times_raw();
    Ok(chunk
        .components()
        .values()
        .filter(move |column| column.descriptor.component.as_str() == component)
        .flat_map(move |column| {
            times
                .iter()
                .enumerate()
                .filter(|(index, _)| column.list_array.is_valid(*index))
                .map(move |(index, &t)| (t, column.list_array.value(index)))
        }))
}

fn sorted_framesets(videos: &mut [Vec<Packet>; 4], tolerance: i64) -> Result<Vec<Frameset>> {
    for video in &mut *videos {
        video.sort_by_key(|p| p.t);
        ensure!(
            video.windows(2).all(|p| p[0].t < p[1].t),
            "duplicate video timestamps"
        );
    }
    if videos.iter().any(Vec::is_empty) {
        return Ok(Vec::new());
    }
    let times: Vec<Vec<i64>> = videos
        .iter()
        .map(|v| v.iter().map(|p| p.t).collect())
        .collect();
    let cameras: Vec<&[i64]> = times.iter().map(Vec::as_slice).collect();
    Ok(match_framesets(&cameras, tolerance)?)
}

fn flat(statics: &Statics, key: &str) -> Result<ArrayRef> {
    let mut array = statics
        .get(key)
        .with_context(|| format!("missing static {key}"))?
        .clone();
    loop {
        if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
            array = list.values().clone();
        } else if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
            array = list.values().clone();
        } else {
            break;
        }
    }
    ensure!(array.null_count() == 0, "null static {key}");
    Ok(array)
}
fn numbers<const N: usize>(statics: &Statics, key: &str) -> Result<[f64; N]> {
    let array = cast(&flat(statics, key)?, &DataType::Float64)?;
    let values = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("expected numeric static")?;
    ensure!(
        values.len() == N && values.values().iter().all(|v| v.is_finite()),
        "{key}: expected {N} finite values"
    );
    Ok(std::array::from_fn(|i| values.value(i)))
}
fn string(statics: &Statics, key: &str) -> Result<String> {
    let array = flat(statics, key)?;
    let values = array
        .as_any()
        .downcast_ref::<StringArray>()
        .with_context(|| format!("{key}: expected string"))?;
    ensure!(values.len() == 1, "{key}: expected one string");
    Ok(values.value(0).into())
}
fn calibration(
    statics: &Statics,
    cameras: &[String],
    profile: &RigProfile,
) -> Result<Calibration<f64>> {
    let downscale = profile.downscale;
    let mut parts = Vec::new();
    for (index, entity) in cameras.iter().enumerate() {
        let pinhole = format!("{entity}/pinhole");
        let [w, h] = numbers::<2>(statics, &format!("{pinhole}:Pinhole:resolution"))?;
        let k = numbers::<9>(statics, &format!("{pinhole}:Pinhole:image_from_camera"))?;
        let r = numbers::<9>(statics, &format!("{entity}:Transform3D:mat3x3"))?;
        let t = numbers::<3>(statics, &format!("{entity}:Transform3D:translation"))?;
        let relation = numbers::<1>(statics, &format!("{entity}:Transform3D:relation"))?[0];
        let model = string(
            statics,
            &format!("{pinhole}:simplecv.components.DistortionModel"),
        )?;
        let coeffs = cast(
            &flat(
                statics,
                &format!("{pinhole}:simplecv.components.DistortionCoefficients"),
            )?,
            &DataType::Float64,
        )?;
        let coeffs = coeffs
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("distortion coefficients")?;
        ensure!(
            coeffs.values().iter().all(|v| v.is_finite()),
            "{entity}: non-finite distortion coefficient"
        );
        let radius_key = format!("{entity}:distortion_valid_radius");
        let radius = statics
            .contains_key(&radius_key)
            .then(|| numbers::<1>(statics, &radius_key).map(|v| v[0]))
            .transpose()?;
        parts.push(CameraParts::from_catalog_statics(
            index,
            &CameraStatics {
                resolution_wh: [w, h],
                image_from_camera: k,
                transform_mat3x3: r,
                transform_translation: t,
                transform_relation: relation as i64,
                distortion_model: model,
                distortion_coefficients: coeffs.values().to_vec(),
                distortion_valid_radius: radius,
            },
            i64::from(downscale),
        )?);
    }
    let shift = cast(
        &flat(statics, &format!("{IMU}:applied_time_shift_ns"))?,
        &DataType::Int64,
    )?;
    let shift = shift
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("IMU time shift")?;
    ensure!(shift.len() == 1, "expected one IMU time shift");
    let scalar = |name: &str| {
        numbers::<1>(statics, &format!("{IMU}:simplecv.ImuCalibration:{name}")).map(|v| Some(v[0]))
    };
    let imu = ImuParts::from_catalog_statics(&ImuStatics {
        rate_hz: scalar("rate_hz")?,
        gyro_noise_density: scalar("gyro_noise_density")?,
        accel_noise_density: scalar("accel_noise_density")?,
        gyro_bias_random_walk: scalar("gyro_bias_random_walk")?,
        accel_bias_random_walk: scalar("accel_bias_random_walk")?,
        applied_time_shift_ns: shift.value(0),
    })?;
    Ok(Calibration::from_catalog_parts(&parts, &imu)?)
}
