//! The factory projection and the hemisphere-tile table against the probe's Python oracle
//! (`/tmp/slam-rs-fisheye-360/calib-probe/pano_lut.py`, recon/09), and on the real X4 Air clip.
//!
//! The clip tests are asset-gated and decode with an `ffmpeg` on `PATH`: run
//! `cargo test -p kornia-staging-io --test insta360_pano -- --ignored` (`KORNIA_STAGING_INSV` names the clip). With
//! `--release`, `evidence_pngs_and_timing` also writes tiles and panoramas to `KORNIA_STAGING_PANO_OUT`.
use kornia_image::{Image, ImageSize};
use kornia_staging_imgproc::panorama::{
    assemble_panorama, build_tile_lut, lens_level1, remap_tile, TileGeometry, TileLut, LUT_INVALID,
};
use kornia_staging_io::insta360::{parse_lens_string, Insta360Clip, LensCalibration, WindowCrop};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const V3: &str = "2_2.314940_7095.540_7093.970_3842.370_3863.400_0.100_0.165_90.630_0.000000_0.000000_0.000000_0.96273077_-2.06431413_4.31214333_0.00172895_-0.00121632_15360_7680_131_2.314940_7148.420_7149.100_11546.600_3864.040_-0.074_0.251_88.277_0.000805_0.000066_-0.033123_0.96663713_-1.95924056_3.78486657_0.00081582_0.00241697_15360_7680_131_197632";
const V6: &str = "2_2.314940_7096.760_7095.390_3834.830_3864.120_0.119_0.407_90.629_0.000000_0.000000_0.000000_0.93861496_-1.68402135_2.19166017_3.85762072_0.00000000_-0.00495962_0.00156076_-0.00330722_0.00397486_0.00827898_-0.00125315_0.02416405_-0.00143547_15360_7680_131_2.314940_7149.220_7149.700_11543.300_3858.430_-0.212_0.339_88.279_0.000132_-0.000043_-0.033291_0.95675713_-1.80496192_2.88544273_1.76948690_0.00000000_0.00131924_0.00023973_0.00279902_-0.00570679_0.00069985_0.00064811_0.00799088_0.02677096_15360_7680_131_394240";
const CROP: WindowCrop = WindowCrop {
    src: [7680, 7680],
    dst: [7424, 7424],
    offset: [0, 0],
};
const LENS: ImageSize = ImageSize {
    width: 1920,
    height: 1920,
};
const ORACLE_TILE: TileGeometry = TileGeometry {
    hemisphere_width: 960,
    height: 960,
    margin: 32,
};
const CLIP: &str = "/home/pablo/data/insta360/VID_20261116_125200_00_003.insv";

/// pano_lut.py `project` of tile pixels (margin 32, 1024x960): (tile, tu, tv, u, v) at 1920 (V6 string).
const ORACLE_V6: [(usize, u32, u32, f64, f64); 40] = [
    (0, 0, 0, 949.4141190063, 66.6916695403),
    (0, 0, 240, 294.3453120156, 313.3967569318),
    (0, 0, 479, 14.4263896322, 975.2000785322),
    (0, 0, 959, 969.1915873928, 1866.7998219919),
    (0, 31, 0, 949.4078043665, 66.7691596434),
    (0, 31, 240, 315.8982502347, 337.4698253345),
    (0, 31, 479, 59.3505060230, 974.5552762092),
    (0, 31, 959, 969.1835492150, 1866.7213979452),
    (0, 511, 0, 950.8908421356, 67.5190335021),
    (0, 511, 240, 956.2711743716, 516.6169994925),
    (0, 511, 479, 961.3093314334, 963.8217988763),
    (0, 511, 959, 970.6515701769, 1865.9284068293),
    (0, 800, 0, 952.0792772478, 67.1907838514),
    (0, 800, 240, 1374.9581218617, 445.2197718510),
    (0, 800, 479, 1506.9681431357, 958.0360626754),
    (0, 800, 959, 971.8485593253, 1866.2349699634),
    (0, 1023, 0, 952.3425621985, 66.6611583037),
    (0, 1023, 240, 1611.5846117058, 299.5366683917),
    (0, 1023, 479, 1905.0708734711, 954.4542128183),
    (0, 1023, 959, 972.1238669695, 1866.7661190935),
    (1, 0, 0, 992.8170396273, 59.5663654339),
    (1, 0, 240, 325.2128575326, 283.2138433304),
    (1, 0, 479, 18.3397881911, 935.8858091541),
    (1, 0, 959, 938.2572082165, 1874.5784655220),
    (1, 31, 0, 992.8075290546, 59.6446887645),
    (1, 31, 240, 345.7736282049, 308.3859416222),
    (1, 31, 479, 63.0420474257, 937.3971673986),
    (1, 31, 959, 938.2523985998, 1874.4999230082),
    (1, 511, 0, 994.2707605485, 60.4586808528),
    (1, 511, 240, 981.4291890020, 514.7744888031),
    (1, 511, 479, 968.0704505758, 965.5199095626),
    (1, 511, 959, 939.7653152714, 1873.7775046979),
    (1, 800, 0, 995.4811152121, 60.1690667217),
    (1, 800, 240, 1406.8863550022, 458.2119029294),
    (1, 800, 479, 1519.6707364625, 981.6941908555),
    (1, 800, 959, 940.9589089994, 1874.1385523907),
    (1, 1023, 0, 995.7678348739, 59.6417278321),
    (1, 1023, 240, 1652.6271499170, 318.7860093658),
    (1, 1023, 479, 1923.8734072779, 993.1552343492),
    (1, 1023, 959, 941.2140918548, 1874.6807519987),
];
/// The same for the V3 string.
const ORACLE_V3: [(usize, u32, u32, f64, f64); 40] = [
    (0, 0, 0, 948.4590258559, 66.4983322707),
    (0, 0, 240, 293.7361796459, 313.8521822671),
    (0, 0, 479, 14.5150824393, 975.3263286693),
    (0, 0, 959, 968.2687818152, 1866.8156899534),
    (0, 31, 0, 948.4526851652, 66.5760400342),
    (0, 31, 240, 315.2551012666, 337.9288927443),
    (0, 31, 479, 59.2778637594, 974.7017602504),
    (0, 31, 959, 968.2607181078, 1866.7373165061),
    (0, 511, 0, 949.9360547665, 67.3262877207),
    (0, 511, 240, 955.7451925120, 516.6246465578),
    (0, 511, 479, 960.9207164393, 963.8195307451),
    (0, 511, 959, 969.7283374143, 1865.9464959608),
    (0, 800, 0, 951.1250339143, 66.9958115198),
    (0, 800, 240, 1374.4463738742, 445.0874439541),
    (0, 800, 479, 1506.7584742064, 958.1407617774),
    (0, 800, 959, 970.9253109064, 1866.2541134196),
    (0, 1023, 0, 951.3885809874, 66.4644477525),
    (0, 1023, 240, 1611.4753035055, 298.8363641360),
    (0, 1023, 479, 1905.5407650964, 954.5381047915),
    (0, 1023, 959, 971.2007679242, 1866.7851609170),
    (1, 0, 0, 993.1367126290, 59.5796806342),
    (1, 0, 240, 325.3856450244, 283.0712142054),
    (1, 0, 479, 18.2761673393, 935.9205397788),
    (1, 0, 959, 938.5292700145, 1874.7091220414),
    (1, 31, 0, 993.1271528294, 59.6579055571),
    (1, 31, 240, 345.9042530552, 308.2349982265),
    (1, 31, 479, 62.9792858752, 937.4373080265),
    (1, 31, 959, 938.5244136671, 1874.6304641305),
    (1, 511, 0, 994.5899440241, 60.4715302728),
    (1, 511, 240, 981.5854330054, 514.7752383495),
    (1, 511, 479, 968.0670959645, 965.6272656735),
    (1, 511, 959, 940.0369516332, 1873.9063770737),
    (1, 800, 0, 995.8005290582, 60.1827648124),
    (1, 800, 240, 1407.1820731554, 458.3522826378),
    (1, 800, 479, 1519.8038324228, 981.9485836282),
    (1, 800, 959, 941.2307961570, 1874.2674923890),
    (1, 1023, 0, 996.0875859361, 59.6561797560),
    (1, 1023, 240, 1652.7720059333, 319.0333572312),
    (1, 1023, 479, 1923.8675492111, 993.2361815977),
    (1, 1023, 959, 941.4863106716, 1874.8103991097),
];

fn lenses(text: &str) -> Vec<LensCalibration> {
    parse_lens_string(text, Some(CROP), [1920, 1920]).unwrap()
}

fn lut(lens: &LensCalibration, geometry: TileGeometry, tile: usize) -> TileLut {
    build_tile_lut(geometry, tile, LENS, |d| lens.project_from_panorama(d))
}

#[test]
fn the_factory_projection_reproduces_the_oracle_to_a_micropixel() {
    for (text, oracle) in [(V6, &ORACLE_V6), (V3, &ORACLE_V3)] {
        let lenses = lenses(text);
        for &(k, tu, tv, u, v) in oracle {
            let d = ORACLE_TILE.bearing(k, f64::from(tu), f64::from(tv));
            let [pu, pv] = lenses[k].project_from_panorama(d).unwrap();
            assert!(
                (pu - u).abs() < 1e-6 && (pv - v).abs() < 1e-6,
                "lens {k} ({tu}, {tv}): ({pu}, {pv}) vs ({u}, {v})"
            );
        }
    }
    // R_P_lens of V6 lens 0 (pano_lut.py): within 0.63 deg of the identity, not the identity.
    let r = lenses(V6)[0].pano_from_lens_rotation;
    assert!(
        (r[0][1] + 0.010977876806482231).abs() < 1e-12
            && (r[2][0] - 0.007103414994941127).abs() < 1e-12
    );
}

#[test]
fn the_lens_centres_sit_on_the_panorama_z_axis_about_the_midpoint() {
    // recon/03 section 4: 33.1 mm baseline on the optical axis, 0.1-0.8 mm lateral; lens 0 looks along +Z.
    let lenses = lenses(V6);
    let [c0, c1] = [lenses[0].pano_centre_m, lenses[1].pano_centre_m];
    for k in 0..3 {
        assert!(
            (c0[k] + c1[k]).abs() < 1e-12,
            "centres are symmetric about the origin"
        );
    }
    assert!(
        (c0[2] - 0.01656).abs() < 2e-4,
        "lens 0 at +16.56 mm on Z: {c0:?}"
    );
    assert!(
        c0[0].hypot(c0[1]) < 1e-3,
        "lateral offset below 1 mm: {c0:?}"
    );
}

#[test]
fn the_radial_part_gives_the_recon_radii() {
    // recon/03 section 5.3: lens 0 (V3) image radius 294.9 / 607.6 / 900.2 / 972.2 px at 30 / 60 / 90 / 100 deg.
    let mut lens = lenses(V3)[0].clone();
    lens.tangential = vec![0.0; 2];
    (lens.fx, lens.fy) = (0.5 * (lens.fx + lens.fy), 0.5 * (lens.fx + lens.fy));
    for (deg, radius) in [
        (30.0f64, 294.9),
        (60.0, 607.6),
        (90.0, 900.2),
        (99.9999, 972.2),
    ] {
        let t = deg.to_radians();
        let [u, v] = lens
            .project([t.sin() * 0.6, t.sin() * 0.8, t.cos()])
            .unwrap();
        let r = (u - lens.cx).hypot(v - lens.cy);
        assert!(
            (r - lens.radius_px(t)).abs() < 1e-9 && (r - radius).abs() < 0.1,
            "{deg}: {r}"
        );
    }
    assert!(
        lens.project([1.0, 0.0, -0.18]).is_none(),
        "100.2 deg is past the lens coverage"
    );
}

#[test]
fn the_table_quantises_the_oracle_to_half_a_step() {
    // q = floor(128 u + 0.5) in 1/256 level-1 px: |q - 128 u| <= 1/2 step, plus the 1e-6 px projection error.
    let lenses = lenses(V6);
    let luts = [0, 1].map(|k| lut(&lenses[k], ORACLE_TILE, k));
    for &(k, tu, tv, u, v) in &ORACLE_V6 {
        let s = luts[k].sample[tv as usize * 1024 + tu as usize];
        if u > 1918.0 {
            assert_eq!(
                s, LUT_INVALID,
                "{k} ({tu}, {tv}) is past the last level-1 sample"
            );
            continue;
        }
        let (ex, ey) = (f64::from(s[0]) - 128.0 * u, f64::from(s[1]) - 128.0 * v);
        assert!(
            ex.abs() <= 0.5 + 1e-3 && ey.abs() <= 0.5 + 1e-3,
            "{k} ({tu}, {tv}): {ex} {ey}"
        );
        let row = luts[k].source_row[tv as usize * 1024 + tu as usize];
        assert!((f64::from(row) - v).abs() < 1e-3);
    }
}

#[test]
fn invalid_pixels_sit_at_the_lens_edges() {
    // With the same validity rule, pano_lut.py's projection gives these counts (ref.py, recon/09 lists 189 for the
    // open interval 0 < u < 1919 at margin 32; level 1 has no sample past full-res 1918).
    let lenses = lenses(V6);
    for (margin, counts) in [(32, [0, 244]), (64, [6787, 8427])] {
        let g = TileGeometry {
            margin,
            ..TileGeometry::default()
        };
        for k in 0..2 {
            let t = lut(&lenses[k], g, k);
            assert_eq!(t.invalid_count(), counts[k], "margin {margin} tile {k}");
            let (w, [cx, _]) = (g.width(), g.principal());
            for (i, _) in t
                .sample
                .iter()
                .enumerate()
                .filter(|(_, s)| **s == LUT_INVALID)
            {
                let (tu, tv) = (i % w, i / w);
                // Only more than 95 deg of longitude from the lens axis (the frames cut the lenses at 95.9-99.1 deg,
                // recon/03 section 5.3), and only within 200 rows (37.5 deg) of the equator.
                let lon = (tu as f64 - cx).abs() / g.focal();
                assert!(
                    lon > 95f64.to_radians() && tv.abs_diff(480) < 200,
                    "{tu} {tv}"
                );
            }
            if margin == 32 && k == 1 {
                assert!(t
                    .sample
                    .iter()
                    .enumerate()
                    .all(|(i, s)| *s != LUT_INVALID
                        || (i % w >= 1019 && (438..=504).contains(&(i / w)))));
            }
        }
    }
}

/// Luma (plane 0, no range conversion) of frame `frame` of video stream `track`, decoded by `ffmpeg`.
fn decode(path: &str, track: usize, frame: usize) -> Image<u8, 1> {
    let t = format!("{:.6}", (frame as f64 * 1001.0 - 100.0) / 30000.0);
    let out = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-ss",
            &t,
            "-i",
            path,
            "-map",
            &format!("0:v:{track}"),
            "-frames:v",
            "1",
        ])
        .args(["-f", "rawvideo", "-pix_fmt", "yuvj420p", "-"])
        .output()
        .expect("ffmpeg runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    Image::new(LENS, out.stdout[..1920 * 1920].to_vec()).unwrap()
}

/// The clip and an `ffmpeg`, or `None` with the reason printed.
fn clip() -> Option<(String, Insta360Clip)> {
    let path = std::env::var("KORNIA_STAGING_INSV").unwrap_or_else(|_| CLIP.into());
    if !Path::new(&path).is_file() {
        eprintln!("SKIP: asset {path} (VID_20261116_125200_00_003.insv) is absent");
        return None;
    }
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIP: no ffmpeg on PATH to decode the HEVC tracks");
        return None;
    }
    let clip = Insta360Clip::open(&path).unwrap();
    assert_eq!(clip.lens_track, [1, 0], "lens 0 is track 1 (recon/09)");
    Some((path, clip))
}

fn tiles(path: &str, clip: &Insta360Clip, luts: &[TileLut; 2], frame: usize) -> [Image<u16, 1>; 2] {
    std::array::from_fn(|k| {
        let mut tile = Image::from_size_val(
            ImageSize {
                width: luts[k].geometry.width(),
                height: 960,
            },
            0u16,
        )
        .unwrap();
        let lens = decode(path, clip.lens_track[k], frame);
        remap_tile(&lens_level1(&lens).unwrap(), &luts[k], &mut tile).unwrap();
        tile
    })
}

#[test]
#[ignore = "needs the X4 Air clip VID_20261116_125200_00_003.insv (KORNIA_STAGING_INSV) and ffmpeg"]
fn a_real_frame_pair_matches_the_oracle_remap() {
    let Some((path, clip)) = clip() else { return };
    let lenses = &clip.lenses_v6;
    let luts = [0, 1].map(|k| lut(&lenses[k], ORACLE_TILE, k));
    let tiles = tiles(&path, &clip, &luts, 12);
    for k in 0..2 {
        // pano_lut.py `remap`: bilinear on the full-resolution lens picture at the factory projection, 0 outside.
        let lens = decode(&path, clip.lens_track[k], 12);
        let level1 = lens_level1(&lens).unwrap();
        let (mut sum, mut n, mut big, mut exact) = (0.0, 0usize, 0usize, 0.0f64);
        for (i, &value) in tiles[k].as_slice().iter().enumerate() {
            let d = ORACLE_TILE.bearing(k, (i % 1024) as f64, (i / 1024) as f64);
            let [u, v] = lenses[k].project_from_panorama(d).unwrap();
            if !(0.0..1919.0).contains(&u) || !(0.0..1919.0).contains(&v) {
                continue;
            }
            let bilinear = |u: f64, v: f64, w: usize, pixels: &dyn Fn(usize) -> f64| {
                let (x0, y0) = (u.floor() as usize, v.floor() as usize);
                let (ax, ay) = (u - x0 as f64, v - y0 as f64);
                let p = |x: usize, y: usize| pixels(y * w + x.min(w - 1));
                (1.0 - ay) * ((1.0 - ax) * p(x0, y0) + ax * p(x0 + 1, y0))
                    + ay * ((1.0 - ax) * p(x0, y0 + 1) + ax * p(x0 + 1, y0 + 1))
            };
            let ours = f64::from(value) / 256.0;
            let diff = (ours - bilinear(u, v, 1920, &|i| f64::from(lens.as_slice()[i]))).abs();
            (sum, n, big) = (sum + diff, n + 1, big + usize::from(diff > 16.0));
            if u <= 1918.0 && v <= 1918.0 {
                let float = bilinear(u / 2.0, v / 2.0, 960, &|i| {
                    f64::from(level1.as_slice()[i]) / 256.0
                });
                exact = exact.max((ours - float).abs());
            }
        }
        let mean = sum / n as f64;
        eprintln!("tile {k}: |ours - oracle remap| mean {mean:.3} grey, {big} of {n} above 16; |ours - f64 rule| max {exact:.3}");
        // Ours samples the [1 4 6 4 1]-filtered level 1, the oracle the raw picture: they differ at sharp edges only.
        assert!(
            mean < 2.0 && (big as f64) < 0.01 * n as f64,
            "tile {k}: {mean} {big}"
        );
        // Against the same level 1 in f64, the integer rule loses at most its 1/512 px position and rounding.
        assert!(exact < 1.0, "tile {k}: {exact}");
    }
}

#[test]
#[ignore = "writes evidence PNGs; needs the X4 Air clip VID_20261116_125200_00_003.insv and ffmpeg"]
fn evidence_pngs_and_timing() {
    let Some((path, clip)) = clip() else { return };
    let out = std::env::var("KORNIA_STAGING_PANO_OUT")
        .unwrap_or_else(|_| "/tmp/slam-rs-fisheye-360/f4a-pano-lut/out".into());
    std::fs::create_dir_all(&out).unwrap();
    let g = TileGeometry::default();
    let started = Instant::now();
    let luts = [0, 1].map(|k| lut(&clip.lenses_v6[k], g, k));
    eprintln!(
        "tables: {:.1} ms for both tiles",
        started.elapsed().as_secs_f64() * 1e3
    );
    for frame in [12, 3862, 7700] {
        let t = tiles(&path, &clip, &luts, frame);
        for (k, tile) in t.iter().enumerate() {
            let bytes: Vec<u8> = tile
                .as_slice()
                .iter()
                .map(|&v| ((u32::from(v) + 128) >> 8).min(255) as u8)
                .collect();
            write_png(
                &format!("{out}/tile{k}_{frame:05}.png"),
                g.width(),
                960,
                &bytes,
            );
        }
        let pano = assemble_panorama([&t[0], &t[1]], g).unwrap();
        write_png(
            &format!("{out}/pano_{frame:05}.png"),
            1920,
            960,
            pano.as_slice(),
        );
    }
    // One frame pair: reduce both lenses and gather both tiles, single thread.
    let lens = [0, 1].map(|k| decode(&path, clip.lens_track[k], 3862));
    let mut dst = [0, 1].map(|_| {
        Image::from_size_val(
            ImageSize {
                width: g.width(),
                height: 960,
            },
            0u16,
        )
        .unwrap()
    });
    let (mut reduce, mut gather) = (Vec::new(), Vec::new());
    for _ in 0..20 {
        let t0 = Instant::now();
        let level1 = [0, 1].map(|k| lens_level1(&lens[k]).unwrap());
        let t1 = Instant::now();
        for k in 0..2 {
            remap_tile(&level1[k], &luts[k], &mut dst[k]).unwrap();
        }
        reduce.push((t1 - t0).as_secs_f64() * 1e3);
        gather.push(t1.elapsed().as_secs_f64() * 1e3);
    }
    reduce.sort_by(f64::total_cmp);
    gather.sort_by(f64::total_cmp);
    eprintln!(
        "frame pair, median of 20: widen + reduce {:.2} ms, gather {:.2} ms",
        reduce[10], gather[10]
    );
}

/// 8-bit greyscale PNG with stored (uncompressed) deflate blocks (RFC 2083, RFC 1950/1951).
fn write_png(path: &str, width: usize, height: usize, pixels: &[u8]) {
    let crc_table: Vec<u32> = (0..256u32)
        .map(|n| {
            (0..8).fold(n, |c, _| {
                if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                }
            })
        })
        .collect();
    let crc = |bytes: &[u8]| {
        !bytes.iter().fold(!0u32, |c, &b| {
            crc_table[((c ^ u32::from(b)) & 255) as usize] ^ (c >> 8)
        })
    };
    let raw: Vec<u8> = pixels
        .chunks_exact(width)
        .flat_map(|row| std::iter::once(0).chain(row.iter().copied()))
        .collect();
    let mut z = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (i, b) in blocks.iter().enumerate() {
        z.push(u8::from(i + 1 == blocks.len()));
        z.extend((b.len() as u16).to_le_bytes());
        z.extend((!(b.len() as u16)).to_le_bytes());
        z.extend_from_slice(b);
    }
    let (a, b) = raw.iter().fold((1u32, 0u32), |(a, b), &x| {
        (
            (a + u32::from(x)) % 65521,
            (b + (a + u32::from(x)) % 65521) % 65521,
        )
    });
    z.extend(((b << 16) | a).to_be_bytes());
    let mut ihdr = Vec::new();
    ihdr.extend((width as u32).to_be_bytes());
    ihdr.extend((height as u32).to_be_bytes());
    ihdr.extend([8, 0, 0, 0, 0]);
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [(&b"IHDR"[..], &ihdr[..]), (b"IDAT", &z), (b"IEND", &[])] {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        let c = crc(&png[start..]);
        png.extend(c.to_be_bytes());
    }
    std::fs::write(path, png).unwrap();
}
