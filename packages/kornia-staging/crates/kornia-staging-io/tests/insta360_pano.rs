//! The factory projection and the hemisphere-tile table against values from an independent f64 Python
//! implementation of the factory projection, and on the real X4 Air clip.
//!
//! The clip tests are asset-gated and decode with an `ffmpeg` on `PATH`: run
//! `cargo test -p kornia-staging-io --test insta360_pano -- --ignored` (`KORNIA_STAGING_INSV` names the clip).
use kornia_image::Image;
use kornia_staging_3d::camera::formats::insta360::{parse_lens_string, FactoryLens, WindowCrop};
use kornia_staging_3d::camera::{CameraModel, Mei, MeiDistortion};
use kornia_staging_imgproc::panorama::{
    hemisphere_tile_table, lens_level1, remap, RemapTable, TileGeometry,
};
use kornia_staging_io::insta360::Insta360Clip;

include!("common/x4air_lens.rs");
include!("common/x4air_clip.rs");
const ORACLE_TILE: TileGeometry = TileGeometry {
    hemisphere_width: 960,
    height: 960,
    margin: 32,
};

/// The independent f64 projection of tile pixels (margin 32, 1024x960): (tile, tu, tv, u, v) at 1920 (V6 string).
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

fn lenses(text: &str) -> Vec<FactoryLens> {
    parse_lens_string(text, Some(CROP), [1920, 1920]).unwrap()
}

/// The factory projection of a ray in the panorama frame.
fn project(lens: &FactoryLens, d: [f64; 3]) -> Option<[f64; 2]> {
    let local = lens
        .pano_from_lens
        .rotation
        .inverse_transform_vector(&d.into());
    lens.camera.project(local.into()).ok()
}

const FACTORY_TILE: TileGeometry = TileGeometry {
    hemisphere_width: 960,
    height: 960,
    margin: 64,
};

fn lut(lens: &FactoryLens, geometry: TileGeometry, tile: usize) -> RemapTable {
    hemisphere_tile_table(geometry, tile, LENS, |d| project(lens, d)).unwrap()
}

/// The oracle's bearing in `P` of tile `k`'s pixel `(tu, tv)`: `lon = (tu - cx) / F + k pi`, `lat = (cy - tv) / F`,
/// `(cos lat sin lon, -sin lat, cos lat cos lon)`, `F = hemisphere_width / pi`.
fn bearing(g: TileGeometry, k: usize, tu: f64, tv: f64) -> [f64; 3] {
    let f = g.hemisphere_width as f64 / std::f64::consts::PI;
    let lon = (tu - (g.width() as f64 - 1.0) / 2.0) / f + k as f64 * std::f64::consts::PI;
    let lat = ((g.height as f64 - 1.0) / 2.0 - tv) / f;
    let (sin_lon, cos_lon) = lon.sin_cos();
    let (sin_lat, cos_lat) = lat.sin_cos();
    [cos_lat * sin_lon, -sin_lat, cos_lat * cos_lon]
}

/// FNV-1a 64 of the entries, little-endian words.
fn fnv(samples: &[[u32; 2]]) -> u64 {
    samples
        .iter()
        .flatten()
        .flat_map(|w| w.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
        })
}

#[test]
fn the_tables_equal_the_previous_tile_builder_bit_for_bit() {
    // Hashes of the tables `build_tile_lut` (the per-tile builder this composition replaced, at 1e945306f) made from
    // the same lenses, geometry and lens size.
    let expected: [(&str, usize, [u64; 2]); 4] = [
        ("V6", 64, [0xade3_53c9_6ebe_345f, 0x3316_c18a_d409_f0ba]),
        ("V6", 32, [0x31d8_0768_c8ef_efc2, 0x615d_042b_e633_9d4c]),
        ("V3", 64, [0x4e6f_698f_9556_8403, 0x6e73_4fd8_ea11_abb3]),
        ("V3", 32, [0x0bcd_5f0e_20ed_1af2, 0x0fae_3a3b_fb6f_2dd3]),
    ];
    for (name, margin, hashes) in expected {
        let lenses = lenses(if name == "V6" { V6 } else { V3 });
        for k in 0..2 {
            let table = lut(
                &lenses[k],
                TileGeometry {
                    margin,
                    ..FACTORY_TILE
                },
                k,
            );
            assert_eq!(
                fnv(table.samples()),
                hashes[k],
                "{name} margin {margin} tile {k}"
            );
        }
    }
}

#[test]
fn the_factory_projection_reproduces_the_oracle_to_a_micropixel() {
    for (text, oracle) in [(V6, &ORACLE_V6), (V3, &ORACLE_V3)] {
        let lenses = lenses(text);
        for &(k, tu, tv, u, v) in oracle {
            let d = bearing(ORACLE_TILE, k, f64::from(tu), f64::from(tv));
            let [pu, pv] = project(&lenses[k], d).unwrap();
            assert!(
                (pu - u).abs() < 1e-6 && (pv - v).abs() < 1e-6,
                "lens {k} ({tu}, {tv}): ({pu}, {pv}) vs ({u}, {v})"
            );
        }
    }
    // R_P_lens of V6 lens 0 (independent f64 implementation): within 0.63 deg of the identity, not the identity.
    let r = lenses(V6)[0].pano_from_lens.rotation;
    assert!(
        (r[(0, 1)] + 0.010977876806482231).abs() < 1e-12
            && (r[(2, 0)] - 0.007103414994941127).abs() < 1e-12
    );
}

#[test]
fn the_lens_centres_sit_on_the_panorama_z_axis_about_the_midpoint() {
    // The X4 Air factory string gives a 33.1 mm baseline on the optical axis, 0.1-0.8 mm lateral; lens 0 looks along +Z.
    let lenses = lenses(V6);
    let [c0, c1] = [0, 1].map(|k| lenses[k].pano_from_lens.translation.vector);
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
    // Lens 0 (V3) image radius 294.9 / 607.6 / 900.2 / 972.2 px at 30 / 60 / 90 / 100 deg.
    // The radial part alone: no tangential terms and the mean focal.
    let factory = lenses(V3)[0].camera;
    let [fx, fy, cx, cy, xi] = factory.params();
    let MeiDistortion::V3 { k, .. } = factory.distortion() else {
        panic!("a V3 string gives V3 lenses")
    };
    let f = 0.5 * (fx + fy);
    let lens = Mei::new(
        [f, f, cx, cy, xi],
        factory.half_field_rad(),
        MeiDistortion::V3 { k, p: [0.0; 2] },
    )
    .unwrap();
    let radius_px = |theta: f64| {
        let m = theta.sin() / (theta.cos() + xi);
        let r2 = m * m;
        f * m * (1.0 + k[0] * r2 + k[1] * r2 * r2 + k[2] * r2 * r2 * r2)
    };
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
        let r = (u - cx).hypot(v - cy);
        assert!(
            (r - radius_px(t)).abs() < 1e-9 && (r - radius).abs() < 0.1,
            "{deg}: {r}"
        );
    }
    assert!(
        lens.project([1.0, 0.0, -0.18]).is_err(),
        "100.2 deg is past the lens coverage"
    );
}

#[test]
fn the_table_quantises_the_oracle_to_half_a_step() {
    // q = floor(128 u + 0.5) in 1/256 level-1 px: |q - 128 u| <= 1/2 step, plus the 1e-6 px projection error.
    let lenses = lenses(V6);
    let luts = [0, 1].map(|k| lut(&lenses[k], ORACLE_TILE, k));
    for &(k, tu, tv, u, v) in &ORACLE_V6 {
        let s = luts[k].samples()[tv as usize * 1024 + tu as usize];
        if u > 1918.0 {
            assert_eq!(
                s,
                RemapTable::INVALID,
                "{k} ({tu}, {tv}) is past the last level-1 sample"
            );
            continue;
        }
        let (ex, ey) = (f64::from(s[0]) - 128.0 * u, f64::from(s[1]) - 128.0 * v);
        assert!(
            ex.abs() <= 0.5 + 1e-3 && ey.abs() <= 0.5 + 1e-3,
            "{k} ({tu}, {tv}): {ex} {ey}"
        );
    }
}

#[test]
fn invalid_pixels_sit_at_the_lens_edges() {
    // With the same validity rule, the independent f64 projection gives these counts (189 for the
    // open interval 0 < u < 1919 at margin 32; level 1 has no sample past full-res 1918).
    let lenses = lenses(V6);
    for (margin, counts) in [(32, [0, 244]), (64, [6787, 8427])] {
        let g = TileGeometry {
            margin,
            ..FACTORY_TILE
        };
        for k in 0..2 {
            let t = lut(&lenses[k], g, k);
            let invalid = t
                .samples()
                .iter()
                .filter(|&&s| s == RemapTable::INVALID)
                .count();
            assert_eq!(invalid, counts[k], "margin {margin} tile {k}");
            let (w, cx) = (g.width(), (g.width() as f64 - 1.0) / 2.0);
            for (i, _) in t
                .samples()
                .iter()
                .enumerate()
                .filter(|(_, s)| **s == RemapTable::INVALID)
            {
                let (tu, tv) = (i % w, i / w);
                // Only more than 95 deg of longitude from the lens axis (the frames cut the lenses at 95.9-99.1 deg),
                // and only within 200 rows (37.5 deg) of the equator.
                let lon = (tu as f64 - cx).abs() * std::f64::consts::PI / 960.0;
                assert!(
                    lon > 95f64.to_radians() && tv.abs_diff(480) < 200,
                    "{tu} {tv}"
                );
            }
            if margin == 32 && k == 1 {
                assert!(t
                    .samples()
                    .iter()
                    .enumerate()
                    .all(|(i, s)| *s != RemapTable::INVALID
                        || (i % w >= 1019 && (438..=504).contains(&(i / w)))));
            }
        }
    }
}

fn tiles(
    path: &str,
    clip: &Insta360Clip,
    luts: &[RemapTable; 2],
    frame: usize,
) -> [Image<u16, 1>; 2] {
    std::array::from_fn(|k| {
        let mut tile = Image::from_size_val(luts[k].dst_size(), 0u16).unwrap();
        let lens = decode(path, clip.lens_track[k], frame);
        remap(&lens_level1(&lens).unwrap(), &luts[k], &mut tile).unwrap();
        tile
    })
}

#[test]
#[ignore = "needs the X4 Air clip VID_20261116_125200_00_003.insv (KORNIA_STAGING_INSV) and ffmpeg"]
fn a_real_frame_pair_matches_the_oracle_remap() {
    let Some((path, clip)) = clip() else { return };
    let lenses = clip
        .lenses_v6
        .as_ref()
        .expect("the clip carries a V6 string");
    let luts = [0, 1].map(|k| lut(&lenses[k], ORACLE_TILE, k));
    let tiles = tiles(&path, &clip, &luts, 12);
    for k in 0..2 {
        // The independent `remap`: bilinear on the full-resolution lens picture at the factory projection, 0 outside.
        let lens = decode(&path, clip.lens_track[k], 12);
        let level1 = lens_level1(&lens).unwrap();
        let (mut sum, mut n, mut big, mut exact) = (0.0, 0usize, 0usize, 0.0f64);
        for (i, &value) in tiles[k].as_slice().iter().enumerate() {
            let d = bearing(ORACLE_TILE, k, (i % 1024) as f64, (i / 1024) as f64);
            let [u, v] = project(&lenses[k], d).unwrap();
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
