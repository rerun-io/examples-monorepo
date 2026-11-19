use super::*;
use crate::camera::formats::insta360::{parse_lens_string, FactoryLens, WindowCrop};
use nalgebra::Vector3;

// The real X4 Air strings, from the clip VID_20261116_125200_00_003.insv.
include!("../../../../kornia-staging-io/tests/common/x4air_lens.rs");

pub(super) struct Pinned {
    rotation: [u64; 9],
    centre: [u64; 3],
    intrinsics: [u64; 5],
    lens: [[u64; 2]; 12],
    panorama: [[u64; 2]; 12],
}

/// The rays the pinned pixels were taken at: up to 95.7 degrees off the axis, of several lengths.
fn rays() -> [[f64; 3]; 12] {
    std::array::from_fn(|i| {
        let theta = (i as f64) * 8.7_f64.to_radians() + 0.013;
        let phi = (i as f64) * 0.77 + 0.1;
        let r = 1.0 + 0.3 * i as f64;
        [
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
        ]
    })
}

fn lenses(text: &str) -> Vec<FactoryLens> {
    parse_lens_string(text, Some(CROP), [1920, 1920]).unwrap()
}

fn bits(pixel: Result<[f64; 2], ProjectionReject>) -> [u64; 2] {
    pixel.map_or([u64::MAX; 2], |p| p.map(f64::to_bits))
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn the_parsed_lenses_project_bit_identically_to_the_factory_reader_they_replace() {
    let all: Vec<FactoryLens> = [V3, V6].into_iter().flat_map(lenses).collect();
    for (lens, pinned) in all.iter().zip(&PINNED) {
        let m = lens.pano_from_lens.rotation.matrix();
        assert_eq!(
            std::array::from_fn::<u64, 9, _>(|k| m[(k / 3, k % 3)].to_bits()),
            pinned.rotation
        );
        assert_eq!(
            lens.pano_from_lens
                .translation
                .vector
                .map(f64::to_bits)
                .data
                .0[0],
            pinned.centre
        );
        assert_eq!(lens.camera.params().map(f64::to_bits), pinned.intrinsics);
        for (k, ray) in rays().into_iter().enumerate() {
            assert_eq!(bits(lens.camera.project(ray)), pinned.lens[k], "ray {k}");
            let local = lens
                .pano_from_lens
                .rotation
                .inverse_transform_vector(&Vector3::from(ray));
            assert_eq!(
                bits(lens.camera.project(local.into())),
                pinned.panorama[k],
                "panorama ray {k}"
            );
        }
    }
}

#[test]
fn the_v3_string_gives_the_recon_lens_numbers_at_1920() {
    let lenses = lenses(V3);
    let [l0, l1] = [lenses[0].camera.params(), lenses[1].camera.params()];
    assert!(
        close(l0[2], 960.242, 1e-3) && close(l0[3], 965.681, 1e-3),
        "{l0:?}"
    );
    assert!(
        close(l1[2], 966.509, 1e-3) && close(l1[3], 965.847, 1e-3),
        "{l1:?}"
    );
    // Pixels per radian at the axis, f / (1 + xi): 553.5 and 557.7.
    let px_per_rad = |p: [f64; 5]| 0.5 * (p[0] + p[1]) / (1.0 + p[4]);
    assert!(close(px_per_rad(l0), 553.5, 0.05) && close(px_per_rad(l1), 557.7, 0.05));
    assert!(matches!(
        lenses[0].camera.distortion(),
        MeiDistortion::V3 { .. }
    ));
    assert!(close(
        lenses[0].camera.half_field_rad(),
        100f64.to_radians(),
        1e-15
    ));
    // Lens 1 is turned 179.91 deg against lens 0; the centres are 33.1 mm apart on the optical axis, lens 0 at
    // +z 16.56 mm from the midpoint.
    let relative = lenses[0].pano_from_lens.inverse() * lenses[1].pano_from_lens;
    assert!(close(relative.rotation.angle().to_degrees(), 179.91, 0.01));
    let local0 = lenses[0]
        .pano_from_lens
        .inverse_transform_vector(&lenses[0].pano_from_lens.translation.vector);
    assert!(
        close(local0.z, 0.0165615, 1e-6) && close(local0.x, -0.0004025, 1e-7),
        "{local0:?}"
    );
    let distance = (lenses[0].pano_from_lens.translation.vector
        - lenses[1].pano_from_lens.translation.vector)
        .norm();
    assert!(close(distance, 0.0331, 1e-4), "{distance}");
}

#[test]
fn the_v6_string_keeps_tangential_and_prism_terms() {
    let lenses = lenses(V6);
    assert!(matches!(
        lenses[0].camera.distortion(),
        MeiDistortion::V6 { d } if d[9] != 0.0 && d[12] != 0.0
    ));
    assert!(close(lenses[0].camera.params()[2], 958.292, 1e-3));
    assert!(close(lenses[1].camera.params()[3], 964.396, 1e-3));
}

#[test]
fn without_the_window_crop_the_centre_is_128_calibration_pixels_off() {
    let lenses = parse_lens_string(V3, None, [1920, 1920]).unwrap();
    // 1920/7680 instead of 1920/7424: the centre lands 128 px * 0.25 further from the origin than with the crop.
    assert!(close(
        lenses[0].camera.params()[2],
        (3842.37 + 0.5) * 0.25 - 0.5,
        1e-9
    ));
}

#[test]
fn malformed_or_unsupported_lens_strings_are_named_errors_not_panics() {
    // The 1e30, 2.5 and -1 strings state that many lenses: the count must be a whole number from 1 to 8.
    for (text, reason) in [
        ("", "not a finite number"),
        ("2_1_2", "version 0"),
        ("2_x_197632", "not a finite number"),
        ("1_2.3_197632", "tokens for 1 lenses"),
        ("2_1_2_3_132096", "version 2"),
        ("1e30_1_197632", "not a whole number"),
        ("2.5_1_197632", "not a whole number"),
        ("-1_1_197632", "not a whole number"),
    ] {
        let error = parse_lens_string(text, None, [1920, 1920]).unwrap_err();
        assert!(error.to_string().contains(reason), "{text}: {error}");
    }
    // Non-finite tokens parse as f64 but are refused by name.
    for bad in ["NaN", "inf", "-inf"] {
        let text = V3.replacen("2.314940", bad, 1);
        let error = parse_lens_string(&text, Some(CROP), [1920, 1920]).unwrap_err();
        assert!(error.to_string().contains(&format!("{bad:?}")), "{error}");
    }
    // A lens type without an entry in the field table (132) gets the 190 degree default; 41 (ONE X2) is a 200 degree
    // class and 156 a 195 degree one.
    for (lens_type, half_field_deg) in [("132", 95.0), ("41", 100.0), ("156", 97.5)] {
        let text = V3.replacen("_15360_7680_131_", &format!("_15360_7680_{lens_type}_"), 1);
        let lenses = parse_lens_string(&text, Some(CROP), [1920, 1920]).unwrap();
        assert_eq!(
            lenses[0].camera.half_field_rad(),
            half_field_deg * std::f64::consts::PI / 180.0
        );
        // Each lens states its own type: the first lens's token changed, the second's did not.
        assert_eq!(
            [lenses[0].lens_type, lenses[1].lens_type],
            [lens_type.parse::<u32>().unwrap(), 131]
        );
    }
    // A negative focal makes no camera.
    let text = V3.replacen("_7095.540_", "_-7095.540_", 1);
    let error = parse_lens_string(&text, Some(CROP), [1920, 1920]).unwrap_err();
    assert!(error.to_string().contains("no valid camera"), "{error}");
}

// The Insta360 X5 clip VID_20261113_145021_00_003's V3 string (lens type 113) and window crop.
const X5_V3: &str = "2_2.000000_4267.220_4267.930_2693.490_2692.190_0.073_0.192_89.740_0.000000_0.000000_0.000000_0.19010369_2.01127100_-3.03693724_0.00169806_-0.00122825_10752_5376_113_2.000000_4249.040_4248.200_8059.560_2694.530_-0.294_0.371_90.620_-0.000469_0.000039_-0.032061_0.18495527_2.02530622_-3.03083992_-0.00055454_0.00157740_10752_5376_113_197632";
const X5_CROP: WindowCrop = WindowCrop {
    src: [5376, 5376],
    dst: [5312, 5312],
    offset: [0, 0],
};

#[test]
fn the_x5_string_of_lens_type_113_gives_two_lenses_at_3840() {
    let lenses = parse_lens_string(X5_V3, Some(X5_CROP), [3840, 3840]).unwrap();
    let [l0, l1] = [lenses[0].camera.params(), lenses[1].camera.params()];
    // fx 4267.22 at 5376 px, scaled to the 5312 px window, then to 3840: 4267.22 * 3840 / 5312.
    assert!(close(l0[0], 4267.22 * 3840.0 / 5312.0, 1e-9), "{l0:?}");
    assert!(
        close(l0[2], 1923.830, 1e-3) && close(l0[3], 1922.890, 1e-3),
        "{l0:?}"
    );
    assert!(
        close(l1[2], 1916.652, 1e-3) && close(l1[3], 1924.582, 1e-3),
        "{l1:?}"
    );
    assert_eq!(l0[4], 2.0);
    assert!(close(
        lenses[0].camera.half_field_rad(),
        100f64.to_radians(),
        1e-15
    ));
    // The two lenses look opposite ways and sit about 32 mm apart.
    let relative = lenses[0].pano_from_lens.inverse() * lenses[1].pano_from_lens;
    assert!(close(relative.rotation.angle().to_degrees(), 180.0, 1.0));
    let distance = (lenses[0].pano_from_lens.translation.vector
        - lenses[1].pano_from_lens.translation.vector)
        .norm();
    assert!(close(distance, 0.032, 0.002), "{distance}");
    // A ray just inside 100 degrees off lens 0's axis projects inside the 3840 px image; just outside does not.
    let ray = |deg: f64| [deg.to_radians().sin(), 0.0, deg.to_radians().cos()];
    let [u, _] = lenses[0].camera.project(ray(99.99)).unwrap();
    assert!(u > 3800.0 && u < 3839.0, "{u}");
    assert!(lenses[0].camera.project(ray(100.01)).is_err());
}

// Bits of the pre-move `LensCalibration` (kornia-staging-io `parse_lens_string`, `project`,
// `project_from_panorama`, `pano_from_lens_rotation`, `pano_centre_m`) for the X4 Air strings with the window crop
// at 1920 px, V3 lens 0, V3 lens 1, V6 lens 0, V6 lens 1. `NONE`: no pixel (u64::MAX).
#[rustfmt::skip]
pub(super) const PINNED: [Pinned; 4] = [
    Pinned {
        rotation: [0x3fefff786a0aee6c, 0xbf8684b9036a56e9, 0xbf67bf3c6aedb693, 0x3f868757b3fa1a59, 0x3fefff7e081dce4b, 0x3f5c1327b956a31e, 0x3f679758bec42804, 0xbf5c98701025eb22, 0x3feffff41c1cd155],
        centre: [0xbf3d7fc164c0de8f, 0xbee2f8dc6cf25fb6, 0x3f90f52f3a665ccb],
        intrinsics: [0x409cac36bb25136c, 0x409caa96f3a43170, 0x408e01f01c3f8f02, 0x408e2d72c234f72d, 0x400284ff43419e30],
        lens: [[0x408e3b38bbb84757, 0x408e3331e8ff95a8], [0x408fd9906ee4dc52, 0x40902e3c255f5e78], [0x408da017a2993b66, 0x4091d7528d274324], [0x4087e26dd595e051, 0x4091d64ecff05bfa], [0x408305f269800af9, 0x408dc2515a4d1b6d], [0x408478af1207028b, 0x40843415e5034a46], [0x408e210a177dc9b2, 0x407b14e85e43e428], [0x4095d7a6291fbd07, 0x40804aa2ab25e9f2], [0x409a298fa775c335, 0x408dac242a02de5b], [0x409831a1a0039d50, 0x40979b3c6d165042], [0x408f7ba2f12ad8fa, 0x409cdcfe7cb84061], [0x4074fa084e7738a0, 0x409a4e3814d88797]],
        panorama: [[0x408e4808e4321303, 0x408e2ad67d71d824], [0x408fec5daa790a5e, 0x409027cccd540e79], [0x408dbc0963a3fd6d, 0x4091d3da55e0c05f], [0x4087ff15141d879d, 0x4091dabe852ae9f4], [0x4083127222bedc12, 0x408dda6a53e1de6a], [0x408467c6f9cdcda4, 0x40844959e9baf4f4], [0x408dfb6a2fd89bc0, 0x407b03d861ddc540], [0x4095ca39529ce7d5, 0x40801ab7083838ce], [0x409a2f6c8d9f28fa, 0x408d6980e1a84df6], [0x40984c5471244e9a, 0x4097812190748c3b], [0x408fc996bdb683ba, 0x409cd78e9e5b6976], [0x407584b36a7f677e, 0x409a6554284c8280]],
    },
    Pinned {
        rotation: [0xbfeffc3758627399, 0xbf9eca0176942bd0, 0x3f72183473bebb2c, 0xbf9ecb6b4f8ca3c2, 0x3feffc4a11687dbe, 0xbf52fe280a63ab61, 0xbf71f18ef977ddc8, 0xbf5529200b920567, 0xbfefffea20a3eefc],
        centre: [0x3f3d7fc164c0de8f, 0x3ee2f8dc6cf25fb6, 0xbf90f52f3a665ccb],
        intrinsics: [0x409ce2ead054beae, 0x409ce39ee58469ef, 0x408e3411a7b9611c, 0x408e2ec5bce90c5c, 0x400284ff43419e30],
        lens: [[0x408e6dc830217f2d, 0x408e34904f6bdba2], [0x409007b66d3ebfd4, 0x40903119f22f8cee], [0x408dd200144b7060, 0x4091dd460eb4ed18], [0x40880b3633ba9a23, 0x4091dbbc07095d09], [0x408327fdcc23a736, 0x408dc27579af2a0a], [0x40849c7948946fb0, 0x408421e769fbf674], [0x408e57758748ef43, 0x407ac9d5f5ae2a8a], [0x4096053a1289d5b7, 0x408025413620f192], [0x409a6506bf35e046, 0x408daa4e3b38a287], [0x40986640d4b47851, 0x4097b143b48289f9], [0x408fba67dbab7307, 0x409cf92bd3e802d1], [0x40753bd2c2ab6eac, 0x409a5e9c3c6e8cef]],
        panorama: [[u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [0x408bcb2207851378, 0x409d871652a44f3a], [0x409793270b208082, 0x40996d6cb4c9ba2a]],
    },
    Pinned {
        rotation: [0x3fefff4c617f5163, 0xbf867b91aee8284b, 0xbf7d2ff1f2587403, 0x3f86832bbc5443b3, 0x3fefff7d1ad68987, 0x3f605fbfec35982e, 0x3f7d1878717db2da, 0xbf6103a8d78baf71, 0x3fefffc69100852e],
        centre: [0xbf283a4c83f6ee20, 0x3f0c55a5d39960bd, 0x3f910b438b598728],
        intrinsics: [0x409cad79d218b79e, 0x409cac0f01c3f8f1, 0x408df25682a5f569, 0x408e2ef01c3f8f02, 0x400284ff43419e30],
        lens: [[0x408e2ba17b2784de, 0x408e34af6e3455d3], [0x408fca10de567d7c, 0x40902eec75752119], [0x408d911f629b0b2c, 0x4091d7fff7247879], [0x4087d3540b9c8087, 0x4091d74baa594c11], [0x4082f5e0dc9a6d2a, 0x408dc3021ab46ea5], [0x40846b552e0a6c42, 0x408432436a95732b], [0x408e182047e0e979, 0x407b17bbc16005ea], [0x4095d152e16eeb04, 0x4080512169b34bc8], [0x409a2077faa9b1c7, 0x408dabf4dadac652], [0x40982dbaa84d5641, 0x40979668f4ebdd2c], [0x408f8202fd28b61a, 0x409cdcf2d4038877], [0x4074f6a5374c574a, 0x409a54c73e9f43e8]],
        panorama: [[0x408e4b2758c97274, 0x408e2adbf0bfdcb3], [0x408fef84b0348457, 0x409027d06ac855cb], [0x408dbf41500d740b, 0x4091d3beebda3a95], [0x408802a8a36a89d5, 0x4091da851f56f2c0], [0x408316449190a67e, 0x408dd9e142b20464], [0x40846bf287736ef6, 0x4084488e5fb4a34c], [0x408e002dfb11ceec, 0x407b03606a4d764e], [0x4095cc011a6f40b0, 0x40801b9c3a0d42e9], [0x409a3034739b469e, 0x408d688b00cd937c], [0x40984e5360217726, 0x40977f389047b828], [0x408fd101057e0552, 0x409cd73c0e216a45], [0x40758a27ad5827b8, 0x409a675b87b115d8]],
    },
    Pinned {
        rotation: [0xbfeffc27f24f4c37, 0xbf9ec0cfddc37920, 0x3f78adc234e042ca, 0xbf9ec676dfdda1e8, 0x3feffc3faab661ff, 0xbf6cd75e134bdedd, 0xbf783bfed7597258, 0xbf6e4fa676720272, 0xbfefffccefea954e],
        centre: [0x3f283a4c83f6ee20, 0xbf0c55a5d39960bd, 0xbf910b438b598728],
        intrinsics: [0x409ce3bead054bec, 0x409ce43dcb08d3dd, 0x408e2d3dcb08d3dc, 0x408e232a5f5682a6, 0x400284ff43419e30],
        lens: [[0x408e66f5d21d2015, 0x408e28f50f4cf788], [0x409004492dd44872, 0x40902b48f387a52b], [0x408dcb23a824e1eb, 0x4091d76b2cd0f9d7], [0x4088043c7954bd7e, 0x4091d60676155514], [0x4083204f41494cdf, 0x408db73b21018386], [0x4084949ab80044e1, 0x40841610fedf7ee6], [0x408e51196fb4e48e, 0x407ab2bb1eaa3072], [0x4096027b3a4ce50e, 0x40801c9793fddaaa], [0x409a611ca12e9a78, 0x408da3dc9964803c], [0x409861c024fbccc1, 0x4097acd3138722d3], [0x408fb736ca423d7a, 0x409cf3f1cb46f292], [0x407540f3729c037a, 0x409a5d0029fdf0d7]],
        panorama: [[u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [u64::MAX, u64::MAX], [0x408bc8e6b7e3cccd, 0x409d866042ba0616], [0x409792b3613d87b4, 0x40996d96ceb78fda]],
    },
];
