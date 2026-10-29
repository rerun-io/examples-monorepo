//! Hexel PLY reader: binary little-endian f32 rows of activated parameters,
//! announced by `comment hexels_format 1` (hexels-work contract `hexel-ply.md`).
use anyhow::{Context as _, Result, bail, ensure};
use gsplat_core::{HexelVariant, Hexels};

/// 3D properties of every variant: centre, wxyz rotation, scales, opacity, sharpness, colour field.
const BASE: &str =
    "x y z qw qx qy qz su sv opacity sharpness c0_r c0_g c0_b gu_r gu_g gu_b gv_r gv_g gv_b";

/// 2D image hexels (`hexels_variant 2d`): pixel-space centre and scales, angle, colour field.
const IMAGE: &str = "x y su sv theta sharpness c0_r c0_g c0_b gx_r gx_g gx_b gy_r gy_g gy_b";

/// Vertex properties of a variant in file order.
pub fn property_names(variant: HexelVariant) -> Vec<String> {
    let words = |names: &str| names.split(' ').map(str::to_owned).collect::<Vec<_>>();
    let mut names = words(BASE);
    match variant {
        HexelVariant::V1 => {}
        HexelVariant::V2 => names.extend((0..9).map(|i| format!("a_{}{}", i / 3, i % 3))),
        HexelVariant::V3 => names.extend((0..45).map(|i| format!("sh_{i}"))),
        HexelVariant::Image => names = words(IMAGE),
    }
    names
}

/// A parsed hexel PLY; 2D fits also carry their image size.
pub struct HexelPly {
    pub hexels: Hexels,
    pub image_size: Option<glam::UVec2>,
}

fn header(bytes: &[u8]) -> Option<(&str, &[u8])> {
    const END: &[u8] = b"end_header\n";
    let end = bytes.windows(END.len()).position(|w| w == END)?;
    Some((
        std::str::from_utf8(&bytes[..end]).ok()?,
        &bytes[end + END.len()..],
    ))
}

/// Whether a PLY declares the hexel format; splat PLYs return false.
pub fn is_hexel_ply(bytes: &[u8]) -> bool {
    header(bytes).is_some_and(|(text, _)| text.lines().any(|l| l == "comment hexels_format 1"))
}

/// Strict reader: exact property list and order, finite values, no trailing bytes.
pub fn parse(bytes: &[u8]) -> Result<HexelPly> {
    let (text, body) = header(bytes).context("PLY header has no end_header line")?;
    let mut lines = text.lines();
    ensure!(lines.next() == Some("ply"), "not a PLY file");
    let (mut variant, mut count, mut properties, mut image_size) = (None, None, Vec::new(), None);
    for line in lines {
        let words: Vec<&str> = line.split_whitespace().collect();
        match words.as_slice() {
            ["format", "binary_little_endian", "1.0"] => {}
            ["format", ..] => bail!("hexel PLY must be binary_little_endian 1.0: {line}"),
            ["comment", "hexels_variant", name] => {
                variant = Some(match *name {
                    "v1" => HexelVariant::V1,
                    "v2" => HexelVariant::V2,
                    "v3" => HexelVariant::V3,
                    "2d" => HexelVariant::Image,
                    _ => bail!("unknown hexels_variant {name}"),
                });
            }
            ["comment", "hexels_image", width, height] => {
                image_size = Some(glam::UVec2::new(width.parse()?, height.parse()?));
            }
            ["comment", ..] | [] => {}
            ["element", "vertex", n] => count = Some(n.parse::<usize>()?),
            ["element", ..] => bail!("hexel PLY has one vertex element: {line}"),
            ["property", "float", name] => properties.push((*name).to_owned()),
            _ => bail!("unsupported hexel PLY header line: {line}"),
        }
    }
    let variant = variant.context("hexel PLY lacks comment hexels_variant")?;
    let count = count.context("hexel PLY lacks element vertex")?;
    ensure!(
        properties == property_names(variant),
        "hexel PLY properties do not match the {variant:?} contract order"
    );
    let floats = count
        .checked_mul(variant.stride())
        .context("hexel count overflow")?;
    ensure!(
        body.len() == floats * 4,
        "hexel PLY body has {} bytes, expected {}",
        body.len(),
        floats * 4
    );
    let params: Vec<f32> = body
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(
        params.iter().all(|v| v.is_finite()),
        "hexel PLY has non-finite values"
    );
    ensure!(count > 0, "empty hexel PLY");
    ensure!(
        (variant == HexelVariant::Image) == image_size.is_some_and(|s| s.min_element() > 0),
        "2D hexel PLYs, and only they, need comment hexels_image <width> <height>"
    );
    Ok(HexelPly {
        hexels: Hexels { variant, params },
        image_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ply(variant: &str, names: &[String], rows: &[f32]) -> Vec<u8> {
        let image = if variant == "2d" {
            "comment hexels_image 40 30\n"
        } else {
            ""
        };
        let mut text = format!(
            "ply\nformat binary_little_endian 1.0\ncomment hexels_format 1\ncomment hexels_variant {variant}\n\
             {image}comment hexels_source test\ncomment hexels_step 7\nelement vertex {}\n",
            rows.len() / names.len()
        );
        for name in names {
            text.push_str("property float ");
            text.push_str(name);
            text.push('\n');
        }
        text += "end_header\n";
        let mut bytes = text.into_bytes();
        bytes.extend(rows.iter().flat_map(|v| v.to_le_bytes()));
        bytes
    }

    #[test]
    fn reads_every_variant_in_contract_order() {
        for (name, variant) in [
            ("v1", HexelVariant::V1),
            ("v2", HexelVariant::V2),
            ("v3", HexelVariant::V3),
            ("2d", HexelVariant::Image),
        ] {
            let names = property_names(variant);
            assert_eq!(names.len(), variant.stride());
            let rows: Vec<f32> = (0..2 * names.len()).map(|i| i as f32).collect();
            let bytes = ply(name, &names, &rows);
            assert!(is_hexel_ply(&bytes));
            let parsed = parse(&bytes).unwrap();
            assert_eq!(parsed.hexels.variant, variant);
            assert_eq!(parsed.hexels.len(), 2);
            assert_eq!(parsed.hexels.params, rows);
            let image = (variant == HexelVariant::Image).then_some(glam::UVec2::new(40, 30));
            assert_eq!(parsed.image_size, image);
        }
        assert_eq!(
            property_names(HexelVariant::V2)[20..23],
            ["a_00", "a_01", "a_02"]
        );
        assert_eq!(property_names(HexelVariant::V3)[64], "sh_44");
    }

    #[test]
    fn rejects_swapped_properties_truncation_and_splat_plys() {
        let mut names = property_names(HexelVariant::V1);
        let rows = vec![0.5; names.len()];
        assert!(parse(&ply("v1", &names, &rows[..rows.len() - 1])).is_err());
        names.swap(7, 8);
        assert!(parse(&ply("v1", &names, &rows)).is_err());
        assert!(parse(&ply("v4", &property_names(HexelVariant::V1), &rows)).is_err());
        let splat = include_bytes!("../tests/one-vertex-ply-header.txt");
        assert!(!is_hexel_ply(splat));
    }

    #[tokio::test]
    #[ignore = "integration: GPU"]
    async fn hexel_ply_renders_colour_depth_and_normal_through_the_cli_renderer() {
        // One frontal hexel at z = 2 facing the camera: o = 0.9, sharp edges.
        let mut row = vec![0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.9, 200.0];
        row.extend([0.8, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("model.ply");
        std::fs::write(&path, ply("v1", &property_names(HexelVariant::V1), &row)).unwrap();
        let scene = crate::PlyScene::load(&path).await.unwrap();
        assert_eq!(scene.hexels.as_ref().unwrap().len(), 1);
        assert_eq!(scene.center, glam::Vec3::new(0.0, 0.0, 2.0));
        let camera = crate::camera::CameraSpec::from_nerf(glam::Mat4::IDENTITY, 1.0, 64, 48);
        // NeRF cameras look down -z; turn this one around to face +z.
        let camera = crate::camera::CameraSpec {
            world_from_camera: glam::Mat4::IDENTITY.to_cols_array_2d(),
            ..camera
        };
        let settings = crate::settings::RenderSettings::default();
        let size = glam::UVec2::new(64, 48);
        let mut renderer =
            crate::engines::ours(&scene, size, &settings, None, glam::Vec3::splat(0.5))
                .await
                .unwrap();
        renderer.render(&camera, crate::Output::Surface).unwrap();
        let rgba = renderer.read_rgba().unwrap();
        let surface = renderer.read_surface().unwrap();
        let centre = 24 * 64 + 32;
        let expected = [0.8 * 0.9 + 0.05, 0.4 * 0.9 + 0.05, 0.2 * 0.9 + 0.05, 0.9];
        for (actual, expected) in rgba[centre * 4..centre * 4 + 4].iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        assert!((surface.depth[centre][0] - 2.0).abs() < 1e-5);
        assert!((surface.depth[centre][1] - 2.0).abs() < 1e-5);
        assert!(
            (glam::Vec4::from(surface.normal[centre]) - glam::Vec4::new(0.0, 0.0, -1.0, 0.0))
                .length()
                < 1e-5
        );
        assert_eq!(rgba[..4], [0.5, 0.5, 0.5, 0.0]);
        renderer.render(&camera, crate::Output::Float).unwrap();
        assert_eq!(renderer.read_rgba().unwrap(), rgba);
        assert!(renderer.read_surface().is_err());
    }

    #[tokio::test]
    #[ignore = "integration: GPU"]
    async fn image_hexel_ply_renders_at_its_own_size() {
        // One flat 2D hexel at pixel (20, 15); its own weight normalizes to colour c0.
        let row = [
            20.0, 15.0, 6.0, 6.0, 0.3, 10.0, 0.9, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("fit.ply");
        std::fs::write(&path, ply("2d", &property_names(HexelVariant::Image), &row)).unwrap();
        let scene = crate::PlyScene::load(&path).await.unwrap();
        let size = scene.image_size.unwrap();
        assert_eq!(size, glam::UVec2::new(40, 30));
        let camera =
            crate::camera::CameraSpec::from_nerf(glam::Mat4::IDENTITY, 1.0, size.x, size.y);
        let settings = crate::settings::RenderSettings::default();
        let mut renderer = crate::engines::ours(&scene, size, &settings, None, glam::Vec3::ZERO)
            .await
            .unwrap();
        renderer.render(&camera, crate::Output::Float).unwrap();
        let rgba = renderer.read_rgba().unwrap();
        let centre = (15 * 40 + 20) * 4;
        let weight = 1.0 / (1.0 + (-10.0f32).exp());
        let scale = weight / (weight + 1e-4);
        for (actual, expected) in rgba[centre..centre + 3].iter().zip([0.9, 0.5, 0.1]) {
            assert!(
                (actual - expected * scale).abs() < 1e-5,
                "{actual} != {expected}"
            );
        }
    }
}
