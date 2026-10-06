//! Shared camera model and explicit benchmark path syntax.
pub use gsplat_render::camera::*;
#[derive(Debug, Clone)]
pub enum CameraPath {
    Orbit(usize),
    TestViews(std::path::PathBuf),
    Specs(std::path::PathBuf),
    Colmap(std::path::PathBuf),
    /// Mip-NeRF 360 holdout: every eighth image after sorting filenames.
    ColmapTest(std::path::PathBuf),
}
impl std::str::FromStr for CameraPath {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value == "held" {
            return Ok(Self::Orbit(1));
        }
        let (kind, path) = value
            .split_once(':')
            .ok_or("path must be orbit:N, held, test-views:FILE, specs:FILE, colmap:DIR, or colmap-test:DIR")?;
        if path.is_empty() {
            return Err("camera path payload is empty".into());
        }
        match kind {
            "orbit" => {
                let n = path
                    .parse::<usize>()
                    .map_err(|_| "orbit count must be a positive integer")?;
                if n == 0 {
                    Err("orbit count must be positive".into())
                } else {
                    Ok(Self::Orbit(n))
                }
            }
            "test-views" => Ok(Self::TestViews(path.into())),
            "specs" => Ok(Self::Specs(path.into())),
            "colmap" => Ok(Self::Colmap(path.into())),
            "colmap-test" => Ok(Self::ColmapTest(path.into())),
            _ => Err(format!("unknown camera path prefix {kind:?}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_camera_path_syntax() {
        assert!(matches!(
            "orbit:300".parse::<CameraPath>().unwrap(),
            CameraPath::Orbit(300)
        ));
        assert!("colmap-test:scene/sparse/0".parse::<CameraPath>().is_ok());
        for invalid in ["orbit300", "orbit:0", "orbit:-1", "colmap:", "testview:x"] {
            assert!(invalid.parse::<CameraPath>().is_err(), "{invalid}");
        }
    }
}
