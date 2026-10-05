//! Checked-in SLAM profiles and command-line overrides.

use slam_rs::config::VioConfig;

use super::{MSDMO_CONFIG, SlamError};

/// Which VIO profile the SLAM stage runs: a checked-in overlay of VIO configuration keys (`slam-rs/configs/profiles/*.json`),
/// applied over [`MSDMO_CONFIG`] through the same path as [`super::SlamConfig::overrides`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SlamProfile {
    /// PR #270's `fast.json`.
    Fast,
    /// `live.json`, the runtime default: `fast`, then a keyframe's joint solve runs beside the next frameset's
    /// frontend (slam-rs D84) and takes at most 5 LM iterations. Inside every accuracy band. Full s66, lossless, against
    /// `fast`: 1333 tracked framesets both, ATE 6.15 vs 8.73 mm, RPE over 30 framesets 5.6 mm / 0.17 deg vs 8.3 mm / 0.20 deg;
    /// the 10 s clip 167 both, 3.23 vs 3.15 mm. Cap B, full system in realtime (60 s): 29.3 Hz while tracking with the source
    /// at 29.4/s, compute mean 26.0 / p95 54.2 ms (fast: 25.4 Hz, 32.3 / 73.6 ms).
    #[default]
    Live,
    /// `live30.json`, opt-in: `live` with a 70 px detection grid (about 40 % less frontend work). Full s66: 1333
    /// tracked, ATE 6.93 mm; Cap B full system: 29.6 Hz while tracking (= the source), compute mean 18.9 / p95 35.7 ms. Outside
    /// the 10 s clip's band: ATE 5.00 mm (band 3.8 mm), and one of seven 15 s cold-start sub-clips starts tracking 2.3 s later.
    Live30,
}

impl SlamProfile {
    /// The profile's overlay: a JSON object of VIO configuration keys (as in [`super::SlamConfig::overrides`]) and values.
    pub fn overlay(self) -> &'static str {
        match self {
            Self::Fast => include_str!("../../../../../slam-rs/configs/profiles/fast.json"),
            Self::Live => include_str!("../../../../../slam-rs/configs/profiles/live.json"),
            Self::Live30 => include_str!("../../../../../slam-rs/configs/profiles/live30.json"),
        }
    }

    /// The profile's name (`fast`, `live`, `live30`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Live => "live",
            Self::Live30 => "live30",
        }
    }
}

impl std::str::FromStr for SlamProfile {
    type Err = SlamError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "fast" => Ok(Self::Fast),
            "live" => Ok(Self::Live),
            "live30" => Ok(Self::Live30),
            other => Err(SlamError::Config(format!("unknown SLAM profile {other} (fast, live, live30)"))),
        }
    }
}

/// One `KEY=VALUE` VIO configuration override (`robocap-live --slam-set`, `slam_bench --set`): the value as JSON, or as a
/// string when it is not JSON. The key is resolved when the overrides are applied ([`super::SlamEstimator::with_profile`]).
///
/// # Errors
///
/// [`SlamError::Config`] without an `=`.
pub fn parse_override(setting: &str) -> Result<(String, serde_json::Value), SlamError> {
    let (key, value) = setting.split_once('=').ok_or_else(|| SlamError::Config(format!("{setting}: expected KEY=VALUE")))?;
    let value = serde_json::from_str(value).unwrap_or_else(|_| serde_json::Value::String(value.to_string()));
    Ok((key.to_string(), value))
}

/// [`MSDMO_CONFIG`] with `profile`'s overlay and then `overrides` applied.
pub(super) fn profile_config(profile: SlamProfile, overrides: &[(String, serde_json::Value)]) -> Result<VioConfig, SlamError> {
    let mut overlay: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(profile.overlay()).map_err(|e| SlamError::Config(format!("profile {}: {e}", profile.as_str())))?;
    for (key, value) in overrides {
        let key = key.strip_prefix("config.").unwrap_or(key);
        let key = if key.starts_with("port.") { key.to_owned() } else { format!("config.{key}") };
        overlay.insert(key, value.clone());
    }
    VioConfig::with_overlay(MSDMO_CONFIG, &serde_json::Value::Object(overlay).to_string()).map_err(|e| SlamError::Config(e.to_string()))
}
