//! Stable markers shared by the live producer and panel parser.
/// Prefix of structured capture health in the scheduler line.
pub const CAPTURE_HEALTH: &str = " | capture_health ";
/// Signal-driven shutdown has begun.
pub const STOPPING: &str = "robocap-live: stopping";
/// Camera capture has ended.
pub const CAPTURE_STOPPED: &str = "robocap-live: capture stopped";

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn markers_match_the_shared_log_contract() {
        let line = "robocap-live: stopping; robocap-live: capture stopped | capture_health {}";
        assert!(line.starts_with(STOPPING));
        assert!(line.contains(CAPTURE_STOPPED));
        assert_eq!(line.split_once(CAPTURE_HEALTH).unwrap().1, "{}");
    }
}
