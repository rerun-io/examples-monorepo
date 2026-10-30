# Uncalibrated camera handling

Camera readers check required calibration fields before accessing them. Missing,
empty, wrong-sized or nonfinite intrinsics/extrinsics produce a `SKIP_CAMERA`
line naming the segment, camera path and fields. The shared reader feeds both
per-camera visibility/overlays and image cross-checks, so an uncalibrated camera
is omitted consistently. Missing detector tracks or empty windows are also
logged and omitted.

OAK RGB uses the calibrated rig pose with its identity camera-to-rig transform;
its pinhole, resolution and 14 lens coefficients must exist. Missing RGB
calibration/pose samples omit the OAK views rather than failing the body fit.
Each OAK mono camera checks its own extrinsics and lens calibration before
projection. Cross-checks without an OAK camera remain unscored, not invented.

On `selfcap__2025-12-14__1c094794__ep001`, `rig_02/cam_00` has pinhole data
but zero translation and rotation rows. Other exo cameras are calibrated. A
full local export from the runner's completed fit succeeded, logging that
camera and omitting it; exo rigs 00, 01, 03 and OAK rig 04 cameras 00–02 remain.
Artifact: `/tmp/soma-fit/skip-uncalibrated/selfcap__2025-12-14__1c094794__ep001.rrd`.
No catalog write or fit rerun was needed. Unit checks cover an intrinsics-only
flagged exo camera and OAK's explicit identity-extrinsic exception. Ruff and
changed-file type checks passed. No full gate.
