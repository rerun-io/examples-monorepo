"""Ultrawide rectification, pose resolution, raycast, and depth-storage geometry."""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import pyarrow as pa
from arkitscenes_download.ingest.distortion import AppleRadialPolynomial
from arkitscenes_download.ingest.paths import TIMELINE
from jaxtyping import Float32, Int64, UInt8, UInt16
from numpy import ndarray
from scipy.spatial.transform import Rotation

from gauss_surf.catalog import SegmentReader, TimedeltaNs, _single_instance, match_exact_timestamps, table_timestamps
from gauss_surf.contracts import RIG_QUATERNION_COLUMN, RIG_TRANSLATION_COLUMN, ULTRAWIDE_QUATERNION_COLUMN, ULTRAWIDE_TRANSLATION_COLUMN


@dataclass(frozen=True, slots=True)
class AppleUndistortion:
    """One reusable destination-to-source remap and its rectified pinhole."""

    K_rect_33: Float32[ndarray, "3 3"]
    """Rectified pinhole intrinsics shaped ``3 3``."""
    source_x_hw: Float32[ndarray, "h w"]
    """Distorted source-image x coordinate for each output pixel."""
    source_y_hw: Float32[ndarray, "h w"]
    """Distorted source-image y coordinate for each output pixel."""


@dataclass(frozen=True, slots=True)
class CameraPoses:
    """Wide and ultrawide world poses resolved on target timelines."""

    world_from_wide_n44: Float32[ndarray, "n_wide 4 4"]
    """Resolved wide-camera poses."""
    world_from_ultrawide_n44: Float32[ndarray, "n_uw 4 4"]
    """Resolved ultrawide-camera poses."""
    ultrawide_staleness_ms_n: Float32[ndarray, "n_uw"]
    """Age of each selected causal rig pose in milliseconds."""


@dataclass(frozen=True, slots=True)
class CameraPoseTrack:
    """One loaded rig-pose track and static ultrawide calibration."""

    timestamps_n: TimedeltaNs
    """Sorted rig-pose timestamps."""
    world_from_rig_n44: Float32[ndarray, "n_poses 4 4"]
    """RDF rig-to-world transforms at every pose timestamp."""
    wide_from_ultrawide_44: Float32[ndarray, "4 4"]
    """Static RDF ultrawide-to-wide transform."""


def world_from_pose(
    translation_3: Float32[ndarray, "3"], quaternion_xyzw_4: Float32[ndarray, "4"]
) -> Float32[ndarray, "4 4"]:
    """Build an RDF camera-to-world transform from Rerun pose components."""
    world_from_camera_44: Float32[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    world_from_camera_44[:3, :3] = Rotation.from_quat(quaternion_xyzw_4).as_matrix().astype(np.float32)
    world_from_camera_44[:3, 3] = translation_3
    return world_from_camera_44


def load_camera_pose_track(
    reader: SegmentReader,
    *,
    wide_from_ultrawide_44: Float32[ndarray, "4 4"] | None = None,
) -> CameraPoseTrack:
    """Fetch a rig-pose track and its static ultrawide calibration once."""
    pose_table: pa.Table = reader.pose_table(RIG_TRANSLATION_COLUMN, RIG_QUATERNION_COLUMN)
    pose_timestamps_n: TimedeltaNs = table_timestamps(pose_table)
    world_from_rig_n44: Float32[ndarray, "n_poses 4 4"] = np.stack(
        [
            world_from_pose(
                np.asarray(_single_instance(row[RIG_TRANSLATION_COLUMN], RIG_TRANSLATION_COLUMN), dtype=np.float32),
                np.asarray(_single_instance(row[RIG_QUATERNION_COLUMN], RIG_QUATERNION_COLUMN), dtype=np.float32),
            )
            for row in pose_table.to_pylist()
        ]
    )
    calibration_44: Float32[ndarray, "4 4"]
    if wide_from_ultrawide_44 is None:
        static_table: pa.Table = (
            reader.segment_view()
            .reader(index=TIMELINE, fill_latest_at=True)
            .select(TIMELINE, ULTRAWIDE_TRANSLATION_COLUMN, ULTRAWIDE_QUATERNION_COLUMN)
            .limit(1)
            .to_arrow_table()
        )
        if static_table.num_rows != 1:
            raise SystemExit("catalog segment has no static wide-from-ultrawide calibration")
        static_row: dict[str, Any] = static_table.to_pylist()[0]
        calibration_44 = world_from_pose(
            np.asarray(_single_instance(static_row[ULTRAWIDE_TRANSLATION_COLUMN], ULTRAWIDE_TRANSLATION_COLUMN), dtype=np.float32),
            np.asarray(_single_instance(static_row[ULTRAWIDE_QUATERNION_COLUMN], ULTRAWIDE_QUATERNION_COLUMN), dtype=np.float32),
        )
    else:
        calibration_44 = np.asarray(wide_from_ultrawide_44, dtype=np.float32)
    return CameraPoseTrack(pose_timestamps_n, world_from_rig_n44, calibration_44)


def build_apple_undistortion(
    K_uw_33: Float32[ndarray, "3 3"],
    coefficients_8: Float32[ndarray, "8"],
    distortion_center_reference_xy: Float32[ndarray, "2"],
    reference_dimensions_wh: tuple[int, int],
    image_wh: tuple[int, int],
) -> AppleUndistortion:
    """Build an Apple radial-polynomial remap at the video resolution.

    Args:
        K_uw_33: float32 distorted-camera intrinsics shaped ``3 3``.
        coefficients_8: float32 Apple percent-magnification coefficients shaped ``8``.
        distortion_center_reference_xy: float32 calibration center shaped ``2``.
        reference_dimensions_wh: Calibration reference width and height.
        image_wh: Output image width and height.

    Returns:
        Reusable rectified pinhole and destination-to-source remap.
    """
    reference_width: int = reference_dimensions_wh[0]
    reference_height: int = reference_dimensions_wh[1]
    width: int = image_wh[0]
    height: int = image_wh[1]
    if min(reference_width, reference_height, width, height) < 1:
        raise ValueError("Reference and image dimensions must be positive")
    K_input_33: Float32[ndarray, "3 3"] = np.asarray(K_uw_33, dtype=np.float32)
    coefficients: Float32[ndarray, "8"] = np.asarray(coefficients_8, dtype=np.float32).reshape(-1)
    pixel_x_hw: Float32[ndarray, "h w"]
    pixel_y_hw: Float32[ndarray, "h w"]
    pixel_x_hw, pixel_y_hw = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    if np.all(coefficients == 0.0):
        return AppleUndistortion(K_input_33.copy(), pixel_x_hw, pixel_y_hw)

    model: AppleRadialPolynomial = AppleRadialPolynomial(coefficients)
    center_reference_xy: Float32[ndarray, "2"] = np.asarray(distortion_center_reference_xy, dtype=np.float32)
    reference_per_image_xy: Float32[ndarray, "2"] = np.array([reference_width / width, reference_height / height], dtype=np.float32)
    center_image_xy: Float32[ndarray, "2"] = center_reference_xy / reference_per_image_xy
    corners_reference_42: Float32[ndarray, "4 2"] = np.array(
        [[0.0, 0.0], [float(reference_width), 0.0], [0.0, float(reference_height)], [float(reference_width), float(reference_height)]],
        dtype=np.float32,
    )
    radius_max_reference: float = float(np.max(np.linalg.norm(corners_reference_42 - center_reference_xy, axis=1)))
    output_delta_image_hw2: Float32[ndarray, "h w 2"] = np.stack(
        [pixel_x_hw - center_image_xy[0], pixel_y_hw - center_image_xy[1]],
        axis=-1,
    )
    def source_map(focal_scale: float) -> Float32[ndarray, "h w 2"]:
        """Map one uniform rectified focal scale into distorted source pixels."""
        rectified_delta_reference_hw2: Float32[ndarray, "h w 2"] = output_delta_image_hw2 * reference_per_image_xy / focal_scale
        rectified_radius_hw: Float32[ndarray, "h w"] = np.linalg.norm(rectified_delta_reference_hw2, axis=-1).astype(np.float32)
        rectified_normalized_hw: Float32[ndarray, "h w"] = rectified_radius_hw / radius_max_reference
        distorted_normalized_hw: Float32[ndarray, "h w"] = model.rectified_to_distorted_radius(rectified_normalized_hw)
        radial_scale_hw: Float32[ndarray, "h w"] = np.divide(
            distorted_normalized_hw,
            rectified_normalized_hw,
            out=np.ones_like(distorted_normalized_hw),
            where=rectified_normalized_hw > 0.0,
        )
        source_reference_hw2: Float32[ndarray, "h w 2"] = center_reference_xy + rectified_delta_reference_hw2 * radial_scale_hw[..., None]
        return (source_reference_hw2 / reference_per_image_xy).astype(np.float32, copy=False)

    def is_in_bounds(source_hw2: Float32[ndarray, "h w 2"]) -> bool:
        """Check whether every bilinear sample lies inside the source image."""
        return bool(
            np.all(source_hw2[..., 0] >= 0.0)
            and np.all(source_hw2[..., 0] <= width - 1)
            and np.all(source_hw2[..., 1] >= 0.0)
            and np.all(source_hw2[..., 1] <= height - 1)
        )

    lower_scale: float = 1.0
    upper_scale: float = 1.0
    source_image_hw2: Float32[ndarray, "h w 2"] = source_map(upper_scale)
    while not is_in_bounds(source_image_hw2):
        upper_scale *= 1.05
        if upper_scale > 4.0:
            raise ValueError("Apple radial calibration has no valid full-frame crop")
        source_image_hw2 = source_map(upper_scale)
    for _ in range(18):
        candidate_scale: float = 0.5 * (lower_scale + upper_scale)
        candidate_source_hw2: Float32[ndarray, "h w 2"] = source_map(candidate_scale)
        if is_in_bounds(candidate_source_hw2):
            upper_scale = candidate_scale
            source_image_hw2 = candidate_source_hw2
        else:
            lower_scale = candidate_scale
    K_rect_33: Float32[ndarray, "3 3"] = K_input_33.copy()
    K_rect_33[0, 0] *= upper_scale
    K_rect_33[1, 1] *= upper_scale
    K_rect_33[0, 2] = center_image_xy[0]
    K_rect_33[1, 2] = center_image_xy[1]
    return AppleUndistortion(
        K_rect_33,
        source_image_hw2[..., 0].astype(np.float32, copy=False),
        source_image_hw2[..., 1].astype(np.float32, copy=False),
    )


def undistort_rgb(
    rgb_hw3: UInt8[ndarray, "h w 3"],
    undistortion: AppleUndistortion,
) -> UInt8[ndarray, "h w 3"]:
    """Rectify one RGB frame with a reusable Apple destination-to-source remap.

    Args:
        rgb_hw3: uint8 distorted RGB image shaped ``h w 3``.
        undistortion: Apple remap built for the same image dimensions.

    Returns:
        uint8 rectified RGB image shaped ``h w 3``.
    """
    if rgb_hw3.ndim != 3 or rgb_hw3.shape[2] != 3:
        raise ValueError("RGB frame must have shape (H, W, 3)")
    if undistortion.source_x_hw.shape != rgb_hw3.shape[:2] or undistortion.source_y_hw.shape != rgb_hw3.shape[:2]:
        raise ValueError("Apple remap dimensions must match the RGB frame")
    rectified_hw3: UInt8[ndarray, "h w 3"] = cv2.remap(
        rgb_hw3,
        undistortion.source_x_hw,
        undistortion.source_y_hw,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified_hw3


def depth_meters_to_uint16_mm(depth_m_hw: Float32[ndarray, "h w"]) -> UInt16[ndarray, "h w"]:
    """Quantize metre z-depth to uint16 millimetres with a zero invalid sentinel.

    Positive depths above 65.535 m clamp to 65535 instead of wrapping. Nonfinite,
    zero, and negative inputs become the zero sentinel.

    Args:
        depth_m_hw: float32 metre depth shaped ``h w``.

    Returns:
        uint16 millimetre depth shaped ``h w``.
    """
    valid_hw: ndarray = np.isfinite(depth_m_hw) & (depth_m_hw > 0.0)
    safe_m_hw: Float32[ndarray, "h w"] = np.where(valid_hw, depth_m_hw, 0.0).astype(np.float32, copy=False)
    depth_mm_float_hw: Float32[ndarray, "h w"] = np.clip(safe_m_hw * 1000.0, 0.0, 65535.0)
    depth_mm_hw: UInt16[ndarray, "h w"] = np.rint(depth_mm_float_hw).astype(np.uint16)
    return depth_mm_hw

def compose_world_from_ultrawide(
    world_from_rig_n44: Float32[ndarray, "n 4 4"],
    wide_from_ultrawide_44: Float32[ndarray, "4 4"],
) -> Float32[ndarray, "n 4 4"]:
    """Compose temporal rig poses with the static ultrawide baseline.

    Args:
        world_from_rig_n44: float32 rig-to-world transforms shaped ``n 4 4``.
        wide_from_ultrawide_44: float32 ultrawide-to-wide transform shaped ``4 4``.

    Returns:
        float32 ultrawide-to-world transforms shaped ``n 4 4``.
    """
    if world_from_rig_n44.ndim != 3 or world_from_rig_n44.shape[1:] != (4, 4):
        raise ValueError("world_from_rig must have shape (N, 4, 4)")
    if wide_from_ultrawide_44.shape != (4, 4):
        raise ValueError("wide_from_ultrawide must have shape (4, 4)")
    world_from_ultrawide_n44: Float32[ndarray, "n 4 4"] = np.matmul(
        world_from_rig_n44,
        wide_from_ultrawide_44,
    ).astype(np.float32, copy=False)
    return world_from_ultrawide_n44


def pose_indices_at_or_before(pose_times_n: TimedeltaNs, target_times_n: TimedeltaNs) -> Int64[ndarray, "n_targets"]:
    """Find the nearest causal pose sample for every target timestamp.

    Args:
        pose_times_n: Sorted pose timestamps shaped ``n_poses``.
        target_times_n: Target timestamps shaped ``n_targets``.

    Returns:
        int64 source-pose indices shaped ``n_targets``. Every selected pose is
        at or before its target; v1 deliberately does not interpolate.

    Raises:
        ValueError: If inputs are invalid or a target precedes the first pose.
    """
    if pose_times_n.ndim != 1 or target_times_n.ndim != 1:
        raise ValueError("Pose and target timestamps must be one-dimensional")
    if len(pose_times_n) == 0:
        raise ValueError("Pose timeline must not be empty")
    pose_ns_n: TimedeltaNs = np.asarray(pose_times_n, dtype="timedelta64[ns]")
    target_ns_n: TimedeltaNs = np.asarray(target_times_n, dtype="timedelta64[ns]")
    if np.any(pose_ns_n[1:] < pose_ns_n[:-1]):
        raise ValueError("Pose timestamps must be sorted")
    indices_n: Int64[ndarray, "n_targets"] = (np.searchsorted(pose_ns_n, target_ns_n, side="right") - 1).astype(np.int64)
    if np.any(indices_n < 0):
        first_target: np.timedelta64 = target_ns_n[int(np.flatnonzero(indices_n < 0)[0])]
        raise ValueError(f"target timestamp {first_target} precedes the first rig pose")
    return indices_n


def resolve_camera_poses(
    track: CameraPoseTrack,
    wide_timestamps_n: TimedeltaNs,
    ultrawide_timestamps_n: TimedeltaNs,
    *,
    exact_wide: bool,
) -> CameraPoses:
    """Resolve exact or causal wide poses and causal ultrawide poses."""
    wide_pose_indices_n: Int64[ndarray, "n_wide"] = (
        match_exact_timestamps(track.timestamps_n, wide_timestamps_n)
        if exact_wide
        else pose_indices_at_or_before(track.timestamps_n, wide_timestamps_n)
    )
    uw_pose_indices_n: Int64[ndarray, "n_uw"] = pose_indices_at_or_before(track.timestamps_n, ultrawide_timestamps_n)
    staleness_ns_n: Int64[ndarray, "n_uw"] = (ultrawide_timestamps_n - track.timestamps_n[uw_pose_indices_n]).astype(
        "timedelta64[ns]"
    ).astype(np.int64)
    return CameraPoses(
        world_from_wide_n44=track.world_from_rig_n44[wide_pose_indices_n],
        world_from_ultrawide_n44=compose_world_from_ultrawide(
            track.world_from_rig_n44[uw_pose_indices_n],
            track.wide_from_ultrawide_44,
        ),
        ultrawide_staleness_ms_n=(staleness_ns_n.astype(np.float32) / 1_000_000.0).astype(np.float32),
    )


def raycast_z_depth(
    scene: o3d.t.geometry.RaycastingScene,
    K_rect_33: Float32[ndarray, "3 3"],
    world_from_camera_44: Float32[ndarray, "4 4"],
    image_wh: tuple[int, int],
) -> Float32[ndarray, "h w"]:
    """Raycast a triangle scene into camera-space pinhole z-depth.

    Open3D's pinhole rays use unnormalized camera directions ``(x, y, 1)``.
    Therefore ``t_hit`` is already z-depth for those rays. The explicit
    multiplication by the direction's camera-space z component below pins down
    that convention and remains correct if the ray construction changes.

    Args:
        scene: Open3D tensor raycasting scene containing the world-space mesh.
        K_rect_33: float32 rectified pinhole intrinsics shaped ``3 3``.
        world_from_camera_44: float32 camera-to-world transform shaped ``4 4``.
        image_wh: Output image width and height.

    Returns:
        float32 camera-space z-depth in metres shaped ``h w``. Misses are zero.
    """
    width: int = image_wh[0]
    height: int = image_wh[1]
    if width < 1 or height < 1:
        raise ValueError(f"Image dimensions must be positive, got {image_wh}")
    camera_from_world_44: Float32[ndarray, "4 4"] = np.linalg.inv(world_from_camera_44).astype(np.float32)
    rays_hw6: o3d.core.Tensor = scene.create_rays_pinhole(
        o3d.core.Tensor(K_rect_33.astype(np.float32, copy=False)),
        o3d.core.Tensor(camera_from_world_44),
        width,
        height,
    )
    raycast: dict[str, o3d.core.Tensor] = scene.cast_rays(rays_hw6)
    t_hit_hw: Float32[ndarray, "h w"] = raycast["t_hit"].numpy().astype(np.float32, copy=False)
    ray_directions_hw3: Float32[ndarray, "h w 3"] = rays_hw6.numpy()[..., 3:].astype(np.float32, copy=False)
    camera_directions_hw3: Float32[ndarray, "h w 3"] = np.einsum(
        "ij,hwj->hwi",
        camera_from_world_44[:3, :3],
        ray_directions_hw3,
    ).astype(np.float32, copy=False)
    depth_hw: Float32[ndarray, "h w"] = t_hit_hw * camera_directions_hw3[..., 2]
    depth_hw[~np.isfinite(depth_hw)] = 0.0
    depth_hw[depth_hw <= 0.0] = 0.0
    return depth_hw
