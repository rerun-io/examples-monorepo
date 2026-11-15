"""Metric rig-to-rig relative pose for a loop candidate — the measurement, not the retrieval.

`colsfm.loop_closure` decides *which* rig frames revisit each other; this module decides
*where* one is relative to the other, in metres. It is the piece the loop-closure docstring
measured as the whole remaining problem on RoboCap: retrieval and pair selection there are
right to 39.9 mm with oracle poses, and the shipped two-view-plus-prior-scale measurement
turns that into 609.0 mm — worse than no loops at all.

The design: a local metric map, then generalized absolute pose
---------------------------------------------------------------

For a candidate rig pair `(A, B)`:

1. **Local map.** Triangulate 3-D points expressed in rig frame `A`, from the images of `A`
   itself plus its temporal neighbours `A±1..A±span`. Every ray is placed by the rig
   extrinsics and by the *local* odometry between neighbouring rig frames, which is
   accurate — the drift a loop closure removes accumulates over a session, not over 100 ms.
2. **Generalized PnP.** Feed the 2-D observations those points get in every camera of `B`
   to `pycolmap.estimate_and_refine_generalized_absolute_pose`, which solves for one rig
   pose over all cameras at once and then refines it on every inlier observation. Its
   `rig_from_world` is `B_T_A`, so the edge is its inverse.

Nothing in step 2 reads the prior relation between `A` and `B`, so the edge can contradict
the odometry about distance — which is exactly what a loop closure is for, and exactly what
scaling a two-view direction by the prior baseline cannot do.

`RigPoseEstimator` owns all of it
---------------------------------

The rig frames, the rig geometry, the keypoints, the matcher and the settings are fixed for
a whole run and are read by every step, so they are the estimator's state rather than five
arguments threaded through four functions. The object also owns the two memos that make the
stage affordable: unprojected bearings per image (one camera-model call per keypoint, which
a rig frame with twenty loop candidates would otherwise repeat twenty times) and one local
map per source rig frame, reused across all of that frame's candidates by `measure_group`.

`measure_group` is also the unit of parallelism: two source rig frames share nothing but the
read-only bearings memo, and `pycolmap.estimate_and_refine_generalized_absolute_pose`
releases the GIL, so `colsfm.loop_closure` runs one group per worker thread.

Why not the generalized essential matrix
----------------------------------------

`pycolmap.estimate_generalized_relative_pose` solves the same rig pair in one call, and
with non-zero rig baselines its scale is observable in principle. Measured on a synthetic
copy of RoboCap's rig (4 cameras, a 0.24 m span, a 0.36 m motion, a scene 10 m deep):

| keypoint noise | rotation error | recovered `||t||` (truth 0.359 m) |
|---|---|---|
| 0.0 px | 0.000 deg | 0.359 m |
| 0.2 px | 0.171 deg | **0.168 m** |
| 0.7 px | 0.084 deg | **0.217 m** |

The rotation survives; the scale does not. Metric scale in a generalized essential matrix
comes only from the rig baseline, so it is read off a 0.24 m lever arm against a scene tens
of metres deep, and half a pixel of noise halves it. Triangulating first spends the same
baseline on depth, where a redundant, refined least-squares problem averages the noise away
instead of amplifying it. The generalized essential matrix is still used, but only for its
rotation: `RigPoseConfig.max_direction_disagreement_deg` cross-checks the PnP translation
direction against it and drops a pair the two disagree about. It costs no extra matching —
it is solved from the same-camera correspondences the observation pass already read.

Depth, and why the neighbours matter
------------------------------------

RoboCap's declared stereo pair is 86 mm. A rig hop is 114 mm, so `A±1` alone more than
triples the available baseline and points it along a different axis, and the two side
cameras contribute a third view direction that the front pair never sees. Measured on the
same 488 RoboCap loop rig pairs, scored against the blob's PGO after rigid alignment
(`colsfm.loop_closure`'s docstring has the whole table):

| local map | edges | translation error vs the oracle, median | RMSE |
|---|---|---|---|
| the stereo pair alone (`neighbour_span=0`) | 189 | 197 mm | 457.0 mm |
| **plus `A±1` (the default)** | 425 | **219 mm** | **408.1 mm** |
| plus `A±2` | 448 | 231 mm | 418.5 mm |

`neighbour_span=0` measures fewer pairs because an 86 mm baseline against a scene metres
deep leaves too few landmarks to resection from: it loses 294 of the 488 pairs to the
inlier gate where the default loses 58. A second step of neighbours buys nothing and costs
one more image pair per camera in the matching stage.

What the estimator is and is not good at, measured on the blob's own 90 loop rig pairs:
its translations move decisively off the odometry and towards the blob's own measurement
(median `||t||` 132 mm against the odometry's 355 mm and the blob's 160 mm), while its
rotations agree with the odometry to 1.5 deg and with
`pycolmap.estimate_generalized_relative_pose` — an estimator that shares only the
correspondences — to **0.35 deg**. See `colsfm.loop_closure` for what that costs against a
reference whose own rotations sit 6.5 deg away.

Cross-stereo observations are unconditional
-------------------------------------------

The source rig's anchor image is matched against *both* cameras of the target's declared
stereo pair. The two cameras see the same scene 86 mm apart, so the cross pair is a real
extra observation rather than a duplicate view, and every measured configuration used it;
the switch that used to guard it was never set to False by any caller or test, so it is
gone rather than untested.

Reading a `match_fn` the caller has not matched yet
---------------------------------------------------

`colsfm.pipeline` discovers which image pairs to match by running the whole stage once with
a matcher that returns nothing, batch-matching what was asked for, then running it again
for real. So this module must ask for **every** pair it could use *before* it gates on
anything: a pair requested only after a successful gate would never be matched in
production. `test_every_needed_image_pair_is_requested_even_when_nothing_matches` pins that.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

import numpy as np
import pycolmap
from jaxtyping import Bool, Float64, Int
from numpy import ndarray

from colsfm.loop_pairs import local_map_partners
from colsfm.rig_geometry import RigFrameIndex, RigGeometry

Matches: TypeAlias = Int[ndarray, "n_matches 2"]
"""Raw feature matches: keypoint index in the first image, keypoint index in the second."""

MatchFunction: TypeAlias = Callable[[int, int], Matches]
"""`match_fn(image_id_a, image_id_b) -> matches`, injected by the caller.

Column 0 indexes the keypoints of `image_id_a`, column 1 those of `image_id_b`, which is
what both `pycolmap.Database.read_matches` and the synthetic matchers in the tests give.
"""

Keypoints: TypeAlias = Float64[ndarray, "n_keypoints 2"]
"""Keypoint pixel coordinates, `x` then `y`, in the original image."""

Bearings: TypeAlias = Float64[ndarray, "n_keypoints 3"]
"""Unit-norm bearing vectors, one per keypoint, in the camera's optical frame."""

RigPoseRejectionReason: TypeAlias = Literal["no_observations", "no_pose", "too_few_inliers", "direction_disagreement"]
"""Which gate turned a rig pair down.

The loop-closure stage reports these verbatim in its diagnostics, because a stage that
returns no edge is only explainable if it says which gate emptied it."""


@dataclass(frozen=True, slots=True)
class RigPoseConfig:
    """Gates and thresholds of the metric rig-to-rig estimator."""

    neighbour_span: int = 1
    """Temporal neighbours of the source rig frame that contribute rays to the local map.

    0 triangulates from the rig's own cameras only, which on RoboCap means an 86 mm stereo
    baseline against a scene metres deep. 1 adds `A±1`, i.e. two more views 114 mm away
    along the drive direction, and is the default. Each extra step costs one more image
    pair per camera in the matching stage."""
    min_depth_meters: float = 0.4
    """Landmarks nearer than this to the anchor camera are triangulation failures, not
    scene structure."""
    max_depth_meters: float = 60.0
    """Landmarks beyond this carry no usable depth at a 0.2 m baseline; keeping them only
    lets a near-parallel ray pair dominate the point count."""
    min_triangulation_angle_deg: float = 0.5
    """Smallest angle between the rays of a landmark. COLMAP's own mapper uses 1.5 deg for
    initialisation; 0.5 deg is looser because the rig's baseline is short by construction
    and the generalized PnP re-weights a distant point by its reprojection error anyway."""
    max_landmark_reprojection_px: float = 4.0
    """A landmark must reproject into every contributing view within this, which is the
    matching stage's own RANSAC threshold."""
    ransac_max_error_px: float = 8.0
    """RANSAC inlier threshold of the generalized PnP, in pixels. Twice the matcher's 4.0 px
    because a 2-D-3-D residual carries the landmark's depth error as well as the keypoint's."""
    ransac_confidence: float = 0.9999
    """RANSAC confidence; pycolmap's own `RANSACOptions` default."""
    ransac_max_trials: int = 10_000
    """Trial cap for both RANSAC loops.

    A bare `pycolmap.RANSACOptions()` allows **100 000**, and the generalized essential
    matrix's minimal solver is expensive enough that a degenerate pair then spends tens of
    seconds proving it cannot fit — measured, it is what a first run of this stage on
    RoboCap spends most of its time on. 10 000 is what COLMAP's own
    `TwoViewGeometryOptions` uses."""
    ransac_min_trials: int = 100
    """Trials RANSAC runs before its confidence test may stop it; COLMAP's own
    `TwoViewGeometryOptions` value."""
    min_inliers: int = 30
    """Fewest correspondences a solved pose may rest on, and the loop stage's own
    acceptance floor.

    This is the blob's `min_matches_num` (`docs/spec/generate_association_main.md` §6.6)
    and it is declared here, once: `colsfm.loop_closure.LoopClosureConfig.min_inliers`
    reads it back, so the estimator's solvability floor and the stage's acceptance rule
    cannot drift apart. The ratio half of the blob's rule has no copy here at all — it
    lives in `colsfm.loop_closure.is_good_match`, applied to the counts this estimator
    reports."""
    max_direction_disagreement_deg: float | None = 45.0
    """How far the refined translation direction may sit from the generalized essential
    matrix's before the pair is dropped, or None to skip the cross-check.

    The two estimates share their correspondences but nothing else — one triangulates and
    resections, the other solves an epipolar constraint — so agreement is real evidence and
    disagreement means one of them latched onto a wrong RANSAC support set. 45 deg is loose
    on purpose: on a 0.2 m revisit baseline the direction is the weakest quantity either
    estimator produces (`colsfm.loop_closure` measured the prior and the blob's own result
    44 deg apart on the same pairs), so a tight gate would reject true loops.

    Measured on RoboCap it costs 66 of 921 pairs and buys 10 mm of trajectory RMSE
    (418.4 mm without it, 408.1 mm with it), so it is on by default."""


BEARING_CACHE_IMAGES: Final[int] = 512
"""Images whose unprojected bearings one estimator keeps resident.

The same number `colsfm.matching_trt.FEATURE_CACHE_IMAGES` uses, and for the same
reason: a measurement reads a source rig frame, its temporal neighbours and one
target rig frame, so a few hundred entries hold every group's working set while a
whole RoboCap sequence's would be 4528 float64 arrays of tens of thousands of rows."""

DEFAULT_RIG_POSE_CONFIG: RigPoseConfig = RigPoseConfig()
"""Shared immutable default, so the signatures below hold no constructor call."""


@dataclass(frozen=True, slots=True)
class AnchorLandmarks:
    """One anchor image's share of a local map, packed for the join in `_observations`.

    The two arrays are parallel and `keypoint_indices` is ascending, so a match list joins
    against them with one `searchsorted` instead of a dict lookup per match. They are packed
    once at triangulation, because a source rig frame with twenty loop candidates would
    otherwise re-sort and re-stack the same points twenty times.
    """

    keypoint_indices: Int[ndarray, "n_landmarks"]
    """Ascending anchor keypoint index per landmark."""
    points_in_rig: Float64[ndarray, "n_landmarks 3"]
    """The landmark positions in the source rig frame, in metres, in the same order."""

    def __len__(self) -> int:
        """How many landmarks this anchor contributed.

        Returns:
            The landmark count.
        """
        return len(self.keypoint_indices)


RigLandmarks: TypeAlias = dict[int, AnchorLandmarks]
"""Local map: 3-D points in the rig frame, keyed by the anchor image that owns them."""


@dataclass(frozen=True, slots=True)
class RigPoseEstimate:
    """One measured rig-to-rig relative pose and the evidence behind it."""

    source_T_target: pycolmap.Rigid3d
    """Metric relative pose: where the target rig frame sits in the source's own frame."""
    num_inliers: int
    """2-D-3-D observations the generalized PnP kept."""
    num_observations: int
    """2-D-3-D observations it was offered."""
    num_landmarks: int
    """Landmarks in the source rig's local map, over every anchor image."""
    num_cameras: int
    """Cameras of the target rig frame that contributed an observation."""
    direction_disagreement_deg: float
    """Angle between this translation direction and the generalized essential matrix's, or
    `nan` when that estimate was unavailable (too few correspondences, a degenerate
    configuration pycolmap answered with a panoramic model, or the cross-check switched
    off)."""


@dataclass(frozen=True, slots=True)
class RigPoseRejection:
    """The gate that refused a rig pair.

    Returned instead of a bare None because the loop-closure stage's diagnostics have to say
    *why* a pair was dropped — "no edges" and "no edges because every pair was too far from
    its epipolar direction" are different bugs.
    """

    reason: RigPoseRejectionReason
    """Which gate turned the pair down."""


RigPoseOutcome: TypeAlias = RigPoseEstimate | RigPoseRejection
"""What one rig pair produced: a measurement, or the gate that refused it."""


def _ransac_options(config: RigPoseConfig) -> pycolmap.RANSACOptions:
    """RANSAC settings shared by both generalized estimators.

    Args:
        config: The estimator settings.

    Returns:
        The options, with the trial count capped; see `RigPoseConfig.ransac_max_trials`.
    """
    options: pycolmap.RANSACOptions = pycolmap.RANSACOptions()
    options.max_error = config.ransac_max_error_px
    options.confidence = config.ransac_confidence
    options.min_num_trials = config.ransac_min_trials
    options.max_num_trials = config.ransac_max_trials
    return options


def bearings_of(camera: pycolmap.Camera, keypoints: Keypoints) -> Bearings:
    """Turn keypoint pixels into unit bearing vectors in the camera's optical frame.

    Args:
        camera: The calibrated camera, whose model undoes the distortion.
        keypoints: Float64 keypoint pixels with shape `[n_keypoints, 2]`.

    Returns:
        Float64 unit bearings with shape `[n_keypoints, 3]`.
    """
    normalised: Float64[ndarray, "n_keypoints 2"] = np.asarray(camera.cam_from_img(np.ascontiguousarray(keypoints, dtype=np.float64)), dtype=np.float64)
    homogeneous: Bearings = np.concatenate([normalised, np.ones((len(normalised), 1), dtype=np.float64)], axis=1)
    return homogeneous / np.linalg.norm(homogeneous, axis=1, keepdims=True)


def direction_angle_deg(first: pycolmap.Rigid3d, second: pycolmap.Rigid3d) -> float:
    """Angle between the translation directions of two poses.

    Args:
        first: The first pose.
        second: The second pose.

    Returns:
        The angle in degrees, or `nan` when either translation is degenerate.
    """
    a: Float64[ndarray, "3"] = np.asarray(first.translation, dtype=np.float64)
    b: Float64[ndarray, "3"] = np.asarray(second.translation, dtype=np.float64)
    norms: float = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norms <= 1e-12:
        return float("nan")
    return float(np.rad2deg(np.arccos(np.clip(float(a @ b) / norms, -1.0, 1.0))))


def _triangulate_batch(
    projections: Float64[ndarray, "n_views 3 4"],
    bearings: Float64[ndarray, "n_points n_views 3"],
    focal_lengths: Float64[ndarray, "n_views"],
    config: RigPoseConfig,
) -> tuple[Float64[ndarray, "n_points 3"], Bool[ndarray, "n_points"]]:
    """Triangulate a batch of landmarks seen in the same views, then gate them.

    The linear system is COLMAP's own: for a bearing `b` seen through the 3x4 projection `P`,
    `b x (P X) = 0` gives two independent rows, and the point is the right singular vector
    of the stacked rows with the smallest singular value. Doing it batched is what makes the
    stage affordable — a rig frame anchors a few thousand landmarks and a per-landmark
    `pycolmap.triangulate_multi_view_point` call costs about a second per rig frame.

    Args:
        projections: `cam_T_rig` as a 3x4 matrix per view; view 0 is the anchor camera.
        bearings: Unit bearing per landmark per view, `[n_points, n_views, 3]`.
        focal_lengths: Mean focal length in pixels per view, which turns the angular
            residual of a bearing back into the pixels `max_landmark_reprojection_px` is
            expressed in.
        config: Gates and thresholds.

    Returns:
        The points in the rig frame, `[n_points, 3]`, and a mask of which passed the depth,
        triangulation-angle and reprojection gates.
    """
    rotations: Float64[ndarray, "n_views 3 3"] = projections[:, :, :3]
    translations: Float64[ndarray, "n_views 3"] = projections[:, :, 3]
    rows: Float64[ndarray, "n_points n_rows 4"] = np.concatenate(
        [
            bearings[:, :, 0:1] * projections[None, :, 2, :] - bearings[:, :, 2:3] * projections[None, :, 0, :],
            bearings[:, :, 1:2] * projections[None, :, 2, :] - bearings[:, :, 2:3] * projections[None, :, 1, :],
        ],
        axis=1,
    )
    # A fisheye keypoint outside its model's valid domain unprojects to `nan`, and LAPACK's
    # SVD raises rather than returning one — measured on RoboCap, where a handful of
    # keypoints per image sit past the OPENCV_FISHEYE undistortion's convergence radius. The
    # offending rows are dropped before the solve rather than after it.
    solvable: Bool[ndarray, "n_points"] = np.isfinite(rows).all(axis=(1, 2))
    points: Float64[ndarray, "n_points 3"] = np.zeros((len(bearings), 3), dtype=np.float64)
    keep: Bool[ndarray, "n_points"] = np.zeros(len(bearings), dtype=bool)
    if not solvable.any():
        return points, keep
    null_space: Float64[ndarray, "n_solvable 4"] = np.linalg.svd(rows[solvable])[2][:, -1, :]
    homogeneous: Float64[ndarray, "n_solvable"] = null_space[:, 3]
    usable: Bool[ndarray, "n_solvable"] = np.abs(homogeneous) > 1e-12
    points[solvable] = null_space[:, :3] / np.where(usable, homogeneous, 1.0)[:, None]
    keep[solvable] = usable
    keep &= np.isfinite(points).all(axis=1)

    in_views: Float64[ndarray, "n_points n_views 3"] = np.einsum("vij,nj->nvi", rotations, points) + translations[None, :, :]
    depths: Float64[ndarray, "n_points n_views"] = in_views[:, :, 2]
    keep &= (depths[:, 0] > config.min_depth_meters) & (depths[:, 0] < config.max_depth_meters) & (depths.min(axis=1) > 0.0)

    directions: Float64[ndarray, "n_points n_views 3"] = in_views / np.maximum(np.linalg.norm(in_views, axis=2, keepdims=True), 1e-12)
    residuals_px: Float64[ndarray, "n_points n_views"] = np.linalg.norm(directions - bearings, axis=2) * focal_lengths[None, :]
    keep &= residuals_px.max(axis=1) <= config.max_landmark_reprojection_px

    # The widest angle any two views see the point from, computed in the rig frame: a ray
    # direction expressed in its own camera's frame is not comparable with another camera's.
    centres: Float64[ndarray, "n_views 3"] = -np.einsum("vji,vj->vi", rotations, translations)
    from_centres: Float64[ndarray, "n_points n_views 3"] = points[:, None, :] - centres[None, :, :]
    from_centres = from_centres / np.maximum(np.linalg.norm(from_centres, axis=2, keepdims=True), 1e-12)
    cosines: Float64[ndarray, "n_points n_views n_views"] = np.einsum("nvi,nwi->nvw", from_centres, from_centres)
    widest_angle_deg: Float64[ndarray, "n_points"] = np.rad2deg(np.arccos(np.clip(cosines.min(axis=(1, 2)), -1.0, 1.0)))
    keep &= widest_angle_deg >= config.min_triangulation_angle_deg
    return points, keep


@dataclass(frozen=True, slots=True)
class _EpipolarCorrespondences:
    """Same-camera matches of one rig pair, kept for the generalized essential matrix.

    The observation pass already asks the matcher for every one of these pairs, so the
    direction cross-check reads them off here instead of re-matching and re-joining.
    """

    source_pixels: list[Keypoints]
    """One block of source-image pixels per camera that matched."""
    target_pixels: list[Keypoints]
    """The matching target-image pixels, block for block."""
    camera_indices: list[int]
    """`camera_idx` per correspondence, flat across the blocks."""


@dataclass(frozen=True, slots=True)
class _Observations:
    """The 2-D-3-D correspondences one rig pair offers the generalized resection."""

    points2d: Float64[ndarray, "n_observations 2"]
    """Pixels in the target rig's images."""
    points3d: Float64[ndarray, "n_observations 3"]
    """The matching landmarks, in the source rig's frame."""
    camera_indices: list[int]
    """`camera_idx` of the target camera each observation was seen in."""
    cameras: set[int]
    """Which cameras of the target rig frame contributed."""
    epipolar: _EpipolarCorrespondences
    """The same-camera matches, for the translation-direction cross-check."""


@dataclass(slots=True)
class RigPoseEstimator:
    """Measures rig-to-rig relative poses over one fixed rig, sequence and matcher.

    Build one per run and call `measure_group` once per source rig frame. The estimator is
    safe to share between threads: the only mutated state is the bearing memo, whose entries
    are idempotent (two threads racing on the same image recompute the same array) and whose
    eviction can only cost a recomputation.
    """

    index: RigFrameIndex
    """Rig frames, their prior poses and their images."""
    geometry: RigGeometry
    """The rig's cameras and extrinsics."""
    keypoints: Mapping[int, Keypoints]
    """Keypoint pixels per image id."""
    match_fn: MatchFunction
    """The injected matcher."""
    config: RigPoseConfig = DEFAULT_RIG_POSE_CONFIG
    """Gates and thresholds."""
    _camera_of_image: dict[int, int] = field(init=False, repr=False)
    """`camera_params_id` per image id, inverted from the rig-frame index."""
    _rig_of_image: dict[int, int] = field(init=False, repr=False)
    """`synced_sample_id` per image id, inverted from the rig-frame index."""
    _bearings: OrderedDict[int, Bearings] = field(init=False, repr=False, default_factory=OrderedDict)
    """Memoised unit bearings per image id, most recently used last.

    An LRU of `BEARING_CACHE_IMAGES`, as `colsfm.matching_trt.FeatureCache` is over
    keypoints and descriptors: the memo used to be an unbounded dict, so a RoboCap run
    ended it holding a `[num_keypoints, 3]` float64 array for every one of 4528 images
    at once, for a measurement that only ever reads a few rig frames' worth."""

    def __post_init__(self) -> None:
        """Invert the rig-frame index into the per-image lookups every step needs."""
        self._camera_of_image = {image_id: camera_id for (_, camera_id), image_id in self.index.keyframe_by_rig_camera.items()}
        self._rig_of_image = {image_id: rig_id for (rig_id, _), image_id in self.index.keyframe_by_rig_camera.items()}

    def camera_of(self, image_id: int) -> pycolmap.Camera:
        """The calibrated camera that took one image.

        Args:
            image_id: The image to look up.

        Returns:
            Its `pycolmap.Camera`.

        Raises:
            KeyError: When the image is not part of the indexed rig frames.
        """
        return self.geometry.cameras[self.geometry.position_of(self._camera_of_image[image_id])]

    def bearings(self, image_id: int) -> Bearings:
        """Unit bearings of every keypoint of one image.

        Args:
            image_id: The image to unproject.

        Returns:
            Float64 unit bearings with shape `[n_keypoints, 3]`.
        """
        cached: Bearings | None = self._bearings.get(image_id)
        if cached is not None:
            self._bearings.move_to_end(image_id)
            return cached
        computed: Bearings = bearings_of(self.camera_of(image_id), self.keypoints[image_id])
        self._bearings[image_id] = computed
        if len(self._bearings) > BEARING_CACHE_IMAGES:
            self._bearings.popitem(last=False)
        return computed

    def focal_length_px(self, image_id: int) -> float:
        """Mean focal length in pixels of the camera that took one image.

        Args:
            image_id: The image to look up.

        Returns:
            The camera's mean focal length in pixels; RoboCap's fisheyes sit near 630,
            Galileo's pinholes near 845.
        """
        return float(self.camera_of(image_id).mean_focal_length())

    def local_map(self, rig_id: int) -> RigLandmarks:
        """Triangulate a metric point cloud in one rig frame's own vehicle frame.

        Every camera of the rig frame anchors its own landmarks: a keypoint of the anchor
        image becomes a 3-D point once at least one partner image — the declared stereo
        partner, or the same camera in a temporal neighbour — matches it. Poses come from the
        rig extrinsics and from the *local* odometry between neighbouring rig frames, never
        from any relation to a far-away rig frame.

        Args:
            rig_id: The rig frame to build the map around.

        Returns:
            The landmarks per anchor image id. Every camera of the rig frame that has
            keypoints gets an entry, possibly empty.
        """
        landmarks: RigLandmarks = {}
        for camera_id in self.geometry.camera_ids:
            anchor: int | None = self.index.keyframe(rig_id, camera_id)
            if anchor is None or anchor not in self.keypoints:
                continue
            landmarks[anchor] = self._anchor_landmarks(rig_id, camera_id, anchor)
        return landmarks

    def measure(self, source_rig_id: int, target_rig_id: int, landmarks: RigLandmarks | None = None) -> RigPoseOutcome:
        """Measure where one rig frame sits relative to another, in metres.

        Builds the source rig frame's local metric map (or reuses the one given), matches
        every anchor image against the target rig frame's cameras, and resections the target
        rig as a whole with `pycolmap.estimate_and_refine_generalized_absolute_pose`. No
        relation between the two rig frames is read from the priors, so the result can
        contradict them.

        Args:
            source_rig_id: Rig frame the edge starts at.
            target_rig_id: Rig frame the edge ends at.
            landmarks: A local map for `source_rig_id` built earlier, or None to build one
                here. `measure_group` is the usual way to share one across candidates.

        Returns:
            The metric relative pose and its evidence, or the gate that rejected the pair.
        """
        local_map: RigLandmarks = self.local_map(source_rig_id) if landmarks is None else landmarks
        observations: _Observations = self._observations(source_rig_id, target_rig_id, local_map)
        num_landmarks: int = sum(len(anchor) for anchor in local_map.values())
        num_observations: int = len(observations.points3d)
        if num_observations < self.config.min_inliers:
            return RigPoseRejection(reason="no_observations")

        answer: dict | None = pycolmap.estimate_and_refine_generalized_absolute_pose(
            observations.points2d,
            observations.points3d,
            observations.camera_indices,
            list(self.geometry.cams_from_rig),
            list(self.geometry.cameras),
            _ransac_options(self.config),
            pycolmap.AbsolutePoseRefinementOptions(),
        )
        if answer is None:
            return RigPoseRejection(reason="no_pose")
        num_inliers: int = int(answer["num_inliers"])
        if num_inliers < self.config.min_inliers:
            return RigPoseRejection(reason="too_few_inliers")
        source_T_target: pycolmap.Rigid3d = answer["rig_from_world"].inverse()

        limit_deg: float | None = self.config.max_direction_disagreement_deg
        epipolar: pycolmap.Rigid3d | None = None if limit_deg is None else self._epipolar_pose(observations.epipolar)
        disagreement_deg: float = float("nan") if epipolar is None else direction_angle_deg(source_T_target, epipolar)
        if limit_deg is not None and not np.isnan(disagreement_deg) and disagreement_deg > limit_deg:
            return RigPoseRejection(reason="direction_disagreement")

        return RigPoseEstimate(
            source_T_target=source_T_target,
            num_inliers=num_inliers,
            num_observations=num_observations,
            num_landmarks=num_landmarks,
            num_cameras=len(observations.cameras),
            direction_disagreement_deg=disagreement_deg,
        )

    def measure_group(self, source_rig_id: int, target_rig_ids: tuple[int, ...]) -> list[RigPoseOutcome]:
        """Measure every candidate of one source rig frame against one local map.

        Triangulating the local map is the expensive half of a measurement, and it depends
        only on the source rig frame, so a source with twenty candidates builds it once.
        This is also the unit `colsfm.loop_closure` hands to a worker thread.

        Args:
            source_rig_id: Rig frame the edges start at.
            target_rig_ids: Rig frames to measure against it, in the caller's own order.

        Returns:
            One outcome per entry of `target_rig_ids`, in that order.
        """
        local_map: RigLandmarks = self.local_map(source_rig_id)
        return [self.measure(source_rig_id, target_rig_id, local_map) for target_rig_id in target_rig_ids]

    # ----------------------------------------------------------------------------------
    # the local map
    # ----------------------------------------------------------------------------------

    def _local_map_partners(self, rig_id: int, camera_id: int) -> list[int]:
        """Images whose rays join one anchor image's in the local map.

        Args:
            rig_id: The source rig frame.
            camera_id: The anchor image's camera.

        Returns:
            Image ids of the stereo partner inside the rig frame and of the same camera in
            every temporal neighbour, in a stable order.
        """
        return local_map_partners(self.index, self.geometry, self.config.neighbour_span, rig_id, camera_id)

    def _anchor_tracks(self, rig_id: int, camera_id: int, anchor: int) -> tuple[dict[int, list[tuple[int, int]]], dict[int, Float64[ndarray, "3 4"]]]:
        """Match one anchor image against its partners and collect its feature tracks.

        Args:
            rig_id: The source rig frame.
            camera_id: The anchor image's camera.
            anchor: The anchor image id.

        Returns:
            The observations of each anchor keypoint as `(partner image id, keypoint index)`
            pairs, and the `cam_T_rig` 3x4 projection of every partner that matched.
        """
        world_T_rig: pycolmap.Rigid3d = self.index.world_T_rig[rig_id]
        tracks: dict[int, list[tuple[int, int]]] = {}
        projections: dict[int, Float64[ndarray, "3 4"]] = {}
        for partner in self._local_map_partners(rig_id, camera_id):
            partner_matches: Matches = np.asarray(self.match_fn(anchor, partner), dtype=np.int64).reshape(-1, 2)
            if partner not in self.keypoints or len(partner_matches) == 0:
                continue
            partner_camera: int = self._camera_of_image[partner]
            rig_T_partner_rig: pycolmap.Rigid3d = world_T_rig.inverse() * self.index.world_T_rig[self._rig_of_image[partner]]
            projections[partner] = np.asarray(
                (self.geometry.cams_from_rig[self.geometry.position_of(partner_camera)] * rig_T_partner_rig.inverse()).matrix(), dtype=np.float64
            )
            # `.tolist()` first: iterating a `[n, 2]` int64 array row by row builds two numpy
            # scalars per match, which is 1.1 ms per 1500 matches and ~20 s over a RoboCap run.
            for anchor_index, partner_index in partner_matches.tolist():
                tracks.setdefault(anchor_index, []).append((partner, partner_index))
        return tracks, projections

    def _anchor_landmarks(self, rig_id: int, camera_id: int, anchor: int) -> AnchorLandmarks:
        """Triangulate the landmarks one anchor image contributes to the local map.

        Args:
            rig_id: The source rig frame.
            camera_id: The anchor image's camera.
            anchor: The anchor image id.

        Returns:
            The surviving landmarks, packed and sorted by anchor keypoint index.
        """
        tracks, projections = self._anchor_tracks(rig_id, camera_id, anchor)
        anchor_from_rig: pycolmap.Rigid3d = self.geometry.cams_from_rig[self.geometry.position_of(camera_id)]
        anchor_projection: Float64[ndarray, "3 4"] = np.asarray(anchor_from_rig.matrix(), dtype=np.float64)
        anchor_bearings: Bearings = self.bearings(anchor)
        anchor_focal: float = self.focal_length_px(anchor)

        # Group the tracks by which images they were seen in: every track in a group shares
        # one stack of projection matrices, so the whole group triangulates in one batched
        # least-squares call instead of one pybind call per landmark. A four-camera rig with
        # two neighbours has at most a handful of distinct view sets, so the grouping is
        # cheap and the batches are large.
        grouped: dict[tuple[int, ...], list[int]] = {}
        for anchor_index, observations in tracks.items():
            grouped.setdefault(tuple(image_id for image_id, _ in observations), []).append(anchor_index)

        kept_indices: list[int] = []
        kept_points: list[Float64[ndarray, "3"]] = []
        for view_set, anchor_indices in grouped.items():
            stacked_projections: Float64[ndarray, "n_views 3 4"] = np.stack([anchor_projection, *(projections[image_id] for image_id in view_set)])
            view_focals: Float64[ndarray, "n_views"] = np.array(
                [anchor_focal, *(self.focal_length_px(image_id) for image_id in view_set)], dtype=np.float64
            )
            observed_indices: dict[int, dict[int, int]] = {}
            for anchor_index in anchor_indices:
                for image_id, index_in_partner in tracks[anchor_index]:
                    observed_indices.setdefault(image_id, {})[anchor_index] = index_in_partner
            batch_bearings: Float64[ndarray, "n_points n_views 3"] = np.stack(
                [
                    anchor_bearings[anchor_indices],
                    *(
                        self.bearings(image_id)[[observed_indices[image_id][anchor_index] for anchor_index in anchor_indices]]
                        for image_id in view_set
                    ),
                ],
                axis=1,
            )
            triangulated, keep = _triangulate_batch(stacked_projections, batch_bearings, view_focals, self.config)
            for position, anchor_index in enumerate(anchor_indices):
                if keep[position]:
                    kept_indices.append(anchor_index)
                    kept_points.append(triangulated[position])

        order: Int[ndarray, "n_landmarks"] = np.argsort(np.asarray(kept_indices, dtype=np.int64), kind="stable")
        return AnchorLandmarks(
            keypoint_indices=np.asarray(kept_indices, dtype=np.int64)[order],
            points_in_rig=np.ascontiguousarray(np.asarray(kept_points, dtype=np.float64).reshape(-1, 3)[order]),
        )

    # ----------------------------------------------------------------------------------
    # the measurement
    # ----------------------------------------------------------------------------------

    def _observations(self, source_rig_id: int, target_rig_id: int, landmarks: RigLandmarks) -> _Observations:
        """Match the source rig's anchor images against the target's and pair 2-D with 3-D.

        Every pair is asked of `match_fn` unconditionally, so a caller probing for the pair
        list with a matcher that returns nothing still sees all of them (module docstring).
        The join between a pair's matches and the anchor's landmarks is a `searchsorted`, not
        a lookup per match: a RoboCap rig pair offers a few thousand matches over six image
        pairs, and doing that in Python is most of the stage's runtime.

        Args:
            source_rig_id: Rig frame the landmarks belong to.
            target_rig_id: Rig frame being localised.
            landmarks: The source rig's local map.

        Returns:
            The correspondences, empty when nothing matched.
        """
        pixel_blocks: list[Keypoints] = []
        point_blocks: list[Float64[ndarray, "n_hits 3"]] = []
        camera_indices: list[int] = []
        contributing: set[int] = set()
        epipolar_source: list[Keypoints] = []
        epipolar_target: list[Keypoints] = []
        epipolar_positions: list[int] = []
        for camera_id in self.geometry.camera_ids:
            anchor: int | None = self.index.keyframe(source_rig_id, camera_id)
            if anchor is None:
                continue
            target_cameras: list[int] = [camera_id]
            partner_camera: int | None = self.geometry.stereo_partner.get(camera_id)
            if partner_camera is not None:
                target_cameras.append(partner_camera)
            anchor_landmarks: AnchorLandmarks | None = landmarks.get(anchor)
            for target_camera in target_cameras:
                observed: int | None = self.index.keyframe(target_rig_id, target_camera)
                if observed is None or observed not in self.keypoints:
                    continue
                pair_matches: Matches = np.asarray(self.match_fn(anchor, observed), dtype=np.int64).reshape(-1, 2)
                if len(pair_matches) == 0:
                    continue
                position: int = self.geometry.position_of(target_camera)
                if target_camera == camera_id and anchor in self.keypoints:
                    epipolar_source.append(self.keypoints[anchor][pair_matches[:, 0]])
                    epipolar_target.append(self.keypoints[observed][pair_matches[:, 1]])
                    epipolar_positions.extend([position] * len(pair_matches))
                if anchor_landmarks is None or len(anchor_landmarks) == 0:
                    continue
                insertion: Int[ndarray, "n_matches"] = np.searchsorted(anchor_landmarks.keypoint_indices, pair_matches[:, 0])
                clipped: Int[ndarray, "n_matches"] = np.clip(insertion, 0, len(anchor_landmarks) - 1)
                matched: Bool[ndarray, "n_matches"] = anchor_landmarks.keypoint_indices[clipped] == pair_matches[:, 0]
                hits: int = int(matched.sum())
                if hits == 0:
                    continue
                pixel_blocks.append(self.keypoints[observed][pair_matches[matched, 1]])
                point_blocks.append(anchor_landmarks.points_in_rig[clipped[matched]])
                camera_indices.extend([position] * hits)
                contributing.add(target_camera)
        epipolar: _EpipolarCorrespondences = _EpipolarCorrespondences(
            source_pixels=epipolar_source, target_pixels=epipolar_target, camera_indices=epipolar_positions
        )
        if not point_blocks:
            return _Observations(np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.float64), [], set(), epipolar)
        return _Observations(
            points2d=np.ascontiguousarray(np.concatenate(pixel_blocks), dtype=np.float64),
            points3d=np.ascontiguousarray(np.concatenate(point_blocks), dtype=np.float64),
            camera_indices=camera_indices,
            cameras=contributing,
            epipolar=epipolar,
        )

    def _epipolar_pose(self, correspondences: _EpipolarCorrespondences) -> pycolmap.Rigid3d | None:
        """Solve the generalized essential matrix over the same-camera matches of a rig pair.

        Its rotation and translation *direction* are trustworthy; its magnitude is not, which
        is why the caller only cross-checks a direction against it (module docstring).

        Args:
            correspondences: The same-camera matches `_observations` collected.

        Returns:
            `source_T_target` with an unreliable magnitude, or None when there are too few
            correspondences or pycolmap answers with a panoramic (pure-rotation) model.
        """
        if len(correspondences.camera_indices) < self.config.min_inliers:
            return None
        answer: dict | None = pycolmap.estimate_generalized_relative_pose(
            np.ascontiguousarray(np.vstack(correspondences.source_pixels), dtype=np.float64),
            np.ascontiguousarray(np.vstack(correspondences.target_pixels), dtype=np.float64),
            correspondences.camera_indices,
            correspondences.camera_indices,
            list(self.geometry.cams_from_rig),
            list(self.geometry.cameras),
            _ransac_options(self.config),
        )
        # pycolmap answers a configuration whose rig baseline cannot fix the scale with a
        # `pano2_from_pano1` key instead, i.e. a pure-rotation model; there is no direction in it.
        if answer is None or "rig2_from_rig1" not in answer:
            return None
        return answer["rig2_from_rig1"].inverse()
