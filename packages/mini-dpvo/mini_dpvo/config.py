"""Default configuration for DPVO (Deep Patch Visual Odometry).

Uses YACS :class:`CfgNode` to define a hierarchical, immutable-after-freeze
configuration tree.  All tuneable hyper-parameters for the DPVO system are
declared here with their default values.

Typical usage::

    from mini_dpvo.config import cfg
    cfg.merge_from_file("my_config.yaml")
    cfg.freeze()

See Teed et al. (2022), "Deep Patch Visual Odometry" for details on how
each parameter affects the system.
"""

from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
# Maximum number of keyframes that can be held in the sliding-window buffer.
# Increase for longer sequences; the main DPVO class allocates fixed-size
# GPU tensors of this length at startup.
_C.BUFFER_SIZE = 2048

# ---------------------------------------------------------------------------
# Patch selection  (See Sec. 3.1 of Teed et al. 2022)
# ---------------------------------------------------------------------------
# When True, use gradient-biased sampling: sample 3x candidate patches and
# keep the top-M by image gradient magnitude.  This concentrates patches on
# textured regions where matching is more reliable.
_C.GRADIENT_BIAS = True

# ---------------------------------------------------------------------------
# Visual Odometry core parameters
# ---------------------------------------------------------------------------
# Number of sparse 3x3 patches tracked per frame (M in the paper).
# Higher values improve accuracy at the cost of compute.
_C.PATCHES_PER_FRAME = 80

# Edges whose source patch belongs to a frame older than
# (current_frame - REMOVAL_WINDOW) are pruned each iteration.
_C.REMOVAL_WINDOW = 20

# Only poses within the last OPTIMIZATION_WINDOW frames are optimized
# during bundle adjustment (after initialization).
_C.OPTIMIZATION_WINDOW = 12

# Maximum temporal distance (in frames) for creating measurement edges
# between a patch and an observing frame.  Controls the graph density.
_C.PATCH_LIFETIME = 12

# ---------------------------------------------------------------------------
# Keyframe management  (See Sec. 3.4 of Teed et al. 2022)
# ---------------------------------------------------------------------------
# Index offset from the newest frame used to identify the candidate
# keyframe for removal.  The system checks motion between frames
# (n - KEYFRAME_INDEX - 1) and (n - KEYFRAME_INDEX + 1).
_C.KEYFRAME_INDEX = 4

# If the average bidirectional pixel flow between the candidate frame
# and its neighbours is below this threshold (in pixels), the candidate
# is removed as redundant.  Lower values retain more keyframes.
_C.KEYFRAME_THRESH = 12.5

# ---------------------------------------------------------------------------
# Camera motion model  (See Sec. 3.3 of Teed et al. 2022)
# ---------------------------------------------------------------------------
# Motion model used to predict the initial pose of each new frame.
# 'DAMPED_LINEAR': P_new = P_{n-1} * (P_{n-1} * P_{n-2}^{-1})^damping
# This extrapolates the previous inter-frame motion with exponential damping.
_C.MOTION_MODEL = 'DAMPED_LINEAR'

# Damping factor applied to the log of the relative motion.
# 0.0 = constant position (no motion prediction),
# 1.0 = constant velocity (full extrapolation).
_C.MOTION_DAMPING = 0.5

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
# Use mixed precision (float16) for the correlation and GRU update
# computations.  Significantly reduces GPU memory and improves throughput.
_C.MIXED_PRECISION = True

cfg: CN = _C
