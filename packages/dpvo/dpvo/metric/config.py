"""Configuration of :class:`~dpvo.metric.dpvo.MetricDPVO` and the adapter for ViPE's uppercase yacs keys.

:class:`MetricDPVOConfig` extends :class:`~dpvo.config.DPVOConfig` with the
metric-depth prior and the MetricTrail global-BA backend. Its defaults are
ViPE's DPVO defaults (``vipe/metrictrail/dpvo/config.py``): RANDOM centroids,
a 4096-keyframe buffer, the timestamp-scaled motion model and a re-raise on BA
failure. The three MetricTrail presets (``metrictrail_local``, ``_gba``,
``_gba_refine``) are yacs blocks; :meth:`MetricDPVOConfig.from_yacs` reads one
and :meth:`MetricDPVOConfig.to_yacs` writes it back.

The adapter takes exactly the 43 keys of :data:`YACS_DEFAULTS`. Keys whose code
paths were deleted with the port (``METRIC_DEPTH=False``, ``GBA_REMEASURE_FINAL='one'``,
``REMEASURE_MIN_AGE>0``, ...) are accepted only at the values the presets use.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, TypeAlias

from dpvo.config import DPVOConfig

RemeasureMode: TypeAlias = Literal["off", "two"]
"""Final re-measurement at terminate: none, or passes of 8 and 24 iterations."""

YacsValue: TypeAlias = bool | int | float | str
"""A value of ViPE's DPVO yacs config."""

YACS_DEFAULTS: Mapping[str, YacsValue] = MappingProxyType(
    {
        "BUFFER_SIZE": 4096,
        "CENTROID_SEL_STRAT": "RANDOM",
        "PATCHES_PER_FRAME": 80,
        "REMOVAL_WINDOW": 20,
        "OPTIMIZATION_WINDOW": 12,
        "PATCH_LIFETIME": 12,
        "KEYFRAME_INDEX": 4,
        "KEYFRAME_THRESH": 12.5,
        "MOTION_MODEL": "DAMPED_LINEAR",
        "MOTION_DAMPING": 0.5,
        "MIXED_PRECISION": True,
        "METRIC_DEPTH": False,
        "METRIC_DEPTH_MIN": 0.2,
        "METRIC_DEPTH_MAX": 500.0,
        "METRIC_DEPTH_PRIOR_WEIGHT": 1.0,
        "METRIC_DEPTH_PRIOR_HUBER": 0.3,
        "METRIC_DEPTH_PRIOR_WARMUP": 12,
        "LOOP_CLOSURE": False,
        "BACKEND_THRESH": 64.0,
        "MAX_EDGE_AGE": 1000,
        "GLOBAL_OPT_FREQ": 15,
        "CLASSIC_LOOP_CLOSURE": False,
        "LOOP_CLOSE_WINDOW_SIZE": 3,
        "LOOP_RETR_THRESH": 0.04,
        "METRIC_GLOBAL_BA": False,
        "METRIC_GLOBAL_BA_T0": "active_min",
        "GBA_SCHEDULE_K": 0,
        "CULLED_INTERP": False,
        "GBA_REMEASURE_FINAL": "off",
        "FMAP_STORE": 512,
        "REMEASURE_MIN_AGE": 0,
        "REMEASURE_BURNIN": 0,
        "REMEASURE_WARM_ACTIVE": False,
        "REMEASURE_CARRY_STATE": False,
        "GBA_SCALE_LATENT": False,
        "GBA_SCALE_SMOOTH": 20.0,
        "GBA_SCALE_CLAMP": 0.35,
        "GBA_SCALE_MIN_PATCHES": 32,
        "GBA_SCALE_MIN_PARALLAX_PX": 2.0,
        "GBA_SCALE_LOCAL_USE": "off",
        "LOOP_REVERSE_EDGES": False,
        "KEYFRAME_MAX_CULL": 0,
        "KEYFRAME_MAX_CULL_ROT_DEG": 10.0,
    }
)
"""ViPE's DPVO yacs defaults (``vipe/metrictrail/dpvo/config.py``), all 43 keys."""

_FMAP_STORE_DEFAULT: int = 512
"""``FMAP_STORE`` is inert while the final re-measurement is off; the adapter then holds it at this value."""

_DIRECT_KEYS: Mapping[str, tuple[str, type]] = MappingProxyType(
    {
        "BUFFER_SIZE": ("buffer_size", int),
        "PATCHES_PER_FRAME": ("patches_per_frame", int),
        "REMOVAL_WINDOW": ("removal_window", int),
        "OPTIMIZATION_WINDOW": ("optimization_window", int),
        "PATCH_LIFETIME": ("patch_lifetime", int),
        "KEYFRAME_INDEX": ("keyframe_index", int),
        "KEYFRAME_THRESH": ("keyframe_thresh", float),
        "MOTION_DAMPING": ("motion_damping", float),
        "MIXED_PRECISION": ("mixed_precision", bool),
        "METRIC_DEPTH_MIN": ("depth_min", float),
        "METRIC_DEPTH_MAX": ("depth_max", float),
        "METRIC_DEPTH_PRIOR_WEIGHT": ("prior_weight", float),
        "METRIC_DEPTH_PRIOR_HUBER": ("prior_huber", float),
        "METRIC_DEPTH_PRIOR_WARMUP": ("prior_warmup", int),
        "LOOP_CLOSURE": ("loop_closure", bool),
        "BACKEND_THRESH": ("backend_thresh", float),
        "MAX_EDGE_AGE": ("max_edge_age", int),
        "GLOBAL_OPT_FREQ": ("global_opt_freq", int),
        "LOOP_CLOSE_WINDOW_SIZE": ("loop_close_window_size", int),
        "LOOP_RETR_THRESH": ("loop_retr_thresh", float),
        "METRIC_GLOBAL_BA": ("metric_global_ba", bool),
        "GBA_SCHEDULE_K": ("gba_schedule_k", int),
        "CULLED_INTERP": ("culled_interp", bool),
        "GBA_REMEASURE_FINAL": ("remeasure_final", str),
        "REMEASURE_BURNIN": ("remeasure_burnin", int),
        "REMEASURE_WARM_ACTIVE": ("remeasure_warm_active", bool),
        "REMEASURE_CARRY_STATE": ("remeasure_carry_state", bool),
        "GBA_SCALE_LATENT": ("scale_latent", bool),
        "GBA_SCALE_SMOOTH": ("scale_smooth", float),
        "GBA_SCALE_CLAMP": ("scale_clamp", float),
        "GBA_SCALE_MIN_PATCHES": ("scale_min_patches", int),
        "GBA_SCALE_MIN_PARALLAX_PX": ("scale_min_parallax_px", float),
        "LOOP_REVERSE_EDGES": ("loop_reverse_edges", bool),
        "KEYFRAME_MAX_CULL": ("keyframe_max_cull", int),
        "KEYFRAME_MAX_CULL_ROT_DEG": ("keyframe_max_cull_rot_deg", float),
    }
)
"""yacs keys that map one to one onto a config field, with the field's Python type."""

_FIXED_KEYS: Mapping[str, YacsValue] = MappingProxyType(
    {
        "CENTROID_SEL_STRAT": "RANDOM",
        "MOTION_MODEL": "DAMPED_LINEAR",
        "METRIC_DEPTH": True,
        "CLASSIC_LOOP_CLOSURE": False,
        "REMEASURE_MIN_AGE": 0,
        "GBA_SCALE_LOCAL_USE": "off",
    }
)
"""yacs keys whose other values select code paths that the port deleted; only these values are accepted."""


def _typed(key: str, value: object, kind: type) -> YacsValue:
    """Check one yacs value against its field type; an int widens to float, nothing else converts."""
    is_bool: bool = isinstance(value, bool)
    if kind is bool and is_bool:
        return value  # pyrefly: ignore[bad-return]
    if kind is int and isinstance(value, int) and not is_bool:
        return value
    if kind is float and isinstance(value, int | float) and not is_bool:
        return float(value)
    if kind is str and isinstance(value, str):
        return value
    raise ValueError(f"yacs key {key} must be {kind.__name__}, got {value!r}")


@dataclass(frozen=True)
class MetricDPVOConfig(DPVOConfig):
    """DPVO with a metric-depth prior and the MetricTrail global-BA backend.

    The backend fields of :class:`~dpvo.config.DPVOConfig` follow from the
    metric ones (see ``__post_init__``): with ``metric_global_ba`` the pruned
    factors are stored, the global BA does not normalize and frees every pose but
    frame 0; ``fmap_memory`` is 36 unless the final re-measurement is on.
    """

    buffer_size: int = 4096
    gradient_bias: bool = False
    store_inactive: bool = False
    motion_scale_by_tstamps: bool = True
    raise_on_ba_failure: bool = True

    depth_min: float = 0.2
    """Smallest metric depth (m) that enters the prior."""
    depth_max: float = 500.0
    """Largest metric depth (m) that enters the prior."""
    prior_weight: float = 1.0
    """Weight ``lam`` of the log-inverse-depth unary in the local and global BA."""
    prior_huber: float = 0.3
    """Huber threshold of the unary, in log inverse depth."""
    prior_warmup: int = 12
    """Keyframes after which the unary turns on (it stays on)."""

    metric_global_ba: bool = False
    """Proximity loop edges feed a global BA that keeps the metric scale and uses the prior."""
    culled_interp: bool = False
    """At terminate, spread each kept keyframe's BA correction over the culled frames before it."""
    remeasure_final: RemeasureMode = "off"
    """At terminate, re-measure every stored and active factor and run the global BA (passes of 8 and 24)."""
    remeasure_burnin: int = 0
    """Re-measurements without a BA at the start of each pass."""
    remeasure_warm_active: bool = False
    """The first pass starts the active edges' GRU state from their tracking-time state."""
    remeasure_carry_state: bool = False
    """The second pass keeps the first pass's GRU state."""

    scale_latent: bool = False
    """The global BA sees the prior rescaled by a smoothed per-keyframe log-scale latent."""
    scale_smooth: float = 20.0
    """Weight of the first-difference penalty on the latent."""
    scale_clamp: float = 0.35
    """Bound on ``|s|`` and gate on the per-keyframe offset ``|m|`` (log units)."""
    scale_min_patches: int = 32
    """Keyframes with fewer valid prior patches have no data term."""
    scale_min_parallax_px: float = 2.0
    """Keyframes whose mean translation-only flow into the previous keyframe is smaller have no data term."""

    loop_reverse_edges: bool = False
    """Add the reverse edges of every accepted proximity loop pair (new patches seen in the old frame)."""
    keyframe_max_cull: int = 0
    """Keep a keyframe candidate after this many consecutive culls (with ``keyframe_max_cull_rot_deg``); 0 = off."""
    keyframe_max_cull_rot_deg: float = 10.0
    """Rotation path (deg) the culled run must also cover before the guard keeps a candidate."""

    def __post_init__(self) -> None:
        DPVOConfig.__post_init__(self)
        mgba: bool = self.metric_global_ba
        if self.classic_loop_closure:
            raise ValueError("MetricDPVO forbids classic_loop_closure")
        if self.loop_closure != mgba:
            raise ValueError("MetricDPVO needs loop_closure == metric_global_ba (proximity edges feed the metric global BA)")
        if not 0.0 < self.depth_min < self.depth_max:
            raise ValueError(f"invalid metric depth range: {self.depth_min}, {self.depth_max}")
        if self.prior_weight < 0.0 or not self.prior_huber > 0.0:
            raise ValueError(f"prior_weight must be >= 0 and prior_huber > 0, got {self.prior_weight}, {self.prior_huber}")
        if self.remeasure_final not in ("off", "two"):
            raise ValueError(f"remeasure_final must be 'off' or 'two', got {self.remeasure_final!r}")
        if not mgba and (self.gba_schedule_k > 0 or self.remeasure_final != "off" or self.scale_latent):
            raise ValueError("gba_schedule_k, remeasure_final and scale_latent need metric_global_ba")
        if self.loop_reverse_edges and (not mgba or self.remeasure_final == "off"):
            raise ValueError("loop_reverse_edges needs metric_global_ba and remeasure_final != 'off' (the long fmap ring)")
        if self.remeasure_burnin < 0 or self.keyframe_max_cull < 0 or self.keyframe_max_cull_rot_deg < 0.0:
            raise ValueError("remeasure_burnin, keyframe_max_cull and keyframe_max_cull_rot_deg must be >= 0")
        # Core backend fields derived from the metric ones
        if self.store_inactive != mgba or self.normalize_global_ba == mgba:
            raise ValueError("MetricDPVO needs store_inactive == metric_global_ba and normalize_global_ba == (not metric_global_ba)")
        if self.global_ba_t0 != ("first" if mgba else "active_min"):
            raise ValueError("MetricDPVO needs global_ba_t0 = 'first' with metric_global_ba, else 'active_min'")
        if self.remeasure_final == "off" and self.fmap_memory != 36:
            raise ValueError("fmap_memory is 36 unless remeasure_final is on")

    @classmethod
    def from_yacs(cls, block: Mapping[str, object]) -> "MetricDPVOConfig":
        """Build the config from a full ViPE DPVO yacs block (uppercase keys).

        Args:
            block: Exactly the 43 keys of :data:`YACS_DEFAULTS`, e.g. a preset's
                ``dpvo:`` block merged over the defaults.

        Raises:
            ValueError: A key is unknown or absent, a value has the wrong type, a
                key selects a deleted code path, or the cross-key rules fail.
        """
        unknown: list[str] = sorted(set(block) - set(YACS_DEFAULTS))
        absent: list[str] = sorted(set(YACS_DEFAULTS) - set(block))
        if unknown or absent:
            raise ValueError(f"DPVO yacs block does not match the 43 keys: unknown={unknown}, absent={absent}")
        for key, fixed in _FIXED_KEYS.items():
            if _typed(key, block[key], type(fixed)) != fixed:
                raise ValueError(f"yacs key {key} must be {fixed!r} (other values are not ported), got {block[key]!r}")
        fields: dict[str, YacsValue] = {field: _typed(key, block[key], kind) for key, (field, kind) in _DIRECT_KEYS.items()}

        if fields["remeasure_final"] not in ("off", "two"):
            raise ValueError(f"GBA_REMEASURE_FINAL must be 'off' or 'two' (other values are not ported), got {fields['remeasure_final']!r}")
        mgba: bool = bool(fields["metric_global_ba"])
        t0_mode: YacsValue = _typed("METRIC_GLOBAL_BA_T0", block["METRIC_GLOBAL_BA_T0"], str)
        if t0_mode != ("all" if mgba else "active_min"):
            raise ValueError(f"METRIC_GLOBAL_BA_T0 must be 'all' with METRIC_GLOBAL_BA, else 'active_min'; got {t0_mode!r}")
        fmap_store: YacsValue = _typed("FMAP_STORE", block["FMAP_STORE"], int)
        remeasure_on: bool = fields["remeasure_final"] != "off"
        if not remeasure_on and fmap_store != _FMAP_STORE_DEFAULT:
            raise ValueError(f"FMAP_STORE is inert without GBA_REMEASURE_FINAL and must stay {_FMAP_STORE_DEFAULT}, got {fmap_store}")
        return cls(
            **fields,  # pyrefly: ignore[bad-argument-type]
            store_inactive=mgba,
            normalize_global_ba=not mgba,
            global_ba_t0="first" if mgba else "active_min",
            fmap_memory=int(fmap_store) if remeasure_on else 36,
        )

    def to_yacs(self) -> dict[str, YacsValue]:
        """The 43-key yacs block that :meth:`from_yacs` turns back into this config.

        Raises:
            ValueError: The config holds a value yacs cannot express (for example
                ``gradient_bias=True`` or ``motion_scale_by_tstamps=False``).
        """
        block: dict[str, YacsValue] = dict(_FIXED_KEYS)
        for key, (field, _kind) in _DIRECT_KEYS.items():
            block[key] = getattr(self, field)
        block["METRIC_GLOBAL_BA_T0"] = "all" if self.global_ba_t0 == "first" else "active_min"
        block["FMAP_STORE"] = self.fmap_memory if self.remeasure_final != "off" else _FMAP_STORE_DEFAULT
        ordered: dict[str, YacsValue] = {key: block[key] for key in YACS_DEFAULTS}
        if MetricDPVOConfig.from_yacs(ordered) != self:
            raise ValueError("config holds values the ViPE yacs keys cannot express")
        return ordered
