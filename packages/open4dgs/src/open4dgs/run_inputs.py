"""What a training run reads, as types: where the clip comes from, how the init cloud is built and where the prior maps
come from. ``open4dgs.apis.train.Config`` turns its flat CLI flags into these once (and refuses flags that do not go
together there), so the pipeline branches on a type, not on flags.

- clip: ``CatalogWindow`` (a ``frame_index`` window, ``read_catalog_clip``) or ``SelfCapWindow`` (phones on
  ``video_time``, ``read_timed_clip``; optionally each phone's own capture times for the trainer);
- init: ``BodyInit`` (body-surface samples), ``LayerInit`` (the segment's init-cloud layer) or ``RrdInit`` (an
  init-cloud recording);
- prior: ``PriorSource`` (the ``moge2_prior`` layer; ``compute_missing``: MoGe-2 runs when it does not match), or
  None for no prior.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from jaxtyping import Float64
from numpy import ndarray

from open4dgs.body_init import body_surface_init
from open4dgs.catalog_clip import MultiViewClip, read_catalog_clip
from open4dgs.holdout import select_holdout
from open4dgs.init_cloud import InitCloud, check_init_matches_run, read_init_cloud, read_layer_init_cloud
from open4dgs.prior import ClipPriors, PriorConfig
from open4dgs.prior_layer import PRIOR_LAYER, ClipReader, compute_prior_layer, read_clip_priors, select_cameras


@dataclass(frozen=True, slots=True)
class CatalogWindow:
    """``num_frames`` frames of a catalog segment from ``frame_start`` on ``frame_index``."""

    catalog_url: str
    dataset: str
    segment_id: str
    frame_start: int
    num_frames: int
    cameras: tuple[str, ...] | None
    max_side: int | None
    roi_pad: float | None
    """People ROI cameras with this pad; None = full frames."""
    reader: ClassVar[ClipReader] = "catalog"

    def read(self) -> tuple[MultiViewClip, Float64[ndarray, "cams frames"] | None]:
        """The clip; no per-camera capture times (the trainer uses frame / fps)."""
        clip: MultiViewClip = read_catalog_clip(
            self.catalog_url,
            self.dataset,
            self.segment_id,
            self.frame_start,
            self.num_frames,
            cameras=self.cameras,
            max_side=self.max_side,
            roi="people" if self.roi_pad is not None else None,
            roi_pad=self.roi_pad if self.roi_pad is not None else 0.15,
        )
        return clip, None


@dataclass(frozen=True, slots=True)
class SelfCapWindow:
    """``num_frames`` frames of a SelfCap segment from ``time_start_s`` on ``video_time`` (``open4dgs.selfcap_clip``)."""

    catalog_url: str
    dataset: str
    segment_id: str
    time_start_s: float
    num_frames: int
    cameras: tuple[str, ...] | None
    max_side: int | None
    roi: bool
    """Person-centred virtual cameras framing the joints."""
    capture_times: bool
    """Give the trainer each phone's own frame times (the dataset's sync) instead of the reference grid."""
    reader: ClassVar[ClipReader] = "selfcap"

    def read(self) -> tuple[MultiViewClip, Float64[ndarray, "cams frames"] | None]:
        """The clip and, with ``capture_times``, each camera's frame times (seconds, clip order)."""
        from open4dgs.selfcap_clip import read_timed_clip

        timed = read_timed_clip(
            self.catalog_url, self.dataset, self.segment_id, self.time_start_s, self.num_frames, cameras=self.cameras, max_side=self.max_side, roi=self.roi
        )
        if not self.capture_times:
            return timed.clip, None
        print(f"[open4dgs] capture times: offset to the reference grid (ms) {np.round((timed.times_s - timed.times_s[0]) * 1e3, 1).max(axis=1).tolist()}")
        return timed.clip, timed.times_s


ClipSource: TypeAlias = CatalogWindow | SelfCapWindow


@dataclass(frozen=True, slots=True)
class BodyInit:
    """Body-surface samples on the SMPL-X meshes, coloured from the training cameras."""

    dataset: str
    segment_id: str
    samples_per_person: int
    keyframe_step: int

    def build(self, clip: MultiViewClip, train_names: tuple[str, ...], holdout_names: tuple[str, ...]) -> InitCloud:
        return body_surface_init(
            clip, train_names, dataset=self.dataset, segment_id=self.segment_id, samples_per_person=self.samples_per_person, keyframe_step=self.keyframe_step
        )

    def label(self, init: InitCloud) -> str:
        """The run property ``init_method``."""
        return init.method


@dataclass(frozen=True, slots=True)
class LayerInit:
    """The segment's init-cloud layer ``layer`` (its held-out cameras and window must be the run's)."""

    catalog_url: str
    dataset: str
    segment_id: str
    layer: str

    def build(self, clip: MultiViewClip, train_names: tuple[str, ...], holdout_names: tuple[str, ...]) -> InitCloud:
        init, init_holdout = read_layer_init_cloud(self.catalog_url, self.dataset, self.segment_id, self.layer)
        check_init_matches_run(init, init_holdout, holdout_names, int(clip.frame_indices[0]), len(clip.frame_indices))
        if init.fps != clip.fps:
            # Layer times are normalised over the window and velocities are per video frame, so fps only maps time to
            # seconds; it must be the clip's, which the trainer renders with (frame / clip.fps). The MAMMA init_roma
            # layer records 60 while the segment's capture fps is 30: without this, frames past 0.95 s have no Gaussians.
            print(f"[open4dgs] init layer fps {init.fps} != clip fps {clip.fps}: using the clip fps")
            init = dataclasses.replace(init, fps=float(clip.fps))
        return init

    def label(self, init: InitCloud) -> str:
        return f"layer:{self.layer}:{init.method}"


@dataclass(frozen=True, slots=True)
class RrdInit:
    """An init-cloud recording (``open4dgs.apis.selfcap --stage init``) of the run's window."""

    path: Path

    def build(self, clip: MultiViewClip, train_names: tuple[str, ...], holdout_names: tuple[str, ...]) -> InitCloud:
        init: InitCloud = read_init_cloud(self.path)
        if init.frame_end != len(clip.frame_indices) - 1 or init.source_frame_start != int(clip.frame_indices[0]):
            raise ValueError(f"{self.path}: init window {init.source_frame_start}+{init.frame_end + 1} does not match the clip")
        return init

    def label(self, init: InitCloud) -> str:
        return f"rrd:{init.method}"


InitSource: TypeAlias = BodyInit | LayerInit | RrdInit

ReadPriors = Callable[[MultiViewClip, str, str, str, tuple[str, ...]], ClipPriors]
ComputePriors = Callable[[MultiViewClip, str, str], ClipPriors]


@dataclass(frozen=True, slots=True)
class PriorSource:
    """The ``moge2_prior`` layer of ``segment_id`` on ``catalog_url``/``dataset``: read there and, with
    ``compute_missing`` (``--prior auto``), computed and registered there when it is missing or of another grid."""

    catalog_url: str
    dataset: str
    segment_id: str
    compute_missing: bool
    config: PriorConfig
    """At the run's steps (``PriorConfig.at_steps``)."""
    output_dir: Path
    """Where a computed layer is written (``<dir>/<dataset>/<segment>.moge2_prior-<grid>.rrd``)."""
    batch_size: int
    clip_source: str
    """``<catalog>/<dataset>`` of the clip, recorded in a computed layer."""
    max_side: int | None
    reader: ClipReader

    def priors(
        self,
        clip: MultiViewClip,
        train_names: tuple[str, ...],
        holdout_names: tuple[str, ...],
        read: ReadPriors = read_clip_priors,
        compute: ComputePriors | None = None,
    ) -> tuple[ClipPriors, ClipPriors | None, str]:
        """The prior maps of the training cameras, those of the held-out cameras (recording only; None when absent) and
        where they came from (``layer:<catalog>/<dataset>`` or ``computed:<catalog>/<dataset>``).

        Without ``compute_missing`` a missing or mismatching layer raises. ``compute(clip, catalog, dataset)`` (default:
        MoGe-2 on every clip camera, registered as the layer) replaces it otherwise.
        """
        url, dataset = self.catalog_url, self.dataset
        try:
            train: ClipPriors = read(clip, url, dataset, self.segment_id, train_names)
        except (ValueError, LookupError) as error:
            if not self.compute_missing:
                raise
            print(f"[open4dgs] prior auto: no {PRIOR_LAYER} layer for this clip on {url}/{dataset} ({error}); computing it now")
            computed: ClipPriors = (compute or self._compute)(clip, url, dataset)
            return select_cameras(computed, train_names), select_cameras(computed, holdout_names), f"computed:{url}/{dataset}"
        print(f"[open4dgs] prior: {PRIOR_LAYER} layer of {url}/{dataset}/{self.segment_id} matches the clip")
        holdout: ClipPriors | None = None
        try:  # held-out maps only for the side-by-side in the run recording; never handed to the trainer
            holdout = read(clip, url, dataset, self.segment_id, holdout_names)
        except ValueError as error:
            print(f"[open4dgs] no held-out prior maps for the recording: {error}")
        return train, holdout, f"layer:{url}/{dataset}"

    def _compute(self, clip: MultiViewClip, url: str, dataset: str) -> ClipPriors:
        priors, path, timings = compute_prior_layer(
            clip,
            self.segment_id,
            source=self.clip_source,
            catalog_url=url,
            dataset=dataset,
            output_dir=self.output_dir,
            max_side=self.max_side,
            reader=self.reader,
            batch_size=self.batch_size,
        )
        print(
            f"[open4dgs] prior auto: MoGe-2 {timings['images']:.0f} images in {timings['moge_s']:.1f} s ({timings['images'] / timings['moge_s']:.1f} img/s), "
            f"write {timings['write_s']:.1f} s, register {timings['register_s']:.1f} s -> {path}, registered as {PRIOR_LAYER} on {url}/{dataset}"
        )
        return priors


def split_cameras(
    camera_names: tuple[str, ...], holdout_cameras: tuple[str, ...] | None, num_holdout: int, seed: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(train names sorted, held-out names): the explicit ``holdout_cameras`` when given, else ``select_holdout``."""
    if holdout_cameras is None:
        return select_holdout(camera_names, num_holdout, seed)
    missing: set[str] = set(holdout_cameras) - set(camera_names)
    if missing:
        raise ValueError(f"holdout cameras {sorted(missing)} are not in the clip {camera_names}")
    return tuple(sorted(set(camera_names) - set(holdout_cameras))), holdout_cameras
