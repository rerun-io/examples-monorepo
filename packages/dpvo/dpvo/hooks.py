"""Extension points of the DPVO tracker.

:class:`~dpvo.dpvo.DPVO` is plain DPVO with three seams that extensions such as
MetricDPVO plug into without copying the tracker:

- :class:`DepthPrior` -- how a new frame's patch depths start and which unary
  prior the bundle adjustment sees. The default :class:`RandomDepthInit` is
  plain DPVO's draw (``rand_like``, then the median of the last three
  keyframes once initialized).
- :class:`GraphPolicy` -- vetoes and side effects of keyframe culling, and
  extra loop edges. The default :class:`DefaultGraphPolicy` does nothing.
- :class:`DPVOObserver` -- callbacks at fixed points of a run (frame done,
  keyframe culled, global BA, terminate) plus a last pass over the trajectory.
  :func:`add_global_observer` attaches observers to every tracker the process
  builds, for callers that do not construct the tracker themselves.

Observer and strategy exceptions propagate to the caller.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

import torch
from jaxtyping import Bool, Float32, Int64, UInt8
from lietorch_mojo import SE3
from torch import Tensor

from dpvo.fastba import BADepthPrior

if TYPE_CHECKING:
    import dpvo.dpvo

BAKind: TypeAlias = Literal["local", "global"]
"""Which bundle adjustment asks for a depth prior: the sliding-window solve or the global one."""


@dataclass(frozen=True, slots=True)
class FrameDepth:
    """Per-frame depth input handed to :meth:`~dpvo.dpvo.DPVO.__call__`; opaque to the core."""

    depth: Float32[Tensor, "ht wd"] | None
    """Depth map at image resolution, or ``None``."""
    mask: Bool[Tensor, "ht wd"] | None
    """Pixels of ``depth`` that may be used, or ``None`` for all."""


@runtime_checkable
class DepthPrior(Protocol):
    """Strategy for patch-depth initialisation and the BA depth prior."""

    def prepare(
        self, dpvo: "dpvo.dpvo.DPVO", image_hw: tuple[int, int], frame: FrameDepth
    ) -> Float32[Tensor, "1 1 h4 w4"] | None:
        """Run before patchify; returns the ``disps`` grid for patchify (``None`` = ones).

        Raise ``ValueError`` to reject the frame; no tracker state has changed yet.
        """
        ...

    def init_patch_depth(self, dpvo: "dpvo.dpvo.DPVO", patches: Float32[Tensor, "1 M 3 ps ps"]) -> None:
        """Write ``patches[:, :, 2]`` in place, and the strategy's frame attributes at slot ``dpvo.n``.

        Runs right after patchify and before any tracker state is written, so a
        ``ValueError`` here rejects the frame cleanly.
        """
        ...

    def ba_prior(self, dpvo: "dpvo.dpvo.DPVO", kind: BAKind) -> BADepthPrior | None:
        """Depth prior for the next BA solve, or ``None`` for none."""
        ...


class RandomDepthInit:
    """Plain DPVO's depth initialisation: uniform random inverse depths, then the recent median.

    Draws ``M`` uniforms from the default generator of the patches' device on
    every frame, even when the median then overwrites them once the tracker is
    initialized; that keeps plain DPVO's RNG stream. No BA prior.
    """

    def prepare(
        self, dpvo: "dpvo.dpvo.DPVO", image_hw: tuple[int, int], frame: FrameDepth
    ) -> Float32[Tensor, "1 1 h4 w4"] | None:
        """No depth grid: patchify's default ones."""
        return None

    def init_patch_depth(self, dpvo: "dpvo.dpvo.DPVO", patches: Float32[Tensor, "1 M 3 ps ps"]) -> None:
        """``rand_like`` per patch; once initialized, the median inverse depth of the last three keyframes."""
        patches[:, :, 2] = torch.rand_like(patches[:, :, 2, 0, 0, None, None])
        if dpvo.is_initialized:
            s: Float32[Tensor, ""] = torch.median(dpvo.pg.patches_[dpvo.n - 3 : dpvo.n, :, 2])
            patches[:, :, 2] = s

    def ba_prior(self, dpvo: "dpvo.dpvo.DPVO", kind: BAKind) -> BADepthPrior | None:
        """No prior."""
        return None


@runtime_checkable
class GraphPolicy(Protocol):
    """Strategy for keyframe culling and extra loop edges."""

    def force_keep(self, dpvo: "dpvo.dpvo.DPVO") -> bool:
        """Veto this call's keyframe cull; asked once per keyframe() call, before the flow test decides."""
        ...

    def on_cull(self, dpvo: "dpvo.dpvo.DPVO", k: int) -> None:
        """Keyframe slot ``k`` is culled; runs before the buffers are compacted."""
        ...

    def on_keep(self, dpvo: "dpvo.dpvo.DPVO", k: int, forced: bool) -> None:
        """Keyframe slot ``k`` stays; ``forced`` means the flow test wanted to cull it."""
        ...

    def extra_loop_edges(
        self, dpvo: "dpvo.dpvo.DPVO", lii: Int64[Tensor, "e"], ljj: Int64[Tensor, "e"]
    ) -> tuple[Int64[Tensor, "r"], Int64[Tensor, "r"]] | None:
        """Edges ``(kk, jj)`` to append right after the proximity loop edges ``(lii, ljj)``, or ``None``."""
        ...


class DefaultGraphPolicy:
    """Plain DPVO: no veto, no side effects, no extra edges."""

    def force_keep(self, dpvo: "dpvo.dpvo.DPVO") -> bool:
        """Never veto."""
        return False

    def on_cull(self, dpvo: "dpvo.dpvo.DPVO", k: int) -> None:
        """Nothing to do."""

    def on_keep(self, dpvo: "dpvo.dpvo.DPVO", k: int, forced: bool) -> None:
        """Nothing to do."""

    def extra_loop_edges(
        self, dpvo: "dpvo.dpvo.DPVO", lii: Int64[Tensor, "e"], ljj: Int64[Tensor, "e"]
    ) -> tuple[Int64[Tensor, "r"], Int64[Tensor, "r"]] | None:
        """No extra edges."""
        return None


@dataclass(frozen=True, slots=True)
class FrameEvent:
    """What :meth:`DPVOObserver.on_frame` learns about a processed frame."""

    tstamp: int
    """Caller timestamp of the frame."""
    frame_index: int
    """``dpvo.counter`` before the call (index among accepted frames)."""
    keyframes_before: int
    """``dpvo.n`` before the call."""
    keyframe_added: bool
    """The frame passed the initialization motion probe and became a keyframe (``n += 1`` ran)."""
    image: UInt8[Tensor, "3 ht wd"]
    """The input image as given (BGR)."""
    intrinsics: Float32[Tensor, "4"]
    """The input intrinsics ``fx fy cx cy`` in pixels."""
    depth: FrameDepth
    """The depth input of the call."""


class DPVOObserver:
    """Callbacks at fixed points of a DPVO run; every method is a no-op until a subclass overrides it."""

    def on_frame(self, dpvo: "dpvo.dpvo.DPVO", event: FrameEvent) -> None:
        """End of ``__call__`` (both exits); never for a frame that the depth prior rejected."""

    def on_cull(self, dpvo: "dpvo.dpvo.DPVO", k: int) -> None:
        """Keyframe slot ``k`` is culled; after the graph policy, before compaction."""

    def on_global_ba(self, dpvo: "dpvo.dpvo.DPVO") -> None:
        """Entry of :meth:`~dpvo.dpvo.DPVO.run_global_ba`, before the solve."""

    def on_terminate(self, dpvo: "dpvo.dpvo.DPVO") -> None:
        """After terminate's 12 final updates, before the trajectory is extracted."""

    def refine_trajectory(self, dpvo: "dpvo.dpvo.DPVO", poses: SE3) -> SE3:
        """Adjust the stacked world-to-camera poses of every frame before they are inverted; identity by default."""
        return poses


ObserverFactory: TypeAlias = Callable[["dpvo.dpvo.DPVO"], DPVOObserver | None]
"""Builds an observer for a new tracker, or returns ``None`` to skip it."""

_GLOBAL_OBSERVER_FACTORIES: list[ObserverFactory] = []


def add_global_observer(factory: ObserverFactory) -> Callable[[], None]:
    """Attach observers to every DPVO built from now on.

    ``DPVO.__init__`` calls each registered factory, in registration order, with
    the new instance and keeps the non-``None`` results after its own observers.

    Args:
        factory: Called once per new tracker.

    Returns:
        A remover; calling it unregisters ``factory`` (a second call does nothing).
    """
    _GLOBAL_OBSERVER_FACTORIES.append(factory)

    def remove() -> None:
        if factory in _GLOBAL_OBSERVER_FACTORIES:
            _GLOBAL_OBSERVER_FACTORIES.remove(factory)

    return remove


def global_observers(dpvo: "dpvo.dpvo.DPVO") -> list[DPVOObserver]:
    """Observers that the registered global factories build for ``dpvo``."""
    built: list[DPVOObserver | None] = [factory(dpvo) for factory in list(_GLOBAL_OBSERVER_FACTORIES)]
    return [observer for observer in built if observer is not None]
