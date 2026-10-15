"""Backend node (SPEC 5.5, placed in its own process): connects each new keyframe, relocalises lost frames, optimises the map."""

import os
from collections.abc import Mapping

import lietorch
import numpy as np
import torch
from dagkit.node import Emit, PortType, RunContext, Tick
from jaxtyping import Float32, Int64
from mast3r.model import AsymmetricMASt3R
from numpy import ndarray

from mast3r_slam.models.retrieval import CHECKPOINT, RetrievalDatabase
from mast3r_slam.nodes.factor_graph import FactorGraph
from mast3r_slam.nodes.shared import SharedKeyframes, SharedSlam, SharedStates
from mast3r_slam.types import BackendTask, CameraView, Frame, KeyframeAdded, KeyframeView, LogLine, MapUpdate, Mode, SlamConfig, TrackedFrame


class Backend:
    """Optimise the map, relocalise lost frames.

    One task per tick: ``KeyframeAdded`` adds matching factors to the previous keyframe and to retrieval candidates, then
    runs global Gauss-Newton; ``RelocRequest`` tries to relocalise the shared current frame against the retrieval database.
    Each task's result leaves as a ``MapUpdate`` (and a new keyframe's ``KeyframeView``, the retrieval and relocalisation
    verdicts as ``LogLine``s); once TERMINATED (realtime stop) queued tasks are dropped and counted.

    At flush the final keyframe poses leave on ``keyframes``, one per keyframe at its own frame time (``emits_late``).
    The shared buffers carry the map both ways; after each task (dropped ones too) the backend publishes its GPU writes
    and acknowledges the task on the ``shared`` resource, which the lossless frontend waits for.
    """

    inputs: Mapping[str, PortType] = {"task": BackendTask}
    outputs: Mapping[str, PortType] = {"map": MapUpdate, "keyframes": TrackedFrame, "keyframe": KeyframeView, "log": LogLine}
    emits_late: bool = True

    def __init__(self, params: SlamConfig) -> None:
        # Built in the backend process before its first CUDA use: the global solve allocates edge-sized buffers that fragment
        # the default allocator (2.5 GB reserved but unallocated at a 120 s OOM). The process only writes into tensors the
        # main process shares by CUDA IPC and exports none, so its own allocator may use expandable segments.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        self.params: SlamConfig = params
        self.times: dict[int, int] = {}
        """Keyframe slot -> time of its frame."""
        self.dropped: int = 0

    def reset(self, ctx: RunContext) -> None:
        model: AsymmetricMASt3R = ctx.resources["mast3r"]  # type: ignore[assignment]
        self.shared: SharedSlam = ctx.resources["shared"]  # type: ignore[assignment]
        self.keyframes: SharedKeyframes = self.shared.keyframes
        self.states: SharedStates = self.shared.states
        K = self.keyframes.K if self.params.use_calib else None
        self.graph: FactorGraph = FactorGraph(model, self.keyframes, self.params.local_opt, self.params.matching, K, device=self.keyframes.device)
        self.retrieval: RetrievalDatabase = RetrievalDatabase(CHECKPOINT, backbone=model, device=self.keyframes.device)
        self.times, self.dropped = {}, 0

    def _solve(self) -> None:
        self.graph.solve_GN_calib() if self.params.use_calib else self.graph.solve_GN_rays()

    def _connect(self, idx: int) -> list[int]:
        """Factors from keyframe ``idx`` to its predecessor and to its retrieval candidates, then a global solve; returns the
        candidates."""
        frame: Frame = self.keyframes[idx]
        retrieved: list[int] = self.retrieval.update(frame, add_after_query=True, k=self.params.retrieval.k, min_thresh=self.params.retrieval.min_thresh)
        candidates: set[int] = {idx - 1, *retrieved} if idx > 0 else set(retrieved)
        candidates.discard(idx)
        kf_idx: list[int] = list(candidates)
        if kf_idx:
            self.graph.add_factors(kf_idx, [idx] * len(kf_idx), self.params.local_opt.min_match_frac)
        self._solve()
        return retrieved

    def _relocalise(self, frame: Frame) -> bool:
        """Append the lost frame as a keyframe connected to its retrieval candidates; undo it when no edge holds."""
        with self.keyframes.lock:
            retrieval = self.params.retrieval
            kf_idx: list[int] = self.retrieval.update(frame, add_after_query=False, k=retrieval.k, min_thresh=retrieval.min_thresh)
            if not kf_idx:
                return False
            self.keyframes.append(frame)
            n_kf: int = len(self.keyframes)
            if not self.graph.add_factors([n_kf - 1] * len(kf_idx), kf_idx, self.params.reloc.min_match_frac, is_reloc=self.params.reloc.strict):
                self.keyframes.pop_last()
                return False
            self.retrieval.update(frame, add_after_query=True, k=retrieval.k, min_thresh=retrieval.min_thresh)
            matched = lietorch.Sim3(self.keyframes.world_sim3_cam[kf_idx[0]].clone())  # the matched keyframe's pose as the start
            self.keyframes.update_world_sim3_cams(matched, torch.tensor([n_kf - 1], device=self.keyframes.world_sim3_cam.device, dtype=torch.long))
            self._solve()
            return True

    def _poses(self) -> Float32[ndarray, "k 8"]:
        with self.keyframes.lock:
            return self.keyframes.world_sim3_cam[: len(self.keyframes), 0].cpu().numpy()

    def step(self, tick: Tick, out: Emit) -> None:
        task: BackendTask = tick.value  # type: ignore[assignment]
        if self.states.get_mode() == Mode.TERMINATED:  # realtime stop: the frontend no longer waits for this work
            self.dropped += 1
        elif isinstance(task, KeyframeAdded):
            retrieved: list[int] = self._connect(task.index)
            self.times[task.index] = tick.t_ns
            out.put("keyframe", tick.t_ns, KeyframeView(CameraView.of(self.keyframes[task.index], task.index)))
            out.put("log", tick.t_ns, LogLine(f"Database retrieval {task.index}: {retrieved}"))
        elif self.states.get_mode() == Mode.RELOC:
            frame: Frame = self.states.get_frame()
            if frame.frame_id != task.frame_index:
                raise RuntimeError(f"the shared current frame is frame {frame.frame_id}, the request is for frame {task.frame_index}")
            out.put("log", tick.t_ns, LogLine(f"Relocalizing frame {frame.frame_id} against the keyframes"))
            if self._relocalise(frame):
                self.times[len(self.keyframes) - 1] = tick.t_ns
                with self.states.lock:  # a realtime stop (TERMINATED) that came during the relocalisation stays
                    if self.states.get_mode() == Mode.RELOC:
                        self.states.set_mode(Mode.TRACKING)
                out.put("log", tick.t_ns, LogLine("Relocalized successfully"))
            else:
                out.put("log", tick.t_ns, LogLine("Failed to relocalize", "WARN"))
        ii: Int64[ndarray, " e"] = self.graph.ii.cpu().numpy().astype(np.int64)
        out.put("map", tick.t_ns, MapUpdate(self._poses(), ii, self.graph.jj.cpu().numpy().astype(np.int64)))
        self.shared.acknowledge()  # after the map was read: the frontend's next keyframe is not in this update
        torch.cuda.empty_cache()  # each global solve allocates edge-sized buffers: cached blocks would fragment the GPU

    def flush(self, out: Emit) -> None:
        poses: Float32[ndarray, "k 8"] = self._poses()
        with self.keyframes.lock:
            frame_index: list[int] = self.keyframes.dataset_idx[: len(poses)].tolist()
        for slot in sorted(self.times, key=lambda k: (self.times[k], k)):
            if slot < len(poses):
                out.put("keyframes", self.times[slot], TrackedFrame(frame_index[slot], poses[slot]))
        self.times = {}
        if self.dropped:
            print(f"mast3r-slam backend: dropped {self.dropped} queued tasks at stop (realtime)")
