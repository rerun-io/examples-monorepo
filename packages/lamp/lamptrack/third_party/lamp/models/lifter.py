# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""3D pose lifter for temporal snippets."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import smplx
import torch
from jaxtyping import Float, Float32
from numpy import ndarray
from torch import Tensor

from lamptrack.third_party.lamp.core.se3 import invert
from lamptrack.third_party.lamp.core.types import Skeleton

logger: logging.Logger = logging.getLogger(__name__)

# The public SMPL multi-view checkpoints expect four camera views.
_DEFAULT_NUM_VIEWS: int = 4
_SMPL_KEYPOINT_COUNT: int = 17
_SMPL_JOINT_COUNT: int = 24
_SMPL_BETA_COUNT: int = 10

# Half-extent of the synthetic floor square passed to the model.
_FLOOR_PLANE_EXTENT_M: float = 25.0

_SMPL_PELVIS: int = 0
_SMPL_L_KNEE: int = 4
_SMPL_R_KNEE: int = 5
_SMPL_L_ANKLE: int = 7
_SMPL_R_ANKLE: int = 8


@dataclass(slots=True)
class SnippetData:
    """Per-track temporal snippet handed from tracker to lifter."""

    person_id: int
    snippet_timestamps_ns: list[int]
    view_cam_indices: list[int | None] = field(default_factory=list[int | None])
    kp2ds_per_view: list[Float32[ndarray, "time 17 3"]] = field(default_factory=list)
    Ts_gw_cam_per_view: list[Float32[ndarray, "time 4 4"]] = field(default_factory=list)
    cam_params_per_view: list[Float32[ndarray, "time params"]] = field(default_factory=list)
    # `T_gravityWorld_world` snapshot at lift time, retained for downstream
    # consumers; the lifter keeps its outputs in the gravity-world frame.
    T_gravityWorld_world: Float32[ndarray, "4 4"] = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )


@dataclass(slots=True)
class LifterSettings:
    """Outlier-rejection thresholds + snippet shape for the lifter."""

    snippet_length: int = 20
    min_pose_depth: float = 1.0
    max_pose_depth: float = 5.0
    # Threshold applied at snippet-build time to convert per-keypoint
    # detection scores to the model's binary `1.0 / 0.0` confidence channel.
    kp_thres_for_binary: float = 0.5
    # Pelvis-distance cap (meters) used by `LampTracker.merge_lifted_tracks`
    # to collapse per-camera duplicate tracks of the same person.
    merge_threshold_m: float = 0.3


class _CapturedLampNet(torch.nn.Module):
    """CUDA-Graph replay wrapper around `LampNet`."""

    def __init__(
        self,
        eager_model: torch.nn.Module,
        capture_batch_size: int,
        snippet_length: int,
        num_views: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._eager_model = eager_model
        self._capture_b: int = capture_batch_size
        self._device: torch.device = device
        # Static buffers are updated in place before each CUDA graph replay.
        self._static_x: list[Float32[Tensor, "batch time 17 3"]] = [
            torch.zeros(
                capture_batch_size,
                snippet_length,
                _SMPL_KEYPOINT_COUNT,
                3,
                device=device,
            )
            for _ in range(num_views)
        ]
        self._static_cams: list[Float32[Tensor, "batch time 16"]] = [
            torch.zeros(capture_batch_size, snippet_length, 16, device=device)
            for _ in range(num_views)
        ]
        eye_template = torch.eye(4, device=device)
        self._static_Ts: list[Float32[Tensor, "batch time 4 4"]] = [
            eye_template.expand(capture_batch_size, snippet_length, 4, 4).contiguous()
            for _ in range(num_views)
        ]
        self._static_gp: Float32[Tensor, "batch 4 3"] = torch.full(
            (capture_batch_size, 4, 3),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        # Settle cuBLAS workspaces before graph capture.
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream), torch.no_grad():  # pyright: ignore[reportArgumentType]
            for _ in range(3):
                _ = self._eager_model(
                    self._static_x,
                    self._static_cams,
                    self._static_Ts,
                    self._static_gp,
                )
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        # Graph outputs are overwritten on each replay, so callers receive clones.
        self._graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, stream=capture_stream), torch.no_grad():
            self._static_out: dict[str, Tensor] = self._eager_model(
                self._static_x,
                self._static_cams,
                self._static_Ts,
                self._static_gp,
            )

    def forward(
        self,
        x_list: list[Float[Tensor, "batch time 17 3"]],
        cam_params: list[Float[Tensor, "batch time params"]],
        Ts_wc: list[Float[Tensor, "batch time 4 4"]],
        ground_planes: Float[Tensor, "batch 4 3"] | None,
    ) -> dict[str, Tensor]:
        actual_b = x_list[0].shape[0]
        if actual_b > self._capture_b:
            # Out-of-range batches fall back to the eager model.
            logger.warning(
                "Captured lifter: B=%d > capture_batch_size=%d; "
                "falling back to eager forward. Bump `capture_batch_size` "
                "in `from_checkpoint(...)` to use the graph for larger batches.",
                actual_b,
                self._capture_b,
            )
            return self._eager_model(x_list, cam_params, Ts_wc, ground_planes)

        # Copy the caller's inputs into the static input buffers.
        for src, dst in zip(x_list, self._static_x, strict=True):
            dst[:actual_b].copy_(src)
        for src, dst in zip(cam_params, self._static_cams, strict=True):
            dst[:actual_b].copy_(src)
        for src, dst in zip(Ts_wc, self._static_Ts, strict=True):
            dst[:actual_b].copy_(src)
        if ground_planes is not None:
            self._static_gp[:actual_b].copy_(ground_planes)
        else:
            self._static_gp.fill_(float("nan"))

        self._graph.replay()

        # Clone so the caller's tensors survive later graph replays.
        return {k: v[:actual_b].clone() for k, v in self._static_out.items()}


class Lifter:
    """Runs the LAMP model on per-person temporal snippets."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        settings: LifterSettings,
    ) -> None:
        # `model` may be `LampNet` or a `_CapturedLampNet` wrapper.
        self._model: torch.nn.Module = model
        self._device: torch.device = device
        self._settings: LifterSettings = settings
        # Used to re-skin fused SMPL parameters at render time.
        self._smpl_model: smplx.SMPL | None = None
        # NaNs mark the floor height as unknown.
        self._ground_planes_unknown: Float32[Tensor, "1 4 3"] = torch.full(
            (1, 4, 3), float("nan"), dtype=torch.float32, device=device
        )
        self._floor_plane: Float32[Tensor, "1 4 3"] | None = None

    def set_floor_plane(self, z: float | None) -> None:
        """Set or clear the fixed floor plane passed to the model."""
        if z is None:
            self._floor_plane = None
            return
        d = _FLOOR_PLANE_EXTENT_M
        corners = torch.tensor(
            [
                [[-d, -d, z], [d, -d, z], [d, d, z], [-d, d, z]],
            ],
            dtype=self._ground_planes_unknown.dtype,
            device=self._device,
        )
        self._floor_plane = corners

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        smpl_model_path: str | Path,
        *,
        device: str = "cuda:0",
        settings: LifterSettings | None = None,
        capture_cuda_graph: bool = False,
        capture_batch_size: int = 8,
    ) -> Lifter:
        """Build a lifter from a `LampNet` state-dict checkpoint."""
        from lamptrack.third_party.lamp.models.model_loader import build_lampnet_from_checkpoint

        resolved = cls.resolve_device(device)
        if settings is None:
            settings = LifterSettings()

        lampnet = build_lampnet_from_checkpoint(
            checkpoint_path=ckpt_path,
            smpl_model_path=smpl_model_path,
            device=resolved,
        )

        lifter = cls(model=lampnet, device=resolved, settings=settings)
        lifter._smpl_model = lampnet.smpl
        lifter._warmup()

        if capture_cuda_graph:
            if resolved.type != "cuda":
                logger.info(
                    "CUDA Graph capture requested but resolved device is %s; using eager model.",
                    resolved,
                )
            else:
                logger.info(
                    "Capturing LampNet into CUDA Graph (B=%d, F=%d, V=%d, K=%d)...",
                    capture_batch_size,
                    lifter.snippet_length,
                    lifter.expected_num_views,
                    _SMPL_KEYPOINT_COUNT,
                )
                t0 = time.perf_counter()
                lifter._model = _CapturedLampNet(
                    eager_model=lifter._model,
                    capture_batch_size=capture_batch_size,
                    snippet_length=lifter.snippet_length,
                    num_views=lifter.expected_num_views,
                    device=resolved,
                )
                logger.info(
                    "CUDA Graph capture: %d ms",
                    int((time.perf_counter() - t0) * 1000),
                )
        return lifter

    def _warmup(
        self,
        max_batch_size: int = 20,
        passes: int = 3,
    ) -> None:
        """Warm CUDA kernels for common batch sizes."""
        if self._device.type != "cuda":
            return
        K = _SMPL_KEYPOINT_COUNT
        T = int(self._settings.snippet_length)
        num_views = _DEFAULT_NUM_VIEWS
        t0 = time.perf_counter()
        try:
            for _ in range(passes):
                for b in range(1, max_batch_size + 1):
                    x = [
                        torch.zeros(b, T, K, 3, device=self._device)
                        for _ in range(num_views)
                    ]
                    cams = [
                        torch.zeros(b, T, 16, device=self._device)
                        for _ in range(num_views)
                    ]
                    Ts = [
                        torch.eye(4, device=self._device)
                        .expand(b, T, 4, 4)
                        .contiguous()
                        for _ in range(num_views)
                    ]
                    # Re-use the cached NaN ground-plane tensor.
                    gp = self._ground_planes_unknown if b == 1 else self._ground_planes_unknown.expand(b, -1, -1).contiguous()
                    with torch.no_grad():
                        _ = self._model(x, cams, Ts, gp)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(
                "Warmed Lifter kernel cache (%d passes x B=1..%d, %.0f ms)",
                passes,
                max_batch_size,
                elapsed_ms,
            )
        except Exception as exc:
            logger.warning("Lifter warmup forward failed (continuing): %s", exc)

    @staticmethod
    def resolve_device(preferred: str) -> torch.device:
        """Return `torch.device(preferred)` or fall back to CPU if CUDA is absent."""
        if preferred.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(preferred)

    @property
    def snippet_length(self) -> int:
        return self._settings.snippet_length

    @property
    def expected_num_views(self) -> int:
        """Number of camera views expected by the loaded public checkpoint."""
        return _DEFAULT_NUM_VIEWS

    @property
    def smpl_faces(self) -> ndarray | None:
        """SMPL face topology `(13776, 3)` int32, or None for non-SMPL lifters."""
        for candidate in (self._model, getattr(self._model, "_eager_model", None)):
            if candidate is None:
                continue
            smpl = getattr(candidate, "smpl", None)
            if smpl is None:
                continue
            faces = getattr(smpl, "faces", None)
            if faces is not None:
                return np.asarray(faces, dtype=np.int32)
        return None

    def forward_smpl_geometry(
        self,
        betas: Float32[ndarray, "batch 10"],
        global_orient_rotmat: Float32[ndarray, "batch 3 3"],
        body_pose_rotmat: Float32[ndarray, "batch 23 3 3"],
        transl: Float32[ndarray, "batch 3"],
    ) -> tuple[Float32[ndarray, "batch 24 3"], Float32[ndarray, "batch 6890 3"]]:
        """Batched SMPL forward over B persons -> world-frame joints and vertices."""
        if self._smpl_model is None:
            raise RuntimeError(
                "forward_smpl_geometry requires the smplx model from "
                "`Lifter.from_checkpoint`; this lifter was constructed without one."
            )
        from lamptrack.third_party.lamp.models.model_utils import smpl_forward_joints_lamp_outputs

        B = int(betas.shape[0])
        with torch.no_grad():
            betas_t = torch.as_tensor(
                betas, dtype=torch.float32, device=self._device
            ).reshape(B, 10)
            body_pose_t = torch.as_tensor(
                body_pose_rotmat, dtype=torch.float32, device=self._device
            ).reshape(B, 1, 23, 3, 3)
            global_orient_t = torch.as_tensor(
                global_orient_rotmat, dtype=torch.float32, device=self._device
            ).reshape(B, 1, 1, 3, 3)
            transl_t = torch.as_tensor(
                transl, dtype=torch.float32, device=self._device
            ).reshape(B, 1, 3)
            joints, verts = smpl_forward_joints_lamp_outputs(
                self._smpl_model,
                betas=betas_t,
                body_pose_rotmat=body_pose_t,
                global_orient_rotmat=global_orient_t,
                transl=transl_t,
                return_verts=True,
            )
            assert verts is not None
            return (
                joints[:, 0].detach().cpu().numpy().astype(np.float32, copy=False),
                verts[:, 0].detach().cpu().numpy().astype(np.float32, copy=False),
            )

    def lift(self, snippet: SnippetData) -> Skeleton:
        """Run the model on `snippet` and return the latest snippet step.

        Outputs stay in the gravity-world frame; persistence and visualization
        consume the same frame directly.
        """
        all_steps = self.lift_all_steps(snippet)
        return all_steps[-1][1]

    def lift_all_steps(self, snippet: SnippetData) -> list[tuple[int, Skeleton]]:
        """Run the model and return all snippet steps, oldest first."""
        return self.lift_all_steps_batched({0: snippet})[0]

    def lift_all_steps_batched(
        self,
        snippets: dict[int, SnippetData],
    ) -> dict[int, list[tuple[int, Skeleton]]]:
        """Run one batched model forward over all lift-eligible snippets."""
        if not snippets:
            return {}

        # Preserve insertion order so per-batch index matches the dict
        # iteration order — keeps the (B, ...) -> per-person output mapping
        # deterministic.
        person_ids: list[int] = list(snippets.keys())

        # All snippets in one batch must share snippet length and view count.
        first = snippets[person_ids[0]]
        snippet_length = len(first.kp2ds_per_view[0])
        num_views = len(first.kp2ds_per_view)
        # Catch caller mistakes before they surface as opaque tensor-shape errors.
        if num_views != self.expected_num_views:
            raise ValueError(
                f"snippet has {num_views} views but the model expects "
                f"{self.expected_num_views}"
            )
        for pid in person_ids:
            snip = snippets[pid]
            if len(snip.kp2ds_per_view) != num_views:
                raise ValueError(
                    f"all batched snippets must share num_views; "
                    f"got {len(snip.kp2ds_per_view)} for pid={pid}, expected {num_views}"
                )
            if len(snip.kp2ds_per_view[0]) != snippet_length:
                raise ValueError(
                    f"all batched snippets must share snippet_length; "
                    f"got {len(snip.kp2ds_per_view[0])} for pid={pid}, "
                    f"expected {snippet_length}"
                )

        x_list, cam_list, Ts_list = self._snippets_to_batched_torch_lists(
            snippets, person_ids
        )
        # Pick the `ground_planes` input. NaNs mean "floor unknown"; a selected
        # plane feeds its height and a "floor known" bit to the model.
        B = len(person_ids)
        base_gp = (
            self._floor_plane
            if self._floor_plane is not None
            else self._ground_planes_unknown
        )
        ground_planes = base_gp if B == 1 else base_gp.expand(B, -1, -1).contiguous()
        with torch.no_grad():
            out = self._model(x_list, cam_list, Ts_list, ground_planes)

        # One host transfer per logical output array, then slice per person.
        skel_w_t: Tensor = out["skel_w"]  # (B, T, 24, 3)
        skel_w_batch = skel_w_t.detach().cpu().numpy().astype(np.float32, copy=False)
        # Optional SMPL body mesh `(B, T, 6890, 3)`.
        verts_w_batch: Float32[ndarray, "batch time 6890 3"] | None = None
        if "verts_w" in out:
            verts_w_batch = (
                out["verts_w"].detach().cpu().numpy().astype(np.float32, copy=False)
            )

        required = {"transl", "global_orient_rotmat", "body_pose_rotmat", "betas"}
        missing = required - set(out.keys())
        if missing:
            raise RuntimeError(
                "Lifter model output is missing required SMPL keys: "
                f"{sorted(missing)}; got keys: {sorted(out.keys())}"
            )
        transl = out["transl"].detach().cpu().numpy().astype(np.float32, copy=False)
        go_rot = (
            out["global_orient_rotmat"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )  # (B, T, 1, 3, 3)
        body_rot = (
            out["body_pose_rotmat"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )  # (B, T, 23, 3, 3)
        B_t, T_t = transl.shape[:2]
        Ts_w_pelvis_batch = np.tile(np.eye(4, dtype=np.float32), (B_t, T_t, 1, 1))
        Ts_w_pelvis_batch[:, :, :3, :3] = go_rot[:, :, 0]
        Ts_w_pelvis_batch[:, :, :3, 3] = transl
        local_joints_batch = np.concatenate([go_rot, body_rot], axis=2)
        shape_batch = out["betas"].detach().cpu().numpy().astype(np.float32, copy=False)

        # Non-finite outputs are rejected downstream in `is_outlier_pose`.
        T = skel_w_batch.shape[1]
        n_joints = skel_w_batch.shape[2]
        if n_joints != _SMPL_JOINT_COUNT:
            raise RuntimeError(f"expected 24 SMPL joints, got {n_joints}")
        results: dict[int, list[tuple[int, Skeleton]]] = {}
        for bi, pid in enumerate(person_ids):
            snippet = snippets[pid]
            snippet_timestamps = snippet.snippet_timestamps_ns
            if len(snippet_timestamps) != T:
                raise ValueError(
                    f"snippet_timestamps_ns len ({len(snippet_timestamps)}) "
                    f"does not match model output T ({T}) for pid={pid}"
                )
            shape_arr = shape_batch[bi]
            out_list: list[tuple[int, Skeleton]] = []
            for tidx in range(T):
                joints_rot_mat = local_joints_batch[bi, tidx]
                # Shape is per-snippet not per-step; copy so each Skeleton
                # owns its own array (fusion's running average mutates in
                # place).
                verts_arr = (
                    verts_w_batch[bi, tidx].copy()
                    if verts_w_batch is not None
                    else np.zeros((0, 3), dtype=np.float32)
                )
                skel = Skeleton(
                    kp_world=skel_w_batch[bi, tidx].copy(),
                    kp_score=np.ones(n_joints, dtype=np.float32),
                    T_world_pelvis=Ts_w_pelvis_batch[bi, tidx].copy(),
                    shape=shape_arr.copy(),
                    joints_rot_mat=joints_rot_mat.copy(),
                    verts_w=verts_arr,
                )
                out_list.append((int(snippet_timestamps[tidx]), skel))
            results[pid] = out_list
        return results

    # Internal helpers

    def _snippets_to_batched_torch_lists(
        self,
        snippets: dict[int, SnippetData],
        person_ids: list[int],
    ) -> tuple[
        list[Float32[Tensor, "batch time 17 3"]],
        list[Float32[Tensor, "batch time params"]],
        list[Float32[Tensor, "batch time 4 4"]],
    ]:
        """Stack per-view per-person tensors for one batched model forward."""
        num_views = len(snippets[person_ids[0]].kp2ds_per_view)
        x_list: list[Float32[Tensor, "batch time 17 3"]] = []
        cam_list: list[Float32[Tensor, "batch time params"]] = []
        Ts_list: list[Float32[Tensor, "batch time 4 4"]] = []
        for v in range(num_views):
            # Build one host stack per view to minimize CPU->GPU transfers.
            kp_np = np.stack(
                [
                    np.ascontiguousarray(snippets[pid].kp2ds_per_view[v])
                    for pid in person_ids
                ],
                axis=0,
            )
            cam_np = np.stack(
                [
                    np.ascontiguousarray(snippets[pid].cam_params_per_view[v])
                    for pid in person_ids
                ],
                axis=0,
            )
            Ts_np = np.stack(
                [
                    np.ascontiguousarray(snippets[pid].Ts_gw_cam_per_view[v])
                    for pid in person_ids
                ],
                axis=0,
            )
            x_list.append(torch.from_numpy(kp_np).to(self._device))
            cam_list.append(torch.from_numpy(cam_np).to(self._device))
            Ts_list.append(torch.from_numpy(Ts_np).to(self._device))
        return x_list, cam_list, Ts_list


# Outlier rejection


def is_outlier_pose(
    skeleton_world: Float32[ndarray, "24 3"],
    T_world_cams: dict[int, Float32[ndarray, "4 4"]],
    T_gravityWorld_world: Float32[ndarray, "4 4"],
    *,
    min_depth: float = 0.5,
    max_depth: float = 5.0,
    leg_vertical_cos_thres: float = 0.5,
) -> bool:
    """Reject lifted SMPL poses with bad depth or non-vertical lower legs."""
    # Degenerate snippets can produce malformed or non-finite skeletons.
    if (
        skeleton_world.shape[0] < _SMPL_JOINT_COUNT
        or not np.isfinite(skeleton_world).all()
    ):
        return True

    # Empty cam map -> no observers, skip the depth check rather than rejecting.
    if not T_world_cams:
        return False

    pelvis = np.asarray(skeleton_world[_SMPL_PELVIS], dtype=np.float32)
    pelvis_h = np.array([pelvis[0], pelvis[1], pelvis[2], 1.0], dtype=np.float32)

    T_gw_inv = invert(T_gravityWorld_world)

    # Ego-centric depth check across cameras.
    is_ego = True
    for T_world_cam in T_world_cams.values():
        # Transform the gravity-world pelvis into this camera frame.
        T_cam_world = invert(T_world_cam)
        T_cam_gw = T_cam_world @ T_gw_inv
        p_cam = (T_cam_gw @ pelvis_h)[:3]
        cam_dist = float(np.linalg.norm(p_cam))
        is_ego_cam = (cam_dist < min_depth) or (cam_dist > max_depth)
        is_ego = is_ego and is_ego_cam

    if is_ego:
        return True

    # Leg-bone vertical check using SMPL knee-to-ankle bones.
    l_upper, l_lower = _SMPL_L_KNEE, _SMPL_L_ANKLE
    r_upper, r_lower = _SMPL_R_KNEE, _SMPL_R_ANKLE
    bone_l = skeleton_world[l_upper].astype(np.float32, copy=False) - skeleton_world[
        l_lower
    ].astype(np.float32, copy=False)
    bone_r = skeleton_world[r_upper].astype(np.float32, copy=False) - skeleton_world[
        r_lower
    ].astype(np.float32, copy=False)
    cos_l = float(bone_l[2]) / max(float(np.linalg.norm(bone_l)), 1e-12)
    cos_r = float(bone_r[2]) / max(float(np.linalg.norm(bone_r)), 1e-12)
    leg_outlier = (cos_l <= leg_vertical_cos_thres) and (
        cos_r <= leg_vertical_cos_thres
    )
    return leg_outlier
