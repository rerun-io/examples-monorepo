"""Minimal 3DGS trainer: gsplat MCMC on one Rerun-catalog ARKitScenes segment.

Views come from the preloaded Rerun dataloader adapter, Gaussians initialize
from the segment's gt mesh, and training streams `GaussianSplats3D` (the 0.36
archetype) plus loss/PSNR curves to the Rerun viewer on an `iteration`
timeline, mirroring the gsplat-rust-renderer training dashboard layout.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from gsplat import MCMCStrategy, export_splats, rasterization
from jaxtyping import Bool, Float32, Float64, UInt8
from numpy import ndarray
from ppisp import PPISP, PPISPConfig
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig, log_pinhole
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure

from rerun_gsplat.apis.mesh_init import ColoredPoints, GaussianInit, gaussians_from_points, load_gt_mesh
from rerun_gsplat.apis.segment_views import SegmentViewsConfig, SplatView, load_segment_views

SH_C0: float = 0.28209479177387814
"""Degree-0 spherical harmonics basis constant."""
SH_DEGREE: int = 3
"""Spherical-harmonics degree. Fixed: Rerun's `sh_coefficients` component is
exactly the 15 degree-1..3 RGB coefficients, so other degrees can't be logged."""
SPLATS_ENTITY: str = "world/splats"
"""Entity path for the trained Gaussians (matches the rust dashboard)."""


@dataclass(frozen=True, slots=True)
class Config:
    """Train a splat on one catalog segment."""

    rr_config: RerunTyroConfig
    """Rerun viewer/save/headless wiring."""
    views: SegmentViewsConfig = field(default_factory=SegmentViewsConfig)
    """Catalog source for training views."""
    max_steps: int = 1_000
    """Total optimization steps (short iteration default; the v1 gate run used 7k)."""
    cap_max: int = 1_000_000
    """MCMC maximum Gaussian count."""
    lr_decay_steps: int | None = None
    """Steps over which the means lr decays to 1% (defaults to ``max_steps``).
    Debug knob: lets a short run wear a long run's lr/noise profile."""
    noise_lr: float = 5e5
    """MCMC position-noise learning rate (gsplat default 5e5)."""
    init_max_points: int = 200_000
    """Cap on gt-mesh vertices used for initialization."""
    holdout_every: int = 10
    """Every Nth view goes to the validation split."""
    log_every: int = 500
    """Steps between splat snapshots, eval renders, and metric points."""
    eval_views_logged: int = 2
    """Validation views whose GT/render pairs stream to the viewer."""
    frustum_image_plane_distance: float = 0.15
    """Image-plane depth of the static training-camera frustums, meters."""
    frustum_jpeg_quality: int = 75
    """JPEG quality for the frustum image planes."""
    ssim_lambda: float = 0.2
    """Weight of the DSSIM term (L1 gets 1 - this)."""
    depth_lambda: float = 0.1
    """Weight of the metric depth L1 term (0 disables depth supervision)."""
    opacity_reg: float = 0.001
    """MCMC opacity regularization weight. gsplat's example default (0.01) death-spirals
    RGB-only runs here: it sinks opacities, MCMC's opacity-gated noise then ejects the
    transparent splats from the scene (verified: ultrawide 30k collapsed to 10 dB; at
    0.001 the same config holds 22 dB). Depth-supervised runs tolerate either."""
    scale_reg: float = 0.01
    """MCMC scale regularization weight."""
    use_ppisp: bool = True
    """Learn photometric compensation (nv-tlabs PPISP) applied to the rendered image
    before the photometric loss: per-frame exposure + color homography, per-camera
    vignetting + CRF. Absorbs the capture's auto-exposure / white-balance drift so
    the splats don't have to."""
    ppisp_reg_scale: float = 1.0
    """Global multiplier on PPISP's six regularization weights. 1.0 keeps the
    nv-tlabs defaults (anchor the compensation near identity); LichtFeld-Studio
    ships the same ratios scaled by 0.001."""
    ppisp_camera_terms: bool = True
    """Ablation knob: False passes camera_idx=None so only the per-frame
    exposure/color terms train (no vignetting/CRF; eval is then fully raw)."""
    ply_out: Path = Path("/tmp/rerun-gsplat/splats.ply")
    """Where the final splat PLY is written."""
    seed: int = 0
    """Sampling seed for view order."""


def send_dashboard_blueprint(config: Config) -> None:
    """Mirror the gsplat-rust-renderer training dashboard layout."""
    eval_pairs: list[rrb.Horizontal] = [
        rrb.Horizontal(
            rrb.Spatial2DView(origin=f"eval/view_{index}/ground_truth", name="Ground truth", contents=["$origin/**"]),
            rrb.Spatial2DView(origin=f"eval/view_{index}/render", name="Render", contents=["$origin/**"]),
            name=f"Eval view {index}",
        )
        for index in range(config.eval_views_logged)
    ]
    eval_views: rrb.Grid = rrb.Grid(*eval_pairs, grid_columns=1, name="Eval views")
    view3d: rrb.Spatial3DView = rrb.Spatial3DView(origin="world", contents=["world/**"], name="Splats")
    graph_views: list[rrb.TimeSeriesView] = [
        rrb.TimeSeriesView(origin="loss", contents=["loss/**"], name="Loss"),
        rrb.TimeSeriesView(origin="psnr", contents=["psnr/**"], name="PSNR"),
        rrb.TimeSeriesView(origin="splats", contents=["splats/**"], name="Splats"),
    ]
    if config.use_ppisp:
        graph_views.append(rrb.TimeSeriesView(origin="ppisp", contents=["ppisp/**"], name="PPISP"))
    graphs: rrb.Horizontal = rrb.Horizontal(*graph_views)
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(view3d, eval_views, column_shares=[3.0, 2.0]),
                graphs,
                row_shares=[4.0, 1.0],
            ),
            rrb.BlueprintPanel(state="collapsed"),
            rrb.SelectionPanel(state="collapsed"),
            rrb.TimePanel(state="collapsed"),
            auto_layout=False,
            auto_views=False,
        )
    )


def log_camera_frustums(views: list[SplatView], image_plane_distance: float, jpeg_quality: int) -> None:
    """Log every training camera as a static frustum with its image plane.

    Uses simplecv's ``log_pinhole`` (poses are RDF/OpenCV, matching ingest's
    baked convention) and puts the JPEG-compressed view frame on the image
    plane, like the gsplat-rust-renderer training-camera ring.
    """
    for index, view in enumerate(views):
        # simplecv expects float64 camera parameters (its k_matrix is float64).
        cam_t_world: Float64[ndarray, "4 4"] = view.cam_t_world_44.numpy().astype(np.float64)
        intrinsics: Intrinsics = Intrinsics.from_k_matrix(
            camera_conventions="RDF",
            k_matrix=view.k_33.numpy().astype(np.float64),
            height=int(view.rgb_hwc.shape[0]),
            width=int(view.rgb_hwc.shape[1]),
        )
        camera: PinholeParameters = PinholeParameters(
            name=f"view_{index:03d}",
            extrinsics=Extrinsics(cam_R_world=cam_t_world[:3, :3], cam_t_world=cam_t_world[:3, 3]),
            intrinsics=intrinsics,
        )
        cam_path: str = f"world/cameras/view_{index:03d}"
        log_pinhole(camera, cam_log_path=Path(cam_path), image_plane_distance=image_plane_distance, static=True)
        rr.log(f"{cam_path}/pinhole/image", rr.Image(view.rgb_hwc.numpy()).compress(jpeg_quality=jpeg_quality), static=True)


def log_splats(params: torch.nn.ParameterDict, sh_active: bool) -> None:
    """Log the current Gaussians as the 0.36 `GaussianSplats3D` archetype."""
    with torch.no_grad():
        quats_wxyz: Float32[Tensor, "n 4"] = torch.nn.functional.normalize(params["quats"], dim=-1)
        rgb_n3: Float32[Tensor, "n 3"] = (params["sh0"][:, 0, :] * SH_C0 + 0.5).clamp(0.0, 1.0)
        alpha_n: Float32[Tensor, "n"] = params["opacities"].sigmoid()
        colors_n4: UInt8[ndarray, "n 4"] = (torch.cat([rgb_n3, alpha_n[:, None]], dim=1) * 255.0).to(torch.uint8).cpu().numpy()
        rr.log(
            SPLATS_ENTITY,
            rr.GaussianSplats3D(
                centers=params["means"].cpu().numpy(),
                quaternions=quats_wxyz[:, [1, 2, 3, 0]].cpu().numpy(),
                scales=params["scales"].exp().cpu().numpy(),
                colors=colors_n4,
                # Rerun stores SH coefficients as float16; cast on-GPU to halve the transfer.
                sh_coefficients=params["shN"].to(torch.float16).cpu().numpy() if sh_active else None,
            ),
        )


def render(params: torch.nn.ParameterDict, view: SplatView, sh_degree_active: int, device: str) -> tuple[Float32[Tensor, "h w 3"], Float32[Tensor, "h w"], dict]:
    """Rasterize the current Gaussians through one view: RGB + expected depth."""
    height: int = int(view.rgb_hwc.shape[0])
    width: int = int(view.rgb_hwc.shape[1])
    outputs: tuple[Tensor, Tensor, dict] = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=torch.cat([params["sh0"], params["shN"]], dim=1),
        sh_degree=sh_degree_active,
        viewmats=view.cam_t_world_44[None].to(device),
        Ks=view.k_33[None].to(device),
        width=width,
        height=height,
        render_mode="RGB+ED",
    )
    return outputs[0][0, :, :, :3], outputs[0][0, :, :, 3], outputs[2]


def sample_rendered_depth(rendered_depth_hw: Float32[Tensor, "h w"], view: SplatView, device: str) -> Float32[Tensor, "dh dw"]:
    """Bilinearly sample the rendered depth at the sensor depth map's pixel centers.

    Both pinholes share the camera pose, so a depth pixel maps to RGB pixel
    coordinates through the exact affine ``K_rgb @ K_lowres^-1`` (the corpus's
    lowres principal point differs from a pure rescale by ~0.4 px, so a plain
    resize would be subtly misregistered).
    """
    depth_height: int = int(view.depth_m_hw.shape[0])
    depth_width: int = int(view.depth_m_hw.shape[1])
    height: int = int(rendered_depth_hw.shape[0])
    width: int = int(rendered_depth_hw.shape[1])
    k_rgb: Float32[Tensor, "3 3"] = view.k_33.to(device)
    k_lo: Float32[Tensor, "3 3"] = view.k_lowres_33.to(device)
    u_lo: Float32[Tensor, "dw"] = torch.arange(depth_width, device=device, dtype=torch.float32)
    v_lo: Float32[Tensor, "dh"] = torch.arange(depth_height, device=device, dtype=torch.float32)
    u_rgb: Float32[Tensor, "dw"] = (u_lo - k_lo[0, 2]) * (k_rgb[0, 0] / k_lo[0, 0]) + k_rgb[0, 2]
    v_rgb: Float32[Tensor, "dh"] = (v_lo - k_lo[1, 2]) * (k_rgb[1, 1] / k_lo[1, 1]) + k_rgb[1, 2]
    # grid_sample normalized coordinates (align_corners=False): pixel center i -> (2i + 1)/size - 1.
    grid_x: Float32[Tensor, "dh dw"] = ((2.0 * u_rgb + 1.0) / width - 1.0).expand(depth_height, depth_width)
    grid_y: Float32[Tensor, "dh dw"] = ((2.0 * v_rgb + 1.0) / height - 1.0)[:, None].expand(depth_height, depth_width)
    grid: Float32[Tensor, "1 dh dw 2"] = torch.stack([grid_x, grid_y], dim=-1)[None]
    sampled: Float32[Tensor, "1 1 dh dw"] = torch.nn.functional.grid_sample(
        rendered_depth_hw[None, None], grid, mode="bilinear", align_corners=False
    )
    return sampled[0, 0]


def masked_depth_l1(rendered_depth_hw: Float32[Tensor, "h w"], view: SplatView, device: str) -> Float32[Tensor, ""]:
    """Mean absolute metric-depth error over the view's valid sensor pixels."""
    sampled: Float32[Tensor, "dh dw"] = sample_rendered_depth(rendered_depth_hw, view, device)
    valid: Bool[Tensor, "dh dw"] = view.depth_valid_hw.to(device)
    return ((sampled - view.depth_m_hw.to(device)).abs() * valid).sum() / valid.sum().clamp(min=1)


def eval_psnr(
    params: torch.nn.ParameterDict, val_views: list[SplatView], config: Config, step: int, device: str, ppisp_module: PPISP | None
) -> float:
    """Mean PSNR over the validation split; also logs the first GT/render pairs.

    With PPISP active, the primary PSNR compares the compensated render
    (``frame_idx=-1``: identity per-frame terms + learned per-camera
    vignetting/CRF — holdout frames have no learned per-frame parameters) and
    the uncompensated PSNR is logged alongside as ``psnr/eval_raw``.
    PSNR accumulates on-device (one host sync at the end) and logged renders
    quantize to uint8 on the GPU before transfer.
    """
    psnr_sum: Float32[Tensor, ""] = torch.zeros((), device=device)
    raw_psnr_sum: Float32[Tensor, ""] = torch.zeros((), device=device)
    depth_mae_sum: Float32[Tensor, ""] = torch.zeros((), device=device)
    depth_view_count: int = 0
    per_camera_psnr_sums: dict[int, Float32[Tensor, ""]] = {}
    per_camera_counts: dict[int, int] = {}
    with torch.no_grad():
        for index, view in enumerate(val_views):
            rendered: Float32[Tensor, "h w 3"]
            rendered_depth: Float32[Tensor, "h w"]
            rendered, rendered_depth, _ = render(params, view, SH_DEGREE, device)
            target: Float32[Tensor, "h w 3"] = view.rgb_hwc.to(device).float() / 255.0
            if ppisp_module is not None:
                raw_mse: Float32[Tensor, ""] = torch.mean((rendered.clamp(0.0, 1.0) - target) ** 2).clamp(min=1e-10)
                raw_psnr_sum = raw_psnr_sum - 10.0 * torch.log10(raw_mse)
                rendered = ppisp_module(rendered.contiguous(), camera_idx=view.camera_index if config.ppisp_camera_terms else None, frame_idx=-1)
            rendered = rendered.clamp(0.0, 1.0)
            mse: Float32[Tensor, ""] = torch.mean((rendered - target) ** 2).clamp(min=1e-10)
            view_psnr: Float32[Tensor, ""] = -10.0 * torch.log10(mse)
            psnr_sum = psnr_sum + view_psnr
            per_camera_psnr_sums[view.camera_index] = per_camera_psnr_sums.get(view.camera_index, torch.zeros((), device=device)) + view_psnr
            per_camera_counts[view.camera_index] = per_camera_counts.get(view.camera_index, 0) + 1
            if bool(view.depth_valid_hw.any()):
                depth_mae_sum = depth_mae_sum + masked_depth_l1(rendered_depth, view, device)
                depth_view_count += 1
            if index < config.eval_views_logged:
                rr.log(f"eval/view_{index}/render", rr.Image((rendered * 255.0).to(torch.uint8).cpu().numpy()))
    mean_psnr: float = float(psnr_sum) / len(val_views)
    rr.log("psnr/eval", rr.Scalars(mean_psnr))
    if ppisp_module is not None:
        rr.log("psnr/eval_raw", rr.Scalars(float(raw_psnr_sum) / len(val_views)))
    per_camera_note: str = ""
    if len(per_camera_counts) > 1:
        # camera_index 0/1 = wide/ultrawide by construction of camera="both" loads.
        for camera_index, count in sorted(per_camera_counts.items()):
            camera_name: str = ("wide", "ultrawide")[camera_index]
            camera_psnr: float = float(per_camera_psnr_sums[camera_index]) / count
            rr.log(f"psnr/eval_{camera_name}", rr.Scalars(camera_psnr))
            per_camera_note += f", {camera_name} {camera_psnr:.2f} dB"
    depth_note: str = ""
    if depth_view_count:
        mean_depth_mae: float = float(depth_mae_sum) / depth_view_count
        rr.log("loss/depth_eval_mae_m", rr.Scalars(mean_depth_mae))
        depth_note = f", depth MAE {mean_depth_mae * 100.0:.1f} cm"
    print(f"step {step}: val PSNR {mean_psnr:.2f} dB{per_camera_note}{depth_note} over {len(val_views)} views")
    return mean_psnr


def main(config: Config) -> float:
    """Load data, train, log to Rerun, save the PLY; returns the final val PSNR."""
    device: str = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("rerun-gsplat training requires CUDA")

    views: list[SplatView] = load_segment_views(config.views)
    val_views: list[SplatView] = views[:: config.holdout_every]
    train_views: list[SplatView] = [view for index, view in enumerate(views) if index % config.holdout_every != 0]
    print(f"{len(train_views)} train / {len(val_views)} val views")

    mesh_points: ColoredPoints = load_gt_mesh(config.views)
    init: GaussianInit = gaussians_from_points(mesh_points, max_points=config.init_max_points)
    sh_coeff_count: int = (SH_DEGREE + 1) ** 2
    sh0_init: Float32[Tensor, "n 1 3"] = ((init.rgbs_n3 - 0.5) / SH_C0)[:, None, :]
    params: torch.nn.ParameterDict = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(init.means_n3.to(device)),
            "scales": torch.nn.Parameter(init.log_scales_n3.to(device)),
            "quats": torch.nn.Parameter(init.quats_n4.to(device)),
            "opacities": torch.nn.Parameter(init.logit_opacities_n.to(device)),
            "sh0": torch.nn.Parameter(sh0_init.to(device)),
            "shN": torch.nn.Parameter(torch.zeros(len(init.means_n3), sh_coeff_count - 1, 3, device=device)),
        }
    )

    # Scene scale normalizes the means learning rate (gsplat convention).
    cam_centers: Float32[ndarray, "v 3"] = np.stack([np.linalg.inv(v.cam_t_world_44.numpy())[:3, 3] for v in views])
    scene_scale: float = float(np.linalg.norm(cam_centers - cam_centers.mean(axis=0), axis=1).max()) * 1.1
    learning_rates: dict[str, float] = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    optimizers: dict[str, torch.optim.Optimizer] = {
        name: torch.optim.Adam([params[name]], lr=learning_rate, eps=1e-15) for name, learning_rate in learning_rates.items()
    }
    means_scheduler: torch.optim.lr_scheduler.ExponentialLR = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / (config.lr_decay_steps or config.max_steps))
    )
    strategy: MCMCStrategy = MCMCStrategy(cap_max=config.cap_max, noise_lr=config.noise_lr)
    strategy.check_sanity(params, optimizers)
    strategy_state: dict = strategy.initialize_state()

    # PPISP compensates the render for the capture's photometric variation before
    # the loss. The controller stays off: holdout eval then uses frame_idx=-1
    # (identity per-frame terms + learned camera terms), and training never enters
    # the scene-freeze distillation phase that would fight the MCMC strategy.
    ppisp_module: PPISP | None = None
    ppisp_optimizers: list[torch.optim.Optimizer] = []
    ppisp_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
    if config.use_ppisp:
        reg: float = config.ppisp_reg_scale
        ppisp_module = PPISP(
            num_cameras=2 if config.views.camera == "both" else 1,
            num_frames=len(train_views),
            config=PPISPConfig(
                use_controller=False,
                exposure_mean=1.0 * reg,
                vig_center=0.02 * reg,
                vig_channel=0.1 * reg,
                vig_non_pos=0.01 * reg,
                color_mean=1.0 * reg,
                crf_channel=0.1 * reg,
                # Mirror the means-lr schedule length instead of ppisp's fixed
                # 30k decay horizon, so short runs decay too.
                scheduler_decay_max_steps=config.lr_decay_steps or config.max_steps,
            ),
        )
        ppisp_optimizers = ppisp_module.create_optimizers()
        ppisp_schedulers = ppisp_module.create_schedulers(ppisp_optimizers, config.max_steps)

    send_dashboard_blueprint(config)
    rr.set_time("iteration", sequence=0)
    log_camera_frustums(views, image_plane_distance=config.frustum_image_plane_distance, jpeg_quality=config.frustum_jpeg_quality)
    for index, view in enumerate(val_views[: config.eval_views_logged]):
        rr.log(f"eval/view_{index}/ground_truth", rr.Image(view.rgb_hwc.numpy()), static=True)
    log_splats(params, sh_active=False)

    rng: np.random.Generator = np.random.default_rng(seed=config.seed)
    final_psnr: float = 0.0
    for step in range(config.max_steps):
        rr.set_time("iteration", sequence=step)
        view_index: int = int(rng.integers(len(train_views)))
        view: SplatView = train_views[view_index]
        sh_degree_active: int = min(step // 1000, SH_DEGREE)
        rendered: Float32[Tensor, "h w 3"]
        rendered_depth: Float32[Tensor, "h w"]
        info: dict
        rendered, rendered_depth, info = render(params, view, sh_degree_active, device)
        if ppisp_module is not None:
            # ppisp flattens with .view() before .contiguous(), so the sliced
            # RGB plane must be made contiguous here. Its CRF stage clamps to
            # [0, 1] internally.
            rendered = ppisp_module(rendered.contiguous(), camera_idx=view.camera_index if config.ppisp_camera_terms else None, frame_idx=view_index)
        target: Float32[Tensor, "h w 3"] = view.rgb_hwc.to(device).float() / 255.0

        l1_loss: Tensor = torch.abs(rendered - target).mean()
        ssim_result: Tensor | tuple[Tensor, Tensor] = structural_similarity_index_measure(
            rendered.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1.0
        )
        assert isinstance(ssim_result, Tensor)
        ssim_value: Tensor = ssim_result
        loss: Tensor = (1.0 - config.ssim_lambda) * l1_loss + config.ssim_lambda * (1.0 - ssim_value)
        depth_loss: Tensor = torch.zeros((), device=device)
        if config.depth_lambda > 0.0 and bool(view.depth_valid_hw.any()):
            depth_loss = masked_depth_l1(rendered_depth, view, device)
            loss = loss + config.depth_lambda * depth_loss
        loss = loss + config.opacity_reg * params["opacities"].sigmoid().mean()
        loss = loss + config.scale_reg * params["scales"].exp().mean()
        if ppisp_module is not None:
            loss = loss + ppisp_module.get_regularization_loss()

        strategy.step_pre_backward(params=params, optimizers=optimizers, state=strategy_state, step=step, info=info)
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for ppisp_optimizer in ppisp_optimizers:
            ppisp_optimizer.step()
            ppisp_optimizer.zero_grad(set_to_none=True)
        for ppisp_scheduler in ppisp_schedulers:
            ppisp_scheduler.step()
        means_scheduler.step()
        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=strategy_state, step=step, info=info, lr=float(means_scheduler.get_last_lr()[0])
        )

        if step % config.log_every == 0 or step == config.max_steps - 1:
            rr.log("loss/total", rr.Scalars(float(loss)))
            if config.depth_lambda > 0.0:
                rr.log("loss/depth_m", rr.Scalars(float(depth_loss)))
            rr.log("splats/num_splats", rr.Scalars(len(params["means"])))
            # Parameter health: a diverging run shows up here before PSNR dies.
            with torch.no_grad():
                nan_count: int = int(sum(torch.isnan(p).sum() for p in params.values()))
                max_scale_m: float = float(params["scales"].max().exp())
                mean_opacity: float = float(params["opacities"].sigmoid().mean())
                max_mean_abs: float = float(params["means"].abs().max())
            rr.log("splats/max_scale_m", rr.Scalars(max_scale_m))
            rr.log("splats/mean_opacity", rr.Scalars(mean_opacity))
            if ppisp_module is not None:
                with torch.no_grad():
                    exposure_span_ev: float = float(ppisp_module.exposure_params.max() - ppisp_module.exposure_params.min())
                    color_latent_mean_abs: float = float(ppisp_module.color_params.abs().mean())
                rr.log("ppisp/exposure_span_ev", rr.Scalars(exposure_span_ev))
                rr.log("ppisp/color_latent_mean_abs", rr.Scalars(color_latent_mean_abs))
            print(
                f"step {step}: loss {float(loss):.4f} (l1 {float(l1_loss):.4f}, depth {float(depth_loss):.4f}), "
                f"nan {nan_count}, max_scale {max_scale_m:.3f} m, mean_opacity {mean_opacity:.3f}, max_mean {max_mean_abs:.1f} m"
            )
            log_splats(params, sh_active=sh_degree_active > 0)
            final_psnr = eval_psnr(params, val_views, config, step, device, ppisp_module)

    config.ply_out.parent.mkdir(parents=True, exist_ok=True)
    export_splats(
        means=params["means"].detach(),
        scales=params["scales"].detach(),
        quats=params["quats"].detach(),
        opacities=params["opacities"].detach(),
        sh0=params["sh0"].detach(),
        shN=params["shN"].detach(),
        format="ply",
        save_to=str(config.ply_out),
    )
    if ppisp_module is not None:
        # The PLY holds uncompensated radiance; the photometric model rides alongside.
        ppisp_out: Path = config.ply_out.with_suffix(".ppisp.pt")
        torch.save(ppisp_module.state_dict(), ppisp_out)
        print(f"ppisp state (not baked into the PLY) saved to {ppisp_out}")
    print(f"final: {len(params['means'])} splats, val PSNR {final_psnr:.2f} dB, PLY at {config.ply_out}")
    return final_psnr
