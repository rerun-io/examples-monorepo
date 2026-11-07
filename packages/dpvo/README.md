# DPVO
A streamlined implementation of [DPVO](https://github.com/princeton-vl/DPVO) (Deep Patch Visual Odometry). Using Rerun viewer, Pixi and Gradio for easy use.

<p align="center">
  <img src="media/mini-dpvo.gif" alt="example output" width="720" />
</p>

## Installation

This package lives inside the [examples-monorepo](https://github.com/rerun-io/examples-monorepo). All dependencies are managed by Pixi via the root `pixi.toml`.

```bash
# From the monorepo root:
pixi run -e dpvo --frozen dpvo-demo
```

## Notes

- This package uses the prebuilt `lietorch` package from Pixi instead of building the vendored copy in-tree.
- `lietorch` is ABI-coupled to the exact PyTorch/libtorch build. If `pixi` upgrades `pytorch` without a matching `lietorch` build, imports will fail with undefined-symbol errors from `lietorch_backends`.
- The current working combination is driven by `pyproject.toml`/`pixi.lock`; avoid changing `pytorch`, `libtorch`, `torchvision`, or `lietorch` independently.
- `torch_scatter` was intentionally removed from the runtime dependency set and replaced with local scatter helpers implemented with native PyTorch ops in `dpvo/scatter_utils.py`.
- This was not a cleanup preference. It was a compatibility decision required to move onto the newer PyTorch/CUDA stack needed by the prebuilt `lietorch` package and the current 5090-tested environment.
- In practice: reintroducing `torch_scatter` means re-checking wheel availability and ABI compatibility against the exact `pytorch`/`libtorch`/CUDA combination in this repo. Do not assume a PyPI or PyG wheel will be compatible just because it exists for a nearby Torch/CUDA version.
- That `torch_scatter` replacement is a narrow compatibility workaround for the current app/demo path and has not been broadly validated outside this workflow yet.
- The package still builds its own CUDA extensions (`dpvo._cuda_corr` and `dpvo._cuda_ba`) locally through the `_build-dpvo-cuda-kernels` task, which runs automatically as a `depends-on` step of `dpvo-demo`, `dpvo-app`, and `dpvo-eval-euroc`. You can also build them directly with `pixi run -e dpvo --frozen _build-dpvo-cuda-kernels`.
- Example assets and checkpoints are now fetched through `hf download`, not `huggingface-cli`.
- `pixi run -e dpvo --frozen dpvo-demo` has been validated on an RTX 5090 with the current lockfile.
- Rerun and GUI flows require a valid display session. In headless shells, Rerun will fail unless `DISPLAY`/`XAUTHORITY` are set.

## Extension points

The tracker (`dpvo.dpvo.DPVO`) is plain DPVO with a few seams. With the defaults it runs exactly as before; see
`docs/architecture.md` for the call order.

### `dpvo.hooks`

- `DepthPrior` (default `RandomDepthInit`, plain DPVO's draw): `prepare(dpvo, image_hw, frame)` returns the `disps`
  grid for patchify, `init_patch_depth(dpvo, patches)` writes the new patches' depths before any state changes (a
  `ValueError` from either rejects the frame), `ba_prior(dpvo, kind)` gives the local or global BA a
  `dpvo.fastba.BADepthPrior` (or `None`).
- `GraphPolicy` (default `DefaultGraphPolicy`): `force_keep` vetoes a keyframe cull, `on_cull`/`on_keep` see the
  decision, `extra_loop_edges` appends edges after the proximity loop edges.
- `DPVOObserver`: no-op callbacks `on_frame(dpvo, FrameEvent)` (end of `__call__`, both exits), `on_cull`,
  `on_global_ba` (entry of `run_global_ba`), `on_terminate` and `refine_trajectory(dpvo, poses)`. Observer exceptions
  propagate.
- `add_global_observer(factory) -> remover`: `DPVO.__init__` calls every registered factory with the new tracker and
  keeps the non-`None` observers, for callers that do not build the tracker themselves (MetricTrail's
  `metrictrail.capture`: ViPE builds the tracker).

Pass them as `DPVO(cfg, network, ht, wd, *, device=torch.device("cuda"), depth_prior=None, graph_policy=None,
observers=())`. `device` places the network and every buffer; the native ops then come from `dpvo.kernels`.

### `dpvo.kernels`

Kernel ops dispatch on the input tensor's `device.type`, with no copies and no syncs. CUDA is built in: the
`dpvo._cuda_corr` and `dpvo._cuda_ba` extensions (`_build-dpvo-cuda-kernels`), imported on the first CUDA lookup.
Another backend package registers itself at import time:

```python
import dpvo.kernels

dpvo.kernels.register("mps", corr=my_corr_module, ba=my_ba_module)
```

A backend is any object (a module of functions is enough) that has the extension signatures, the
`CorrKernels` and `BAKernels` protocols:

| Group | Op | Signature |
| --- | --- | --- |
| corr | `forward` | `(fmap1, fmap2, coords, ii, jj, radius: int) -> (corr,)` |
| corr | `backward` | `(fmap1, fmap2, coords, ii, jj, grad, radius: int) -> (fmap1_grad, fmap2_grad)` |
| corr | `patchify_forward` | `(net, coords, radius: int) -> (patches,)` |
| corr | `patchify_backward` | `(net, coords, grad, radius: int) -> (net_grad,)` |
| ba | `forward` | `(poses, patches, intrinsics, target, weight, lmbda, depth_prior, depth_prior_weight, depth_prior_lambda: float, depth_prior_huber: float, ii, jj, kk, M: int, t0: int, t1: int, iterations: int, eff_impl: bool) -> ()`; updates `poses` and `patches` in place |
| ba | `neighbors` | `(ii, jj) -> (ix, jx)` |
| ba | `reproject` | `(poses, patches, intrinsics, ii, jj, kk) -> coords` |
| ba | `solve_system` | `(J_Ginv_i, J_Ginv_j, ii, jj, res, ep: float, lm: float, freen: int) -> (delta,)` |

`corr_kernels(device)` and `ba_kernels(device)` return the registered group; an unregistered device type raises
`LookupError`.

### `dpvo.metric`

`MetricDPVO(cfg: MetricDPVOConfig, network, ht, wd, *, device, observers=())` is ViPE MetricTrail's DPVO on these
hooks: a per-frame metric-depth prior (patch depths from the depth map, a log-inverse-depth unary in the local and
global BA after a warm-up), proximity loop edges into a metric global BA, a keyframe-cull guard, reverse loop edges,
culled-frame interpolation and a final re-measured global BA. `__call__(tstamp, image, intrinsics, metric_depth=None,
metric_mask=None)` needs the depth; `metric_prior_diagnostics()` returns the counters ViPE checks.
`MetricDPVOConfig.from_yacs(block)` / `to_yacs()` map ViPE's 43 uppercase yacs keys (`YACS_DEFAULTS`); the presets
`metrictrail_local`, `metrictrail_gba` and `metrictrail_gba_refine` run ViPE's operations in ViPE's order.
MetricTrail (`packages/metrictrail`) runs it through `recipes/vipe-src`'s `vipe.metrictrail.dpvo` shims.

## Demo
Hosted Demos can be found in either lightning studio or huggingface spaces

<a target="_blank" href="https://lightning.ai/pablovelagomez1/studios/mini-dpvo">
  <img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio"/>
</a>

<a href='https://huggingface.co/spaces/pablovela5620/MiniDPVO'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

To run the full demo pipeline (from the monorepo root):
```
pixi run -e dpvo --frozen dpvo-demo
```

To launch the Gradio web UI:
```
pixi run -e dpvo --frozen dpvo-app
```

Look in the root `pixi.toml` file to see exactly what each command does under `[feature.dpvo.tasks]`.

## Acknowledgements
Original Code and paper from [DPVO](https://github.com/princeton-vl/DPVO)
```
@article{teed2023deep,
  title={Deep Patch Visual Odometry},
  author={Teed, Zachary and Lipson, Lahav and Deng, Jia},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
