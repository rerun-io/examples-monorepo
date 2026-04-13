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
- The package still builds its own CUDA extensions (`dpvo._cuda_corr` and `dpvo._cuda_ba`) locally through `pixi run post-install`.
- Example assets and checkpoints are now fetched through `hf download`, not `huggingface-cli`.
- `pixi run dpvo-demo` has been validated on an RTX 5090 with the current lockfile.
- Rerun and GUI flows require a valid display session. In headless shells, Rerun will fail unless `DISPLAY`/`XAUTHORITY` are set.

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
