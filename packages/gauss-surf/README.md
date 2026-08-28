# gauss-surf

Surface-aligned Gaussian splatting for ARKitScenes, end to end from a Rerun
catalog: select sharp frames, infer MoGe v2 normals, rectify the ultrawide
camera with mesh depth, train a direct [gsplat](https://github.com/nerfstudio-project/gsplat)
recipe with PGSR-style planar regularization, and publish the splat plus its
rendered depth/normal layers back to the catalog — one segment in about four
minutes on an RTX 5090.

Uses [Rerun](https://rerun.io/) for data loading and visualization and
[Pixi](https://pixi.sh/latest/) for installation. Everything reads from and
writes to catalog layers produced by the sibling
[`arkitscenes-download`](../arkitscenes-download) and
[`prompt-da`](../prompt-da) packages.

<p align="center">
  <img src="media/gauss-surf-github.gif" alt="Splat chase-cam with live video, splat depth, and splat normal layers" width="960" />
</p>

## Quickstart

```bash
# 1. Serve the catalog (terminal 1) and register a segment's layers into it
pixi run -e gauss-surf gauss-surf-serve
pixi run -e gauss-surf gauss-surf-register-segment --video-id 48018538 --default-orientation landscape

# 2. Generate the PromptDA depth layer for the segment (once per segment)
pixi run -e prompt-da-stream prompt-da-arkitscenes-register --video-id 48018538 \
  --catalog-url rerun+http://127.0.0.1:51236 --dataset-name arkitscenes-v2

# 3. Run the whole pipeline: frame selection -> MoGe normals -> ultrawide signals -> training
pixi run gauss-surf-segment-pipeline 48018538 arkitscenes-v2 my-first-run

# 4. Open the viewer on the result
rerun "rerun+http://127.0.0.1:51236/dataset/<dataset-id>?segment_id=48018538"
```

Each pipeline stage is also its own task (`gauss-surf-frame-selection`,
`gauss-surf-moge-normals`, `gauss-surf-ultrawide-signals`,
`gauss-surf-train-gsplat`) — `pixi task list -e gauss-surf` lists everything.

## Corpus run: frame selection + chosen-frame PromptDA

The first two stages run over many segments in one process each, writing one
layer RRD per segment straight into the corpus layer tree and registering it.
Existing outputs are skipped (`--force` regenerates); a segment that fails is
logged and the batch continues, so a run can be stopped and resumed at will.

```bash
CATALOG=rerun+http://<host>:51235
NAS=/mnt/nas/datasets/arkitscenes/arkitscenes.2026.07.22   # layer-major corpus root

# 1. Queue: one segment id per line (e.g. every landscape gt-clean segment)
pixi run -e prompt-da-stream python -c "
from rerun.catalog import CatalogClient; import pyarrow as pa
t = pa.Table.from_batches(CatalogClient('$CATALOG').get_dataset('arkitscenes-v2-gt-clean').segment_table().collect())
print('\n'.join(sorted(str(r['rerun_segment_id']) for r in t.to_pylist()
      if int(r['property:capture:orientation_quarter_turns_ccw'][0]) == 0)))" > queue.txt

# 2. Frame selection (~6 s/segment) — must finish before PromptDA reads its chosen frames
pixi run -e gauss-surf gauss-surf-frame-selection --video-ids-file queue.txt \
  --catalog-url $CATALOG --output-dir $NAS/frame_selection

# 3. PromptDA at the chosen frames (~35-50 s/segment; fine to run concurrently, it trails stage 2)
pixi run -e prompt-da-stream prompt-da-arkitscenes-chosen --video-ids-file queue.txt \
  --catalog-url $CATALOG --output-dir $NAS/promptda

# 4. Sub-datasets are snapshots: rebuild gt-clean so it sees the new layers
pixi run -e arkitscenes-download arkitscenes-download-gt-subdataset --catalog-url $CATALOG
```

Run long batches in a tmux session started over `ssh <host>` (a tmux server
started from an agent shell dies with that shell). The catalog is in-memory: if
the server restarts, re-register from the layer tree
(`register_catalog.py --rrd-dir $NAS`, then `dataset.register(<layer>/*.rrd)` for
the derived layers) — the RRDs on disk are the durable product. Segments with
fewer than `--min-chosen` sharp windows (default 200) are rejected by design and
get no PromptDA layer.

## What the trainer does

- **Catalog-native input**: all cameras, poses, video, PromptDA depth, and MoGe
  normals stream from the Rerun dataset through a GPU-resident quantized cache
  (NVDEC-decoded RGB, uint16-mm depth, uint8 normals).
- **Two-call gsplat training**: 7k iterations with depth, normal-prior,
  normal-consistency, and flatness losses on top of the standard photometric
  term; densification runs on gsplat's `DefaultStrategy` with absgrad.
- **Fused publication**: after training, one inference-only render pass emits an
  eight-channel signal image per camera (RGB, depth, normals, alpha) and writes
  three catalog layers — `splat`, `splat_depth`, `splat_triage` — registered
  in-process, no intermediate exports.
- **Holdout evaluation**: PSNR/SSIM plus depth-MAE triage against ARKit and
  PromptDA depth on frames the trainer never saw.

## Acknowledgements

Built on [gsplat](https://github.com/nerfstudio-project/gsplat), with geometry
losses adapted from [PGSR](https://github.com/zju3dv/PGSR), normals from
[MoGe](https://github.com/microsoft/MoGe), and data from
[ARKitScenes](https://github.com/apple/ARKitScenes).

```bibtex
@article{ye2024gsplat,
  title={gsplat: An Open-Source Library for Gaussian Splatting},
  author={Ye, Vickie and Li, Ruilong and Kerr, Justin and Turkulainen, Matias and Yi, Brent and Pan, Zhuoyang and Seiskari, Otto and Ye, Jianbo and Hu, Jeffrey and Tancik, Matthew and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2409.06765},
  year={2024}
}

@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}

@article{wang2025moge2,
  title={MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details},
  author={Wang, Ruicheng and Xu, Sicheng and Yang, Yue and Dai, Yizhong and Xiu, Wenxiang and Tong, Xin and Yang, Jiaolong},
  journal={arXiv preprint arXiv:2507.02546},
  year={2025}
}

@inproceedings{baruch2021arkitscenes,
  title={ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data},
  author={Baruch, Gilad and Chen, Zhuoyuan and Dehghan, Afshin and Dimry, Tal and Feigin, Yuri and Fu, Peter and Gebauer, Thomas and Joffe, Brandon and Kurz, Daniel and Schwartz, Arik and Shulman, Elad},
  booktitle={NeurIPS Datasets and Benchmarks},
  year={2021}
}
```
