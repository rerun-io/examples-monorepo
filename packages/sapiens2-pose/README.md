# Sapiens2: Pose Estimation
### ICLR 2026

Top-down 308-keypoint human pose estimation. Detects people with DETR, runs a lightweight Sapiens2 inference path on each crop, and visualizes the result in an embedded Rerun viewer.

- **Code:** [github.com/facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)
- **Models:** [Sapiens2 collection](https://huggingface.co/facebook/sapiens2)
- **Paper:** https://openreview.net/pdf?id=IVAlYCqdvW

## Monorepo usage

```bash
pixi install -e sapiens2-pose
pixi run -e sapiens2-pose sapiens2-pose-app
```

The first launch downloads the DETR detector and Sapiens2 pose checkpoints from Hugging Face.
