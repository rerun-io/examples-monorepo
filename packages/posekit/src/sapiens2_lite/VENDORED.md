# Sapiens2 pose network source

This is a build-only vendored dependency of posekit (formerly `packages/sapiens2-pose`).
The public API is `posekit.models.sapiens` (`SapiensPoseConfig`). The export,
quantization, calibration and checkpoint code that we wrote is typed posekit code in
`posekit.models.sapiens_net`; only Meta's network code is here. It runs in the
`posekit` environment; try it with
`pixi run -e posekit --frozen posekit-video-pose --rr-config.headless --rr-config.save /tmp/sapiens.rrd --video-path assets/examples/hands_video.mp4 yolox onnx sapiens`.

Source: [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2) at
`0e51c12d7c7257d88431b2d50e523a7b03004854` (2026-04-28), under the Sapiens2 License in
`LICENSE.md` (unchanged).

| File | Upstream | Changes |
| --- | --- | --- |
| `backbone.py` | `sapiens/backbones/standalone/sapiens2.py` | Unused `typing` imports removed. |
| `keypoints308.py` | `sapiens/pose/configs/_base_/keypoints308.py` | None. |
| `pose.py` | `sapiens/pose/src/models/heads/pose_heatmap_head.py`, `sapiens/pose/src/models/core/pose_topdown_estimator.py`, `sapiens/pose/src/datasets/utils.py` (`parse_pose_metainfo`) | Condensed to inference: `ModelSpec` table per size, heatmap head, top-down model, safetensors loader. No registry, training, codecs or CPU crop/decode helpers (posekit crops and decodes on the GPU). |

The model math and checkpoint layout are unchanged. Weights download on first use into
the Hugging Face cache: `facebook/sapiens2-pose-{0.4b,0.8b,1b}`
(`sapiens2_<size>_pose.safetensors`, latest revision).

The pre-port goldens (`/mnt/nas/ssd-8tb/posekit/goldens/sapiens2-v1`, README beside them)
hold the outputs of the original package on fixed person boxes; `tests/test_sapiens_golden.py`
compares this copy against them.
