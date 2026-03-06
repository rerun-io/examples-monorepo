## WiLoR-mini: Simplifying WiLoR into a Python package

**Original repository: [WiLoR](https://github.com/rolpotamias/WiLoR), thanks to the authors for sharing**

I have simplified WiLoR, focusing on the inference process. Now it can be installed via pip and used directly, and it will automatically download the model.

## Updates
- PyTorch/CUDA: Upgraded to PyTorch >= 2.7 with CUDA 12.8 wheels, enabling RTX 50xx/Blackwell support. Pinned `ultralytics==8.3.162` to avoid the weights_only change in PyTorch 2.7+.
- Packaging: Modern `pyproject.toml` with Hatchling; project name `wilor-nano` (import path `wilor_nano`); requires Python >= 3.11.
- Typing: Added comprehensive type hints using jaxtyping across models, pipelines, and utils. Package-wide runtime checks via beartype are enabled only in the Pixi dev environment (`PIXI_ENVIRONMENT_NAME=dev`).
- Models: ViT attention uses `torch.nn.functional.scaled_dot_product_attention`; MANO via `ManoSimpleLayer`; rotations handled with 6D→rotmat and `roma`; tensor shapes documented in signatures.
- Pipeline: Encapsulated inference in `WiLorHandPose3dEstimationPipeline` with typed `TypedDict` outputs, YOLO-based hand detection, and automatic HF model downloads; camera projection utilities are typed NumPy/Torch.
- Reproducibility: Managed with Pixi and a committed `pixi.lock`. Dev tasks/tools available in the `dev` environment (ruff, pytest). Examples:
  - `pixi shell -e dev`
  - `pixi run -e dev pytest`
  - `pixi run -e dev ruff check .`

### Pixi Tasks
- `image-example`: Runs the demo image inference using `assets/img.png`.
  - `pixi run image-example`
  - Equivalent raw command: `pixi run python tools/wilor_inference.py --image-path assets/img.png`
- `video-example`: Runs the demo video inference using `assets/video.mp4`.
  - `pixi run video-example`
  - Equivalent raw command: `pixi run python tools/wilor_inference.py --video-path assets/video.mp4`

Notes:
- These tasks are defined under the Pixi `dev` feature and resolve automatically if unique. They invoke `tools/wilor_inference.py` which logs results with Rerun and downloads pretrained weights on first run.
- List available tasks: `pixi task list`

### How to use?
Note: make sure you are using Python3.10
* install: `pip install git+https://github.com/warmshao/WiLoR-mini`
* Usage:
```python
import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)
img_path = "assets/img.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
outputs = pipe.predict(image)

```
For more usage examples, please refer to: `tests/test_pipelines.py`

### Demo
<video src="https://github.com/user-attachments/assets/ca7329fe-0b66-4eb6-87a5-4cb5cbe9ec43" controls="controls" width="300" height="500">您的浏览器不支持播放该视频！</video>
