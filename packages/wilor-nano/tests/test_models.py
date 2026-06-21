# @Time    : 2024/10/29
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : WiLoR-mini
# @FileName: test_models.py
import os
import time
from pathlib import Path

import pytest

PRETRAINED_MODELS_DIR = Path(__file__).parents[1] / "src" / "wilor_nano" / "pretrained_models"
RUN_PROFILER_TESTS = os.environ.get("RUN_WILOR_PROFILER_TESTS") == "1"


def test_wilor_model():

    import cv2
    import numpy as np
    import torch

    from wilor_nano.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16

    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)
    img_path = "assets/img.png"
    image = cv2.imread(img_path)
    assert image is not None
    outputs = pipe.predict(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand_bboxs = []
    is_rights = []
    for i in range(len(outputs)):
        hand_bboxs.append(outputs[i]["hand_bbox"])
        is_rights.append(outputs[i]["is_right"])
    is_rights_array = np.asarray(is_rights, dtype=np.int64)
    for _ in range(1):
        t0 = time.time()
        outputs = pipe.predict_with_bboxes(image, np.array(hand_bboxs), is_rights_array)
        print(time.time() - t0)
    assert outputs


@pytest.mark.skipif(not RUN_PROFILER_TESTS, reason="manual profiler test; set RUN_WILOR_PROFILER_TESTS=1 to run")
def test_vit_profiler():
    import torch
    import torch.profiler

    from wilor_nano.models.vit import vit
    # 创建模型和数据
    mano_mean_path = str(PRETRAINED_MODELS_DIR / "mano_mean_params.npz")
    mano_model_path = str(PRETRAINED_MODELS_DIR / "mano_clean" / "MANO_RIGHT.pkl")
    model = vit(mano_mean_path=mano_mean_path, mano_model_path=mano_model_path)  # 假设你有一个名为 ViT 的模型
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16
    data = torch.randn(2, 3, 256, 192).to(device, dtype=dtype)  # 示例输入，8是batch size
    model.to(device, dtype=dtype)
    model.eval()
    # 使用 Profiler 进行性能分析
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_flops=True
    ) as prof:
        # 运行模型
        model(data)

    with torch.no_grad():
        for _ in range(100):
            t0 = time.time()
            model(data)
            print(time.time() - t0)
    # 打印分析结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


@pytest.mark.skipif(not RUN_PROFILER_TESTS, reason="manual profiler test; set RUN_WILOR_PROFILER_TESTS=1 to run")
def test_wilor_profiler():
    import torch
    import torch.profiler

    from wilor_nano.models.wilor import WiLor
    # 创建模型和数据
    mano_mean_path = str(PRETRAINED_MODELS_DIR / "mano_mean_params.npz")
    mano_model_path = str(PRETRAINED_MODELS_DIR / "mano_clean" / "MANO_RIGHT.pkl")
    FOCAL_LENGTH = 5000
    IMAGE_SIZE = 256
    model = WiLor(
        mano_root_dir=Path(mano_model_path).parent,
        mano_mean_path=mano_mean_path,
        focal_length=FOCAL_LENGTH,
        image_size=IMAGE_SIZE,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float16
    data = torch.randn(2, 256, 256, 3).to(device, dtype=dtype)  # 示例输入，8是batch size
    model.to(device, dtype=dtype)
    model.eval()
    # 使用 Profiler 进行性能分析
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            profile_memory=True,
            with_flops=True
    ) as prof:
        # 运行模型
        model(data)

    with torch.no_grad():
        for _ in range(100):
            t0 = time.time()
            model(data)
            print(time.time() - t0)
    # 打印分析结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == '__main__':
    # test_wilor_model()
    test_vit_profiler()
    # test_wilor_profiler()
