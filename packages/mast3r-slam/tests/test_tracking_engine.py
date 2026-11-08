"""The bf16 TensorRT tracking engine against the eager fp32 forward on real frames.

Needs a CUDA GPU, the MASt3R checkpoint and the example video. The first run on a machine exports and builds the
engines for each image size (about 3 min at 224, longer at 512); later runs load them from trtkit's cache. 224 crops
every video to a square; 512 keeps the video's aspect ratio (512x288 for this 16:9 video), so it covers a size the
engines are built for on first use.
"""

from pathlib import Path
from typing import Literal

import pytest

torch = pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")
pytest.importorskip("tensorrt", reason="the engine runs on TensorRT")

PACKAGE_DIR: Path = Path(__file__).parent.parent
CHECKPOINT: Path = PACKAGE_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
VIDEO: Path = PACKAGE_DIR / "data" / "normal-apt-tour.mp4"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


@pytest.mark.parametrize("img_size", [224, 512])
def test_tracking_engine_matches_eager_forward(monkeypatch: pytest.MonkeyPatch, img_size: Literal[224, 512]) -> None:
    """Every output of the engine stays within cosine 0.999 of the eager fp32 forward, the precision gate.

    A second use of the same size reuses the loaded engine instead of loading it again.
    """
    for asset in (CHECKPOINT, VIDEO):
        if not asset.exists():
            pytest.skip(f"missing {asset.name} (pixi run -e mast3r-slam mast3r-slam-example-quick downloads it)")
    monkeypatch.chdir(PACKAGE_DIR)
    import lietorch_mojo as lietorch

    import mast3r_slam.mast3r_utils as mast3r_utils
    from mast3r_slam.config import load_config
    from mast3r_slam.dataloader import load_dataset
    from mast3r_slam.frame import Frame, create_frame

    load_config("config/fast.yaml")
    model = mast3r_utils.load_mast3r().eval()
    dataset = load_dataset(str(VIDEO), img_size=img_size)
    dataset.subsample(5)
    img_hw: tuple[int, int] = dataset.get_img_shape()[0]
    identity = lietorch.Sim3.Identity(1, device="cuda:0")

    def forward(keyframe_idx: int, frame_idx: int) -> list:
        keyframe: Frame = create_frame(keyframe_idx, dataset.read_resized(keyframe_idx), identity)
        keyframe.feat, keyframe.pos, _ = model._encode_image(keyframe.rgb_tensor, keyframe.img_true_shape)
        frame: Frame = create_frame(frame_idx, dataset.read_resized(frame_idx), identity)
        outputs = mast3r_utils.mast3r_asymmetric_inference(model, frame, keyframe)
        assert frame.feat is not None
        assert frame.pos is not None
        return [*(t.clone() for t in outputs), frame.feat.clone(), frame.pos.clone()]

    pairs: list[tuple[int, int]] = [(0, 3), (40, 43), (120, 125)]
    with torch.inference_mode():
        eager = {pair: forward(*pair) for pair in pairs}
        mast3r_utils.use_tracking_engine(model, img_hw)
        try:
            loaded = mast3r_utils._TRACKING_ENGINES[id(model), img_hw]
            mast3r_utils.use_tracking_engine(model, img_hw)
            assert mast3r_utils._TRACKING_ENGINES[id(model), img_hw] is loaded
            for pair in pairs:
                for name, got, want in zip(("X", "C", "D", "Q", "feat", "pos"), forward(*pair), eager[pair], strict=True):
                    assert got.shape == want.shape, name
                    cosine: float = torch.nn.functional.cosine_similarity(got.flatten().double(), want.flatten().double(), dim=0).item()
                    assert cosine >= 0.999, f"{name} at {pair}: cosine {cosine:.5f}"
        finally:
            mast3r_utils._TRACKING_FORWARD.pop(id(model), None)
            mast3r_utils._TRACKING_ENGINES.pop((id(model), img_hw), None)


@pytest.mark.parametrize("img_size", [224, 512])
def test_pair_decoder_engine_matches_eager_decoder(monkeypatch: pytest.MonkeyPatch, img_size: Literal[224, 512]) -> None:
    """The backend's batched pair decoder stays within cosine 0.999 of the eager fp32 decoder, in both directions."""
    for asset in (CHECKPOINT, VIDEO):
        if not asset.exists():
            pytest.skip(f"missing {asset.name} (pixi run -e mast3r-slam mast3r-slam-example-quick downloads it)")
    monkeypatch.chdir(PACKAGE_DIR)
    import mast3r_slam.mast3r_utils as mast3r_utils
    from mast3r_slam.config import load_config
    from mast3r_slam.dataloader import load_dataset

    load_config("config/fast.yaml")
    model = mast3r_utils.load_mast3r().eval()
    dataset = load_dataset(str(VIDEO), img_size=img_size)
    dataset.subsample(5)
    img_hw: tuple[int, int] = dataset.get_img_shape()[0]
    true_shape = torch.tensor([img_hw], device="cuda")
    with torch.inference_mode():
        encoded = [model._encode_image(dataset.read_resized(i).rgb_tensor.cuda(), true_shape) for i in (0, 4, 40, 45)]
        feat = torch.cat([f for f, _, _ in encoded])
        pos = torch.cat([p for _, p, _ in encoded])
        shapes = [true_shape] * 2
        args = (feat[[0, 2]], pos[[0, 2]], feat[[1, 3]], pos[[1, 3]], shapes, shapes)
        eager = mast3r_utils.mast3r_decode_symmetric_batch(model, *args)
        mast3r_utils.use_pair_decoder_engine(model, img_hw)
        try:
            engine = mast3r_utils.mast3r_decode_symmetric_batch(model, *args)
        finally:
            mast3r_utils._PAIR_DECODER.pop(id(model), None)
        for name, got, want in zip(("X", "C", "D", "Q"), engine, eager, strict=True):
            assert got.shape == want.shape, name
            for direction in range(4):
                cosine: float = torch.nn.functional.cosine_similarity(got[direction].flatten().double(), want[direction].flatten().double(), dim=0).item()
                assert cosine >= 0.999, f"{name}[{direction}]: cosine {cosine:.5f}"
