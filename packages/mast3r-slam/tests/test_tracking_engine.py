"""The bf16 TensorRT tracking engine against the eager fp32 forward on real frames.

Needs a CUDA GPU, the MASt3R checkpoint and the example video. The first run on a machine exports and builds the
engine (about 90 s); later runs load it from trtkit's cache.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="MASt3R-SLAM requires PyTorch")
pytest.importorskip("tensorrt", reason="the engine runs on TensorRT")

PACKAGE_DIR: Path = Path(__file__).parent.parent
CHECKPOINT: Path = PACKAGE_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
VIDEO: Path = PACKAGE_DIR / "data" / "normal-apt-tour.mp4"

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


def test_tracking_engine_matches_eager_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every output of the engine stays within cosine 0.999 of the eager fp32 forward, the precision gate."""
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
    dataset = load_dataset(str(VIDEO), img_size=224)
    dataset.subsample(5)
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
        mast3r_utils.use_tracking_engine(model, (224, 224))
        try:
            for pair in pairs:
                for name, got, want in zip(("X", "C", "D", "Q", "feat", "pos"), forward(*pair), eager[pair], strict=True):
                    assert got.shape == want.shape, name
                    cosine: float = torch.nn.functional.cosine_similarity(got.flatten().double(), want.flatten().double(), dim=0).item()
                    assert cosine >= 0.999, f"{name} at {pair}: cosine {cosine:.5f}"
        finally:
            mast3r_utils._TRACKING_FORWARD.pop(id(model), None)
