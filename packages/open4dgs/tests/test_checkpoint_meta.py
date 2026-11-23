"""A saved checkpoint rebuilds the model it was trained as (``CheckpointMeta``): the settings, cameras and capture times
come back, and ``open4dgs-export-asog --evaluate`` scores that model, not one rebuilt from defaults."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("trigs", reason="TRiGS comes from recipes/open4dgs-trigs-src (open4dgs envs)")

from test_asog import FPS, NUM_FRAMES, synthetic_params  # noqa: E402

from open4dgs.apis.export_asog import evaluation_models  # noqa: E402
from open4dgs.splat_export import ASOG_CUT_FORMAT  # noqa: E402
from open4dgs.trigs_adapter import LEGACY_KEYFRAME_STEP, CheckpointMeta, TrainConfig, TrainedClip  # noqa: E402

CAMERAS: int = 3


def non_default_meta() -> CheckpointMeta:
    """Every setting a rebuild from defaults would get wrong: SH degree 1, keyframe step 2, rolling shutter with
    per-camera signs, learned time offsets, the prior's RGB+ED render and the dataset's capture times."""
    world_T_cam = np.tile(np.eye(4, dtype=np.float32), (CAMERAS, 1, 1))
    world_T_cam[:, 0, 3] = np.arange(CAMERAS, dtype=np.float32)
    K = np.tile(np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]], dtype=np.float32), (CAMERAS, 1, 1))
    return CheckpointMeta(
        train=TrainConfig(sh_degree=1, learn_time_offsets=True, rs_readout_ms=(10.0, 0.0, 20.0), rs_sign=(1.0, 1.0, -1.0), seed=7),
        sh_degree=1,
        fps=FPS,
        num_frames=NUM_FRAMES,
        keyframe_step=2,
        holdout_positions=(1,),
        render_depth=True,
        world_T_cam=world_T_cam,
        K=K,
        sizes=((64, 48), (64, 48), (32, 24)),
        capture_times=tuple(tuple(f / FPS + 0.001 * c for c in range(CAMERAS)) for f in range(NUM_FRAMES)),
    )


def saved_model(tmp_path: Path, device: torch.device) -> tuple[TrainedClip, Path]:
    model = TrainedClip.from_params(synthetic_params(), non_default_meta(), device)
    path: Path = tmp_path / "trigs_ckpt.pt"
    model.save(path)
    return model, path


def assert_same_model(a: TrainedClip, b: TrainedClip) -> None:
    assert a.trigs_config == b.trigs_config
    assert a.sh_degree == b.sh_degree and a.fps == b.fps and a.num_frames == b.num_frames
    assert a.capture_times == b.capture_times
    assert torch.equal(a.cameras.viewmats, b.cameras.viewmats) and torch.equal(a.cameras.Ks, b.cameras.Ks)
    assert torch.equal(a.cameras.rs_signs, b.cameras.rs_signs)
    assert a.trainer._sequence_t_max == b.trainer._sequence_t_max


def test_checkpoint_round_trips_every_training_setting(tmp_path: Path) -> None:
    model, path = saved_model(tmp_path, torch.device("cpu"))
    loaded = TrainedClip.load(path, torch.device("cpu"))

    assert_same_model(model, loaded)
    assert loaded.trigs_config.keyframe_stride == 2 and loaded.trigs_config.render_depth
    assert loaded.trigs_config.learn_camera_time_offsets and loaded.trigs_config.rs_readout_ms == 20.0
    assert loaded.cameras.rs_signs.tolist() == [0.5, 0.0, -1.0]
    assert all(torch.equal(model.params[k], loaded.params[k]) for k in model.params)


def test_export_asog_evaluates_the_trained_model(tmp_path: Path) -> None:
    """The three models ``--evaluate`` scores carry the checkpoint's settings; only the cut05 variant changes the cut."""
    model, path = saved_model(tmp_path, torch.device("cpu"))
    decoded = {k: v.detach() + 1.0 for k, v in model.params.items()}
    models = evaluation_models(path, decoded, clip=None, holdout=(), device=torch.device("cpu"))

    assert_same_model(model, models["uncompressed"])
    assert_same_model(model, models["decoded"])
    assert torch.equal(models["decoded"].params["means"], decoded["means"])
    assert models["decoded_cut05"].trigs_config == dataclasses.replace(model.trigs_config, temporal_active_threshold=ASOG_CUT_FORMAT)


def test_a_checkpoint_without_meta_reads_with_the_old_defaults(tmp_path: Path) -> None:
    path: Path = tmp_path / "trigs_ckpt.pt"
    torch.save({"params": synthetic_params(), "sh_degree": 3, "fps": FPS, "num_frames": NUM_FRAMES}, path)
    loaded = TrainedClip.load(path, torch.device("cpu"))

    assert loaded.trigs_config == TrainConfig().to_trigs(FPS, LEGACY_KEYFRAME_STEP, ())
    assert loaded.capture_times is None and len(loaded.cameras) == 0


@pytest.mark.integration
def test_a_loaded_checkpoint_renders_bit_identically(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (gsplat rasterization)")
    model, path = saved_model(tmp_path, torch.device("cuda"))
    loaded = TrainedClip.load(path, torch.device("cuda"))
    for cam in range(CAMERAS):
        for frame in (0, NUM_FRAMES // 2):
            (rgb_a, alpha_a), (rgb_b, alpha_b) = model.render(model.cameras, cam, frame), loaded.render(loaded.cameras, cam, frame)
            assert torch.equal(rgb_a, rgb_b) and torch.equal(alpha_a, alpha_b)
