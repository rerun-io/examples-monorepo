from pathlib import Path

import av
import numpy as np
import pytest
from jaxtyping import Float, UInt8
from numpy import ndarray

from simplecv.camera_parameters import PinholeParameters
from simplecv.data.exoego.mamma import (
    MammaConfig,
    MammaSequence,
    calibration_dir_for_sequence,
    discover_camera_names,
    discover_sequence_names,
    pinhole_from_camera_npz,
    video_dir_for_sequence,
)

SEQUENCE_NAME: str = "mamma_markerless_test/indoors/seq_a"
CAMERA_NAMES: tuple[str, ...] = ("A001", "B001")
VIDEO_WIDTH: int = 32
VIDEO_HEIGHT: int = 32

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


def _have_smplx_model() -> bool:
    return (PROJECT_ROOT / "simplecv" / "data" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz").exists()


def _write_synthetic_mp4(path: Path, num_frames: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container: av.container.OutputContainer = av.open(str(path), mode="w")
    stream: av.video.stream.VideoStream = container.add_stream("h264", rate=30, options={"preset": "ultrafast", "crf": "0"})
    stream.width = VIDEO_WIDTH
    stream.height = VIDEO_HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.max_b_frames = 0
    for frame_idx in range(num_frames):
        pixels: UInt8[ndarray, "h w 3"] = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), frame_idx * 40, dtype=np.uint8)
        frame: av.VideoFrame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _write_camera_npz(path: Path, name: str, *, translation_m: float = 1.6, native_size: tuple[int, int] = (VIDEO_WIDTH, VIDEO_HEIGHT)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cam_ext: Float[ndarray, "4 4"] = np.eye(4, dtype=np.float32)
    cam_ext[:3, 3] = np.float32(translation_m)
    cam_int: Float[ndarray, "3 3"] = np.array(
        [[100.0, 0.0, native_size[0] / 2], [0.0, 100.0, native_size[1] / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez(
        path,
        cam_name=name,
        cam_int=cam_int,
        cam_ext=cam_ext,
        cam_img_w=native_size[0],
        cam_img_h=native_size[1],
    )


def _write_synthetic_sequence(root: Path, sequence_name: str = SEQUENCE_NAME, video_dir_name: str = "videos_light") -> Path:
    sequence_dir: Path = root / sequence_name
    meta_dir: Path = sequence_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.savez(meta_dir / "global.npz", seq_name=sequence_name.rsplit("/", maxsplit=1)[-1], fps=30, frame_start=0, frame_end=3)
    for camera_name in CAMERA_NAMES:
        _write_camera_npz(meta_dir / f"{camera_name}.npz", camera_name)
        _write_synthetic_mp4(sequence_dir / video_dir_name / f"{camera_name}.mp4")
    return sequence_dir


def test_discovery_and_video_dir_preference(tmp_path: Path) -> None:
    sequence_dir: Path = _write_synthetic_sequence(tmp_path)
    assert discover_sequence_names(tmp_path) == [SEQUENCE_NAME]
    assert calibration_dir_for_sequence(sequence_dir).name == "meta"
    assert video_dir_for_sequence(sequence_dir).name == "videos_light"
    assert discover_camera_names(sequence_dir) == CAMERA_NAMES

    # The AV1 mirror wins over the shipped yuv444 variants once present.
    for camera_name in CAMERA_NAMES:
        _write_synthetic_mp4(sequence_dir / "videos_av1" / f"{camera_name}.mp4")
    assert video_dir_for_sequence(sequence_dir).name == "videos_av1"


def test_pinhole_from_camera_npz_normalizes_millimeters(tmp_path: Path) -> None:
    npz_path: Path = tmp_path / "A001.npz"
    video_path: Path = tmp_path / "A001.mp4"
    _write_camera_npz(npz_path, "A001", translation_m=1600.0)
    _write_synthetic_mp4(video_path)

    pinhole: PinholeParameters = pinhole_from_camera_npz(npz_path, video_path)
    assert pinhole.name == "A001"
    np.testing.assert_allclose(pinhole.extrinsics.cam_t_world, np.full(3, 1.6), rtol=1e-6)


def test_pinhole_from_camera_npz_rescales_k_to_video_resolution(tmp_path: Path) -> None:
    npz_path: Path = tmp_path / "A001.npz"
    video_path: Path = tmp_path / "A001.mp4"
    _write_camera_npz(npz_path, "A001", native_size=(VIDEO_WIDTH * 2, VIDEO_HEIGHT * 2))
    _write_synthetic_mp4(video_path)

    pinhole: PinholeParameters = pinhole_from_camera_npz(npz_path, video_path)
    assert (pinhole.intrinsics.width, pinhole.intrinsics.height) == (VIDEO_WIDTH, VIDEO_HEIGHT)
    assert pinhole.intrinsics.fl_x == pytest.approx(50.0)
    assert pinhole.intrinsics.cx == pytest.approx(VIDEO_WIDTH / 2)


def test_mamma_sequence_is_exo_only(tmp_path: Path) -> None:
    _write_synthetic_sequence(tmp_path)
    config: MammaConfig = MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME)
    sequence: MammaSequence = MammaSequence(config)

    assert sequence.ego_sequence is None
    assert sequence.exo_sequence is not None
    assert sequence.exo_sequence.exo_video_names == list(CAMERA_NAMES)
    assert sorted(sequence.stream_timestamps_ns) == [f"exo/{name}" for name in CAMERA_NAMES]
    assert len(sequence) == 3

    sample = sequence[0]
    assert sample.ego_bgr_list is None
    assert sample.exo_bgr_list is not None
    assert len(sample.exo_bgr_list) == len(CAMERA_NAMES)
    assert sample.exo_bgr_list[0].shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)

    assert MammaSequence.num_sequences_for_config(config) == 1
    episode_identities: list[str] = [type(seq).sequence_identity_for_config(seq.config).recording_id for seq in MammaSequence.iter_episode_sequences(config)]
    assert episode_identities == ["mamma__mamma_markerless_test__indoors__seq_a"]


def _write_params_npz(path: Path, num_frames: int, seed: int, gender: str = "neutral") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng: np.random.Generator = np.random.default_rng(seed)
    np.savez(
        path,
        poses=rng.normal(scale=0.1, size=(num_frames, 165)).astype(np.float32),
        betas=rng.normal(scale=0.5, size=16).astype(np.float32),
        trans=rng.normal(size=(num_frames, 3)).astype(np.float32),
        num_betas=16,
        flat_hand_mean=False,
        gender=gender,
        mocap_frame_rate=30,
    )


def _write_eval_gt_npz(path: Path, *, num_frames: int = 3, genders: tuple[str, ...] = ("male", "female")) -> None:
    """Synthetic ``gt/global.npz`` in the eval subsets' MoSh GT layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_people: int = len(genders)
    rng: np.random.Generator = np.random.default_rng(7)
    np.savez(
        path,
        seq_name="seq",
        frames_len=num_frames,
        fps=30,
        people_len=n_people,
        subject_name=np.array([f"{50000 + i}" for i in range(n_people)]),
        gender=np.array(genders),
        shape=np.zeros((n_people, 300)),
        pose_world=rng.normal(scale=0.05, size=(num_frames, n_people, 165)),
        pose_trans_world=rng.normal(size=(num_frames, n_people, 3)),
        flat_hand_mean=True,
        v_template=np.zeros((n_people, 10475, 3)),
        method_used="mosh_v_template",
    )


def test_mamma_load_labels_builds_multi_person_smplx_stack(tmp_path: Path) -> None:
    sequence_dir: Path = _write_synthetic_sequence(tmp_path)
    _write_params_npz(sequence_dir / "pred" / "params_00.npz", num_frames=3, seed=0)
    _write_params_npz(sequence_dir / "pred" / "params_01.npz", num_frames=3, seed=1)

    sequence: MammaSequence = MammaSequence(MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME))
    labels = sequence.exoego_labels
    assert labels is not None
    assert labels.smplx_stack is not None
    assert labels.smplx_stack.model_type == "smplx"
    assert labels.smplx_stack.betas.shape == (2, 16)
    assert labels.smplx_stack.poses.shape == (3, 2, 165)
    assert labels.smplx_stack.trans.shape == (3, 2, 3)
    assert labels.smplx_stack.gender == "neutral"
    assert labels.smplx_stack.transl_pivot == "root_joint"
    assert labels.xyzc_stack.shape == (3, 133, 4)
    if _have_smplx_model():
        # COCO-133 keypoints are derived from the SMPL-X forward (person 0).
        assert np.all(np.isfinite(labels.xyzc_stack[..., :3]))
        assert np.all(labels.xyzc_stack[..., 3] == 1.0)
    else:
        assert np.all(np.isnan(labels.xyzc_stack[..., :3]))

    # Per-frame sampling passes the stack through whole.
    sample = sequence[0]
    assert sample.labels is not None
    assert sample.labels.smplx_stack is labels.smplx_stack


def test_mamma_load_labels_without_pred_returns_none(tmp_path: Path) -> None:
    _write_synthetic_sequence(tmp_path)
    sequence: MammaSequence = MammaSequence(MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME))
    assert sequence.exoego_labels is None


def test_mamma_load_labels_from_eval_gt_warns_on_mixed_genders(tmp_path: Path) -> None:
    sequence_dir: Path = _write_synthetic_sequence(tmp_path)
    _write_eval_gt_npz(sequence_dir / "gt" / "global.npz")

    # SmplxStack carries one gender for all people, so a mixed-gender GT must
    # degrade loudly (person 0 wins) rather than silently.
    with pytest.warns(UserWarning, match="mix body-model genders"):
        sequence: MammaSequence = MammaSequence(MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME))
    labels = sequence.exoego_labels
    assert labels is not None
    assert labels.smplx_stack is not None
    assert labels.smplx_stack.gender == "male"
    assert labels.smplx_stack.flat_hand_mean is True
    assert labels.smplx_stack.betas.shape == (2, 300)
    assert labels.smplx_stack.poses.shape == (3, 2, 165)
    assert labels.smplx_stack.trans.shape == (3, 2, 3)
    assert labels.smplx_stack.v_template is not None
    assert labels.smplx_stack.v_template.shape == (2, 10475, 3)
    assert labels.xyzc_stack.shape == (3, 133, 4)


def test_mamma_pred_params_warn_on_mixed_genders(tmp_path: Path) -> None:
    sequence_dir: Path = _write_synthetic_sequence(tmp_path)
    _write_params_npz(sequence_dir / "pred" / "params_00.npz", num_frames=3, seed=0, gender="neutral")
    _write_params_npz(sequence_dir / "pred" / "params_01.npz", num_frames=3, seed=1, gender="female")

    with pytest.warns(UserWarning, match="mix body-model genders"):
        sequence: MammaSequence = MammaSequence(MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME))
    labels = sequence.exoego_labels
    assert labels is not None
    assert labels.smplx_stack is not None
    assert labels.smplx_stack.gender == "neutral"


@pytest.mark.skipif(not _have_smplx_model(), reason="SMPL-X model file not available under simplecv/data/body_models/")
def test_log_smplx_batch_streams_meshes_in_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mesh logger must forward + send in bounded frame chunks, covering every frame.

    Materializing a whole sequence of vertices before sending costs ~126 KB/frame
    (SMPL-X) — several GB on long EPFL sessions — so this pins the streaming shape.
    """
    import rerun as rr

    import simplecv.ops.smplx.smplx_torch as smplx_torch
    from simplecv.apis.view_exoego import log_smplx_batch

    sequence_dir: Path = _write_synthetic_sequence(tmp_path)
    _write_params_npz(sequence_dir / "pred" / "params_00.npz", num_frames=3, seed=0)
    sequence: MammaSequence = MammaSequence(MammaConfig(root_directory=tmp_path, sequence_name=SEQUENCE_NAME))

    monkeypatch.setattr(smplx_torch, "SMPLX_FORWARD_CHUNK_FRAMES", 2)
    forward_chunk_frames: list[int] = []
    original_forward_batched = smplx_torch.SmplxLayerTorch.forward_batched

    def _spy_forward_batched(self: smplx_torch.SmplxLayerTorch, poses: ndarray, transl: ndarray) -> smplx_torch.SmplxForwardResult:
        forward_chunk_frames.append(poses.shape[0])
        return original_forward_batched(self, poses, transl)

    monkeypatch.setattr(smplx_torch.SmplxLayerTorch, "forward_batched", _spy_forward_batched)
    send_columns_calls: list[str] = []
    monkeypatch.setattr(rr, "send_columns", lambda entity_path, indexes, columns: send_columns_calls.append(str(entity_path)))

    rr.init("test_log_smplx_batch_chunks", spawn=False)
    log_smplx_batch(
        exoego_sequence=sequence,
        smplx_parent_log_path=Path("world/gt"),
        timeline="video_time",
        timestamps_ns=(np.arange(3, dtype=np.int64) * 33_333_333),
        log_smplx=True,
    )
    assert forward_chunk_frames == [2, 1]
    assert len(send_columns_calls) == 2
