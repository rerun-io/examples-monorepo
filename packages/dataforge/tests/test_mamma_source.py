"""MAMMA raw readers against hand-written release files (the key sets of the sampled release NPZs)."""

from pathlib import Path

import numpy as np
import pytest

from dataforge.datasets.mamma_source import MammaCamera, SequenceLabels, discover_sequences, read_camera, read_eval_labels, read_pred_labels

K: list[list[float]] = [[1000.0, 0.0, 100.0], [0.0, 1000.0, 50.0], [0.0, 0.0, 1.0]]


def write_camera(path: Path, *, translation: tuple[float, float, float] = (0.0, 0.0, 0.0), lens: bool = True, **overrides: np.ndarray) -> Path:
    """A camera NPZ with the release's keys and dtypes (IOI cameras carry ``vicon_radial_2``, iPhones do not)."""
    cam_ext: np.ndarray = np.eye(4, dtype=np.float32)
    cam_ext[:3, 3] = translation
    arrays: dict[str, np.ndarray] = {
        "cam_name": np.array("IOI_03"),
        "cam_int": np.array(K, dtype=np.float32),
        "cam_ext": cam_ext,
        "cam_img_w": np.array(2056.0),
        "cam_img_h": np.array(1504.0),
        "cam_portrait": np.array(False),
    }
    if lens:
        arrays["vicon_radial_2"] = np.array([100.0, 50.0, 1e-4, 0.0, 0.0])
    arrays.update(overrides)
    np.savez(path, **arrays)  # pyrefly: ignore[bad-argument-type] (no array is named allow_pickle)
    return path


def test_read_camera_keeps_the_release_calibration(tmp_path: Path) -> None:
    camera: MammaCamera = read_camera(write_camera(tmp_path / "IOI_03.npz", translation=(0.5, -0.25, 3.0)))
    assert camera.name == "IOI_03"
    assert (camera.width, camera.height) == (2056, 1504)
    np.testing.assert_array_equal(camera.K, np.array(K))
    np.testing.assert_array_equal(camera.cam_T_world[:3, 3], [0.5, -0.25, 3.0])
    assert camera.lens is not None


def test_read_camera_converts_millimetre_extrinsics(tmp_path: Path) -> None:
    camera: MammaCamera = read_camera(write_camera(tmp_path / "IOI_03.npz", translation=(1500.0, -250.0, 3000.0)))
    np.testing.assert_allclose(camera.cam_T_world[:3, 3], [1.5, -0.25, 3.0])


def test_an_iphone_camera_has_no_lens_model(tmp_path: Path) -> None:
    assert read_camera(write_camera(tmp_path / "A001.npz", lens=False)).lens is None


def test_a_malformed_camera_names_its_file(tmp_path: Path) -> None:
    path: Path = write_camera(tmp_path / "IOI_03.npz", cam_int=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="IOI_03.npz"):
        read_camera(path)


def test_the_vicon_lens_maps_distorted_to_undistorted_pixels(tmp_path: Path) -> None:
    # Centre (100, 50), k1 = 1e-4: the distorted pixel (110, 50) is 10 px out, r^2 = 100, so it scales by 1.01 to (110.1, 50).
    camera: MammaCamera = read_camera(write_camera(tmp_path / "IOI_03.npz"))
    assert camera.lens is not None
    np.testing.assert_allclose(camera.lens.undistort(np.array([[110.0, 50.0]])), [[110.1, 50.0]])
    np.testing.assert_allclose(camera.lens.distort(np.array([[110.1, 50.0]])), [[110.0, 50.0]], atol=1e-9)


def test_a_point_behind_the_camera_leaves_the_rest_of_the_batch_projected(tmp_path: Path) -> None:
    # A whole sequence of keypoints goes through one call: one invalid point must not stop the others' iterations.
    camera: MammaCamera = read_camera(write_camera(tmp_path / "IOI_03.npz"))
    edge: np.ndarray = np.array([[0.04, 0.03, 1.0]])
    alone: np.ndarray = camera.project(edge)
    together: np.ndarray = camera.project(np.concatenate([edge, [[0.0, 0.0, -1.0]]]))
    assert np.isfinite(alone).all()
    np.testing.assert_allclose(together[0], alone[0])
    assert np.isnan(together[1]).all()


def test_project_applies_the_pinhole_then_the_lens(tmp_path: Path) -> None:
    # The camera sits at the world origin: (0.0101, 0, 1) m lands on the undistorted pixel (110.1, 50), which the lens moves to (110, 50).
    camera: MammaCamera = read_camera(write_camera(tmp_path / "IOI_03.npz"))
    np.testing.assert_allclose(camera.project(np.array([[0.0101, 0.0, 1.0]])), [[110.0, 50.0]], atol=1e-9)
    assert np.isnan(camera.project(np.array([[0.0, 0.0, -1.0]]))).all()  # behind the camera


def write_eval_global(path: Path, *, method: str = "mosh_v_template", template: bool = True, frames: int = 4, pose_frames: int | None = None) -> Path:
    """An eval ``gt/global.npz``: two people listed in subject order, float64 parameters as the release stores them."""
    rng: np.random.Generator = np.random.default_rng(0)
    arrays: dict[str, np.ndarray] = {
        "seq_name": np.array("250225_WestCoastSwing_Basic_Whip_03675_03676_1"),
        "frames_len": np.array(frames),
        "fps": np.array(30),
        "people_len": np.array(2),
        "subject_name": np.array(["03676", "03675"]),
        "gender": np.array(["female", "male"]),
        "shape": rng.normal(size=(2, 300)),
        "pose_world": rng.normal(size=(pose_frames or frames, 2, 165)),
        "pose_trans_world": rng.normal(size=(pose_frames or frames, 2, 3)),
        "flat_hand_mean": np.array(True),
        "keywords": np.array(["Whip", "Basic"]),
        "method_used": np.array(method),
    }
    if template:
        arrays["v_template"] = rng.normal(size=(2, 10475, 3))
    np.savez(path, **arrays)  # pyrefly: ignore[bad-argument-type] (no array is named allow_pickle)
    return path


def test_read_eval_labels_keeps_each_subject_in_release_order(tmp_path: Path) -> None:
    path: Path = write_eval_global(tmp_path / "global.npz")
    labels: SequenceLabels = read_eval_labels(path)
    with np.load(path) as source:
        assert (labels.fps, labels.frames, labels.method) == (30.0, 4, "mosh_v_template")
        assert [person.subject for person in labels.people] == ["03676", "03675"]
        assert [person.gender for person in labels.people] == ["female", "male"]
        for index, person in enumerate(labels.people):
            np.testing.assert_array_equal(person.pose, source["pose_world"][:, index])
            np.testing.assert_array_equal(person.trans, source["pose_trans_world"][:, index])
            np.testing.assert_array_equal(person.betas, source["shape"][index])
            assert person.v_template is not None and person.flat_hand_mean
            np.testing.assert_array_equal(person.v_template, source["v_template"][index])


def test_beta_only_ground_truth_has_no_template(tmp_path: Path) -> None:
    labels: SequenceLabels = read_eval_labels(write_eval_global(tmp_path / "global.npz", method="mosh_betas", template=False))
    assert labels.method == "mosh_betas"
    assert all(person.v_template is None for person in labels.people)


def test_a_template_fit_without_its_template_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global.npz"):
        read_eval_labels(write_eval_global(tmp_path / "global.npz", template=False))


def test_labels_shorter_than_the_sequence_name_the_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global.npz"):
        read_eval_labels(write_eval_global(tmp_path / "global.npz", frames=4, pose_frames=3))


def write_pred(path: Path, *, frames: int = 4, rate: float = 30.0) -> Path:
    """A markerless ``pred/params_XX.npz`` (float32, 16 betas, mean hand added); each file's values are its own."""
    rng: np.random.Generator = np.random.default_rng(int(path.stem.removeprefix("params_")))
    np.savez(
        path,
        betas=rng.normal(size=16).astype(np.float32),
        poses=rng.normal(size=(frames, 165)).astype(np.float32),
        trans=rng.normal(size=(frames, 3)).astype(np.float32),
        gender=np.array("neutral"),
        mocap_frame_rate=np.array(rate),
        num_betas=np.array(16),
        flat_hand_mean=np.array(False),
    )
    return path


def write_meta_global(path: Path, *, frames: int = 4) -> Path:
    np.savez(
        path,
        seq_name=np.array("050825_WestCoastSwing_CutOff_03688_03689_1"),
        frames_len=np.array(frames),
        fps=np.array(30),
        method_used=np.array("mamma_pred"),
    )
    return path


def test_read_pred_labels_takes_one_person_per_file(tmp_path: Path) -> None:
    pred: list[Path] = [write_pred(tmp_path / "params_00.npz"), write_pred(tmp_path / "params_01.npz")]
    labels: SequenceLabels = read_pred_labels(write_meta_global(tmp_path / "global.npz"), pred)
    assert (labels.fps, labels.frames, labels.method) == (30.0, 4, "mamma_pred")
    assert [person.subject for person in labels.people] == ["params_00", "params_01"]
    for path, person in zip(pred, labels.people, strict=True):
        with np.load(path) as source:
            np.testing.assert_array_equal(person.pose, source["poses"])
            np.testing.assert_array_equal(person.trans, source["trans"])
            np.testing.assert_array_equal(person.betas, source["betas"])
        assert person.gender == "neutral" and not person.flat_hand_mean and person.v_template is None
    assert not np.array_equal(labels.people[0].pose, labels.people[1].pose)


def test_a_prediction_at_another_rate_names_its_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="params_00.npz"):
        read_pred_labels(write_meta_global(tmp_path / "global.npz"), [write_pred(tmp_path / "params_00.npz", rate=60.0)])


def test_a_markerless_sequence_missing_a_person_is_skipped(tmp_path: Path) -> None:
    # The release names a couple dance with both dancers' ids; it ships params_00 and params_01.
    directory: Path = tmp_path / "mamma_markerless_dance" / "050825_WestCoastSwing_CutOff_03688_03689_1"
    for sub in ("meta", "pred", "videos_crf16"):
        (directory / sub).mkdir(parents=True)
    write_meta_global(directory / "meta/global.npz")
    write_camera(directory / "meta/IOI_01.npz")
    (directory / "videos_crf16/IOI_01.mp4").write_bytes(b"stand-in")
    write_pred(directory / "pred/params_01.npz")
    found, skipped = discover_sequences(tmp_path, "markerless-dance")
    assert found == []
    assert skipped == [(directory, "predictions ['params_01'], the release ships ['params_00', 'params_01']")]
