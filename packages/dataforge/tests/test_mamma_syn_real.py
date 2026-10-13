"""MammaSyn on the real shards: the SMPL-X forward against the shipped vertices, and a frame-limited convert per dataset group."""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import column_rows, raw_asset, read_back, recording_properties
from PIL import Image

from dataforge import paths, schema
from dataforge.datasets.mamma import MammaConfig
from dataforge.datasets.mamma_body import BODY_FILES, PosedPerson, SmplxBody
from dataforge.datasets.mamma_syn import in_camera
from dataforge.datasets.mamma_syn_source import (
    BEDLAM_DATASET,
    SYN_DIR,
    BedlamLabels,
    BedlamView,
    RenderLabels,
    RenderView,
    ShardReader,
    SynFamily,
    SynSample,
    assemble_bedlam,
    assemble_render,
    load_people,
    read_bedlam_view,
    read_render_view,
    sequence_samples,
)

pytestmark = pytest.mark.golden

MAMMA_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_ROOT", "/mnt/nas/datasets/mamma"))
MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_MODEL_ROOT", str(paths.raw_root() / "mamma-body-models")))
HANDS: str = "MammaSyn-Hands/interhand_2-6_NC_200_00/be_NB-NZj2zRkeQ_seq_000002"
BEDLAM_BATCH: str = "20221024_10_100_batch01handhair_zoom_suburb_d"
PREVIEW: int = 6
SAMPLES: dict[str, tuple[SynFamily, str]] = {
    "hands": ("syn", "interhand_2-6_NC_200_00/be_NB-NZj2zRkeQ_seq_000002"),
    "interactions": ("syn", "inter-x_train_close_1_NC_200_00_contact/be_5X4U1G5bRFik_seq_000000"),
    "singles": ("syn", "b1_all_2-6_C_200_00/be_zKWfBuy3QUaE_seq_000000"),
    "bedlam": ("syn-bedlam", f"{BEDLAM_BATCH}/seq_000000"),
}
"""One sequence of each group: Hands (five people, a dropped frame), Inter-X couples, BEDLAM bodies in MAMMA's rig, a BEDLAM zoom clip."""


def body() -> SmplxBody:
    for model in BODY_FILES:
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))
    return SmplxBody(MODEL_ROOT)


def first_sample(directory: Path, key: str, tmp_path: Path) -> tuple[SynSample, bytes]:
    raw_asset("MammaSyn shards", directory / "metadata.json")
    sample: SynSample = sequence_samples(sorted(directory.glob("*.tar")), tmp_path / "index", key)[0]
    with ShardReader(directory) as shards:
        return sample, shards.read(sample.labels)


def test_a_render_is_the_locked_head_forward_of_its_shipped_parameters(tmp_path: Path) -> None:
    # The shipped vertices are in the view's camera frame, the parameters in the world: E @ forward must give them back.
    directory: Path = MAMMA_ROOT / SYN_DIR / HANDS
    sample, payload = first_sample(directory, HANDS, tmp_path)
    shipped: dict[int, dict[str, object]] = load_people(payload, "first sample")
    labels: RenderLabels = assemble_render([RenderView(sample.camera, sample.frame, read_render_view(payload, "first sample"))], {sample.camera: (2056, 1504)})
    smplx_body: SmplxBody = body()
    for person, key in zip(labels.people, sorted(shipped), strict=True):
        posed: PosedPerson = smplx_body.pose(person, np.zeros(1, dtype=np.int64), mesh_stride=1)
        cam_ext: np.ndarray = labels.cameras[0].cam_T_world
        vertices: np.ndarray = posed.vertices[0].astype(np.float64) @ cam_ext[:3, :3].T + cam_ext[:3, 3]
        assert np.abs(vertices - shipped[key]["vertices3d"]).max() < 1e-4  # metres: float32 forward
        pixels: np.ndarray = vertices @ labels.cameras[0].K.T
        assert np.abs(pixels[:, :2] / pixels[:, 2:] - shipped[key]["vertices2d"]).max() < 0.05


def test_a_bedlam_body_is_the_v1_1_neutral_forward_moved_by_its_own_cam_ext(tmp_path: Path) -> None:
    directory: Path = MAMMA_ROOT / SYN_DIR / BEDLAM_DATASET / BEDLAM_BATCH
    sample, payload = first_sample(directory, f"{BEDLAM_DATASET}/{BEDLAM_BATCH}", tmp_path)
    shipped: dict[int, dict[str, object]] = load_people(payload, "first sample")
    labels: BedlamLabels = assemble_bedlam([BedlamView(sample.frame, read_bedlam_view(payload, "first sample"))])
    smplx_body: SmplxBody = body()
    for person, transforms in zip(labels.people, labels.cam_T_body, strict=True):
        posed: PosedPerson = in_camera(smplx_body.pose(person, np.zeros(1, dtype=np.int64), mesh_stride=1), transforms)
        pixels: np.ndarray = posed.vertices[0].astype(np.float64) @ labels.K[0].T
        assert np.abs(pixels[:, :2] / pixels[:, 2:] - shipped[person.index]["vertices2d"]).max() < 0.05


@pytest.mark.parametrize("sample", list(SAMPLES))
def test_a_preview_of_each_group_converts(sample: str, tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del nvenc_ffmpeg
    family, key = SAMPLES[sample]
    body()
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family=family, root=MAMMA_ROOT, model_root=MODEL_ROOT, sequences=(key,), frame_limit=PREVIEW, index_root=tmp_path / "index").setup()
    ((identity, source),) = dataset.discover()
    dataset.convert(identity, source, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    assert set(targets) == {"base", "body_pose", "body_mesh", "projections", "masks", "contacts", "visibility"}
    base = read_back(targets["base"])
    capture = recording_properties(base, "capture")
    episode = recording_properties(base, "episode")
    assert capture["num_frames"] == PREVIEW
    error_mm: object = episode["joints3d_max_error_mm"]
    assert isinstance(error_mm, float) and error_mm < 0.1  # the shipped joints, reproduced by the forward
    cameras: object = capture["num_cameras"]
    assert isinstance(cameras, int)
    for rig in range(1, cameras + 1):
        assert len(column_rows(base, f"{schema.video_path(rig, 0)}:VideoStream:sample")) <= PREVIEW  # a camera may lack a frame
    subjects: list[str] = base.reader(index=None, contents="/__properties/**").to_arrow_table().to_pylist()[0]["property:episode:subjects"]
    assert len(subjects) == episode["num_people"]
    pose = read_back(targets["body_pose"])
    for subject in subjects:
        rows = column_rows(pose, f"{schema.person_coco133_xyz_path(int(subject.removeprefix('person_')))}:Points3D:positions")
        assert len(rows) == PREVIEW


def test_a_bedlam_closeup_preview_is_upright_its_keypoints_on_its_masks(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Closeup batches ship portrait renders turned to landscape under portrait intrinsics: turned back, each person's projected
    body keypoints land on that person's mask (as shipped, almost none do)."""
    del nvenc_ffmpeg
    body()
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    key: str = "20221012_1_500_batch01hand_closeup_highSchoolGym/seq_000166"
    dataset = MammaConfig(family="syn-bedlam", root=MAMMA_ROOT, model_root=MODEL_ROOT, sequences=(key,), frame_limit=PREVIEW, index_root=tmp_path / "index").setup()
    ((identity, source),) = dataset.discover()
    dataset.convert(identity, source, force=False)
    targets: dict[str, Path] = dataset.targets(identity)
    base = read_back(targets["base"])
    resolution = column_rows(base, f"{schema.pinhole_path(1, 0)}:Pinhole:resolution").column(1).to_pylist()[0][0]
    assert list(resolution) == [720.0, 1280.0]
    masks = read_back(targets["masks"])
    projections = read_back(targets["projections"])
    subjects: list[str] = base.reader(index=None, contents="/__properties/**").to_arrow_table().to_pylist()[0]["property:episode:subjects"]
    for subject in subjects:
        person: int = int(subject.removeprefix("person_"))
        blobs = column_rows(masks, f"{schema.person_mask_path(1, 0, person)}:EncodedImage:blob").column(1).to_pylist()
        points = column_rows(projections, f"{schema.person_projection_path(1, 0, person)}:Points2D:positions").column(1).to_pylist()
        hits, inside = 0, 0
        for blob, row in zip(blobs, points, strict=True):
            quarter: np.ndarray = np.asarray(Image.open(io.BytesIO(bytes(blob[0]))))
            for x, y in np.asarray(row, dtype=np.float64)[:17]:  # COCO-133's 17 body keypoints
                u, v = int(x // 4), int(y // 4)
                if 0 <= u < quarter.shape[1] and 0 <= v < quarter.shape[0]:
                    inside += 1
                    hits += int(quarter[v, u] != 0)
        assert inside > 0 and hits / inside > 0.6, f"{subject}: {hits} of {inside} body keypoints on the mask"
