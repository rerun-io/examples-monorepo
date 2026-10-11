"""MAMMA on the real release: a frame-limited convert of one sequence per family, and the Vicon lens against the eval masks."""

import io
import os
import tarfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import column_rows, raw_asset, read_back, recording_properties, static_values
from PIL import Image

from dataforge import paths, schema
from dataforge.datasets.mamma import MammaConfig, MammaDataset
from dataforge.datasets.mamma_body import MODEL_FILES, PosedPerson, SmplxBody
from dataforge.datasets.mamma_layers import MAX_PIXELS
from dataforge.datasets.mamma_source import Family, MammaCamera, SequenceLabels, read_camera, read_eval_labels

pytestmark = pytest.mark.golden

MAMMA_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_ROOT", "/mnt/nas/datasets/mamma"))
MODEL_ROOT: Path = Path(os.environ.get("DATAFORGE_MAMMA_MODEL_ROOT", str(paths.raw_root() / "mamma-body-models")))
PREVIEW: int = 30
SAMPLES: dict[str, tuple[Family, str]] = {
    "singles": ("eval", "singles/230929_WhiteRabbit_CatchBall_50048_1"),
    "extra": ("eval", "extra/070525_BedlamLab_Eval_Solo_Dancing_00202_1"),
    "dance": ("eval", "dance/250225_WestCoastSwing_Basic_Whip_03675_03676_1"),
    "markerless-dance": ("markerless-dance", "050825_WestCoastSwing_CutOff_03688_03689_1"),
    "multiple-people": ("markerless-multiple-people", "260216_MultiMama_3_accidental_bump_000111_1"),
    "iphones": ("markerless-iphones", "indoors/crossing_arms"),
}
"""One sequence of each kind: 4112x3008 male Singles, Extra with markers, 32-camera Dance with masks, the three markerless rigs."""


def models() -> None:
    for model in MODEL_FILES.values():
        raw_asset("MAMMA SMPL-X model (dataforge-download mamma)", model.local(MODEL_ROOT))


@pytest.mark.parametrize("sample", list(SAMPLES))
def test_a_preview_of_each_family_converts(sample: str, tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family, key = SAMPLES[sample]
    models()
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = MammaConfig(family=family, root=MAMMA_ROOT, model_root=MODEL_ROOT, sequences=(key,), frame_limit=PREVIEW).setup()
    assert isinstance(dataset, MammaDataset)
    ((identity, sequence),) = dataset.discover()
    raw_asset("MAMMA release sequence", sequence.labels_dir / "global.npz")
    dataset.convert(identity, sequence, force=False)
    targets: dict[str, Path] = {layer: target for layer, target in dataset.targets(identity).items() if target.is_file()}
    expected: set[str] = {"base", "body_pose", "body_mesh", "projections"}
    expected |= {"masks"} if sample in ("singles", "extra", "dance") else set()
    expected |= {"markers"} if sample == "extra" else set()
    assert set(targets) == expected

    labels: SequenceLabels = sequence.labels()
    base = read_back(targets["base"])
    capture = recording_properties(base, "capture")
    assert (capture["num_frames"], capture["source_num_frames"], capture["num_cameras"]) == (PREVIEW, labels.frames, len(sequence.cameras))
    for rig, name in enumerate(sequence.cameras, start=1):
        camera: MammaCamera = read_camera(sequence.calibration(name))
        samples = column_rows(base, f"{schema.video_path(rig, 0)}:VideoStream:sample")
        assert samples[schema.TIMELINE].cast("int64").to_pylist() == [round(frame * 10**9 / labels.fps) for frame in range(PREVIEW)]  # nearest ns
        values = static_values(base, schema.cam_path(rig, 0))
        width: int = int(values.get("stored_width", camera.width))  # pyrefly: ignore[bad-argument-type] (an Arrow integer)
        height: int = int(values.get("stored_height", camera.height))  # pyrefly: ignore[bad-argument-type] (an Arrow integer)
        assert width * height <= MAX_PIXELS
    pose = read_back(targets["body_pose"])
    for person in range(len(labels.people)):
        assert len(column_rows(pose, f"{schema.person_coco133_xyz_path(person)}:Points3D:positions")) == PREVIEW


def raster(pixels: np.ndarray, faces: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The silhouette of a mesh's projected triangles, one at a time (``fillPoly`` fills overlapping polygons by parity)."""
    image: np.ndarray = np.zeros(size[::-1], dtype=np.uint8)
    triangles: np.ndarray = pixels[faces]
    for triangle in np.round(triangles[np.isfinite(triangles).all(axis=(1, 2))] * 16).astype(np.int32):
        cv2.fillConvexPoly(image, triangle, 1, lineType=cv2.LINE_8, shift=4)
    return image.astype(bool)


def test_the_vicon_lens_fits_the_eval_masks_better_than_a_pinhole() -> None:
    """The GT meshes of eval Dance projected with and without the lens, scored against the shipped person masks.

    Scored on the views where the lens moves the body most: near the image centre both agree, and an occluded person's
    mask (visible pixels only) lowers both scores alike. The research run (2026-10-10) found IoU 0.858 against 0.805.
    """
    models()
    directory: Path = MAMMA_ROOT / "mamma_eval_dance/250225_WestCoastSwing_Basic_Whip_03675_03676_1"
    labels: SequenceLabels = read_eval_labels(raw_asset("MAMMA eval Dance GT", directory / "gt/global.npz"))
    frames: np.ndarray = np.arange(0, labels.frames, 15, dtype=np.int64)  # multiples of the mesh stride
    body = SmplxBody(MODEL_ROOT)
    posed: list[PosedPerson] = [body.pose(person, frames) for person in labels.people]
    assert body.faces is not None
    candidates: list[tuple[float, str, int, int]] = []  # (mean lens shift in pixels, camera, frame row, person)
    cameras: dict[str, tuple[MammaCamera, MammaCamera]] = {}
    for path in sorted((directory / "gt").glob("IOI_*.npz")):
        camera: MammaCamera = read_camera(path)
        assert camera.lens is not None
        pinhole_camera: MammaCamera = replace(camera, lens=None)
        cameras[camera.name] = (camera, pinhole_camera)
        for row in range(len(frames)):
            boxes: list[tuple[np.ndarray, np.ndarray]] = []
            for pose in posed:
                pixels: np.ndarray = pinhole_camera.project(pose.vertices[row].astype(np.float64))
                boxes.append((np.nanmin(pixels, axis=0), np.nanmax(pixels, axis=0)))
            for person, pose in enumerate(posed):
                pixels = pinhole_camera.project(pose.vertices[row].astype(np.float64))
                low, high = boxes[person]
                inside: bool = bool(np.isfinite(pixels).all() and (low > 0).all() and (high < [camera.width, camera.height]).all())
                apart: bool = all(
                    (other_high < low).any() or (other_low > high).any() for index, (other_low, other_high) in enumerate(boxes) if index != person
                )
                if inside and apart:
                    shift: float = float(np.linalg.norm(camera.project(pose.vertices[row].astype(np.float64)) - pixels, axis=-1).mean())
                    candidates.append((shift, camera.name, row, person))
    scores: dict[str, list[float]] = {"lens": [], "pinhole": []}
    for _, name, row, person in sorted(candidates, reverse=True)[:12]:
        with tarfile.open(raw_asset("MAMMA eval Dance masks", directory / f"masks/{name}_masks.tar")) as tar:
            member = tar.extractfile(f"mask_{frames[row]:04d}_{person + 1:02d}.png")
            assert member is not None
            mask: np.ndarray = np.asarray(Image.open(io.BytesIO(member.read())).convert("L")) > 0
        for model, projector in zip(("lens", "pinhole"), cameras[name], strict=True):
            silhouette = raster(projector.project(posed[person].vertices[row].astype(np.float64)), body.faces, (projector.width, projector.height))
            scores[model].append(float((silhouette & mask).sum() / (silhouette | mask).sum()))
    lens, pinhole = np.mean(scores["lens"]), np.mean(scores["pinhole"])
    print(f"\nmask IoU over the {len(scores['lens'])} most distorted of {len(candidates)} clear person views: lens {lens:.3f}, pinhole {pinhole:.3f}")
    assert lens > pinhole + 0.02 and lens > 0.8
