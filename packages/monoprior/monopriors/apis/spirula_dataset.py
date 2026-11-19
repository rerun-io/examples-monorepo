"""Export a colsfm cube-face reconstruction as a spirula training folder.

colsfm writes a COLMAP text model whose image names are keys ``<face>/<t_ns>`` with no pixels on disk. This
export decodes exactly the registered frames from the cube-faces layer and writes the folder spirula reads:
``images/<face>/<t_ns>.jpg``, ``masks/<face>/<t_ns>.png`` (255 = keep, 0 = person, from the people-mask layer)
and ``sparse/0/{cameras,images,points3D}.txt`` with the image names pointing at the files. It is a third-party
input format at the boundary, an export, never a source of truth.
"""

import shutil
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float64, UInt8
from numpy import ndarray

from monopriors.cube_faces import CubeFace
from monopriors.cube_layer import iter_framesets, open_rrd, read_cube_faces, read_people_masks

COLMAP_PIXEL_CENTRE_SHIFT_PX: float = 0.5
"""Added to cx, cy and POINTS2D on export: the faces layer and colsfm put the first pixel centre at 0.0, spirula (COLMAP) at 0.5."""


@dataclass(frozen=True, slots=True)
class ColmapCamera:
    """One ``cameras.txt`` row."""

    camera_id: int
    """COLMAP camera id."""
    model: str
    """COLMAP camera model name."""
    width: int
    """Image width, pixels."""
    height: int
    """Image height, pixels."""
    params: Float64[ndarray, " n_params"]
    """Model parameters in COLMAP order."""


@dataclass(frozen=True, slots=True)
class ColmapImageKey:
    """The face and stamp a colsfm image name ``<face>/<t_ns>`` stands for."""

    face: str
    """Face name, e.g. ``cube6_front0``."""
    t_ns: int
    """Frameset stamp on the layer's timeline."""

    @property
    def file_name(self) -> str:
        """The training image's path relative to ``images/``."""
        return f"{self.face}/{self.t_ns}.jpg"


def read_colmap_cameras(text: str) -> dict[int, ColmapCamera]:
    """Parse a COLMAP ``cameras.txt``."""
    cameras: dict[int, ColmapCamera] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields: list[str] = line.split()
        camera: ColmapCamera = ColmapCamera(int(fields[0]), fields[1], int(fields[2]), int(fields[3]), np.asarray(fields[4:], dtype=np.float64))
        cameras[camera.camera_id] = camera
    return cameras


def colmap_camera_line(camera: ColmapCamera, principal_point_shift_px: float) -> str:
    """A ``cameras.txt`` row for a PINHOLE camera (fx fy cx cy) with cx, cy moved by ``principal_point_shift_px``.

    Raises:
        ValueError: For a non-PINHOLE camera.
    """
    if camera.model != "PINHOLE":
        raise ValueError(f"camera {camera.camera_id} is {camera.model}; the export writes PINHOLE only")
    params: Float64[ndarray, "4"] = camera.params + np.array([0.0, 0.0, principal_point_shift_px, principal_point_shift_px])
    return " ".join([str(camera.camera_id), camera.model, str(camera.width), str(camera.height), *(repr(float(value)) for value in params)])


def keep_mask(people: UInt8[ndarray, "h w"]) -> UInt8[ndarray, "h w"]:
    """spirula mask: 255 = keep, 0 = a person (``people`` 1)."""
    return np.where(people == 0, 255, 0).astype(np.uint8)


def _image_records(text: str) -> Iterator[tuple[str, str]]:
    """``images.txt`` as (header line, points line) pairs; the points line may be empty."""
    lines: list[str] = [line for line in text.splitlines() if not line.startswith("#")]
    # Trailing blank lines past the last record would pair up with nothing.
    while len(lines) % 2 == 1 and lines and not lines[-1].strip():
        lines.pop()
    if len(lines) % 2:
        raise ValueError("images.txt: odd number of record lines")
    for index in range(0, len(lines), 2):
        yield lines[index], lines[index + 1]


def parse_image_key(name: str) -> ColmapImageKey:
    """Read a colsfm image name ``<face>/<t_ns>`` (no extension).

    Raises:
        ValueError: If the name is not ``<face>/<integer>``.
    """
    face, separator, stamp = name.partition("/")
    if not separator or not face or not stamp.isdigit():
        raise ValueError(f"image name {name!r} is not <face>/<t_ns>")
    return ColmapImageKey(face=face, t_ns=int(stamp))


def rewrite_image_names(text: str, *, points_shift_px: float) -> tuple[str, dict[int, ColmapImageKey], dict[int, int]]:
    """Point every image record of ``images.txt`` at its exported file ``<face>/<t_ns>.jpg``, POINTS2D moved by ``points_shift_px``.

    Returns:
        The rewritten text, the key per image id and the camera id per image id.
    """
    out: list[str] = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    keys: dict[int, ColmapImageKey] = {}
    camera_ids: dict[int, int] = {}
    for header, points in _image_records(text):
        fields: list[str] = header.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"images.txt: malformed record {header[:80]!r}")
        key: ColmapImageKey = parse_image_key(fields[9].strip())
        image_id: int = int(fields[0])
        keys[image_id] = key
        camera_ids[image_id] = int(fields[8])
        out.append(" ".join([*fields[:9], key.file_name]))
        if points_shift_px:  # (X, Y, POINT3D_ID) triples: move X and Y
            points = " ".join(v if i % 3 == 2 else repr(float(v) + points_shift_px) for i, v in enumerate(points.split()))
        out.append(points)
    return "\n".join(out) + "\n", keys, camera_ids


@dataclass
class SpirulaDatasetConfig:
    """Export colsfm's registered cube-face frames, people masks and model as a spirula training folder."""

    colsfm_sparse: Path
    """colsfm's ``sparse/`` COLMAP text model (``cameras.txt``, ``images.txt``, ``points3D.txt``)."""
    faces_rrd: Path
    """The cube-faces layer ``.rrd``."""
    masks_rrd: Path | None
    """The people-mask layer ``.rrd`` (``<face entity>/pinhole/people_mask``, 1 = person); None keeps every pixel."""
    dataset_dir: Path
    """Output spirula dataset folder."""
    timeline: str = "video_time"
    """Timeline of the layers."""
    jpeg_quality: int = 95
    """JPEG quality of the training images."""
    shift_principal_point: bool = True
    """Add :data:`COLMAP_PIXEL_CENTRE_SHIFT_PX` to cx, cy and every POINTS2D observation (integer pixel centres -> COLMAP's,
    spirula's); poses are unchanged."""

    def __post_init__(self) -> None:
        """Validate ranges."""
        if not 50 <= self.jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality must be in [50, 100], got {self.jpeg_quality}")


def _check_cameras(cameras: dict[int, ColmapCamera], keys: dict[int, ColmapImageKey], camera_ids: dict[int, int], faces: dict[str, CubeFace]) -> None:
    """Every registered image's camera is a PINHOLE of its face's size, close to the layer's K."""
    for image_id, key in keys.items():
        if key.face not in faces:
            raise ValueError(f"image {image_id} names face {key.face!r}, not in the faces layer ({sorted(faces)})")
        camera: ColmapCamera = cameras[camera_ids[image_id]]
        face: CubeFace = faces[key.face]
        if camera.model != "PINHOLE" or (camera.width, camera.height) != (face.width, face.height):
            raise ValueError(
                f"camera {camera.camera_id} is {camera.model} {camera.width}x{camera.height}; face {face.name} is PINHOLE {face.width}x{face.height}"
            )
        layer_params: Float64[ndarray, "4"] = np.array([face.k_matrix[0, 0], face.k_matrix[1, 1], face.k_matrix[0, 2], face.k_matrix[1, 2]])
        if np.max(np.abs(camera.params - layer_params)) > 1.0:
            print(f"warning: camera {camera.camera_id} ({face.name}) params {camera.params} differ from the layer's {layer_params} by more than 1 px")


def _write_jpeg(path: Path, rgb: UInt8[ndarray, "h w 3"], quality: int) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"could not write {path}")
    path.chmod(0o644)


def _write_mask(path: Path, keep: UInt8[ndarray, "h w"]) -> None:
    if not cv2.imwrite(str(path), keep):
        raise OSError(f"could not write {path}")
    path.chmod(0o644)


def main(config: SpirulaDatasetConfig) -> None:
    """Write images, masks and the renamed model for every image colsfm registered."""
    shift_px: float = COLMAP_PIXEL_CENTRE_SHIFT_PX if config.shift_principal_point else 0.0
    images_text, keys, camera_ids = rewrite_image_names((config.colsfm_sparse / "images.txt").read_text(), points_shift_px=shift_px)
    cameras: dict[int, ColmapCamera] = read_colmap_cameras((config.colsfm_sparse / "cameras.txt").read_text())
    wanted: dict[str, set[int]] = {}
    for key in keys.values():
        wanted.setdefault(key.face, set()).add(key.t_ns)
    sparse_dir: Path = config.dataset_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    with open_rrd(config.faces_rrd, "faces") as faces_layer:
        all_faces: tuple[CubeFace, ...] = read_cube_faces(faces_layer)
        _check_cameras(cameras, keys, camera_ids, {face.name: face for face in all_faces})
        faces: tuple[CubeFace, ...] = tuple(face for face in all_faces if face.name in wanted)
        for face in faces:
            (config.dataset_dir / "images" / face.name).mkdir(parents=True, exist_ok=True)
        n_images: int = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            writes: list[Future[None]] = []
            for frameset in iter_framesets(faces_layer, faces, config.timeline, set().union(*wanted.values())):
                for face in faces:
                    if frameset.t_ns in wanted[face.name]:
                        path: Path = config.dataset_dir / "images" / face.name / f"{frameset.t_ns}.jpg"
                        writes.append(pool.submit(_write_jpeg, path, frameset.rgb_by_face[face.name], config.jpeg_quality))
                        n_images += 1
            for write in writes:
                write.result()
    if n_images != len(keys):
        raise ValueError(f"wrote {n_images} images but colsfm registered {len(keys)}: some registered stamps are not in the faces layer")
    with open_rrd(config.masks_rrd, "masks") if config.masks_rrd is not None else nullcontext() as masks_layer:
        for face in faces:
            people_by_t: dict[int, UInt8[ndarray, "h w"]] | None = None
            if masks_layer is not None:
                people_by_t = read_people_masks(masks_layer, face, config.timeline, wanted[face.name])
                missing: set[int] = wanted[face.name] - set(people_by_t)
                if missing:
                    raise ValueError(f"{face.name}: no people_mask for {len(missing)} registered frames, e.g. {sorted(missing)[:3]}")
            (config.dataset_dir / "masks" / face.name).mkdir(parents=True, exist_ok=True)
            for t_ns in sorted(wanted[face.name]):
                keep: UInt8[ndarray, "h w"] = keep_mask(people_by_t[t_ns]) if people_by_t is not None else np.full((face.height, face.width), 255, np.uint8)
                _write_mask(config.dataset_dir / "masks" / face.name / f"{t_ns}.png", keep)
    (sparse_dir / "images.txt").write_text(images_text)
    used_camera_ids: set[int] = set(camera_ids.values())
    camera_lines: list[str] = ["# Camera list with one line of data per camera:", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    camera_lines += [colmap_camera_line(camera, shift_px) for camera in cameras.values() if camera.camera_id in used_camera_ids]
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    shutil.copyfile(config.colsfm_sparse / "points3D.txt", sparse_dir / "points3D.txt")
    for path in sparse_dir.iterdir():
        path.chmod(0o644)
    print(
        f"exported {n_images} images, {n_images} masks ({'people' if config.masks_rrd is not None else 'keep all'}), cx/cy +{shift_px} px, {len(used_camera_ids)} cameras over {len(faces)} faces -> {config.dataset_dir}"
    )
