"""MammaSyn readers: the WebDataset shards of MAMMA's synthetic training set, rebuilt into sequences.

Each sequence directory (``MammaSyn/<group>/<dataset>/<sequence>/``) holds numbered plain tars and a ``metadata.json``. A sample
is one camera view of one frame, three members under one key: ``<key>.png`` (RGBA render), ``<key>.mask.png`` (instance mask,
person + 1) and ``<key>.data.pyd`` (a pickled dict of the people seen in that view). MAMMA's own renders key a sample
``.../<sequence>/<camera>/<frame>``; BEDLAM_MASKS_WD, the BEDLAM release repacked, keys it ``.../png/<clip>/<clip>_<frame>``,
one camera per clip. See ``docs/mamma.md``.
"""

import io
import pickle
import tarfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float64, Int64
from numpy import ndarray
from serde import serde
from serde.json import from_json, to_json

from dataforge import writing
from dataforge.datasets.mamma_source import Gender, MammaCamera, PersonFit, decode

SynFamily: TypeAlias = Literal["syn", "syn-bedlam"]
SYN_DIR: str = "MammaSyn"
"""The synthetic set's directory under the release root (the Hugging Face dataset ``Intelligent-Systems/MammaSyn``)."""
BEDLAM_DATASET: str = "MammaSyn-Singles/BEDLAM_MASKS_WD"
"""The BEDLAM release repacked: its own family (``syn-bedlam``); every other dataset directory holds MAMMA's renders."""
RENDER_FPS: int = 6
"""MAMMA renders its synthetic sequences at 6 fps (MAMMA paper, ``docs/DATASETS.md``): frame N is at N / 6 s."""

MEMBER_KINDS: tuple[str, ...] = ("png", "mask.png", "data.pyd")
"""The three members of every sample: render, instance mask, labels."""
PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {(module, name) for module in ("numpy.core.multiarray", "numpy._core.multiarray") for name in ("_reconstruct", "scalar")}
    | {("numpy", "dtype"), ("numpy", "ndarray")}
)
"""Everything a label pickle may build besides plain containers: numpy arrays and scalars (all a release pickle refers to;
numpy 1 wrote ``numpy.core``, numpy 2 writes ``numpy._core``)."""


@dataclass(frozen=True, slots=True)
class SynSequence:
    """One MammaSyn sequence directory of MAMMA's renders: its shards hold every camera of one scene."""

    group: str
    """``Hands``, ``Interactions`` or ``Singles`` (the ``MammaSyn-<group>`` directory)."""
    dataset: str
    """The dataset directory (``harmony4d_train_1_NC_200_00_contact``): the motion source, rig spread and render batch."""
    directory: Path
    """The sequence directory (``be_HsuS3iLSSWWZ_seq_000000``)."""

    @property
    def parts(self) -> tuple[str, str]:
        return (self.dataset, self.directory.name)

    @property
    def shards(self) -> list[Path]:
        return sorted(self.directory.glob("*.tar"))

    @property
    def metadata(self) -> Path:
        return self.directory / "metadata.json"


def discover_renders(root: Path) -> tuple[list[SynSequence], list[tuple[Path, str]]]:
    """Every render sequence under ``root/MammaSyn``, and each other sequence directory with why it was skipped."""
    found: list[SynSequence] = []
    skipped: list[tuple[Path, str]] = []
    for group_dir in sorted((root / SYN_DIR).glob("MammaSyn-*")):
        for dataset_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            if f"{group_dir.name}/{dataset_dir.name}" == BEDLAM_DATASET:
                continue
            for directory in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                sequence: SynSequence = SynSequence(group_dir.name.removeprefix("MammaSyn-"), dataset_dir.name, directory)
                if not sequence.shards:
                    skipped.append((directory, "no shards"))
                elif not sequence.metadata.is_file():
                    skipped.append((directory, "no metadata.json"))
                else:
                    found.append(sequence)
    return found, skipped


@serde
@dataclass(frozen=True, slots=True)
class ShardMember:
    """Where one member's bytes sit: a shard of the sequence directory and a byte range in it (the tars are uncompressed)."""

    shard: str
    """Tar file name in the sequence directory."""
    offset: int
    size: int


@serde
@dataclass(frozen=True, slots=True)
class SynSample:
    """One camera view of one frame."""

    camera: str
    """The member's directory: the IOI camera of a render, the clip of a BEDLAM sample (one camera each)."""
    frame: int
    """Source frame number from the member name (renders: 6 fps numbering; BEDLAM: 30 fps numbering, every 5th)."""
    image: ShardMember
    mask: ShardMember
    labels: ShardMember


def index_shards(shards: list[Path]) -> list[SynSample]:
    """Every sample of ``shards`` (one sequence directory's tars), sorted by camera then frame; reads tar headers only.

    Raises:
        ValueError: A member has an unknown kind, appears twice, or a sample lacks one of ``MEMBER_KINDS``; names the shard.
    """
    members: dict[tuple[str, int], dict[str, ShardMember]] = defaultdict(dict)
    shard_of: dict[tuple[str, int], str] = {}
    for shard in shards:
        with tarfile.open(shard, "r:") as tar:
            for info in tar:
                if not info.isfile():
                    continue
                camera, leaf = info.name.split("/")[-2:]
                stem, _, kind = leaf.partition(".")
                if kind not in MEMBER_KINDS:
                    raise ValueError(f"{shard}: {info.name} is not one of the sample members {MEMBER_KINDS}")
                key: tuple[str, int] = (camera, int(stem.rsplit("_", 1)[-1]))
                if kind in members[key]:
                    raise ValueError(f"{shard}: {camera}/{leaf} appears twice in the sequence")
                members[key][kind] = ShardMember(shard.name, info.offset_data, info.size)
                shard_of[key] = shard.name
    samples: list[SynSample] = []
    for (camera, frame), found in sorted(members.items()):
        missing: list[str] = [kind for kind in MEMBER_KINDS if kind not in found]
        if missing:
            raise ValueError(f"{shard_of[(camera, frame)]}: sample {camera}/{frame:04d} has no {', '.join(missing)}")
        samples.append(SynSample(camera, frame, found["png"], found["mask.png"], found["data.pyd"]))
    return samples


def read_member(directory: Path, member: ShardMember) -> bytes:
    """One member's bytes from the shard in ``directory``."""
    with ShardReader(directory) as reader:
        return reader.read(member)


class ShardReader:
    """Reads members from a sequence's shards, each shard opened once (an open per member would cost the NAS an NFS lookup)."""

    def __init__(self, directory: Path) -> None:
        self.directory: Path = directory
        self.stack: ExitStack = ExitStack()
        self.streams: dict[str, BinaryIO] = {}

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def read(self, member: ShardMember) -> bytes:
        stream: BinaryIO | None = self.streams.get(member.shard)
        if stream is None:
            stream = self.streams[member.shard] = self.stack.enter_context((self.directory / member.shard).open("rb"))
        stream.seek(member.offset)
        payload: bytes = stream.read(member.size)
        if len(payload) != member.size:
            raise ValueError(f"{self.directory / member.shard}: {len(payload)} of {member.size} bytes at {member.offset}")
        return payload


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ShardStamp:
    """A shard as indexed: a changed size or mtime means the cached index is stale."""

    name: str
    size: int
    mtime_ns: int


@serde(deny_unknown_fields=True)
@dataclass(frozen=True, slots=True)
class ShardIndex:
    """A sequence directory's samples, cached as JSON so a rerun reads no tar headers (BEDLAM's clips are only found that way)."""

    shards: list[ShardStamp]
    samples: list[SynSample]


def sequence_samples(directory: Path, index_root: Path, key: str) -> list[SynSample]:
    """The samples of the shards in ``directory``, from the index cached at ``index_root/<key>.json`` unless a shard changed."""
    stamps: list[ShardStamp] = [ShardStamp(shard.name, (stat := shard.stat()).st_size, stat.st_mtime_ns) for shard in sorted(directory.glob("*.tar"))]
    cache: Path = index_root / f"{key}.json"
    if cache.is_file():
        index: ShardIndex = from_json(ShardIndex, cache.read_text())
        if index.shards == stamps:
            return index.samples
    samples: list[SynSample] = index_shards(sorted(directory.glob("*.tar")))
    with writing.atomic_write(cache) as temp:
        temp.write_text(to_json(ShardIndex(stamps, samples)))
    return samples


class NumpyOnlyUnpickler(pickle.Unpickler):
    """Unpickles plain containers and numpy arrays only: a label pickle naming any other global is refused, not run."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) not in PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"the label pickle refers to {module}.{name}, which is not a numpy array type")
        return super().find_class(module, name)


def load_people(payload: bytes, origin: str) -> dict[int, dict[str, object]]:
    """A ``data.pyd``: the people of one view, keyed by person index. ``origin`` names the sample in errors."""
    try:
        people: object = NumpyOnlyUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, EOFError, ValueError) as error:
        raise ValueError(f"{origin}: {error}") from error
    if not isinstance(people, dict) or not all(isinstance(key, int) and isinstance(record, dict) for key, record in people.items()):
        raise ValueError(f"{origin}: a label pickle holds a dict of person index to record")
    return people


@serde
@dataclass(frozen=True, slots=True)
class RenderPersonLabel:
    """One person in one view of a MAMMA render: the fields the converter keeps (the rest are derived from these or the image)."""

    pose_world: Float64[ndarray, "165"]
    """SMPL-X axis-angle pose, world frame (the release's 165 layout)."""
    shape: Float64[ndarray, "16"]
    """Betas of the locked-head neutral model."""
    trans_world: Float64[ndarray, "3"]
    """smplx ``transl``, world metres."""
    cam_int: Float64[ndarray, "3 3"]
    """The view's K, native pixels."""
    cam_ext: Float64[ndarray, "4 4"]
    """The view's world-to-camera transform (OpenCV axes)."""
    gender: Gender
    joints3d: Float64[ndarray, "127 3"]
    """The model's joints (no face contour), world metres: the check that the parameters reproduce the render's body."""


def read_render_view(payload: bytes, origin: str) -> dict[int, RenderPersonLabel]:
    """The people of one render view, typed."""
    return {key: decode(RenderPersonLabel, origin, record) for key, record in load_people(payload, origin).items()}


@dataclass(frozen=True, slots=True)
class RenderView:
    """One sample's labels: the people one camera sees at one frame."""

    camera: str
    frame: int
    people: dict[int, RenderPersonLabel]


@dataclass(frozen=True, slots=True)
class RenderLabels:
    """A render sequence's labels on its frame rows: every person, every camera."""

    frames: Int64[ndarray, "t"]
    """Source frame numbers of the rows (6 fps numbering); a frame no view has is a gap, not a row."""
    people: list[PersonFit]
    """One fit per person index, over the rows; absent where no view labels the person."""
    joints3d: list[Float64[ndarray, "t 127 3"]]
    """Each person's shipped joints over the rows, NaN where absent."""
    cameras: list[MammaCamera]
    """The fixed cameras, in name order (pinhole: a render has no lens)."""


def assemble_render(views: list[RenderView], sizes: dict[str, tuple[int, int]]) -> RenderLabels:
    """Join the views of a render sequence: a person's parameters must agree in every view that sees them, and a camera's
    calibration in every frame. ``sizes`` gives each camera's image width and height (from its PNGs).

    Raises:
        ValueError: Two views disagree on a person, a camera moves, or a camera sees nobody (no calibration).
    """
    ordered: list[RenderView] = sorted(views, key=lambda view: (view.camera, view.frame))
    frames: Int64[ndarray, "t"] = np.array(sorted({view.frame for view in views}), dtype=np.int64)
    row_of: dict[int, int] = {int(frame): row for row, frame in enumerate(frames)}
    calibration: dict[str, tuple[Float64[ndarray, "3 3"], Float64[ndarray, "4 4"]]] = {}
    for view in ordered:
        for person in view.people.values():
            known: tuple[Float64[ndarray, "3 3"], Float64[ndarray, "4 4"]] = calibration.setdefault(view.camera, (person.cam_int, person.cam_ext))
            if not (np.array_equal(known[0], person.cam_int) and np.array_equal(known[1], person.cam_ext)):
                raise ValueError(f"{view.camera}: at frame {view.frame} the camera moves (a MAMMA render's cameras are fixed)")
    blind: list[str] = sorted(set(sizes) - set(calibration))
    if blind:
        raise ValueError(f"cameras {blind} see nobody in any frame: their labels carry no calibration")
    cameras: list[MammaCamera] = [MammaCamera(name, K, cam_T_world, *sizes[name], lens=None) for name, (K, cam_T_world) in sorted(calibration.items())]

    people: list[PersonFit] = []
    joints3d: list[Float64[ndarray, "t 127 3"]] = []
    for key in sorted({key for view in views for key in view.people}):
        pose: Float64[ndarray, "t 165"] = np.zeros((len(frames), 165))
        trans: Float64[ndarray, "t 3"] = np.zeros((len(frames), 3))
        joints: Float64[ndarray, "t 127 3"] = np.full((len(frames), 127, 3), np.nan)
        present: Bool[ndarray, "t"] = np.zeros(len(frames), dtype=bool)
        seen_by: dict[int, str] = {}
        first: RenderPersonLabel | None = None
        for view in ordered:
            label: RenderPersonLabel | None = view.people.get(key)
            if label is None:
                continue
            row: int = row_of[view.frame]
            if first is None:
                first = label
            if not (np.array_equal(label.shape, first.shape) and label.gender == first.gender):
                raise ValueError(f"person {key}: the shape or gender changes at frame {view.frame} ({view.camera})")
            if present[row]:
                if not (np.array_equal(pose[row], label.pose_world) and np.array_equal(trans[row], label.trans_world) and np.array_equal(joints[row], label.joints3d)):
                    raise ValueError(f"person {key}: the views disagree at frame {view.frame}: {seen_by[row]} and {view.camera}")
                continue
            pose[row], trans[row], joints[row], present[row], seen_by[row] = label.pose_world, label.trans_world, label.joints3d, True, view.camera
        assert first is not None  # every key comes from some view
        people.append(
            PersonFit(
                subject=f"person_{key:02d}",
                gender=first.gender,
                model="locked_head_neutral",
                flat_hand_mean=True,
                pose=pose,
                trans=trans,
                betas=first.shape,
                v_template=None,
                present=present,
            )
        )
        joints3d.append(joints)
    return RenderLabels(frames=frames, people=people, joints3d=joints3d, cameras=cameras)
