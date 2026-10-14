"""MammaSyn readers: the WebDataset shards of MAMMA's synthetic training set, rebuilt into sequences.

Each sequence directory (``MammaSyn/<group>/<dataset>/<sequence>/``) holds numbered plain tars and a ``metadata.json``. A sample
is one camera view of one frame, three members under one key: ``<key>.png`` (RGBA render), ``<key>.mask.png`` (instance mask,
person + 1) and ``<key>.data.pyd`` (a pickled dict of the people seen in that view). MAMMA's own renders key a sample
``.../<sequence>/<camera>/<frame>``; BEDLAM_MASKS_WD, the BEDLAM release repacked, keys it ``.../png/<clip>/<clip>_<frame>``,
one camera per clip. See ``docs/mamma.md``.
"""

import io
import os
import pickle
import tarfile
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, TypeAlias

import numpy as np
from jaxtyping import Bool, Float32, Float64, Int64
from numpy import ndarray
from serde import field, serde
from serde.json import to_json

from dataforge import writing
from dataforge.datasets.mamma_source import VERTICES, Gender, MammaCamera, PersonFit, decode, flags
from dataforge.records import NumpyOnlyUnpickler, read_json

SynFamily: TypeAlias = Literal["syn", "syn-bedlam"]
SYN_DIR: str = "MammaSyn"
"""The synthetic set's directory under the release root (the Hugging Face dataset ``Intelligent-Systems/MammaSyn``)."""
BEDLAM_DATASET: str = "MammaSyn-Singles/BEDLAM_MASKS_WD"
"""The BEDLAM release repacked: its own family (``syn-bedlam``); every other dataset directory holds MAMMA's renders."""
RENDER_FPS: int = 6
"""MAMMA renders its synthetic sequences at 6 fps (MAMMA paper, ``docs/DATASETS.md``): frame N is at N / 6 s."""

MEMBER_KINDS: tuple[str, ...] = ("png", "mask.png", "data.pyd")
"""The three members of every sample: render, instance mask, labels."""
READ_STREAMS: int = 4
"""Spans of a sequence's shards copied at once to local disk ahead of its convert. Measured on the NAS (2026-10-11): sequential reads give about
180 MB/s with 1 stream, 340 with 4, 300 with 8; random reads (a tar header walk, member by member) about 250 per second."""
READ_SPAN: int = 256 << 20
"""Bytes of a shard one stream copies in order."""


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


@serde
@dataclass(frozen=True, slots=True)
class SynMetadata:
    """A sequence or batch directory's ``metadata.json`` (third party: unknown fields allowed)."""

    format: str
    num_samples: int
    """Camera-view samples in the directory's shards."""


def discover_renders(root: Path) -> tuple[list[SynSequence], list[tuple[Path, str]]]:
    """Every render sequence under ``root/MammaSyn``, and each other sequence directory with why it was skipped (no shards, no
    ``metadata.json``, or one that declares no samples: the release ships two such empty directories)."""
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
                elif read_json(sequence.metadata, SynMetadata).num_samples == 0:
                    skipped.append((directory, "empty: metadata.json declares no samples"))
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


def stage_shards(shards: list[Path], into: Path) -> list[Path]:
    """Copy ``shards`` into ``into`` in order, ``READ_STREAMS`` spans of ``READ_SPAN`` at a time, each copy keeping its shard's size
    and mtime (the stamps the index cache is keyed on); a copy already there with them is kept. Returns the copies.

    A sequence is read from a local copy because a header walk over NFS is one random read per member, and the page cache does
    not keep a read-ahead: on dl-server kswapd reclaims it within seconds while the convert's encoders run, with memory free.
    """
    into.mkdir(parents=True, exist_ok=True)
    stats: dict[Path, os.stat_result] = {shard: shard.stat() for shard in shards}
    pending: list[Path] = []
    for shard, stat in stats.items():
        copy: Path = into / shard.name
        kept: os.stat_result | None = copy.stat() if copy.is_file() else None
        if kept is None or (kept.st_size, kept.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
            pending.append(shard)
            with (into / f".{shard.name}.part").open("wb") as created:
                created.truncate(stat.st_size)

    with ThreadPoolExecutor(max_workers=READ_STREAMS) as executor:
        spans: list[Future[None]] = [
            executor.submit(copy_span, shard, into / f".{shard.name}.part", start, min(READ_SPAN, stats[shard].st_size - start))
            for shard in pending
            for start in range(0, stats[shard].st_size, READ_SPAN)
        ]
        for span in spans:
            span.result()
    for shard in pending:
        part: Path = into / f".{shard.name}.part"
        os.utime(part, ns=(stats[shard].st_atime_ns, stats[shard].st_mtime_ns))
        part.rename(into / shard.name)
    return [into / shard.name for shard in shards]


def copy_span(source: Path, target: Path, start: int, length: int) -> None:
    """Copy ``length`` bytes at ``start`` of ``source`` into the same place of the existing file ``target``.

    Raises:
        ValueError: ``source`` ends first (it changed since it was sized); the pre-sized ``target`` would otherwise keep zeros.
    """
    buffer: memoryview = memoryview(bytearray(8 << 20))
    copied: int = 0
    with source.open("rb", buffering=0) as reader, target.open("r+b") as writer:
        reader.seek(start)
        writer.seek(start)
        while copied < length and (count := reader.readinto(buffer[: min(len(buffer), length - copied)])):
            writer.write(buffer[:count])  # buffered: writes all of it
            copied += count
    if copied != length:
        raise ValueError(f"{source}: {copied} of {length} bytes at {start}")


READ_GAP: int = 1 << 20
"""Bytes of other members a read runs through rather than seek past: a seek on the NAS costs more than reading ~1 MB."""
RUN_READS: int = 8
"""Runs ``read_members`` reads at once. Its runs are short random reads, one NAS round trip each (over 250 ms under load,
2026-10-12): read one after another, a BEDLAM clip's ~30 label pickles took about 10 s."""


def read_members(directory: Path, members: list[ShardMember], gap: int = READ_GAP) -> dict[tuple[str, int], bytes]:
    """``members`` read into memory, keyed by (shard, offset): members less than ``gap`` apart come in one read, so a BEDLAM clip,
    whose samples lie in a few runs of its batch's shards, costs a few reads instead of three per sample; ``RUN_READS`` runs at
    once. A gap of 0 reads no bytes but the members' (the label pickles alone, between PNGs that can be under ``READ_GAP``)."""
    runs: list[list[ShardMember]] = []
    for member in sorted(members, key=lambda member: (member.shard, member.offset)):
        last: ShardMember | None = runs[-1][-1] if runs else None
        if last is not None and last.shard == member.shard and member.offset - (last.offset + last.size) <= gap:
            runs[-1].append(member)
        else:
            runs.append([member])

    def read_run(run: list[ShardMember]) -> dict[tuple[str, int], bytes]:
        start: int = run[0].offset
        with ShardReader(directory) as reader:
            payload: bytes = reader.read(ShardMember(run[0].shard, start, max(member.offset + member.size for member in run) - start))
        return {(member.shard, member.offset): payload[member.offset - start : member.offset - start + member.size] for member in run}

    found: dict[tuple[str, int], bytes] = {}
    with ThreadPoolExecutor(max_workers=RUN_READS) as executor:
        for read in executor.map(read_run, runs):
            found.update(read)
    return found


class ShardReader:
    """Reads members from a sequence's shards, each shard opened once (an open per member would cost the NAS an NFS lookup), or
    from ``members`` already read (``read_members``)."""

    def __init__(self, directory: Path, members: dict[tuple[str, int], bytes] | None = None) -> None:
        self.directory: Path = directory
        self.members: dict[tuple[str, int], bytes] | None = members
        self.stack: ExitStack = ExitStack()
        self.streams: dict[str, BinaryIO] = {}

    def __enter__(self) -> "ShardReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stack.close()

    def read(self, member: ShardMember) -> bytes:
        """The member's bytes; a member read ahead must be the whole member (a labels-only read caches only a PNG's header)."""
        if self.members is not None:
            payload: bytes = self.members[(member.shard, member.offset)]
        else:
            stream: BinaryIO | None = self.streams.get(member.shard)
            if stream is None:
                stream = self.streams[member.shard] = self.stack.enter_context((self.directory / member.shard).open("rb"))
            stream.seek(member.offset)
            payload = stream.read(member.size)
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


def sequence_samples(shards: list[Path], index_root: Path, key: str) -> list[SynSample]:
    """The samples of ``shards`` (a directory's tars in name order, or their local copies), from the index cached at
    ``index_root/<key>.json`` unless a shard changed."""
    stamps: list[ShardStamp] = [ShardStamp(shard.name, (stat := shard.stat()).st_size, stat.st_mtime_ns) for shard in shards]
    cache: Path = index_root / f"{key}.json"
    if cache.is_file():
        index: ShardIndex = read_json(cache, ShardIndex)
        if index.shards == stamps:
            return index.samples
    samples: list[SynSample] = index_shards(shards)
    with writing.atomic_write(cache) as temp:
        temp.write_text(to_json(ShardIndex(stamps, samples)))
    return samples


def load_people(payload: bytes, origin: str) -> dict[int, dict[str, object]]:
    """A ``data.pyd``: the people of one view, keyed by person index. ``origin`` names the sample in errors."""
    try:
        people: object = NumpyOnlyUnpickler(io.BytesIO(payload)).load()
    except (pickle.UnpicklingError, EOFError, ValueError) as error:
        raise ValueError(f"{origin}: {error}") from error
    if not isinstance(people, dict) or not all(isinstance(key, int) and isinstance(record, dict) for key, record in people.items()):
        raise ValueError(f"{origin}: a label pickle holds a dict of person index to record")
    return people


def vertex_flags(value: object) -> Bool[ndarray, "n"]:
    """A per-vertex 0/1 column (the release ships int32 in its renders, float64 in BEDLAM) as one flag per vertex."""
    return flags(value).reshape(-1)


def partner_distance(value: object) -> Float32[ndarray, "n"] | None:
    """``sdf_vertices``: one distance per vertex, or None for the empty list the release ships where it has none.

    Raises:
        ValueError: A shipped distance is not finite (only the empty list means none was shipped).
    """
    distances: ndarray = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.isfinite(distances).all():
        raise ValueError(f"sdf_vertices holds {int((~np.isfinite(distances)).sum())} non-finite distances")
    return distances if len(distances) else None


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
    person_visitiblity_rate: float = field(deserializer=float)
    """The release's per-view visibility score, under its own name and definition."""
    floor_contact_mask: Bool[ndarray, "10475"]
    """True where the vertex touches the floor; the same in every view of a frame."""
    vertex_visibility: Bool[ndarray, "10475"] = field(deserializer=vertex_flags)
    """True where this view sees the vertex."""
    sdf_vertices: Float32[ndarray, "10475"] | None = field(default=None, deserializer=partner_distance)
    """Each vertex's signed distance to the other person's surface, metres, negative inside; the same in every view of a frame.
    Only Harmony4D, Inter-X, InteractionCouple and LatinDance10 ship it."""


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
class PersonContact:
    """One person's shipped contact over a sequence's rows."""

    floor: Bool[ndarray, "t 10475"]
    """True where a vertex touches the floor; all False where the person is absent."""
    partner: int | None
    """The person a distance is to: the other one of a two-person scene; None where the source ships no distance."""
    partner_distance_m: Float32[ndarray, "t 10475"] | None
    """Each vertex's signed distance to ``partner``'s surface, metres, negative inside; NaN rows where none is shipped."""


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
    contacts: list[PersonContact]
    """Each person's contact over the rows, in ``people`` order."""


def assemble_render(views: list[RenderView], sizes: dict[str, tuple[int, int]]) -> RenderLabels:
    """Join the views of a render sequence: a person's parameters must agree in every view that sees them, and a camera's
    calibration in every frame. ``sizes`` gives each camera's image width and height (from its PNGs).

    Raises:
        ValueError: Two views disagree on a person (contact included), a camera moves, a camera sees nobody (no calibration), or a
            scene without exactly two people ships a distance to the other person.
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

    keys: list[int] = sorted({key for view in views for key in view.people})
    if len(keys) != 2 and any(label.sdf_vertices is not None for view in views for label in view.people.values()):
        raise ValueError(f"{len(keys)} people ship a distance to the other person: only a two-person scene says whom it is to")
    people: list[PersonFit] = []
    joints3d: list[Float64[ndarray, "t 127 3"]] = []
    contacts: list[PersonContact] = []
    for key in keys:
        pose: Float64[ndarray, "t 165"] = np.zeros((len(frames), 165))
        trans: Float64[ndarray, "t 3"] = np.zeros((len(frames), 3))
        joints: Float64[ndarray, "t 127 3"] = np.full((len(frames), 127, 3), np.nan)
        floor: Bool[ndarray, "t 10475"] = np.zeros((len(frames), VERTICES), dtype=bool)
        distance: Float32[ndarray, "t 10475"] = np.full((len(frames), VERTICES), np.nan, dtype=np.float32)
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
            shipped: Float32[ndarray, "10475"] = np.full(VERTICES, np.nan, dtype=np.float32) if label.sdf_vertices is None else label.sdf_vertices
            if present[row]:
                if not (
                    np.array_equal(pose[row], label.pose_world)
                    and np.array_equal(trans[row], label.trans_world)
                    and np.array_equal(joints[row], label.joints3d)
                    and np.array_equal(floor[row], label.floor_contact_mask)
                    and np.array_equal(distance[row], shipped, equal_nan=True)
                ):
                    raise ValueError(f"person {key}: the views disagree at frame {view.frame}: {seen_by[row]} and {view.camera}")
                continue
            pose[row], trans[row], joints[row], present[row], seen_by[row] = label.pose_world, label.trans_world, label.joints3d, True, view.camera
            floor[row], distance[row] = label.floor_contact_mask, shipped
        assert first is not None  # every key comes from some view
        people.append(
            PersonFit(
                subject=f"person_{key:02d}",
                index=key,
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
        shipped_any: bool = bool(np.isfinite(distance).any())
        contacts.append(PersonContact(floor, (set(keys) - {key}).pop() if shipped_any else None, distance if shipped_any else None))
    return RenderLabels(frames=frames, people=people, joints3d=joints3d, cameras=cameras, contacts=contacts)


@serde
@dataclass(frozen=True, slots=True)
class BedlamPersonLabel:
    """One person in one BEDLAM view: the fields the converter keeps."""

    pose_world: Float32[ndarray, "165"]
    """SMPL-X axis-angle pose in the person's own world."""
    shape: Float32[ndarray, "11"]
    """Betas of SMPL-X v1.1 neutral (BEDLAM fits every body to the neutral model, whatever its gender label)."""
    trans_world: Float64[ndarray, "3"]
    """smplx ``transl`` in the person's own world, metres."""
    cam_int: Float64[ndarray, "3 3"]
    """The view's K, native pixels (a zoom clip's changes every frame)."""
    cam_ext: Float64[ndarray, "4 4"]
    """The person's world to the camera: BEDLAM places each body in a world of its own (OpenCV axes)."""
    gender: Gender
    """The body's gender as BEDLAM labels it; not the model."""
    joints3d: Float64[ndarray, "127 3"]
    """The model's joints (no face contour) in the camera frame: the check that the parameters reproduce the render's body."""
    motion_info: list[str]
    """Where the motion comes from, six strings: the action, a number, the AMASS file, then three numbers."""
    person_visitiblity_rate: float = field(deserializer=float)
    """The release's per-view visibility score, under its own name and definition (BEDLAM writes an unseen person's as the int 0)."""
    floor_contact_mask: Bool[ndarray, "10475"]
    """True where the vertex touches the floor."""
    vertex_visibility: Bool[ndarray, "10475"] = field(deserializer=vertex_flags)
    """True where the camera sees the vertex."""

    def __post_init__(self) -> None:
        if len(self.motion_info) != 6:
            raise ValueError(f"motion_info holds {len(self.motion_info)} strings, BEDLAM ships 6")


def read_bedlam_view(payload: bytes, origin: str) -> dict[int, BedlamPersonLabel]:
    """The people of one BEDLAM view, typed."""
    return {key: decode(BedlamPersonLabel, origin, record) for key, record in load_people(payload, origin).items()}


@dataclass(frozen=True, slots=True)
class BedlamView:
    """One sample of a BEDLAM clip: the people its one camera sees at one frame."""

    frame: int
    people: dict[int, BedlamPersonLabel]


@dataclass(frozen=True, slots=True)
class BedlamLabels:
    """A BEDLAM clip's labels on its frame rows, in the camera frame (the only frame every body shares)."""

    frames: Int64[ndarray, "t"]
    """Source frame numbers (30 fps numbering, every 5th shipped)."""
    K: Float64[ndarray, "t 3 3"]
    """The camera's intrinsics at each row; a row whose view labels nobody takes the nearest labelled row's."""
    people: list[PersonFit]
    """One fit per person, parameters in the person's own world; absent where the view leaves them out."""
    cam_T_body: list[Float64[ndarray, "t 4 4"]]
    """Each person's world to the camera at each row, NaN where absent."""
    joints3d: list[Float64[ndarray, "t 127 3"]]
    """Each person's shipped joints in the camera frame, NaN where absent."""
    contacts: list[PersonContact]
    """Each person's floor contact over the rows (BEDLAM ships no distance between people)."""
    motions: list[list[str]]
    """Each person's ``motion_info``."""


def assemble_bedlam(views: list[BedlamView]) -> BedlamLabels:
    """Join a BEDLAM clip's views (one per frame, in any order) into per-person fits and per-row camera intrinsics.

    Raises:
        ValueError: The people of one view disagree on the intrinsics, a person's shape, gender or motion changes, or no view labels
            anyone.
    """
    ordered: list[BedlamView] = sorted(views, key=lambda view: view.frame)
    frames: Int64[ndarray, "t"] = np.array([view.frame for view in ordered], dtype=np.int64)
    known: dict[int, Float64[ndarray, "3 3"]] = {}
    for row, view in enumerate(ordered):
        for key, person in sorted(view.people.items()):
            first_K: Float64[ndarray, "3 3"] = known.setdefault(row, person.cam_int)
            if not np.array_equal(first_K, person.cam_int):
                raise ValueError(f"frame {view.frame}: person {key} has other intrinsics than the view's first person")
    if not known:
        raise ValueError("no view of the clip labels anyone")
    K: Float64[ndarray, "t 3 3"] = np.stack([known[min(known, key=lambda labelled: abs(labelled - row))] for row in range(len(ordered))])

    keys: list[int] = sorted({key for view in ordered for key in view.people})
    people: list[PersonFit] = []
    cam_T_body: list[Float64[ndarray, "t 4 4"]] = []
    joints3d: list[Float64[ndarray, "t 127 3"]] = []
    contacts: list[PersonContact] = []
    motions: list[list[str]] = []
    for key in keys:
        pose: Float64[ndarray, "t 165"] = np.zeros((len(ordered), 165))
        trans: Float64[ndarray, "t 3"] = np.zeros((len(ordered), 3))
        transforms: Float64[ndarray, "t 4 4"] = np.full((len(ordered), 4, 4), np.nan)
        joints: Float64[ndarray, "t 127 3"] = np.full((len(ordered), 127, 3), np.nan)
        floor: Bool[ndarray, "t 10475"] = np.zeros((len(ordered), VERTICES), dtype=bool)
        present: Bool[ndarray, "t"] = np.zeros(len(ordered), dtype=bool)
        first: BedlamPersonLabel | None = None
        for row, view in enumerate(ordered):
            label: BedlamPersonLabel | None = view.people.get(key)
            if label is None:
                continue
            if first is None:
                first = label
            if not (np.array_equal(label.shape, first.shape) and label.gender == first.gender and label.motion_info == first.motion_info):
                raise ValueError(f"person {key}: the shape, gender or motion changes at frame {view.frame}")
            pose[row], trans[row], transforms[row], joints[row], present[row] = label.pose_world, label.trans_world, label.cam_ext, label.joints3d, True
            floor[row] = label.floor_contact_mask
        assert first is not None  # every key comes from some view
        people.append(
            PersonFit(
                subject=f"person_{key:02d}",
                index=key,
                gender=first.gender,
                model="v1_1_neutral",
                flat_hand_mean=True,
                pose=pose,
                trans=trans,
                betas=first.shape.astype(np.float64),
                v_template=None,
                present=present,
            )
        )
        cam_T_body.append(transforms)
        joints3d.append(joints)
        contacts.append(PersonContact(floor, None, None))
        motions.append(first.motion_info)
    return BedlamLabels(frames=frames, K=K, people=people, cam_T_body=cam_T_body, joints3d=joints3d, contacts=contacts, motions=motions)


BEDLAM_NUMBERING_FPS: int = 30
"""BEDLAM numbers its frames at its 30 fps render rate and ships every 5th: frame N is at N / 30 s, samples 6 Hz apart."""
BEDLAM_STRIDE: int = BEDLAM_NUMBERING_FPS // RENDER_FPS
"""Numbered frames between two shipped BEDLAM samples."""
INDEX_WORKERS: int = 4
"""Batches whose shard headers are read at once on the first discovery (about 3 s per shard over NFS; cached after)."""


@dataclass(frozen=True, slots=True)
class BedlamClip:
    """One BEDLAM clip: one camera, its samples in a batch directory's shards."""

    directory: Path
    """The batch directory (``20221010_3-10_500_batch01hand_zoom_suburb_d``: people, clips, camera motion, scene)."""
    clip: str
    """``seq_000123``."""
    samples: list[SynSample]
    """The clip's samples in frame order."""

    @property
    def parts(self) -> tuple[str, str]:
        return (self.directory.name, self.clip)

    @property
    def turns(self) -> int:
        """Clockwise quarter turns from the shipped PNGs to the camera's image: BEDLAM renders its closeup batches in portrait
        and ships them turned a quarter counter-clockwise, K still for the portrait image (BEDLAM's loaders turn them back)."""
        return 1 if "closeup" in self.directory.name else 0


def discover_bedlam(root: Path, index_root: Path, batches: set[str] | None = None) -> tuple[list[BedlamClip], list[str]]:
    """Every clip of the BEDLAM batches (or of ``batches``) and the problems found; indexes uncached batches' shard headers.

    A batch's samples must number what its ``metadata.json`` declares; a batch that does not is a problem, not a source of clips.
    """
    directories: list[Path] = sorted(path for path in (root / SYN_DIR / BEDLAM_DATASET).iterdir() if path.is_dir() and (batches is None or path.name in batches))

    def index(directory: Path) -> tuple[Path, list[SynSample]]:
        return directory, sequence_samples(sorted(directory.glob("*.tar")), index_root, f"{BEDLAM_DATASET}/{directory.name}")

    clips: list[BedlamClip] = []
    problems: list[str] = []
    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as executor:
        for directory, samples in executor.map(index, directories):
            declared: int = read_json(directory / "metadata.json", SynMetadata).num_samples
            if declared != len(samples):
                problems.append(f"{directory}: {len(samples)} samples in the shards, metadata.json declares {declared}")
                continue
            by_clip: dict[str, list[SynSample]] = defaultdict(list)
            for sample in samples:
                by_clip[sample.camera].append(sample)
            clips += [BedlamClip(directory, clip, by_clip[clip]) for clip in sorted(by_clip)]
    return clips, problems
