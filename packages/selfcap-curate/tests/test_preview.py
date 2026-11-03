"""The preview layer of synthetic segments served by an in-process catalog: window, rigs and content."""

import struct
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
import rerun as rr
import rerun.blueprint as rrb
from dataforge.datasets.selfcap import SelfcapConfig, SelfcapDataset
from dataforge.selfcap_schema import SOMA_MESH, SOMA_XLO
from dataforge.writing import recording_to
from jaxtyping import Int64, UInt8
from numpy import ndarray
from rerun.catalog import DatasetEntry
from rerun.chunk import Chunk, RrdReader
from rerun.server import Server

from selfcap_curate.apis.preview import Config, main
from selfcap_curate.preview import PreviewOptions, PreviewResult, build_segment_preview

FRAME_COUNT: int = 40
FRAME_NS: int = 100_000_000
"""Forty frames every 100 ms: the source videos span 0.0-3.9 s."""
KEYFRAME_INTERVAL: int = 7
"""The fixture encoder starts a GOP every seven frames."""
SOURCE_TIMES_NS: Int64[ndarray, "n"] = np.arange(FRAME_COUNT, dtype=np.int64) * FRAME_NS
SEGMENT: str = "selfcap__2025-12-20__25aeb66a__ep001"
OTHER: str = "selfcap__2025-12-21__1dbe8c49__ep002"


@pytest.fixture(scope="module")
def av1_samples(tmp_path_factory: pytest.TempPathFactory) -> list[bytes]:
    """A tiny real AV1 stream, one temporal unit per frame, keyframe every seven frames."""
    target: Path = tmp_path_factory.mktemp("av1") / "source.ivf"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=192x108:rate=10", "-frames:v", str(FRAME_COUNT),
         "-c:v", "libsvtav1", "-preset", "12", "-g", str(KEYFRAME_INTERVAL), "-pix_fmt", "yuv420p", "-svtav1-params", "pred-struct=1:enable-overlays=0:scd=0", "-f", "ivf", str(target)],
        check=True,
    )
    data: bytes = target.read_bytes()
    position: int = struct.unpack_from("<H", data, 6)[0]
    samples: list[bytes] = []
    while position < len(data):
        size: int = struct.unpack_from("<I", data, position)[0]
        samples.append(data[position + 12 : position + 12 + size])
        position += 12 + size
    assert len(samples) == FRAME_COUNT
    return samples


@dataclass(frozen=True, slots=True)
class SyntheticSegment:
    """What one synthetic base holds; the defaults are a calibrated four-phone session."""

    id: str = SEGMENT
    """Segment id."""
    phones: int = 4
    """Exo rigs before the OAK; three puts the OAK on rig_03 and the Quest on rig_04."""
    posed_phones: tuple[int, ...] = (0, 1, 2, 3)
    """Rigs whose ``cam_00`` has a pose."""
    exo_size: tuple[int, int] = (1920, 1080)
    """Pinhole resolution of the phones."""
    ego_size: tuple[int, int] = (1280, 720)
    """Pinhole resolution of the OAK."""
    soma_mesh: bool = False
    """The segment has a SOMA mesh."""
    soma_xlo: bool = False
    """The segment has the low-vertex SOMA mesh."""
    oak_rig: int | None = None
    """Rig of the OAK, the Quest one after it; None puts it right after the phones."""
    phone_imu: bool = False
    """``rig_03`` has an IMU although it is a phone."""
    keyframe_flags: bool = True
    """The videos carry ``VideoStream:is_keyframe``, one row per keyframe as the mp4 reader writes it."""


DEFAULT_SEGMENT: SyntheticSegment = SyntheticSegment()


def write_segment(path: Path, segment: SyntheticSegment, samples: list[bytes]) -> Path:
    """Log one session's rigs, videos, body and meshes on ``video_time``."""
    times = rr.TimeColumn("video_time", duration=SOURCE_TIMES_NS.astype("timedelta64[ns]"))
    ego_rig: int = segment.phones if segment.oak_rig is None else segment.oak_rig
    quest_rig: int = ego_rig + 1
    keyframe_times = rr.TimeColumn("video_time", duration=SOURCE_TIMES_NS[::KEYFRAME_INTERVAL].astype("timedelta64[ns]"))

    def video(recording: rr.RecordingStream, rig: int, cam: int = 0) -> None:
        entity: str = f"/world/rig_{rig:02d}/cam_{cam:02d}/pinhole/video"
        recording.log(entity, rr.VideoStream(codec=rr.VideoCodec.AV1), static=True)
        recording.send_columns(entity, indexes=[times], columns=rr.VideoStream.columns(sample=[np.frombuffer(sample, np.uint8) for sample in samples]))
        if segment.keyframe_flags:
            recording.send_columns(entity, indexes=[keyframe_times], columns=rr.VideoStream.columns(is_keyframe=[True] * len(SOURCE_TIMES_NS[::KEYFRAME_INTERVAL])))

    def pinhole(recording: rr.RecordingStream, rig: int, size: tuple[int, int]) -> None:
        recording.log(
            f"/world/rig_{rig:02d}/cam_00/pinhole",
            rr.Pinhole(image_from_camera=[[1000.0, 0.0, size[0] / 2], [0.0, 1100.0, size[1] / 2], [0.0, 0.0, 1.0]], width=size[0], height=size[1]),
            static=True,
        )

    with recording_to(path, recording_id=segment.id) as recording:
        recording.send_property("capture", rr.AnyValues(collector="a person"))
        for rig in range(segment.phones):
            recording.log(f"/world/rig_{rig:02d}", rr.AnyValues(kind="exo"), static=True)
            pinhole(recording, rig, segment.exo_size)
            video(recording, rig)
            if rig in segment.posed_phones:
                recording.log(f"/world/rig_{rig:02d}/cam_00", rr.Transform3D(translation=[float(rig), 0.0, 1.0]), static=True)
        if segment.phone_imu:
            recording.send_columns("/world/rig_03/imu_00", indexes=[times], columns=rr.Scalars.columns(scalars=np.zeros(FRAME_COUNT)))
        recording.log(f"/world/rig_{ego_rig:02d}", rr.AnyValues(kind="ego"), static=True)
        recording.log(f"/world/rig_{quest_rig:02d}", rr.AnyValues(kind="quest"), static=True)
        pinhole(recording, ego_rig, segment.ego_size)
        video(recording, ego_rig)
        video(recording, ego_rig, cam=1)
        video(recording, quest_rig, cam=1)
        recording.send_columns(f"/world/rig_{ego_rig:02d}/imu_00", indexes=[times], columns=rr.Scalars.columns(scalars=np.zeros(FRAME_COUNT)))
        recording.log(
            f"/world/rig_{ego_rig:02d}/cam_00/pinhole/skeleton",
            rr.AnnotationContext([rr.ClassDescription(info=rr.AnnotationInfo(id=1, label="body"), keypoint_connections=[(0, 1)])]),
            static=True,
        )
        recording.send_columns(
            f"/world/rig_{ego_rig:02d}/cam_00/pinhole/skeleton",
            indexes=[times],
            columns=rr.Points2D.columns(positions=np.tile([[640.0, 360.0], [1280.0, 720.0]], (FRAME_COUNT, 1)), radii=np.tile([2.0, -3.0], FRAME_COUNT)).partition([2] * FRAME_COUNT),
        )
        recording.log(f"/world/rig_{ego_rig:02d}", rr.Transform3D(translation=[0.0, 0.0, 0.1]), static=True)
        recording.send_columns(
            f"/world/rig_{quest_rig:02d}",
            indexes=[times],
            columns=rr.Transform3D.columns(translation=np.tile([0.0, 1.6, 0.0], (FRAME_COUNT, 1))),
        )
        # A body that moves along +x with the clock: x = frame index, plus one joint that never tracks.
        positions: list[list[list[float]]] = [[[float(frame), 1.0, 2.0], [float(frame), 1.0, 4.0], [float("nan"), 0.0, 0.0]] for frame in range(FRAME_COUNT)]
        recording.log("/world/skeleton", rr.AnnotationContext([rr.ClassDescription(info=rr.AnnotationInfo(id=1, label="body"), keypoint_connections=[(0, 1)])]), static=True)
        recording.send_columns(
            "/world/skeleton",
            indexes=[times],
            columns=rr.Points3D.columns(positions=np.asarray(positions, np.float32).reshape(-1, 3)).partition([3] * FRAME_COUNT),
        )
        if segment.soma_mesh:
            recording.send_columns(SOMA_MESH, indexes=[times], columns=rr.Scalars.columns(scalars=np.zeros(FRAME_COUNT)))
        if segment.soma_xlo:
            recording.send_columns(
                SOMA_XLO,
                indexes=[times],
                columns=rr.Mesh3D.columns(vertex_positions=np.tile(np.eye(3, dtype=np.float32), (FRAME_COUNT, 1))).partition([3] * FRAME_COUNT),
            )
    return path


@dataclass(slots=True)
class Catalog:
    """An in-process catalog, one test long."""

    url: str
    """Where ``Config.catalog_url`` points."""
    entry: DatasetEntry
    """The dataset of the test."""
    root: Path
    """Parent of the layer files."""
    samples: list[bytes]
    """The AV1 stream every synthetic video carries."""

    def add(self, segment: SyntheticSegment) -> None:
        """Register one synthetic segment as its base layer."""
        path: Path = write_segment(self.root / f"{segment.id}.rrd", segment, self.samples)
        self.entry.register([path.as_uri()], layer_name="base").wait()


@pytest.fixture
def catalog(tmp_path: Path, av1_samples: list[bytes]) -> Iterator[Catalog]:
    with Server(port=0) as server:
        yield Catalog(server.url(), server.client().create_dataset("selfcap-test"), tmp_path, av1_samples)


def chunks_of(path: Path) -> list[Chunk]:
    reader: RrdReader = RrdReader(path)
    return list(reader.stream(store=reader.recordings()[0]).to_chunks())


def column(path: Path, entity: str, component: str) -> list:
    """Every row of one component of one entity, in time order, from a written file."""
    rows: list[tuple[int, object]] = []
    for chunk in chunks_of(path):
        batch: pa.RecordBatch = chunk.to_record_batch()
        if str(chunk.entity_path) == entity and component in batch.schema.names:
            stamps: list[int] = batch.column("preview_time").cast(pa.int64()).to_pylist() if "preview_time" in batch.schema.names else [0] * batch.num_rows
            rows.extend(zip(stamps, batch.column(component).to_pylist(), strict=True))
    return [value for _, value in sorted(rows, key=lambda row: row[0])]


def stamps_of(path: Path, entity: str, component: str) -> list[int]:
    """Sorted ``preview_time`` values of one component of one entity."""
    found: list[int] = []
    for chunk in chunks_of(path):
        batch: pa.RecordBatch = chunk.to_record_batch()
        if str(chunk.entity_path) == entity and component in batch.schema.names:
            found.extend(batch.column("preview_time").cast(pa.int64()).to_pylist())
    return sorted(found)


def test_the_layer_holds_the_middle_window_on_its_own_timeline(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(SyntheticSegment())
    output: Path = tmp_path / "preview"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=2.0))
    target: Path = output / f"{SEGMENT}.rrd"
    # Span 0.0-3.9 s, middle 1.95 s, 2 s window: [0.95, 2.95) s holds frames 10-29.
    expected: list[int] = [frame * FRAME_NS for frame in range(10, 30)]
    assert stamps_of(target, "/preview/body/skeleton", "Points3D:positions") == expected
    timelines: set[str] = {name for chunk in chunks_of(target) for name in chunk.timeline_names}
    assert timelines == {"preview_time"}, "only the window exists, on its own timeline"
    reader: RrdReader = RrdReader(target)
    assert reader.recordings()[0].recording_id == SEGMENT
    assert not any(str(chunk.entity_path).startswith("/__properties") for chunk in chunks_of(target)), "a layer restates no segment properties"
    # Frames 10-29 average to x = 19.5; the two tracked joints to y = 1, z = 3; the NaN joint counts for nothing.
    translation = column(target, "/preview/center", "Transform3D:translation")
    assert translation == [[pytest.approx([19.5, 1.0, 3.0])]]


def build_one(tmp_path: Path, catalog: Catalog, segment: SyntheticSegment = DEFAULT_SEGMENT, **options: object) -> Path:
    """Register one segment and write its preview with a 2 s window."""
    catalog.add(segment)
    output: Path = tmp_path / "preview"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=2.0, **options))
    return output / f"{segment.id}.rrd"


def test_the_exo_camera_model_is_scaled_to_the_preview_video(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog)
    assert column(target, "/preview/exo_cam", "Transform3D:translation") == [[pytest.approx([2.0, 0.0, 1.0])]], "rig_02's pose"
    # 1920 -> 480 is a quarter: fx 1000 -> 250, fy 1100 -> 275, cx 960 -> 240, cy 540 -> 135 (Mat3x3 is column-major).
    assert column(target, "/preview/exo_cam/pinhole", "Pinhole:image_from_camera") == [[pytest.approx([250.0, 0.0, 0.0, 0.0, 275.0, 0.0, 240.0, 135.0, 1.0])]]
    assert column(target, "/preview/exo_cam/pinhole", "Pinhole:resolution") == [[pytest.approx([480.0, 270.0])]]


def test_the_ego_keypoints_follow_the_preview_image_scale(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog)
    entity: str = "/preview/exo_cam/pinhole/ego/keypoints"
    assert stamps_of(target, entity, "Points2D:positions") == [frame * FRAME_NS for frame in range(10, 30)]
    # 1280 -> 480 is 3/8: (640, 360) -> (240, 135); a radius in pixels (2) scales, a UI radius (negative) stays.
    assert column(target, entity, "Points2D:positions")[0] == [pytest.approx([240.0, 135.0]), pytest.approx([480.0, 270.0])]
    assert column(target, entity, "Points2D:radii")[0] == [pytest.approx(0.75), pytest.approx(-3.0)]
    assert any("AnnotationContext:context" in chunk.to_record_batch().schema.names for chunk in chunks_of(target) if str(chunk.entity_path) == entity)


def test_only_the_moving_rig_poses_are_copied_each_with_an_empty_points_column(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog)
    assert stamps_of(target, "/world/rig_05", "Transform3D:translation") == [frame * FRAME_NS for frame in range(10, 30)]
    # Rerun previews prefetch transforms only on the timeline with the most rows, so the pose needs a visualizer query.
    assert column(target, "/world/rig_05", "Points3D:positions") == [[]] * 20
    world: set[str] = {str(chunk.entity_path) for chunk in chunks_of(target) if str(chunk.entity_path).startswith("/world")}
    assert world == {"/world/rig_05"}, "static rigs load in previews from the base"


def test_the_soma_xlo_mesh_is_copied_for_the_window(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog, SyntheticSegment(soma_mesh=True, soma_xlo=True))
    assert stamps_of(target, "/preview/body/soma_xlo", "Mesh3D:vertex_positions") == [frame * FRAME_NS for frame in range(10, 30)]


def write_ivf(path: Path, samples: list[bytes], size: tuple[int, int]) -> Path:
    """The IVF container around AV1 temporal units, as ffmpeg reads it."""
    with open(path, "wb") as stream:
        stream.write(b"DKIF" + struct.pack("<HHIHHIIII", 0, 32, 0x31305641, size[0], size[1], 30, 1, len(samples), 0))
        for index, sample in enumerate(samples):
            stream.write(struct.pack("<IQ", len(sample), index) + sample)
    return path


def decoded_gray(path: Path) -> UInt8[ndarray, "n h w"]:
    """Decode an AV1 file with ffmpeg to 480x270 gray frames."""
    raw: bytes = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", "libdav1d", "-i", str(path), "-vf", "scale=480:270:flags=area", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, 270, 480)


def preview_samples(target: Path, entity: str) -> list[bytes]:
    return [bytes(row[0] if isinstance(row, list) else row) for row in column(target, entity, "VideoStream:sample")]


def test_the_videos_are_re_encoded_to_the_preview_size_and_keep_their_window_frames(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog)
    source: UInt8[ndarray, "n h w"] = decoded_gray(write_ivf(tmp_path / "source.ivf", catalog.samples, (192, 108)))
    for entity in ("/preview/exo_cam/pinhole/video", "/preview/exo_cam/pinhole/ego/video"):
        assert stamps_of(target, entity, "VideoStream:sample") == [frame * FRAME_NS for frame in range(10, 30)]
        assert column(target, entity, "VideoStream:codec") == [[rr.components.VideoCodec.AV1.value]]
        low: Path = write_ivf(tmp_path / "low.ivf", preview_samples(target, entity), (480, 270))
        probed: str = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "csv=p=0", str(low)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert probed == "av1,480,270"
        frames: UInt8[ndarray, "n h w"] = decoded_gray(low)
        assert len(frames) == 20
        # The window starts mid-GOP (frame 7 is the keyframe before frame 10): each frame must still be its own source frame.
        for index, frame in enumerate(frames):
            distance: list[float] = [float(np.abs(frame.astype(np.int16) - candidate.astype(np.int16)).mean()) for candidate in source]
            assert int(np.argmin(distance)) == 10 + index
    # Both videos stack in one view: the ego branch starts 4 px below the exo image.
    assert column(target, "/preview/exo_cam/pinhole/ego", "Transform3D:translation") == [[pytest.approx([0.0, 274.0, 0.0])]]


@pytest.mark.parametrize(
    ("segment", "options", "exo_translation"),
    [
        pytest.param(SyntheticSegment(phones=3, posed_phones=(0, 1, 2)), {}, [2.0, 0.0, 1.0], id="three-phone session: the OAK is rig_03"),
        pytest.param(SyntheticSegment(posed_phones=(0, 1, 3)), {}, [1.0, 0.0, 1.0], id="rig_02 unposed: the next posed phone"),
        pytest.param(SyntheticSegment(phones=3, posed_phones=(0,)), {}, [0.0, 0.0, 1.0], id="only rig_00 posed, three phones"),
        pytest.param(SyntheticSegment(phones=3, posed_phones=(0, 1, 2), oak_rig=4), {}, [2.0, 0.0, 1.0], id="three phones, the OAK on rig_04 by its kind"),
        pytest.param(SyntheticSegment(phone_imu=True), {}, [2.0, 0.0, 1.0], id="an IMU on phone rig_03 does not make it the OAK"),
        pytest.param(SyntheticSegment(), {"exo": 0, "ego": 4}, [0.0, 0.0, 1.0], id="named rigs win"),
    ],
)
def test_auto_rigs_pick_the_oak_by_its_kind_and_the_first_posed_phone(tmp_path: Path, catalog: Catalog, segment: SyntheticSegment, options: dict, exo_translation: list[float]) -> None:
    target: Path = build_one(tmp_path, catalog, segment, **options)
    assert column(target, "/preview/exo_cam", "Transform3D:translation") == [[pytest.approx(exo_translation)]]
    assert len(stamps_of(target, "/preview/exo_cam/pinhole/ego/keypoints", "Points2D:positions")) == 20, "the OAK's keypoints: it has them, the Quest has not"
    assert len(stamps_of(target, "/preview/exo_cam/pinhole/ego/video", "VideoStream:sample")) == 20


def test_an_unposed_exo_phone_gets_no_camera_model_and_keeps_its_video(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog, SyntheticSegment(posed_phones=()))
    entities: set[str] = {str(chunk.entity_path) for chunk in chunks_of(target)}
    assert "/preview/exo_cam" not in entities and "/preview/exo_cam/pinhole" not in entities
    assert len(stamps_of(target, "/preview/exo_cam/pinhole/video", "VideoStream:sample")) == 20


def test_a_frame_rate_thins_the_body_and_the_videos_alike(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog, fps=5.0)
    # 5 Hz over a 10 Hz source: every second frame from the window start, frames 10, 12, ... 28.
    expected: list[int] = [frame * FRAME_NS for frame in range(10, 30, 2)]
    assert stamps_of(target, "/preview/body/skeleton", "Points3D:positions") == expected
    assert stamps_of(target, "/preview/exo_cam/pinhole/video", "VideoStream:sample") == expected
    assert stamps_of(target, "/preview/exo_cam/pinhole/ego/video", "VideoStream:sample") == expected
    assert stamps_of(target, "/world/rig_05", "Transform3D:translation") == expected
    assert stamps_of(target, "/preview/exo_cam/pinhole/ego/keypoints", "Points2D:positions") == expected


def test_a_window_longer_than_the_video_takes_the_whole_video(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(SyntheticSegment())
    output: Path = tmp_path / "preview"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=30.0))
    assert stamps_of(output / f"{SEGMENT}.rrd", "/preview/body/skeleton", "Points3D:positions") == SOURCE_TIMES_NS.tolist()


def run_failing(tmp_path: Path, catalog: Catalog, broken: SyntheticSegment, capsys: pytest.CaptureFixture[str]) -> tuple[str, Path]:
    """Build a broken segment beside a good one: the run writes the good one and exits nonzero at the end."""
    good: SyntheticSegment = SyntheticSegment(id=OTHER)
    catalog.add(good)
    catalog.add(broken)
    output: Path = tmp_path / "preview"
    with pytest.raises(SystemExit) as stopped:
        main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=2.0))
    assert stopped.value.code not in (0, None)
    assert (output / f"{OTHER}.rrd").is_file()
    assert sorted(path.name for path in output.iterdir()) == [f"{OTHER}.rrd"], "no partial output of the failed segment"
    lines: list[str] = capsys.readouterr().out.splitlines()
    return next(line for line in lines if line.startswith(f"preview FAIL {broken.id}")), output


@pytest.mark.parametrize(
    ("broken", "reason"),
    [
        (SyntheticSegment(exo_size=(1280, 720)), "resolution"),
        (SyntheticSegment(ego_size=(1920, 1080)), "resolution"),
        (SyntheticSegment(soma_mesh=True), "soma_xlo"),
        (SyntheticSegment(keyframe_flags=False), "keyframe"),
    ],
    ids=["exo resolution", "ego resolution", "soma mesh without soma_xlo", "videos without keyframe flags"],
)
def test_a_bad_segment_fails_alone_with_one_line(tmp_path: Path, catalog: Catalog, broken: SyntheticSegment, reason: str, capsys: pytest.CaptureFixture[str]) -> None:
    line, _ = run_failing(tmp_path, catalog, broken, capsys)
    assert reason in line


def test_a_build_returns_what_it_wrote(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(DEFAULT_SEGMENT)
    target: Path = tmp_path / "layer.rrd"
    result: PreviewResult = build_segment_preview(catalog.entry, SEGMENT, target, PreviewOptions(seconds=2.0))
    assert result == PreviewResult(
        segment=SEGMENT,
        target=target,
        lo_ns=950_000_000,
        hi_ns=2_950_000_000,
        ego_rig=4,
        exo_rig=2,
        exo_camera_model=True,
        moving_rigs=("/world/rig_05",),
        exo_frames=20,
        ego_frames=20,
        file_bytes=target.stat().st_size,
    )


def test_each_segment_uses_its_own_posed_phone(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(SyntheticSegment(id=OTHER, posed_phones=(0,)))
    target: Path = build_one(tmp_path, catalog)
    assert column(target, "/preview/exo_cam", "Transform3D:translation") == [[pytest.approx([2.0, 0.0, 1.0])]]
    assert column(tmp_path / "preview" / f"{OTHER}.rrd", "/preview/exo_cam", "Transform3D:translation") == [[pytest.approx([0.0, 0.0, 1.0])]]


def test_an_existing_target_is_refused_and_left_untouched(tmp_path: Path, catalog: Catalog, capsys: pytest.CaptureFixture[str]) -> None:
    catalog.add(SyntheticSegment())
    catalog.add(SyntheticSegment(id=OTHER))
    output: Path = tmp_path / "preview"
    output.mkdir()
    existing: Path = output / f"{SEGMENT}.rrd"
    existing.write_bytes(b"registered, do not touch")
    with pytest.raises(SystemExit):
        main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=2.0))
    assert existing.read_bytes() == b"registered, do not touch"
    assert (output / f"{OTHER}.rrd").is_file()
    assert any(line.startswith(f"preview FAIL {SEGMENT}") for line in capsys.readouterr().out.splitlines())


def test_a_segment_without_a_soma_mesh_still_builds(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog)
    assert target.is_file()
    assert "/preview/body/soma_xlo" not in {str(chunk.entity_path) for chunk in chunks_of(target)}


def test_named_segments_and_a_key_file_select_only_those(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(SyntheticSegment())
    catalog.add(SyntheticSegment(id=OTHER))
    named: Path = tmp_path / "named"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=named, seconds=2.0, segment_ids=(OTHER,)))
    assert [path.name for path in named.iterdir()] == [f"{OTHER}.rrd"]
    keys: Path = tmp_path / "keys.txt"
    keys.write_text(f"{SEGMENT.removeprefix('selfcap__')}\n\n")
    from_file: Path = tmp_path / "from_file"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=from_file, seconds=2.0, key_file=keys))
    assert [path.name for path in from_file.iterdir()] == [f"{SEGMENT}.rrd"]
    every: Path = tmp_path / "every"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=every, seconds=2.0, workers=2))
    assert sorted(path.name for path in every.iterdir()) == sorted((f"{SEGMENT}.rrd", f"{OTHER}.rrd"))


def test_an_unknown_segment_stops_the_run_before_any_work(tmp_path: Path, catalog: Catalog) -> None:
    catalog.add(SyntheticSegment())
    output: Path = tmp_path / "preview"
    with pytest.raises(ValueError, match="absent"):
        main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, segment_ids=(OTHER,)))
    assert not output.exists()


def test_a_failed_encode_reports_what_ffmpeg_said(tmp_path: Path, catalog: Catalog, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    broken: Path = tmp_path / "ffmpeg"
    broken.write_text("#!/bin/sh\necho 'Unknown decoder libdav1d' >&2\nexit 1\n")
    broken.chmod(0o755)
    monkeypatch.setenv("DATAFORGE_FFMPEG", str(broken))
    catalog.add(SyntheticSegment())
    output: Path = tmp_path / "preview"
    with pytest.raises(SystemExit):
        main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=output, seconds=2.0))
    assert "Unknown decoder libdav1d" in capsys.readouterr().out
    assert list(output.iterdir()) == [], "the work directory is gone too"


def test_the_written_file_stacks_on_its_segment_as_the_preview_layer(tmp_path: Path, catalog: Catalog) -> None:
    target: Path = build_one(tmp_path, catalog, SyntheticSegment(soma_mesh=True, soma_xlo=True))
    before: set[str] = {name for name in catalog.entry.segment_table().schema().names if name.startswith("property:")}
    catalog.entry.register([target.as_uri()], layer_name="preview").wait()
    assert {name for name in catalog.entry.segment_table().schema().names if name.startswith("property:")} == before, "the layer adds no property column"
    assert "/preview/center" in catalog.entry.segment_store(SEGMENT).schema().entity_paths()
    assert isinstance(SelfcapDataset(SelfcapConfig()).catalog_blueprints(catalog.entry)[1], rrb.table.TableBlueprint)
    # A rebuild over the registered layer copies the source rows only, not the earlier preview's.
    rebuilt: Path = tmp_path / "rebuilt"
    main(Config(catalog_url=catalog.url, dataset="selfcap-test", output_dir=rebuilt, seconds=2.0))
    assert stamps_of(rebuilt / f"{SEGMENT}.rrd", "/world/rig_05", "Transform3D:translation") == [frame * FRAME_NS for frame in range(10, 30)]


@pytest.mark.parametrize("bad", [{"workers": 0}, {"seconds": 0.0}, {"fps": 0.0}], ids=["workers", "seconds", "fps"])
def test_a_config_that_cannot_build_stops_before_any_catalog_read(tmp_path: Path, bad: dict) -> None:
    with pytest.raises(ValueError):
        main(Config(catalog_url="rerun+http://127.0.0.1:9", dataset="selfcap-test", output_dir=tmp_path, **bad))
