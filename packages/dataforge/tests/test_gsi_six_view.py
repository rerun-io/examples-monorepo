"""GSI Egocentric Six-View: download from a loopback copy of GSI's API, and the repack of its rrds into dataforge layers."""

import hashlib
import json
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import requests
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from conftest import assert_raw_root_guarded, read_back, read_chunks, recording_properties, static_values
from jaxtyping import Float32, Float64, Int64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import blueprints, schema, transports, writing
from dataforge.datasets import gsi_six_view_layers
from dataforge.datasets.gsi_six_view import GsiSixViewConfig, GsiSixViewDataset
from dataforge.datasets.gsi_six_view_remote import fetch_signed
from dataforge.datasets.gsi_six_view_source import FLU_R_RDF, VIEWS

EDITION: str = "01a0f166-7994-7b43-a622-a325541c4356"
"""The 10 h edition's id, as GSI's API names it."""
USERNAME: str = "gsi-ots-10h"
PASSWORD: str = "partner-password-SECRET"
"""What the fake API accepts; no test may see it in a message."""
SIGNATURE: str = "SIGNATURE-SECRET"
"""Every signed link carries it; no test may see it in a message."""
EPISODES: tuple[str, ...] = ("ep_v1_aaaa", "ep_v1_bbbb", "ep_v1_cccc")


T0_NS: int = 1_000_000_000
"""``capture_time`` of a fake episode's first frame; GSI's clock is device duration, not wall time."""
FRAME_NS: int = 33_333_333
"""GSI's 30 fps frame step."""
FRAMES: int = 3
"""Frames per fake episode."""
DROPOUT: dict[str, int] = {"ep_v1_cccc": 1}
"""Fake episodes whose SLAM skips a frame, as two of GSI's real ones do (frame 115 of ep_v1_0afe40df…, 147 of ep_v1_23887c95…)."""


def view_rotation(cam: int) -> Float64[ndarray, "3 3"]:
    """A fake ``rig_R_cam`` (FLU to FLU): identity for left_front, then a yaw and a pitch that grow with the camera."""
    return Rotation.from_euler("zy", [12.0 * cam, -7.0 * cam], degrees=True).as_matrix()


def view_translation(cam: int) -> Float64[ndarray, "3"]:
    """A fake ``rig_t_cam`` in metres; zero for left_front, the rig frame."""
    return np.array([0.01 * cam, -0.02 * cam, 0.005 * cam])


def view_intrinsics(cam: int) -> Float64[ndarray, "3 3"]:
    """A fake K per view; the principal point moves with the camera, so a swapped view shows."""
    return np.array([[500.0, 0.0, 960.0 + cam], [0.0, 501.0, 540.0 - cam], [0.0, 0.0, 1.0]])


def world_T_rig(frame: int) -> Float64[ndarray, "4 4"]:
    """A fake SLAM pose of GSI's rig frame: walking along x, turning about z."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 10.0 * frame, degrees=True).as_matrix()
    pose[:3, 3] = [0.1 * frame, 0.0, 1.5]
    return pose


def rig_keypoints(frame: int, side: int) -> Float32[ndarray, "21 3"]:
    """Fake joints in the rig frame: half a metre ahead of left_front, one hand each side, moving with the frame."""
    wrist: Float64[ndarray, "3"] = np.array([0.5, 0.1 - 0.2 * side, -0.2 + 0.01 * frame])
    return (wrist + 0.004 * np.arange(21)[:, None] * np.array([1.0, 0.0, 0.5])).astype(np.float32)


def episode_rrds(episode: str) -> dict[str, bytes]:
    """A two-file GSI episode with the shipped layout: hands.rrd (six views, both hands, calibration, episode properties) and
    slam.rrd (the rig pose on /slam/rig, /camera and /hand, plus two poses that carry only ``capture_time``)."""
    files: dict[str, bytes] = {}
    for name in ("hands.rrd", "slam.rrd"):
        recording = rr.RecordingStream(application_id="gsi_off_the_shelf", recording_id=episode, send_properties=False)
        stream = rr.binary_stream(recording)
        recording.send_recording_name(episode)
        recording.send_recording_start_time_nanos(1_758_686_498_000_000_000)
        recording.log("/", rr.ViewCoordinates.FLU, static=True)
        if name == "hands.rrd":
            recording.send_property(
                "episode", rr.AnyValues(episode_id=episode, caption=f"caption of {episode}", n_frames=FRAMES, duration_s=0.1, start_ns=T0_NS, source_frame_start=100)
            )
            recording.log(
                "/meta/calibration",
                rr.AnyValues(
                    views=list(VIEWS),
                    K=np.stack([view_intrinsics(cam) for cam in range(6)]).reshape(-1),
                    resolution=[1920, 1080] * 6,
                    front_P1=np.arange(12.0),
                    eye_baseline=[0.108],
                    Q_frame=["rectified_optical"],
                    left_side_T_cam_imu=np.eye(4).reshape(-1),
                ),
                static=True,
            )
            for cam, view in enumerate(VIEWS):
                recording.log(f"/camera/{view}", rr.Transform3D(mat3x3=view_rotation(cam), translation=view_translation(cam)), static=True)
                recording.log(
                    f"/camera/{view}/image",
                    rr.Pinhole(image_from_camera=view_intrinsics(cam), resolution=[1920, 1080], camera_xyz=rr.ViewCoordinates.FLU, image_plane_distance=0.02),
                    static=True,
                )
                recording.log(f"/camera/{view}/image", rr.VideoStream(codec=rr.VideoCodec.H265), static=True)
            for side in ("left", "right"):
                recording.log(f"/hand/{side}/keypoints3d", rr.Points3D.from_fields(class_ids=1, keypoint_ids=range(21), radii=0.006), static=True)
            for frame in range(FRAMES):
                recording.set_time("capture_time", duration=np.timedelta64(T0_NS + frame * FRAME_NS, "ns"))
                recording.set_time("frame", sequence=frame)
                recording.set_time("source_frame", sequence=100 + frame)
                for view in VIEWS:
                    recording.log(f"/camera/{view}/image", rr.VideoStream.from_fields(sample=b"\x00\x00\x00\x01" + bytes([frame])))
                for side_index, side in enumerate(("left", "right")):
                    joints: Float32[ndarray, "21 3"] = rig_keypoints(frame, side_index)
                    recording.log(f"/hand/{side}/keypoints3d", rr.Points3D.from_fields(positions=joints))
                    vertices: Float32[ndarray, "778 3"] = joints[0] + np.linspace(0.0, 0.05, 778 * 3, dtype=np.float32).reshape(778, 3)
                    recording.log(f"/hand/{side}/vertices", rr.Mesh3D(vertex_positions=vertices, triangle_indices=[[0, 1, 2], [2, 3, 4]]))
                    recording.log(
                        f"/hand/{side}/mano",
                        rr.AnyValues(
                            hand_pose=np.zeros(135), betas=np.zeros(10), global_orient=np.eye(3).reshape(-1), transl=joints[0].astype(np.float64),
                            is_detected=[True], is_interpolated=[False], is_usable=[frame != 1 or side == "right"],
                        ),
                    )  # fmt: skip
        else:
            recording.log("/slam/path", rr.LineStrips3D([[world_T_rig(frame)[:3, 3] for frame in range(FRAMES)]]), static=True)
            recording.log("/slam/meta", rr.AnyValues(json=['{"method": "fake"}']), static=True)
            recording.set_time("capture_time", duration=np.timedelta64(T0_NS - FRAME_NS, "ns"))
            recording.log("/slam/rig", rr.Transform3D(translation=[9.0, 9.0, 9.0]))
            for frame in range(FRAMES):
                if DROPOUT.get(episode) == frame:
                    continue
                recording.set_time("capture_time", duration=np.timedelta64(T0_NS + frame * FRAME_NS, "ns"))
                recording.set_time("frame", sequence=frame)
                recording.set_time("source_frame", sequence=100 + frame)
                pose: Float64[ndarray, "4 4"] = world_T_rig(frame)
                for entity in ("/slam/rig", "/camera", "/hand"):
                    recording.log(entity, rr.Transform3D(translation=pose[:3, 3], quaternion=Rotation.from_matrix(pose[:3, :3]).as_quat()))
        recording.flush()
        data: bytes | None = stream.read()
        assert data is not None
        files[name] = data
    return files


@dataclass
class FakeGsi:
    """GSI's API and its signed R2 links, served on loopback."""

    files: dict[str, dict[str, bytes]]
    """Episode id → file name → bytes."""
    page_size: int = 2
    """Episodes per ``/v1/datasets/files`` page (GSI's CLI asks for 25)."""
    expired: set[str] = field(default_factory=set)
    """``<episode>/<name>`` links answered 403 the first time, the way an hour-old link is."""
    corrupt: set[str] = field(default_factory=set)
    """``<episode>/<name>`` links that serve as many bytes as the listing promises, but not the same ones."""
    fetched: list[str] = field(default_factory=list)
    """``<episode>/<name>`` of every file body served, in order."""
    reorder_on_refresh: bool = False
    """A page asked for again lists its episodes in reverse order."""
    vanish_on_refresh: set[str] = field(default_factory=set)
    """Episodes a page asked for again no longer lists."""
    pages_served: dict[int, int] = field(default_factory=dict)
    """Times each page offset was served."""
    url: str = ""
    """``http://127.0.0.1:<port>`` once serving."""

    def manifest(self) -> dict[str, object]:
        def entry(episode: str, name: str) -> dict[str, object]:
            data: bytes = self.files[episode][name]
            return {"path": f"episodes/{episode}/rerun/{name}", "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

        return {
            "dataset": "GSI Egocentric Six-View",
            "edition": "10h",
            "release": "v1",
            "source_release": "2026-09-v2",
            "schema_version": "gsi.ots-portable-edition.v1",
            "rrd_contract": "gsi.ots.customer-rrd.v2",
            "privacy_status": "PRIVATE_INTERIM_EXISTING_BLUR",
            "category_method": "caption task family",
            "total_hours": 0.01,
            "total_size_bytes": 1,
            "artifact_path_base": "../../",
            "episodes": [
                {
                    "episode_id": episode,
                    "caption": f"caption of {episode}",
                    "category": "Food preparation",
                    "duration_s": 0.1,
                    "views": 6,
                    "hands_rrd": entry(episode, "hands.rrd"),
                    "slam_rrd": entry(episode, "slam.rrd"),
                }
                for episode in self.files
            ],
        }

    def signed(self, path: str) -> str:
        return f"{self.url}/r2/{path}?X-Amz-Expires=3600&X-Amz-Signature={SIGNATURE}"

    def page(self, offset: int, limit: int) -> dict[str, object]:
        episodes: list[str] = list(self.files)[offset : offset + min(limit, self.page_size)]
        self.pages_served[offset] = self.pages_served.get(offset, 0) + 1
        if self.pages_served[offset] > 1:
            episodes = [episode for episode in episodes if episode not in self.vanish_on_refresh]
            episodes = episodes[::-1] if self.reorder_on_refresh else episodes
        following: int = offset + len(episodes)
        return {
            "edition_id": EDITION,
            "offset": offset,
            "total": len(self.files),
            "next_offset": following if following < len(self.files) else None,
            "items": [
                {
                    "episode_id": episode,
                    "files": [
                        {"service": name.split(".")[0].upper(), "name": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "url": self.signed(f"{episode}/{name}")}
                        for name, data in self.files[episode].items()
                    ],
                }
                for episode in episodes
            ],
        }


@pytest.fixture
def gsi() -> Iterator[FakeGsi]:
    fake = FakeGsi({episode: episode_rrds(episode) for episode in EPISODES})

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, status: int, body: bytes = b"", content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def authorized(self) -> bool:
            return self.headers.get("Authorization") == "Bearer access-token"

        def do_POST(self) -> None:
            body: dict[str, str] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/auth/partner-login":
                accepted: bool = body == {"username": USERNAME, "password": PASSWORD}
                tokens: dict[str, object] = {"access_token": "access-token", "refresh_token": "refresh", "expires_in": 86400, "token_type": "Bearer"}
                self.reply(200, json.dumps(tokens).encode()) if accepted else self.reply(401, b'{"error": {"message": "nope"}}')
            elif self.path == "/v1/datasets/manifest" and self.authorized() and body == {"edition_id": EDITION}:
                self.reply(200, json.dumps({"filename": "gsi-10h-manifest.json", "url": fake.signed("manifest.json"), "expires_at": "x"}).encode())
            else:
                self.reply(404)

        def do_HEAD(self) -> None:
            self.reply(403)  # R2 signs a link for GET only

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query: dict[str, list[str]] = parse_qs(url.query)
            if url.path == "/v1/datasets/files" and self.authorized() and query["edition"] == [EDITION]:
                self.reply(200, json.dumps(fake.page(int(query["offset"][0]), int(query["limit"][0]))).encode())
            elif url.path.startswith("/r2/") and query.get("X-Amz-Signature") == [SIGNATURE]:
                key: str = url.path.removeprefix("/r2/")
                if key == "manifest.json":
                    self.reply(200, json.dumps(fake.manifest()).encode())
                elif key in fake.expired:
                    fake.expired.discard(key)
                    self.reply(403, b"<Error>expired</Error>", "application/xml")
                else:
                    episode, name = key.split("/")
                    fake.fetched.append(key)
                    data: bytes = fake.files[episode][name]
                    self.reply(200, data[:-1] + bytes([data[-1] ^ 1]) if key in fake.corrupt else data, "application/octet-stream")
            else:
                self.reply(404)

        def log_message(self, format: str, *_args: object) -> None:
            """Keep pytest's captured output about dataforge, not about HTTP."""

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


UPSIDE_DOWN: tuple[int, ...] = (2, 3)
"""left_eye and right_eye: GSI stores them upside down; convert turns them 180°."""
HALF_TURN: Float64[ndarray, "4 4"] = np.diag([-1.0, -1.0, 1.0, 1.0])
"""180° about an RDF camera's optical axis."""


@dataclass
class Reencoded:
    """The streams convert sent to the upright re-encode, in place of the GPU encoder the unit lane does not have."""

    calls: list[tuple[int, int, list[int]]] = field(default_factory=list)
    """(camera, samples, frame indices) per call."""


@pytest.fixture(autouse=True)
def reencoded(monkeypatch: pytest.MonkeyPatch) -> Reencoded:
    record = Reencoded()

    def fake(recording: rr.RecordingStream, cam: int, samples: Iterable[bytes | memoryview], *, times_ns: object, frame_indices: object) -> None:
        record.calls.append((cam, sum(1 for _ in samples), np.asarray(frame_indices).tolist()))

    monkeypatch.setattr(gsi_six_view_layers, "write_upright_video", fake)
    return record


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GSI_PARTNER_USERNAME", USERNAME)
    monkeypatch.setenv("GSI_PARTNER_PASSWORD", PASSWORD)


def config(gsi: FakeGsi, root: Path, *sequences: str) -> GsiSixViewConfig:
    return GsiSixViewConfig(root=root, api_url=gsi.url, sequences=sequences or None)


def test_download_fetches_the_manifest_and_only_the_selected_episodes(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    config(gsi, tmp_path, "ep_v1_cccc").setup().download()

    assert json.loads((tmp_path / "manifest.json").read_text()) == gsi.manifest()
    assert gsi.fetched == ["ep_v1_cccc/hands.rrd", "ep_v1_cccc/slam.rrd"]
    for name, data in gsi.files["ep_v1_cccc"].items():
        assert (tmp_path / "episodes/ep_v1_cccc" / name).read_bytes() == data
    assert not (tmp_path / "episodes/ep_v1_aaaa").exists()


def test_a_rerun_fetches_nothing_already_on_disk(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    config(gsi, tmp_path, "ep_v1_aaaa").setup().download()
    config(gsi, tmp_path, "ep_v1_aaaa", "ep_v1_bbbb").setup().download()

    assert gsi.fetched == ["ep_v1_aaaa/hands.rrd", "ep_v1_aaaa/slam.rrd", "ep_v1_bbbb/hands.rrd", "ep_v1_bbbb/slam.rrd"]


def test_an_expired_link_is_signed_again(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    gsi.expired.add("ep_v1_bbbb/slam.rrd")
    config(gsi, tmp_path, "ep_v1_bbbb").setup().download()

    assert (tmp_path / "episodes/ep_v1_bbbb/slam.rrd").read_bytes() == gsi.files["ep_v1_bbbb"]["slam.rrd"]


@pytest.mark.parametrize("episode", ["ep_v1_aaaa", "ep_v1_bbbb"])
def test_a_refreshed_link_is_matched_by_episode_and_name(episode: str, gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    gsi.expired.add(f"{episode}/hands.rrd")
    gsi.reorder_on_refresh = True
    config(gsi, tmp_path, episode).setup().download()

    assert (tmp_path / "episodes" / episode / "hands.rrd").read_bytes() == gsi.files[episode]["hands.rrd"]


def test_a_refreshed_listing_without_the_episode_says_so(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    gsi.expired.add("ep_v1_aaaa/hands.rrd")
    gsi.vanish_on_refresh.add("ep_v1_aaaa")
    with pytest.raises(ValueError, match="no longer lists ep_v1_aaaa/hands.rrd"):
        config(gsi, tmp_path, "ep_v1_aaaa").setup().download()


def test_bytes_that_differ_from_the_listing_never_take_the_name(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    gsi.corrupt.add("ep_v1_aaaa/hands.rrd")
    with pytest.raises(transports.IntegrityError, match="sha256"):
        config(gsi, tmp_path, "ep_v1_aaaa").setup().download()

    assert not list((tmp_path / "episodes").rglob("*.rrd*"))


def test_an_unknown_episode_is_refused_before_any_fetch(gsi: FakeGsi, signed_in: None, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ep_v1_zzzz"):
        config(gsi, tmp_path, "ep_v1_aaaa", "ep_v1_zzzz").setup().download()

    assert gsi.fetched == []


def test_no_password_or_signed_link_reaches_a_message(gsi: FakeGsi, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GSI_PARTNER_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="GSI_PARTNER_PASSWORD"):
        config(gsi, tmp_path).setup().download()

    monkeypatch.setenv("GSI_PARTNER_USERNAME", USERNAME)
    monkeypatch.setenv("GSI_PARTNER_PASSWORD", PASSWORD + "-wrong")
    with pytest.raises(ValueError, match="HTTP 401") as refused:
        config(gsi, tmp_path).setup().download()
    assert PASSWORD not in str(refused.value)

    monkeypatch.setenv("GSI_PARTNER_PASSWORD", PASSWORD)
    gsi.expired.update({"ep_v1_aaaa/hands.rrd"})
    gsi.corrupt.add("ep_v1_aaaa/hands.rrd")
    with pytest.raises(transports.IntegrityError) as corrupt:
        config(gsi, tmp_path, "ep_v1_aaaa").setup().download()
    assert SIGNATURE not in str(corrupt.value) and "access-token" not in str(corrupt.value)


def test_a_failed_connection_never_quotes_the_signed_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    link: str = f"https://r2.example/episodes/a/hands.rrd?X-Amz-Signature={SIGNATURE}"

    def refuse(url: str, **_kwargs: object) -> None:
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(RuntimeError) as failed:
        fetch_signed(link, tmp_path / "hands.rrd", label="GSI a/hands.rrd")
    assert SIGNATURE not in str(failed.value) and failed.value.__cause__ is None and failed.value.__suppress_context__


def converted(gsi: FakeGsi, raw: Path, out: Path, monkeypatch: pytest.MonkeyPatch, episode: str, *, keep_download: bool = False) -> dict[str, Path]:
    """Download and convert one episode; its layer targets."""
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(out))
    dataset = GsiSixViewConfig(root=raw, api_url=gsi.url, sequences=(episode,), keep_download=keep_download).setup()
    dataset.download()
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=False)
    return dataset.targets(identity)


def rows(store: rrc.ChunkStore, entity: str, component: str) -> list:
    """One component of an entity, one value per ``frame_index`` row in order."""
    table = store.reader(index=schema.FRAME_INDEX, contents=entity).to_arrow_table().sort_by(schema.FRAME_INDEX)
    return table.column(next(name for name in table.column_names if name.endswith(component))).to_pylist()


def rig_T_cam(store: rrc.ChunkStore, cam: int) -> Float64[ndarray, "4 4"]:
    """A camera's static ``rig_T_cam`` as written."""
    values: dict[str, object] = static_values(store, schema.cam_path(0, cam))
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.asarray(values["mat3x3"], dtype=np.float64).reshape(3, 3).T
    pose[:3, 3] = np.asarray(values["translation"], dtype=np.float64).reshape(3)
    return pose


def written_world_T_rig(store: rrc.ChunkStore, frame: int) -> Float64[ndarray, "4 4"]:
    """The rig node's temporal pose on one frame, as written."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3] = np.asarray(rows(store, schema.rig_path(0), "Transform3D:mat3x3")[frame], dtype=np.float64).reshape(3, 3).T
    pose[:3, 3] = np.asarray(rows(store, schema.rig_path(0), "Transform3D:translation")[frame], dtype=np.float64).reshape(3)
    return pose


def view_pose(cam: int) -> Float64[ndarray, "4 4"]:
    """A fake view's shipped ``rig_T_cam`` (FLU to FLU) as a pose."""
    pose: Float64[ndarray, "4 4"] = np.eye(4)
    pose[:3, :3], pose[:3, 3] = view_rotation(cam), view_translation(cam)
    return pose


def flu_rdf() -> Float64[ndarray, "4 4"]:
    """The fixed turn from an RDF camera frame to GSI's FLU one, as a pose."""
    turn: Float64[ndarray, "4 4"] = np.eye(4)
    turn[:3, :3] = FLU_R_RDF
    return turn


def test_convert_writes_the_four_exoego_layers_under_the_dataforge_identity(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb")

    assert targets == {layer: tmp_path / "out" / layer / "gsi-six-view__ep_v1_bbbb.rrd" for layer in ("base", "pose", "hand_mesh", "projections")}
    for target in targets.values():
        [recording] = rrc.RrdReader(target).recordings()
        assert (recording.application_id, recording.recording_id) == ("dataforge", "gsi-six-view__ep_v1_bbbb")
    base: rrc.ChunkStore = read_back(targets["base"])
    rig: dict[str, object] = static_values(base, schema.rig_path(0))
    assert (rig["schema_version"], rig["reference"], rig["num_cameras"]) == ("exoego:v2", "cam_00", 6)
    assert [static_values(base, schema.cam_path(0, cam))["name"] for cam in range(6)] == list(VIEWS)
    video = base.reader(index=schema.TIMELINE, contents=schema.video_path(0, 0)).to_arrow_table()
    assert video.column(schema.TIMELINE).cast("int64").to_pylist() == [0, FRAME_NS, 2 * FRAME_NS]
    assert {schema.FRAME_INDEX, "capture_time"} <= set(video.column_names) and "source_frame" not in video.column_names
    assert len(rows(base, schema.rig_path(0), "Transform3D:translation")) == FRAMES, "the two capture-only SLAM poses are gone"
    assert schema.trajectory_path("slam") in {str(chunk.entity_path) for chunk in read_chunks(targets["base"])}
    trail = base.reader(index=schema.TIMELINE, contents=schema.trail_path("slam")).to_arrow_table()
    assert trail.num_rows == FRAMES, "one trail segment per pose; the blueprint's window limits it"
    assert recording_properties(base, "RecordingInfo") == {"name": "gsi-six-view__ep_v1_bbbb", "start_time": 1_758_686_498_000_000_000}
    assert recording_properties(base, "episode")["start_ns"] == T0_NS
    capture: dict[str, object] = recording_properties(base, "capture")
    assert (capture["num_cameras"], capture["num_frames"], capture["camera_names"]) == (6, FRAMES, ",".join(VIEWS))
    assert capture["hands_rrd_sha256"] == hashlib.sha256(gsi.files["ep_v1_bbbb"]["hands.rrd"]).hexdigest()
    assert set(recording_properties(base, "gsi_calibration")) == {"front_P1", "eye_baseline", "Q_frame", "left_side_T_cam_imu"}, "what the cameras already say is dropped"
    for layer in ("pose", "hand_mesh", "projections"):
        assert not any(str(chunk.entity_path).startswith("/camera") for chunk in read_chunks(targets[layer]))


def test_every_camera_keeps_gsis_world_pose_in_an_rdf_frame(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base: rrc.ChunkStore = read_back(converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb")["base"])

    np.testing.assert_allclose(rig_T_cam(base, 0), np.eye(4), atol=1e-6, err_msg="cam_00, left_front, is the rig frame")
    for cam in range(6):
        pinhole: dict[str, object] = static_values(base, schema.pinhole_path(0, cam))
        assert pinhole["camera_xyz"] == [3, 2, 5], "RDF"
        written_K: Float64[ndarray, "3 3"] = np.asarray(pinhole["image_from_camera"]).reshape(3, 3).T
        shipped_K: Float64[ndarray, "3 3"] = view_intrinsics(cam)
        if cam in UPSIDE_DOWN:  # pixel centres on integers, as GSI's (Kalibr) K and basalt.rotate_camera_cw have them
            shipped_K[0, 2], shipped_K[1, 2] = 1919.0 - shipped_K[0, 2], 1079.0 - shipped_K[1, 2]
        np.testing.assert_allclose(written_K, shipped_K, err_msg="K as shipped, its principal point turned with an upside-down view")
        gsi_rig_T_cam: Float64[ndarray, "4 4"] = np.eye(4)
        gsi_rig_T_cam[:3, :3], gsi_rig_T_cam[:3, 3] = view_rotation(cam), view_translation(cam)
        for frame in range(FRAMES):
            expected: Float64[ndarray, "4 4"] = world_T_rig(frame) @ gsi_rig_T_cam @ flu_rdf() @ (HALF_TURN if cam in UPSIDE_DOWN else np.eye(4))
            np.testing.assert_allclose(written_world_T_rig(base, frame) @ rig_T_cam(base, cam), expected, atol=1e-5)


def test_the_hands_reach_the_world_and_every_view_through_its_pinhole(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb")
    pose: rrc.ChunkStore = read_back(targets["pose"])
    mesh: rrc.ChunkStore = read_back(targets["hand_mesh"])
    projections: rrc.ChunkStore = read_back(targets["projections"])

    for frame in range(FRAMES):
        coco: Float32[ndarray, "133 3"] = np.asarray(rows(pose, schema.coco133_xyz_path(), ":positions")[frame], dtype=np.float32)
        for side, slots in ((0, slice(91, 112)), (1, slice(112, 133))):
            expected: Float64[ndarray, "21 3"] = rig_keypoints(frame, side) @ world_T_rig(frame)[:3, :3].T + world_T_rig(frame)[:3, 3]
            np.testing.assert_allclose(coco[slots], expected, atol=1e-5)
        assert np.isnan(coco[:91]).all()
    assert rows(pose, f"{schema.hands_path('left')}/confidence", "Scalars:scalars") == [[1.0], [0.0], [1.0]], "GSI's is_usable"
    left_vertices: list = rows(mesh, schema.hand_mesh_path("left"), "Mesh3D:vertex_positions")
    assert left_vertices[1] == [], "no mesh where GSI marks the hand unusable"
    rig_vertex: Float32[ndarray, "3"] = rig_keypoints(0, 0)[0] + np.linspace(0.0, 0.05, 778 * 3, dtype=np.float32)[:3]
    np.testing.assert_allclose(np.asarray(left_vertices[0][0]), rig_vertex @ world_T_rig(0)[:3, :3].T + world_T_rig(0)[:3, 3], atol=1e-5)
    for cam in range(6):
        pixels: Float32[ndarray, "133 2"] = np.asarray(rows(projections, schema.coco133_uv_projected_path(0, cam), ":positions")[0], dtype=np.float32)
        in_cam: Float64[ndarray, "21 3"] = (rig_keypoints(0, 0) - view_translation(cam)) @ view_rotation(cam)
        optical: Float64[ndarray, "21 3"] = np.stack([-in_cam[:, 1], -in_cam[:, 2], in_cam[:, 0]], axis=1)
        expected_px: Float64[ndarray, "21 2"] = (optical @ view_intrinsics(cam).T)[:, :2] / optical[:, 2:]
        if cam in UPSIDE_DOWN:  # the turned frame shows GSI's pixel (u, v) at (W-1-u, H-1-v)
            expected_px = np.array([1919.0, 1079.0]) - expected_px
        inside = (optical[:, 2] > 0) & (expected_px[:, 0] >= 0) & (expected_px[:, 0] < 1920) & (expected_px[:, 1] >= 0) & (expected_px[:, 1] < 1080)
        np.testing.assert_allclose(pixels[91:112][inside], expected_px[inside], atol=1e-2)
        assert np.isnan(pixels[91:112][~inside]).all()


def test_the_upside_down_views_are_reencoded_upright(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reencoded: Reencoded) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb")
    base: rrc.ChunkStore = read_back(targets["base"])

    assert reencoded.calls == [(cam, FRAMES, list(range(FRAMES))) for cam in UPSIDE_DOWN], "every shipped sample of cams 2 and 3, in frame order"
    entities: set[str] = {str(chunk.entity_path) for chunk in read_chunks(targets["base"])}
    for cam in range(6):
        node: dict[str, object] = static_values(base, schema.cam_path(0, cam))
        if cam in UPSIDE_DOWN:
            assert (node.get("image_rotation_cw_deg"), node["video_codec"]) == (180, "av1")
            assert schema.video_path(0, cam) not in entities, "GSI's upside-down samples are not passed through"
        else:
            assert ("image_rotation_cw_deg" not in node, node["video_codec"]) == (True, "h265")
            assert schema.video_path(0, cam) in entities


def test_a_slam_dropout_hides_the_rig_and_the_world_hands_for_that_frame(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_cccc")
    base: rrc.ChunkStore = read_back(targets["base"])
    pose: rrc.ChunkStore = read_back(targets["pose"])

    assert np.isnan(written_world_T_rig(base, 1)[:3]).all(), "exoego:v2: a NaN world_T_rig on the dropout frame"
    np.testing.assert_allclose(written_world_T_rig(base, 2) @ rig_T_cam(base, 3), world_T_rig(2) @ view_pose(3) @ flu_rdf() @ HALF_TURN, atol=1e-5)
    assert recording_properties(base, "capture")["slam_missing_frames"] == 1
    assert base.reader(index=schema.TIMELINE, contents=schema.trail_path("slam")).to_arrow_table().num_rows == FRAMES - 1
    assert np.isnan(np.asarray(rows(pose, schema.coco133_xyz_path(), ":positions")[1], dtype=np.float32)).all(), "no world position without a pose"
    assert rows(read_back(targets["hand_mesh"]), schema.hand_mesh_path("right"), "Mesh3D:vertex_positions")[1] == []
    pixels = np.asarray(rows(read_back(targets["projections"]), schema.coco133_uv_projected_path(0, 0), ":positions")[1], dtype=np.float32)
    assert np.isfinite(pixels[91:133]).all(), "projections need no SLAM: the hands are in GSI's rig frame"


def test_convert_deletes_the_download_once_every_layer_is_published(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa")

    assert not (tmp_path / "raw/episodes/ep_v1_aaaa").exists()
    assert (tmp_path / "raw/manifest.json").is_file()
    GsiSixViewConfig(root=tmp_path / "raw", api_url=gsi.url, sequences=("ep_v1_aaaa",), skip_converted=True).setup().download()
    assert gsi.fetched == ["ep_v1_aaaa/hands.rrd", "ep_v1_aaaa/slam.rrd"], "--skip-converted fetches nothing for a converted episode"


def test_a_converted_episode_downloads_again_and_rebuilds_with_force(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa")
    published: dict[str, int] = {layer: target.stat().st_mtime_ns for layer, target in targets.items()}

    dataset = config(gsi, tmp_path / "raw", "ep_v1_aaaa").setup()
    dataset.download()  # the request decides, not the published outputs
    assert gsi.fetched == ["ep_v1_aaaa/hands.rrd", "ep_v1_aaaa/slam.rrd"] * 2
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=True)
    assert all(target.stat().st_mtime_ns > published[layer] for layer, target in targets.items()), "every layer rebuilt"
    assert not source.exists()


def test_convert_refuses_an_output_root_beneath_the_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_raw_root_guarded("gsi-six-view", tmp_path / "raw/episodes/ep_v1_aaaa", tmp_path, monkeypatch)


def test_a_kept_download_stays_for_another_reader(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa", keep_download=True)

    assert all((tmp_path / "raw/episodes/ep_v1_aaaa" / name).is_file() for name in ("hands.rrd", "slam.rrd"))


def test_the_canonical_panes_show_each_view_with_its_keypoints_and_both_meshes(tmp_path: Path) -> None:
    dataset = GsiSixViewConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewDataset)
    world, *panes = writing.blueprint_views(dataset.default_blueprint())
    assert isinstance(world, rrb.Spatial3DView) and world.origin == schema.rig_path(0)
    assert "+ /world/**" in list(world.contents)  # type: ignore[arg-type]
    assert [pane.name for pane in panes] == ["left_eye", "right_eye", "left_front", "right_front", "left_side", "right_side"]
    for pane in panes:
        cam: int = VIEWS.index(str(pane.name))
        assert isinstance(pane, rrb.Spatial2DView) and pane.origin == schema.pinhole_path(0, cam)
        assert list(pane.contents) == [  # type: ignore[arg-type]
            schema.video_path(0, cam), schema.coco133_uv_projected_path(0, cam), schema.hand_mesh_path("left"), schema.hand_mesh_path("right")
        ]  # fmt: skip
    for view in (world, writing.blueprint_views(dataset.table_blueprint())[0]):
        assert (view.visualizer_overrides or {})[schema.trail_path("slam")] == blueprints.trail_window_range(), "the trail shows the shared cursor window"
    scene, card = writing.blueprint_views(dataset.table_blueprint())
    assert isinstance(card, rrb.Spatial2DView) and card.origin == schema.pinhole_path(0, 0), "the card decodes left_front only"
    assert all(f"- {schema.video_path(0, cam)}/**" in list(scene.contents) for cam in range(6))  # type: ignore[arg-type]


def hevc_samples(ffmpeg: Path, frames: int) -> list[bytes]:
    """A tiny real H.265 stream split into one Annex-B sample per frame, the way GSI ships its samples."""
    stream: bytes = subprocess.run(
        [str(ffmpeg), "-v", "error", "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=30:duration={frames / 30}", "-frames:v", str(frames),
         "-c:v", "libx265", "-x265-params", "aud=1:repeat-headers=1:bframes=0:log-level=error", "-f", "hevc", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    delimiter: bytes = b"\x00\x00\x00\x01\x46\x01"  # an access-unit delimiter opens every frame
    return [delimiter + sample for sample in stream.split(delimiter) if sample]


@pytest.mark.integration
def test_an_upside_down_view_reencodes_one_upright_sample_per_frame(nvenc_ffmpeg: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.undo()  # the real re-encode, not the unit lane's recorder
    samples: list[bytes] = hevc_samples(nvenc_ffmpeg, 12)
    assert len(samples) == 12
    frames: Int64[ndarray, "f"] = np.arange(12, dtype=np.int64)
    with writing.recording_to(tmp_path / "base.rrd", recording_id="upright") as recording:
        gsi_six_view_layers.write_upright_video(recording, 2, samples, times_ns=frames * FRAME_NS, frame_indices=frames)

    store: rrc.ChunkStore = read_back(tmp_path / "base.rrd")
    video = store.reader(index=schema.TIMELINE, contents=schema.video_path(0, 2)).to_arrow_table()
    assert video.column(schema.TIMELINE).cast("int64").to_pylist() == (np.arange(12) * FRAME_NS).tolist()
    assert video.column(schema.FRAME_INDEX).to_pylist() == list(range(12))
    assert static_values(store, schema.video_path(0, 2))["codec"] == rr.VideoCodec.AV1.value
