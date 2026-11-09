"""GSI Egocentric Six-View: download from a loopback copy of GSI's API, and the repack of its rrds into dataforge layers."""

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from conftest import assert_raw_root_guarded, read_back, read_chunks, recording_properties, static_values
from jaxtyping import Float32, Float64
from numpy import ndarray
from scipy.spatial.transform import Rotation

from dataforge import hands, schema, transports
from dataforge.datasets.gsi_six_view import FLU_R_RDF, VIEWS, GsiSixViewConfig, GsiSixViewDataset
from dataforge.datasets.gsi_six_view_native import CAMERAS, GsiSixViewNativeConfig, GsiSixViewNativeDataset

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


def converted(
    gsi: FakeGsi, raw: Path, out: Path, monkeypatch: pytest.MonkeyPatch, episode: str, *, native: bool = False, keep_download: bool = False
) -> dict[str, Path]:
    """Download and convert one episode into ``gsi-six-view`` (or its native mirror); its layer targets."""
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(out))
    kind: type[GsiSixViewConfig] = GsiSixViewNativeConfig if native else GsiSixViewConfig
    dataset = kind(root=raw, api_url=gsi.url, sequences=(episode,), keep_download=keep_download).setup()
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
    pose[:3, :3] = Rotation.from_quat(np.asarray(rows(store, schema.rig_path(0), "Transform3D:quaternion")[frame]).reshape(4)).as_matrix()
    pose[:3, 3] = np.asarray(rows(store, schema.rig_path(0), "Transform3D:translation")[frame]).reshape(3)
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
        np.testing.assert_allclose(np.asarray(pinhole["image_from_camera"]).reshape(3, 3).T, view_intrinsics(cam), err_msg="K as shipped")
        gsi_rig_T_cam: Float64[ndarray, "4 4"] = np.eye(4)
        gsi_rig_T_cam[:3, :3], gsi_rig_T_cam[:3, 3] = view_rotation(cam), view_translation(cam)
        for frame in range(FRAMES):
            expected: Float64[ndarray, "4 4"] = world_T_rig(frame) @ gsi_rig_T_cam @ flu_rdf()
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
        inside = (optical[:, 2] > 0) & (expected_px[:, 0] >= 0) & (expected_px[:, 0] < 1920) & (expected_px[:, 1] >= 0) & (expected_px[:, 1] < 1080)
        np.testing.assert_allclose(pixels[91:112][inside], expected_px[inside], atol=1e-2)
        assert np.isnan(pixels[91:112][~inside]).all()


def test_a_kept_download_lets_the_native_mirror_convert_the_same_files(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa", keep_download=True)
    assert (tmp_path / "raw/episodes/ep_v1_aaaa/hands.rrd").is_file()

    native: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "native", monkeypatch, "ep_v1_aaaa", native=True)
    assert all(target.is_file() for target in native.values())
    assert not (tmp_path / "raw/episodes/ep_v1_aaaa").exists()


def test_the_native_mirror_repacks_both_files_as_shipped(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb", native=True)

    assert targets == {layer: tmp_path / "out" / layer / "gsi-six-view-native__ep_v1_bbbb.rrd" for layer in ("base", "slam")}
    base: rrc.ChunkStore = read_back(targets["base"])
    slam: rrc.ChunkStore = read_back(targets["slam"])
    assert base.reader(index="frame", contents="/hand/left/keypoints3d").to_arrow_table().num_rows == FRAMES
    assert slam.reader(index="frame", contents="/slam/rig").to_arrow_table().num_rows >= FRAMES
    assert recording_properties(base, "RecordingInfo") == {"name": "gsi-six-view-native__ep_v1_bbbb", "start_time": 1_758_686_498_000_000_000}
    assert recording_properties(base, "capture")["category"] == "Food preparation"
    assert not any(str(chunk.entity_path).startswith("/__properties") for chunk in read_chunks(targets["slam"]))


def test_convert_deletes_the_download_once_every_layer_is_published(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa")

    assert not (tmp_path / "raw/episodes/ep_v1_aaaa").exists()
    assert (tmp_path / "raw/manifest.json").is_file()
    config(gsi, tmp_path / "raw", "ep_v1_aaaa").setup().download()
    assert gsi.fetched == ["ep_v1_aaaa/hands.rrd", "ep_v1_aaaa/slam.rrd"]


@pytest.mark.parametrize("command", ["gsi-six-view", "gsi-six-view-native"])
def test_convert_refuses_an_output_root_beneath_the_download(command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_raw_root_guarded(command, tmp_path / "raw/episodes/ep_v1_aaaa", tmp_path, monkeypatch)


SMALLEST: str = "ep_v1_6a03abdb6245d608f61851028dc83477"
"""The edition's smallest episode, 33 MB of hands.rrd."""


@pytest.mark.integration
def test_the_live_api_serves_an_episode_that_converts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.environ.get("GSI_PARTNER_PASSWORD"):
        pytest.skip("needs GSI's partner sign-in: GSI_PARTNER_USERNAME and GSI_PARTNER_PASSWORD (source ~/.secrets.env)")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = GsiSixViewConfig(root=tmp_path / "raw", sequences=(SMALLEST,)).setup()
    dataset.download()
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=False)

    targets: dict[str, Path] = dataset.targets(identity)
    assert all(target.is_file() for target in targets.values())
    base: rrc.ChunkStore = read_back(targets["base"])
    assert recording_properties(base, "capture")["category"] == "Food preparation"
    assert {schema.video_path(0, cam) for cam in range(6)} <= {str(chunk.entity_path) for chunk in read_chunks(targets["base"])}
    assert not source.exists()


def layout_views(blueprint: rrb.Blueprint) -> list[rrb.View]:
    """Every view of ``blueprint``, in layout order."""
    found: list[rrb.View] = []
    stack: list[object] = [blueprint.root_container]
    while stack:
        node = stack.pop()
        if isinstance(node, rrb.View):
            found.append(node)
        if isinstance(node, rrb.Container):
            stack.extend(reversed(node.contents))  # type: ignore[arg-type]
    return found


def assert_see_through_hands(view: rrb.View) -> None:
    """Both MANO meshes draw with the shared translucent hand albedo, so the keypoints show through them."""
    for side in ("left", "right"):
        mesh: object = (view.visualizer_overrides or {})[f"/hand/{side}/vertices"]
        assert isinstance(mesh, rr.Mesh3D)
        assert mesh.albedo_factor == rr.components.AlbedoFactorBatch(hands.HAND_ALBEDO[side])


def assert_projects_the_hands(pane: rrb.Spatial2DView, camera: str) -> None:
    """A camera pane shows its video and both hands' meshes and keypoints, projected through its pinhole."""
    assert pane.origin == f"/camera/{camera}/image"
    assert set(pane.contents) == {  # type: ignore[arg-type]
        f"+ /camera/{camera}/image",
        *(f"+ /hand/{side}/{part}" for side in ("left", "right") for part in ("vertices", "keypoints3d")),
    }
    assert_see_through_hands(pane)


def test_the_native_world_shows_every_frustum_without_decoding_a_video(tmp_path: Path) -> None:
    dataset = GsiSixViewNativeConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewNativeDataset)
    world, *panes = layout_views(dataset.default_blueprint())
    assert isinstance(world, rrb.Spatial3DView)
    assert list(world.contents) == ["+ /**"]  # type: ignore[arg-type]
    for camera in CAMERAS:
        # GSI logs the VideoStream on the pinhole entity itself; an override that names only the Pinhole visualizer
        # draws the frustum and leaves the video undecoded.
        assert isinstance((world.visualizer_overrides or {})[f"/camera/{camera}/image"], rr.Pinhole)
    assert_see_through_hands(world)
    assert [pane.name for pane in panes] == list(CAMERAS)
    for pane, camera in zip(panes, CAMERAS, strict=True):
        assert isinstance(pane, rrb.Spatial2DView)
        assert_projects_the_hands(pane, camera)


def test_the_native_table_card_projects_the_hands_into_left_front(tmp_path: Path) -> None:
    dataset = GsiSixViewNativeConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewNativeDataset)
    world, pane = layout_views(dataset.table_blueprint())
    assert isinstance(world, rrb.Spatial3DView)
    assert_see_through_hands(world)
    assert isinstance(pane, rrb.Spatial2DView)
    assert_projects_the_hands(pane, "left_front")


def test_the_canonical_panes_show_each_view_with_its_keypoints_and_both_meshes(tmp_path: Path) -> None:
    dataset = GsiSixViewConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewDataset)
    world, *panes = layout_views(dataset.default_blueprint())
    assert isinstance(world, rrb.Spatial3DView) and world.origin == schema.rig_path(0)
    assert "+ /world/**" in list(world.contents)  # type: ignore[arg-type]
    assert [pane.name for pane in panes] == ["left_eye", "right_eye", "left_front", "right_front", "left_side", "right_side"]
    for pane in panes:
        cam: int = VIEWS.index(str(pane.name))
        assert isinstance(pane, rrb.Spatial2DView) and pane.origin == schema.pinhole_path(0, cam)
        assert list(pane.contents) == [  # type: ignore[arg-type]
            schema.video_path(0, cam), schema.coco133_uv_projected_path(0, cam), schema.hand_mesh_path("left"), schema.hand_mesh_path("right")
        ]  # fmt: skip
    scene, card = layout_views(dataset.table_blueprint())
    assert isinstance(card, rrb.Spatial2DView) and card.origin == schema.pinhole_path(0, 0), "the card decodes left_front only"
    assert all(f"- {schema.video_path(0, cam)}/**" in list(scene.contents) for cam in range(6))  # type: ignore[arg-type]
