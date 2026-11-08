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

import pytest
import rerun as rr
import rerun.blueprint as rrb
import rerun.chunk as rrc
from conftest import assert_raw_root_guarded, read_back, read_chunks, recording_properties

from dataforge import hands, transports
from dataforge.datasets.gsi_six_view import CAMERAS, GsiSixViewConfig, GsiSixViewDataset

EDITION: str = "01a0f166-7994-7b43-a622-a325541c4356"
"""The 10 h edition's id, as GSI's API names it."""
USERNAME: str = "gsi-ots-10h"
PASSWORD: str = "partner-password-SECRET"
"""What the fake API accepts; no test may see it in a message."""
SIGNATURE: str = "SIGNATURE-SECRET"
"""Every signed link carries it; no test may see it in a message."""
EPISODES: tuple[str, ...] = ("ep_v1_aaaa", "ep_v1_bbbb", "ep_v1_cccc")


def episode_rrds(episode: str) -> dict[str, bytes]:
    """A two-file GSI episode with the shipped layout: hands.rrd (cameras, hands, episode properties) and slam.rrd (the rig path)."""
    files: dict[str, bytes] = {}
    for name in ("hands.rrd", "slam.rrd"):
        recording = rr.RecordingStream(application_id="gsi_off_the_shelf", recording_id=episode, send_properties=False)
        stream = rr.binary_stream(recording)
        recording.send_recording_name(episode)
        recording.send_recording_start_time_nanos(1_758_686_498_000_000_000)
        recording.log("/", rr.ViewCoordinates.RDF, static=True)
        if name == "hands.rrd":
            recording.send_property("episode", rr.AnyValues(episode_id=episode, caption=f"caption of {episode}", n_frames=3, duration_s=0.1))
            recording.log("/camera/left_front/image", rr.Pinhole(focal_length=100.0, width=64, height=48), static=True)
            for frame in range(3):
                recording.set_time("frame", sequence=frame)
                recording.log("/hand/left/keypoints3d", rr.Points3D([[0.0, 0.0, float(frame)]]))
        else:
            for frame in range(3):
                recording.set_time("frame", sequence=frame)
                recording.log("/slam/rig", rr.Transform3D(translation=[float(frame), 0.0, 0.0]))
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


def converted(gsi: FakeGsi, raw: Path, out: Path, monkeypatch: pytest.MonkeyPatch, episode: str) -> dict[str, Path]:
    """Download and convert one episode; its layer targets."""
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(out))
    dataset = config(gsi, raw, episode).setup()
    dataset.download()
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=False)
    return dataset.targets(identity)


def test_convert_repacks_both_files_under_the_dataforge_identity(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    targets: dict[str, Path] = converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_bbbb")

    assert targets == {layer: tmp_path / "out" / layer / "gsi-six-view__ep_v1_bbbb.rrd" for layer in ("base", "slam")}
    for target in targets.values():
        [recording] = rrc.RrdReader(target).recordings()
        assert (recording.application_id, recording.recording_id) == ("dataforge", "gsi-six-view__ep_v1_bbbb")
    base: rrc.ChunkStore = read_back(targets["base"])
    slam: rrc.ChunkStore = read_back(targets["slam"])
    assert base.reader(index="frame", contents="/hand/left/keypoints3d").to_arrow_table().num_rows == 3
    assert slam.reader(index="frame", contents="/slam/rig").to_arrow_table().num_rows == 3
    assert recording_properties(base, "episode") == {"episode_id": "ep_v1_bbbb", "caption": "caption of ep_v1_bbbb", "n_frames": 3, "duration_s": 0.1}
    assert recording_properties(base, "RecordingInfo") == {"name": "gsi-six-view__ep_v1_bbbb", "start_time": 1_758_686_498_000_000_000}
    capture: dict[str, object] = recording_properties(base, "capture")
    assert {key: capture[key] for key in ("num_cameras", "num_frames", "category", "gsi_rrd_contract", "gsi_source_release")} == {
        "num_cameras": 6,
        "num_frames": 3,
        "category": "Food preparation",
        "gsi_rrd_contract": "gsi.ots.customer-rrd.v2",
        "gsi_source_release": "2026-09-v2",
    }
    assert capture["hands_rrd_sha256"] == hashlib.sha256(gsi.files["ep_v1_bbbb"]["hands.rrd"]).hexdigest()
    assert not any(str(chunk.entity_path).startswith("/__properties") for chunk in read_chunks(targets["slam"]))


def test_convert_deletes_the_download_once_both_layers_are_published(gsi: FakeGsi, signed_in: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    converted(gsi, tmp_path / "raw", tmp_path / "out", monkeypatch, "ep_v1_aaaa")

    assert not (tmp_path / "raw/episodes/ep_v1_aaaa").exists()
    assert (tmp_path / "raw/manifest.json").is_file()
    config(gsi, tmp_path / "raw", "ep_v1_aaaa").setup().download()
    assert gsi.fetched == ["ep_v1_aaaa/hands.rrd", "ep_v1_aaaa/slam.rrd"]


def test_convert_refuses_an_output_root_beneath_the_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert_raw_root_guarded("gsi-six-view", tmp_path / "raw/episodes/ep_v1_aaaa", tmp_path, monkeypatch)


SMALLEST: str = "ep_v1_6a03abdb6245d608f61851028dc83477"
"""The edition's smallest episode, 33 MB of hands.rrd."""


@pytest.mark.integration
def test_the_live_api_serves_an_episode_that_repacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.environ.get("GSI_PARTNER_PASSWORD"):
        pytest.skip("needs GSI's partner sign-in: GSI_PARTNER_USERNAME and GSI_PARTNER_PASSWORD (source ~/.secrets.env)")
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    dataset = GsiSixViewConfig(root=tmp_path / "raw", sequences=(SMALLEST,)).setup()
    dataset.download()
    [(identity, source)] = dataset.discover()
    dataset.convert(identity, source, force=False)

    base: rrc.ChunkStore = read_back(dataset.targets(identity)["base"])
    assert recording_properties(base, "capture")["category"] == "Food preparation"
    assert {f"/camera/{camera}/image" for camera in CAMERAS} <= {str(chunk.entity_path) for chunk in read_chunks(dataset.targets(identity)["base"])}
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


def test_the_default_world_shows_every_frustum_without_decoding_a_video(tmp_path: Path) -> None:
    dataset = GsiSixViewConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewDataset)
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


def test_the_table_card_projects_the_hands_into_left_front(tmp_path: Path) -> None:
    dataset = GsiSixViewConfig(root=tmp_path).setup()
    assert isinstance(dataset, GsiSixViewDataset)
    world, pane = layout_views(dataset.table_blueprint())
    assert isinstance(world, rrb.Spatial3DView)
    assert_see_through_hands(world)
    assert isinstance(pane, rrb.Spatial2DView)
    assert_projects_the_hands(pane, "left_front")
