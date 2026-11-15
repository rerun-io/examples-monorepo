"""The two halves of a conversion every dataset shares: ``pending_layers`` and ``write_layers``.

A toy dataset whose ``convert`` is nothing but those two calls stands in for the
real ones, so the checks hold at every level of the review stack; the raw-root
guard is checked through ``conftest.assert_raw_root_guarded``, as each dataset's
own test module does.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import rerun as rr
import rerun.blueprint as rrb
from conftest import assert_raw_root_guarded, read_chunks

from dataforge import paths
from dataforge.datasets import dataset_defaults
from dataforge.datasets.base import DataforgeDataset, FrameLimitedConfig
from dataforge.identity import SequenceIdentity

LAYERS: tuple[str, ...] = (paths.BASE_LAYER, paths.HAND_POSE_LAYER, paths.BODY_POSE_LAYER, paths.BODY_MESH_LAYER)
IDENTITY: SequenceIdentity = SequenceIdentity("toy", ("subject", "sequence"))


@dataclass
class ToyConfig(FrameLimitedConfig):
    command = "toy"
    _target: type = field(default_factory=lambda: ToyDataset)
    root: Path = Path("raw/toy")


class ToyDataset(DataforgeDataset[ToyConfig, None]):
    """Writes base and hand_pose alone and the two body layers together, from one shared pass."""

    layers = LAYERS

    def download(self) -> None:
        raise NotImplementedError

    def discover(self) -> list[tuple[SequenceIdentity, None]]:
        return [(IDENTITY, None)]

    def convert(self, identity: SequenceIdentity, source: None, *, force: bool) -> Path:
        targets, pending = self.pending_layers(identity, force=force, roots=[self.config.root])
        writers: dict[str, Callable[[rr.RecordingStream], None]] = {
            layer: lambda recording, layer=layer: rr.log(f"/{layer}", rr.TextLog(layer), recording=recording) for layer in LAYERS[:2]
        }

        def together(recordings: dict[str, rr.RecordingStream]) -> None:
            for layer, recording in recordings.items():
                rr.log(f"/{layer}", rr.TextLog(layer), recording=recording)

        self.write_layers(identity, targets, pending, writers, together=together)
        return targets[paths.BASE_LAYER]

    def default_blueprint(self) -> rrb.Blueprint:
        return rrb.Blueprint(rrb.TextLogView(origin="/"))

    def table_blueprint(self) -> rrb.Blueprint:
        return self.default_blueprint()


@pytest.fixture
def output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("DATAFORGE_OUTPUT_ROOT", str(tmp_path / "out"))
    return tmp_path / "out"


def test_every_layer_is_written_once_each_alone_or_together_and_timed(output_root: Path, tmp_path: Path) -> None:
    dataset = ToyDataset(ToyConfig(root=tmp_path / "raw"))
    base: Path = dataset.convert(IDENTITY, None, force=False)
    assert base == output_root / paths.BASE_LAYER / f"{IDENTITY.recording_id}.rrd"
    for layer in LAYERS:
        chunks = read_chunks(output_root / layer / f"{IDENTITY.recording_id}.rrd")
        assert {str(chunk.entity_path) for chunk in chunks if not str(chunk.entity_path).startswith("/__")} == {f"/{layer}"}
    # Only the layers written alone are timed here; ``together`` times its own work.
    assert set(dataset.timer.stage_s) == {f"write:{layer}" for layer in LAYERS[:2]}


def test_a_published_layer_is_pending_again_only_under_force(output_root: Path, tmp_path: Path) -> None:
    dataset = ToyDataset(ToyConfig(root=tmp_path / "raw"))
    dataset.convert(IDENTITY, None, force=False)
    (output_root / paths.HAND_POSE_LAYER / f"{IDENTITY.recording_id}.rrd").unlink()
    targets, pending = dataset.pending_layers(IDENTITY, force=False, roots=[])
    assert list(targets) == list(LAYERS)
    assert pending == [paths.HAND_POSE_LAYER]
    assert dataset.pending_layers(IDENTITY, force=True, roots=[])[1] == list(LAYERS)


def test_a_preview_lands_in_its_own_tree(output_root: Path, tmp_path: Path) -> None:
    dataset = ToyDataset(ToyConfig(root=tmp_path / "raw", frame_limit=8))
    assert dataset.convert(IDENTITY, None, force=False) == output_root / "preview-first8" / paths.BASE_LAYER / f"{IDENTITY.recording_id}.rrd"


def test_pending_layers_without_a_writer_need_together(output_root: Path, tmp_path: Path) -> None:
    dataset = ToyDataset(ToyConfig(root=tmp_path / "raw"))
    targets, pending = dataset.pending_layers(IDENTITY, force=False, roots=[])
    with pytest.raises(ValueError, match="no writer for pending layers"):
        dataset.write_layers(IDENTITY, targets, pending, {})


def test_output_beneath_the_raw_root_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(dataset_defaults, "toy", ToyConfig())
    assert_raw_root_guarded("toy", None, tmp_path, monkeypatch)
