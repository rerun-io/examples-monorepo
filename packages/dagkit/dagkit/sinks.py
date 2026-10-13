"""Sinks with an explicit RecordingStream (decision 8) and the layer publish/register policy (SPEC §2.7)."""

import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, cast

import numpy as np
import rerun as rr
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer

from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.sources import SegmentRef
from dagkit.time import STATIC, GraphError, TimeNs
from dagkit.types import LayerFile, Producer, Type, log_producer

Register: TypeAlias = Literal["new", "replace", "off"]


def log_value(recording: rr.RecordingStream, timeline: str, t_ns: TimeNs, value: Type, root: str) -> None:
    """Log `value.to_rerun(root)` at `t_ns` on `timeline` (STATIC values are logged static)."""
    if t_ns != STATIC:
        recording.set_time(timeline, duration=np.timedelta64(t_ns, "ns"))
    for log in value.to_rerun(root):
        recording.log(log.path, log.data, static=log.static or t_ns == STATIC)


def segment_layers(segment: SegmentRef) -> list[str]:
    """Layer names the catalog holds for `segment`."""
    dataset: DatasetEntry = segment.entry()  # the entry must outlive the query it runs
    rows: list[dict] = dataset.segment_table().to_arrow_table().select(["rerun_segment_id", "rerun_layer_names"]).to_pylist()
    return next((row["rerun_layer_names"] for row in rows if row["rerun_segment_id"] == segment.segment), [])


def _check_new(segment: SegmentRef, layer: str, register: Register) -> None:
    if register == "new" and segment.window_ns is None and layer in segment_layers(segment):
        raise GraphError(f"{segment.dataset}/{segment.segment} already has layer {layer!r}; a refresh needs an explicit --register replace")


def open_layer(segment: SegmentRef, layer: str, producer: Producer, *, output_dir: Path) -> tuple[rr.RecordingStream, Path]:
    """Recording for one layer of one segment, saving to a hidden partial file; logs `/producers/<layer>` first."""
    partial: Path = output_dir / segment.segment / f".{layer}.{uuid.uuid4().hex[:12]}.partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(producer.package, recording_id=segment.segment, send_properties=False)
    recording.save(partial)
    log_producer(recording, layer, producer)
    return recording, partial


def publish_layer_file(partial: Path, segment: SegmentRef, layer: str, *, output_dir: Path, register: Register, stopped: bool = False) -> Path:
    """Move a finished file to the immutable `<layer>.<run_id>.rrd` (0644), then register it per policy.

    Stopped and windowed runs get a `.stopped` / `.window` infix and are never registered.
    """
    run_id: str = partial.name.split(".")[-2] if partial.name.endswith(".partial") else uuid.uuid4().hex[:12]
    infix: str = ".stopped" if stopped else ".window" if segment.window_ns is not None else ""
    final: Path = output_dir / segment.segment / f"{layer}{infix}.{run_id}.rrd"
    if final.exists():
        raise GraphError(f"{final} exists; layer files are immutable")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.chmod(0o644)
    shutil.move(partial, final)
    if register != "off" and not infix:
        _check_new(segment, layer, register)
        on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if register == "replace" else OnDuplicateSegmentLayer.ERROR
        dataset: DatasetEntry = segment.entry()
        dataset.register([final.resolve().as_uri()], layer_name=layer, on_duplicate=on_duplicate).wait()
    return final


class RerunSink:
    """Logs each input's `Type.to_rerun` under its root; the runner has already synced and unwrapped Fenced values."""

    outputs: Mapping[str, PortType] = {}

    def __init__(self, recording: rr.RecordingStream, inputs: Mapping[str, PortType], roots: Mapping[str, str], *, timeline: str = "video_time") -> None:
        self.recording: rr.RecordingStream = recording
        self.inputs: dict[str, PortType] = dict(inputs)
        self.roots: dict[str, str] = dict(roots)
        self.timeline: str = timeline

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        log_value(self.recording, self.timeline, tick.t_ns, cast(Type, tick.value), self.roots[tick.port])

    def flush(self, out: Emit) -> None:
        self.recording.flush()


class LayerSink(RerunSink):
    """One catalog layer of one segment; lossless only. Owns flush, close and publish of the stream from `open_layer`."""

    def __init__(
        self,
        recording: rr.RecordingStream,
        partial: Path,
        segment: SegmentRef,
        layer: str,
        inputs: Mapping[str, PortType],
        roots: Mapping[str, str],
        *,
        register: Register = "new",
    ) -> None:
        super().__init__(recording, inputs, roots, timeline=segment.timeline)
        _check_new(segment, layer, register)
        self.partial: Path = partial
        self.segment: SegmentRef = segment
        self.layer: str = layer
        self.register: Register = register
        self.published: Path | None = None
        self._ctx: RunContext | None = None

    def reset(self, ctx: RunContext) -> None:
        if ctx.mode != "lossless":
            raise GraphError(f"{ctx.node}: LayerSink runs lossless only")
        self._ctx = ctx

    def flush(self, out: Emit) -> None:
        if self.published is not None:
            return
        self.recording.flush()
        self.recording.disconnect()
        stopped: bool = self._ctx is not None and self._ctx.control is not None and self._ctx.control.stopped
        output_dir: Path = self.partial.parent.parent
        self.published = publish_layer_file(self.partial, self.segment, self.layer, output_dir=output_dir, register=self.register, stopped=stopped)

    def abort(self) -> None:
        """Failed run: drop the partial file."""
        self.recording.disconnect()
        self.partial.unlink(missing_ok=True)


class LayerFileSink:
    """Publishes a finished layer file another writer produced (e.g. robocap-live's Rust core) at flush."""

    inputs: Mapping[str, PortType] = {"layer": LayerFile}
    outputs: Mapping[str, PortType] = {}

    def __init__(self, segment: SegmentRef, layer: str, *, output_dir: Path, register: Register = "new") -> None:
        _check_new(segment, layer, register)
        self.segment: SegmentRef = segment
        self.layer: str = layer
        self.output_dir: Path = output_dir
        self.register: Register = register
        self.published: Path | None = None
        self._file: LayerFile | None = None
        self._ctx: RunContext | None = None

    def reset(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, LayerFile)
        self._file = tick.value

    def flush(self, out: Emit) -> None:
        if self._file is None:
            return
        stopped: bool = self._ctx is not None and self._ctx.control is not None and self._ctx.control.stopped
        self.published = publish_layer_file(self._file.path, self.segment, self.layer, output_dir=self.output_dir, register=self.register, stopped=stopped)
        self._file = None
