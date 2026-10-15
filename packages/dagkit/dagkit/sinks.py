import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypeAlias, cast

import rerun as rr
from rerun.catalog import DatasetEntry, OnDuplicateSegmentLayer

from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.sources import SegmentRef
from dagkit.time import GraphError
from dagkit.types import LayerFile, Producer, Type, log_producer, log_value

Register: TypeAlias = Literal["new", "replace", "off"]


def segment_layers(segment: SegmentRef) -> list[str]:
    dataset: DatasetEntry = segment.entry()  # the entry must outlive the query it runs
    rows: list[dict] = dataset.filter_segments([segment.segment]).segment_table().select("rerun_layer_names").to_arrow_table().to_pylist()
    return rows[0]["rerun_layer_names"] if rows else []


def partial_path(output_dir: Path, segment: SegmentRef, layer: str) -> Path:
    """``<output_dir>/<segment>/.<layer>.<run id>.partial`` (its directory made): the name ``publish_layer_file`` reads the run id from."""
    partial: Path = output_dir / segment.segment / f".{layer}.{uuid.uuid4().hex[:12]}.partial"
    partial.parent.mkdir(parents=True, exist_ok=True)
    return partial


def open_layer(segment: SegmentRef, layer: str, producer: Producer, *, output_dir: Path) -> tuple[rr.RecordingStream, Path]:
    partial: Path = partial_path(output_dir, segment, layer)
    recording = rr.RecordingStream(producer.package, recording_id=segment.segment, send_properties=False)
    recording.save(partial)
    log_producer(recording, layer, producer)
    return recording, partial


def publish_layer_file(partial: Path, segment: SegmentRef, layer: str, *, output_dir: Path, register: Register, stopped: bool = False) -> Path:
    run_id: str = partial.name.split(".")[-2] if partial.name.endswith(".partial") else uuid.uuid4().hex[:12]
    infix: str = ".stopped" if stopped else ".window" if segment.window_ns is not None else ""
    final: Path = output_dir / segment.segment / f"{layer}{infix}.{run_id}.rrd"
    if final.exists():
        raise GraphError(f"{final} exists; layer files are immutable")
    final.parent.mkdir(parents=True, exist_ok=True)
    partial.chmod(0o644)
    shutil.move(partial, final)
    if register != "off" and not infix:  # "new" registers with ERROR: the catalog refuses a layer registered since (AlreadyExistsError)
        on_duplicate: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if register == "replace" else OnDuplicateSegmentLayer.ERROR
        dataset: DatasetEntry = segment.entry()
        dataset.register([final.resolve().as_uri()], layer_name=layer, on_duplicate=on_duplicate).wait()
    return final


class RerunSink:
    """Draw its inputs in the viewer."""

    outputs: Mapping[str, PortType] = {}

    def __init__(
        self, recording: rr.RecordingStream, inputs: Mapping[str, PortType], roots: Mapping[str, str], *, timeline: str = "video_time"
    ) -> None:
        self.recording: rr.RecordingStream = recording
        self.inputs: dict[str, PortType] = dict(inputs)
        self.roots: dict[str, str] = dict(roots)
        self.timeline: str = timeline
        self.drawn: dict[str, set[str]] = {}  # input port -> entity paths it logged (the first 64): the guide's links

    def reset(self, ctx: RunContext) -> None:
        pass

    def step(self, tick: Tick, out: Emit) -> None:
        paths: list[str] = log_value(self.recording, self.timeline, tick.t_ns, cast(Type, tick.value), self.roots[tick.port])
        if len(drawn := self.drawn.setdefault(tick.port, set())) < 64:
            drawn.update(paths)

    def flush(self, out: Emit) -> None:
        self.recording.flush()


class LayerFileSink:
    """Publish a finished layer file."""

    inputs: dict[str, PortType] = {"layer": LayerFile}
    outputs: Mapping[str, PortType] = {}

    def __init__(self, segment: SegmentRef, layer: str, *, output_dir: Path, register: Register = "new") -> None:
        if register == "new" and segment.window_ns is None and layer in segment_layers(segment):
            raise GraphError(f"{segment.dataset}/{segment.segment} already has layer {layer!r}; a refresh needs an explicit --register replace")
        self.segment, self.layer, self.output_dir, self.register = segment, layer, output_dir, register
        self.published: Path | None = None
        self._ctx: RunContext | None = None
        self._file: LayerFile | None = None

    def reset(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def step(self, tick: Tick, out: Emit) -> None:
        assert isinstance(tick.value, LayerFile)
        self._file = tick.value

    def flush(self, out: Emit) -> None:
        if self._file is not None:
            stopped: bool = self._ctx is not None and self._ctx.control is not None and self._ctx.control.stopped
            self.published = publish_layer_file(
                self._file.path, self.segment, self.layer, output_dir=self.output_dir, register=self.register, stopped=stopped
            )
            self._file = None


class LayerSink(RerunSink, LayerFileSink):
    """Save its inputs as a catalog layer."""

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
        RerunSink.__init__(self, recording, inputs, roots, timeline=segment.timeline)
        LayerFileSink.__init__(self, segment, layer, output_dir=partial.parent.parent, register=register)
        self.partial: Path = partial
        self._file = LayerFile(partial)

    def reset(self, ctx: RunContext) -> None:
        if ctx.mode != "lossless":
            raise GraphError(f"{ctx.node}: LayerSink runs lossless only")
        self._ctx = ctx

    def flush(self, out: Emit) -> None:
        if self.published is None:
            self.recording.flush()
            self.recording.disconnect()
            LayerFileSink.flush(self, out)

    def abort(self) -> None:
        self.recording.disconnect()
        self.partial.unlink(missing_ok=True)
