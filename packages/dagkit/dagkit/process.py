import multiprocessing as mp
import os
import pickle
import traceback
from collections.abc import Mapping
from dataclasses import replace
from multiprocessing.connection import Connection

from dagkit.graph import Spawn
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.time import GraphError, TimeNs


class _Collect(list[tuple[str, str, TimeNs, object]]):
    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.append(("put", port, t_ns, value))

    def watermark(self, t_ns: TimeNs) -> None:
        self.append(("watermark", "", t_ns, None))


def _child(spawn: Spawn, cpus: tuple[int, ...] | None, conn: Connection) -> None:
    try:
        if cpus and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)  # the child does the work, so the placement's CPUs bind it
        node = spawn.cls(spawn.config)
        while (message := conn.recv())[0] != "close":
            op, arg = message
            out = _Collect()
            if op == "reset":
                node.reset(arg)
            elif op == "step":
                node.step(arg, out)
            else:
                node.flush(out)
            conn.send(("ok", list(out)))  # pickles synchronously: an unpicklable output raises here, not in a feeder thread
    except BaseException as exc:
        text: str = traceback.format_exc()
        try:
            conn.send(("error", (exc, text)))
        except (pickle.PicklingError, TypeError, AttributeError):  # an unpicklable exception still reaches the parent, as text
            conn.send(("error", (GraphError(repr(exc)), text)))


class ProcessNode:
    def __init__(self, spawn: Spawn, cpus: tuple[int, ...] | None = None) -> None:
        self.inputs: Mapping[str, PortType] = spawn.cls.inputs
        self.outputs: Mapping[str, PortType] = spawn.cls.outputs
        self.emits_late: bool = getattr(spawn.cls, "emits_late", False)
        context = mp.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(target=_child, args=(spawn, cpus, child_conn), daemon=True)

    def _call(self, op: str, arg: object, out: Emit) -> None:
        if self._process.exitcode is None and not self._process.is_alive():
            self._process.start()
        self._conn.send((op, arg))  # pickles synchronously: an unpicklable tick or resource raises in the runner thread
        try:
            while not self._conn.poll(1.0):
                if not self._process.is_alive():
                    raise EOFError
            status, payload = self._conn.recv()
        except EOFError:
            raise GraphError(f"process node died (exit code {self._process.exitcode}) without a message") from None
        if status == "error":
            exc, text = payload
            getattr(exc, "add_note", print)(f"raised in the child process during {op}:\n{text}")
            raise exc
        for kind, port, t_ns, value in payload:
            out.put(port, t_ns, value) if kind == "put" else out.watermark(t_ns)

    def reset(self, ctx: RunContext) -> None:
        self._call("reset", replace(ctx, control=None), _Collect())

    def step(self, tick: Tick, out: Emit) -> None:
        self._call("step", tick, out)

    def flush(self, out: Emit) -> None:
        self._call("flush", None, out)

    def close(self) -> None:
        if self._process.is_alive():
            self._conn.send(("close", None))
            self._process.join(5.0)
        self.abort()

    def abort(self) -> None:
        for stop, timeout in ((self._process.terminate, 2.0), (self._process.kill, None)):
            if self._process.is_alive():
                stop()
                self._process.join(timeout)
