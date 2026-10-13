import multiprocessing as mp
import pickle
import traceback
from collections.abc import Mapping
from dataclasses import replace
from multiprocessing.connection import Connection

from dagkit.graph import Spawn
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.time import GraphError, TimeNs


class _Collect:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, TimeNs, object]] = []

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.calls.append(("put", port, t_ns, value))

    def watermark(self, t_ns: TimeNs) -> None:
        self.calls.append(("watermark", "", t_ns, None))


def _child(spawn: Spawn, conn: Connection) -> None:
    try:
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
            conn.send(("ok", out.calls))  # pickles synchronously: an unpicklable output raises here, not in a feeder thread
    except BaseException as exc:
        text: str = traceback.format_exc()
        try:
            conn.send(("error", (exc, text)))
        except (pickle.PicklingError, TypeError, AttributeError):  # an unpicklable exception still reaches the parent, as text
            conn.send(("error", (GraphError(repr(exc)), text)))


class ProcessNode:
    def __init__(self, spawn: Spawn) -> None:
        self.inputs: Mapping[str, PortType] = spawn.cls.inputs
        self.outputs: Mapping[str, PortType] = spawn.cls.outputs
        self.emits_late: bool = getattr(spawn.cls, "emits_late", False)
        context = mp.get_context("spawn")
        self._conn, child_conn = context.Pipe()
        self._process = context.Process(target=_child, args=(spawn, child_conn), daemon=True)

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
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
