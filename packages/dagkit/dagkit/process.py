"""Process placement: a thread-side proxy for a `Spawn` recipe that runs in a child process (SPEC §2.5)."""

import multiprocessing as mp
import pickle
import queue
import traceback
from collections.abc import Mapping
from dataclasses import replace
from multiprocessing.queues import Queue

from dagkit.graph import Spawn
from dagkit.node import Emit, PortType, RunContext, Tick
from dagkit.time import GraphError, TimeNs


class _Collect:
    """The child's `Emit`: records puts and watermarks for the proxy to replay."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, TimeNs, object]] = []

    def put(self, port: str, t_ns: TimeNs, value: object) -> None:
        self.calls.append(("put", port, t_ns, value))

    def watermark(self, t_ns: TimeNs) -> None:
        self.calls.append(("watermark", "", t_ns, None))


def _child(spawn: Spawn, inbox: Queue, outbox: Queue) -> None:
    try:
        node = spawn.cls(spawn.config)
        while (message := inbox.get())[0] != "close":
            op, arg = message
            out = _Collect()
            if op == "reset":
                node.reset(arg)
            elif op == "step":
                node.step(arg, out)
            else:
                node.flush(out)
            outbox.put(("ok", out.calls))
    except BaseException as exc:
        text: str = traceback.format_exc()
        try:
            pickle.dumps(exc)
        except (pickle.PicklingError, TypeError, AttributeError):  # an unpicklable exception still reaches the parent, as text
            exc = GraphError(repr(exc))
        outbox.put(("error", (exc, text)))


class ProcessNode:
    """Proxy node run by a runner thread; the child (spawn context) builds `cls(config)` and runs its contract."""

    def __init__(self, spawn: Spawn) -> None:
        self.inputs: Mapping[str, PortType] = spawn.cls.inputs
        self.outputs: Mapping[str, PortType] = spawn.cls.outputs
        self.emits_late: bool = getattr(spawn.cls, "emits_late", False)
        context = mp.get_context("spawn")
        self._inbox: Queue = context.Queue()
        self._outbox: Queue = context.Queue()
        self._process = context.Process(target=_child, args=(spawn, self._inbox, self._outbox), daemon=True)

    def _call(self, op: str, arg: object, out: Emit) -> None:
        if self._process.exitcode is None and not self._process.is_alive():
            self._process.start()
        self._inbox.put((op, arg))
        while True:
            try:
                status, payload = self._outbox.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._process.is_alive():
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
        """Ask the child to exit, then join → terminate → kill."""
        if self._process.is_alive():
            self._inbox.put(("close", None))
            self._process.join(5.0)
        self.abort()

    def abort(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(2.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
