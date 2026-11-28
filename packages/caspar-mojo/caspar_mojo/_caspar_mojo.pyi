"""Types of the Mojo extension built from `mojo/_caspar_mojo.mojo`."""

from numpy import ndarray

from caspar_mojo.problem import BaOptions
from caspar_mojo.solver import NativeStats, _Packet, _State

def solve(packet: _Packet, options: BaOptions, result: _State) -> NativeStats: ...
def linearize(packet: _Packet, gradient: ndarray, hblk: ndarray) -> float: ...
def device_count() -> int: ...
