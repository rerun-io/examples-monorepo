"""Device-memory ownership in `colsfm.tensorrt_runtime`, through a fake CUDA boundary.

The seam is `DeviceBufferPool`, the device memory one execution context owns, plus
the `GpuPreprocessor` constructor that acquires several resources in a row. What
matters is what they own after a `cudaMalloc` fails, and how many times each
pointer is freed — neither of which a passing GPU run can show.

`FakeCudart` replaces only the two device entry points, delegating everything else
(streams, page-locked host memory) to the real `cuda.bindings.runtime`, so the one
test that builds a real engine still works while its allocations stay fake.

The module needs the `tensorrt` and `cuda-python` wheels because
`colsfm.tensorrt_runtime` imports them at module scope; only the last test needs a
device, and it skips itself without one.
"""

from __future__ import annotations

from typing import Any, Final

import pytest

pytest.importorskip("tensorrt", reason="colsfm.tensorrt_runtime imports tensorrt at module scope")
cudart: Any = pytest.importorskip("cuda.bindings.runtime", reason="colsfm.tensorrt_runtime allocates through cuda-python")

from colsfm import tensorrt_runtime  # noqa: E402  (after importorskip)
from colsfm.tensorrt_runtime import DeviceBufferPool, GpuPreprocessor, compute_capability  # noqa: E402

FIRST_FAKE_POINTER: Final[int] = 0x1000
"""The address the fake boundary hands out first; every later one is a multiple of it."""


class FakeCudart:
    """A stand-in for `cuda.bindings.runtime` whose device allocations are bookkeeping.

    Every call other than `cudaMalloc` and `cudaFree` goes to the real module, so
    a page-locked staging buffer is still real memory and a stream is still a
    stream. Allocations and frees are recorded, so a test can assert that a
    pointer was freed exactly once and that a pointer the pool still claims was
    never freed behind its back.
    """

    def __init__(self, fail_at_allocation: int | None = None) -> None:
        """Arm the fake, optionally failing one allocation.

        Args:
            fail_at_allocation: 1-based index of the `cudaMalloc` call that must
                fail, or None to let every allocation succeed.
        """
        self.fail_at_allocation: int | None = fail_at_allocation
        self.allocation_count: int = 0
        self.live: set[int] = set()
        self.freed: list[int] = []
        self.double_frees: list[int] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate everything this fake does not model to the real `cudart`.

        Args:
            name: Attribute the caller asked for, e.g. `cudaError_t`.

        Returns:
            The real module's attribute.
        """
        return getattr(cudart, name)

    def cudaMalloc(self, byte_count: int) -> tuple[Any, int]:
        """Hand out a fake device address, or fail when the test armed this call.

        Args:
            byte_count: Bytes requested.

        Returns:
            The `(status, pointer)` pair `cudart.cudaMalloc` returns.
        """
        self.allocation_count += 1
        if self.fail_at_allocation == self.allocation_count:
            return (cudart.cudaError_t.cudaErrorMemoryAllocation, 0)
        pointer: int = FIRST_FAKE_POINTER * self.allocation_count
        self.live.add(pointer)
        return (cudart.cudaError_t.cudaSuccess, pointer)

    def cudaFree(self, pointer: int) -> tuple[Any]:
        """Release a fake device address, recording a double free rather than hiding it.

        Args:
            pointer: The address to free.

        Returns:
            The `(status,)` tuple `cudart.cudaFree` returns.
        """
        self.freed.append(pointer)
        if pointer in self.live:
            self.live.remove(pointer)
        else:
            self.double_frees.append(pointer)
        return (cudart.cudaError_t.cudaSuccess,)


def _cuda_device_available() -> bool:
    """Whether a CUDA device answers, so a real engine can be built.

    Returns:
        True when `cudaGetDeviceProperties` succeeds.
    """
    try:
        compute_capability()
    except RuntimeError:
        return False
    return True


def test_a_failed_growth_leaves_the_pool_owning_the_buffer_it_could_not_replace(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `cudaMalloc` that fails must not cost the caller the buffer it already had.

    Freeing first and allocating second left the freed pointer recorded at its old
    capacity: the next call handed that dangling address to TensorRT, and `close`
    freed it a second time.
    """
    fake: FakeCudart = FakeCudart()
    monkeypatch.setattr(tensorrt_runtime, "cudart", fake)
    pool: DeviceBufferPool = DeviceBufferPool()
    original: int = pool.ensure("image", 1024)

    fake.fail_at_allocation = fake.allocation_count + 1
    with pytest.raises(RuntimeError, match="cudaMalloc"):
        pool.ensure("image", 4096)

    assert pool.pointers["image"] == original
    assert pool.capacities["image"] == 1024
    assert original in fake.live
    assert fake.freed == []
    assert pool.ensure("image", 512) == original

    pool.close()
    assert fake.freed == [original]
    assert fake.double_frees == []


def test_a_successful_growth_frees_the_old_buffer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Growing swaps the record first, then returns the old allocation exactly once."""
    fake: FakeCudart = FakeCudart()
    monkeypatch.setattr(tensorrt_runtime, "cudart", fake)
    pool: DeviceBufferPool = DeviceBufferPool()
    original: int = pool.ensure("image", 1024)
    grown: int = pool.ensure("image", 4096)

    assert grown != original
    assert fake.freed == [original]
    assert pool.pointers["image"] == grown
    assert pool.capacities["image"] == 4096

    pool.close()
    assert sorted(fake.freed) == sorted([original, grown])
    assert fake.double_frees == []


def test_closing_twice_frees_nothing_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """`close` is idempotent, with several bindings live."""
    fake: FakeCudart = FakeCudart()
    monkeypatch.setattr(tensorrt_runtime, "cudart", fake)
    pool: DeviceBufferPool = DeviceBufferPool()
    pointers: list[int] = [pool.ensure("image", 64), pool.ensure("keypoints", 128)]

    pool.close()
    pool.close()

    assert sorted(fake.freed) == sorted(pointers)
    assert fake.double_frees == []
    assert pool.pointers == {}
    assert pool.capacities == {}


@pytest.mark.skipif(not _cuda_device_available(), reason="a partly built GpuPreprocessor needs a real engine to get that far")
def test_a_preprocessor_whose_second_allocation_fails_frees_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial construction must not leak: what was acquired is freed, then the error propagates.

    The engine is real — an 8x8 one builds in seconds — but the device
    allocations are fake, so the second `cudaMalloc` can be made to fail on
    demand. Before the fix the first device buffer and the page-locked staging
    both stayed allocated with no owner left to free them.
    """
    fake: FakeCudart = FakeCudart(fail_at_allocation=2)
    released_staging: list[int] = []
    real_close: Any = tensorrt_runtime.PinnedHostBuffer.close

    def recording_close(buffer: tensorrt_runtime.PinnedHostBuffer) -> None:
        """Record that page-locked staging was released, then release it.

        Args:
            buffer: The staging buffer being closed.
        """
        if not buffer.closed:
            released_staging.append(buffer.pointer)
        real_close(buffer)

    monkeypatch.setattr(tensorrt_runtime.PinnedHostBuffer, "close", recording_close)
    monkeypatch.setattr(tensorrt_runtime, "cudart", fake)

    with pytest.raises(RuntimeError, match="cudaMalloc"):
        GpuPreprocessor(height=8, width=8, max_batch=1, optimal_batch=1, stream=0)

    assert fake.freed == [FIRST_FAKE_POINTER]
    assert fake.live == set()
    assert fake.double_frees == []
    assert len(released_staging) == 1
