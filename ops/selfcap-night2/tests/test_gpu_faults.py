"""CUDA context failures are infrastructure, not bad session data."""
import pytest
from infrastructure import is_gpu_fault, is_infrastructure_error


@pytest.mark.parametrize('message', [
    'CUDA error: an illegal memory access was encountered',
    'cudaErrorIllegalAddress',
    'Error 807: MPS server is not ready to accept new MPS client requests',
    'RuntimeError: invalid argument to reset_peak_memory_stats',
])
def test_gpu_context_faults_are_infrastructure(message):
    assert is_gpu_fault(message)
    assert is_infrastructure_error(message)


def test_bad_input_and_oom_are_not_fatal_context_faults():
    assert not is_gpu_fault('SOMA_INPUT_LIMITATION missing body observations')
    assert not is_gpu_fault('CUDA out of memory')
