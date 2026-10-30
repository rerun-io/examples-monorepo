"""Identify executable/schema incompatibilities, never input quality failures."""
def is_gpu_fault(detail: str) -> bool:
    """Fatal device/context errors, including the secondary poisoned-MPS symptoms."""
    text = detail.lower()
    return any(marker in text for marker in (
        'illegal memory access', 'illegal address', 'cudaerrorillegaladdress',
        'mps server is not ready', 'mps connection', 'mps rpc',
        'invalid argument to reset_peak_memory_stats', 'cuda error: misaligned address',
        'cuda error: device-side assert triggered',
    ))


def is_infrastructure_error(detail: str) -> bool:
    return is_gpu_fault(detail) or any(marker in detail for marker in (
        'ImportError:', 'ModuleNotFoundError:',
        "AttributeError: module '",
        'SerdeError: unknown fields:',
    ))
