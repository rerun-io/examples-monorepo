"""Identify executable/schema incompatibilities, never input quality failures."""
def is_infrastructure_error(detail: str) -> bool:
    return any(marker in detail for marker in (
        'ImportError:', 'ModuleNotFoundError:',
        "AttributeError: module '",
        'SerdeError: unknown fields:',
    ))
