"""Shared live-catalog probe for the integration-gated tests."""

import socket

PROBE_CATALOG_URL: str = "rerun+http://127.0.0.1:51299"
"""Ephemeral 0.35 `rerun server` with one registered segment (tmux gsplat-catalog-035)."""


def catalog_reachable() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 51299), timeout=1.0):
            return True
    except OSError:
        return False
