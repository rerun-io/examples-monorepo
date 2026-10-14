"""Tyro-facing entry points; the scripts under ``tools/`` are thin shims over these.

Unknown reference benchmark IDs, recordings absent from this host, unsupported configs and ``--gpu`` without a driver
must produce one sentence and a non-zero exit, not a feed traceback. The shim's :func:`run` converts these errors;
``main`` keeps the tools' typed ValueError/FileNotFoundError contract, asserted by every test via ``main(Config(...))``.
"""

from collections.abc import Callable

import tyro


def run[Config](config_type: type[Config], main: Callable[[Config], None]) -> None:
    """Parse CLI args with Tyro dataclass ``config_type`` and call entry point ``main``.
    Convert tools' bad-input ValueError/FileNotFoundError to SystemExit with the error's own text as one sentence.
    """
    try:
        main(tyro.cli(config_type))
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
