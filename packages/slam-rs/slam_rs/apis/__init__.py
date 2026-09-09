"""Tyro-facing entry points; the scripts under ``tools/`` are thin shims over these.

Every one of these tools reaches a user error the same way — a manifest id that
is not in the manifest, a recording that is not on this machine, a
configuration the port does not run, ``--gpu`` on a host with no driver — and
what the operator should see for all of them is one sentence and a non-zero
exit, not a traceback through the feed. :func:`run` is where that conversion
lives, in the shim rather than in ``main``: every test drives ``main(Config(…))``
and asserts the typed ``ValueError`` or ``FileNotFoundError``, which is the
contract the tools are written against and stays exactly as it is.
"""

from collections.abc import Callable

import tyro


def run[Config](config_type: type[Config], main: Callable[[Config], None]) -> None:
    """Parse the command line and run one tool, turning a user error into one sentence.

    Args:
        config_type: The tool's Tyro config dataclass.
        main: The tool's entry point.

    Raises:
        SystemExit: With the error's own text, for the two typed errors every
            tool raises on bad input.
    """
    try:
        main(tyro.cli(config_type))
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
