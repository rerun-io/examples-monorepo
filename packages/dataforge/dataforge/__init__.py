"""dataforge: download → convert → register → view for every dataset in the monorepo.

Design: docs/dataforge-design-report.html (single-file, open in a browser).
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype import BeartypeConf
    from beartype.claw import beartype_this_package

    # Check function boundaries only. With PEP 526 checks on, every annotated local
    # such as ``x: Float64[ndarray, "n 3"] = ...`` re-evaluates its jaxtyping hint on
    # each execution; jaxtyping returns a new class every time and beartype compiles
    # and caches a new checker for it. This package's inner loops are exactly that
    # shape — the per-chunk video tap and the csv/gt arrays, both of which run once
    # per sample of a multi-million-row sequence.
    beartype_this_package(conf=BeartypeConf(claw_is_pep526=False))
