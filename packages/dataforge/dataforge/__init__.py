"""dataforge: download → convert → register → view for every dataset in the monorepo.

Design: docs/dataforge-design-report.html (single-file, open in a browser).
"""

import os

if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package

    beartype_this_package()
