"""In-place ``rerun rrd compact`` wrapper.

``rerun rrd compact`` merges small chunks within a file (an iterative
clustering process) so the on-disk representation is more compact and
catalog-side reads are more uniform. We run it after every layer
write because (a) the maven_poc reference demo treats it as part of
the contract for any ``.rrd`` that's going to be served via a catalog
and (b) it's cheap (~ms per file at our sizes) and idempotent.

Empirical note: on our local 25 GB corpus the cold-mount wall-clock
isn't measurably affected by compaction (median 36 s with vs. ~37 s
without, both in the same noise range). The total on-disk size shrinks
by ~5 % (25 GB → 24 GB). The maven_poc team's "faster mounts" claim
comes from a cloud-served setup where chunk count drives transfer
cost, not parse cost; in our local in-process server the parse work
dominates regardless. We keep the compaction step anyway because the
cost is negligible and it converges with the rerun-recommended
practice for catalog-bound rrds.

The ``--max-bytes`` / ``--max-rows`` thresholds match the maven_poc
defaults (1 MB / 1 M rows / 1 M unsorted-rows). ``--num-pass`` is
left at the rerun CLI default of 50 (self-terminating once compaction
converges, so the cap rarely binds).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_RERUN_RRD_COMPACT: tuple[str, ...] = (
    "rerun",
    "rrd",
    "compact",
    "--max-bytes",
    "1000000",
    "--max-rows",
    "1000000",
    "--max-rows-if-unsorted",
    "1000000",
)


def compact_rrd(path: Path) -> None:
    """Compact ``path`` in place. Idempotent and safe to call repeatedly."""
    subprocess.run(
        [*_RERUN_RRD_COMPACT, "-o", str(path), str(path)],
        check=True,
    )
