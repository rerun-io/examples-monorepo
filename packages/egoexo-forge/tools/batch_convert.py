"""Deprecated entrypoint.

Ego/exo dataset -> Rerun ``.rrd`` batch conversion moved to dataforge
(``dataforge.apis.convert``). This stub remains only so the
``egoexo-forge-batch-convert`` task does not error.
"""

if __name__ == "__main__":
    print("deprecated: ego/exo batch .rrd conversion moved to dataforge.apis.convert")
