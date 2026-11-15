"""Download one URL to a path unless it is already there, and check its sha256 (stdlib only; used by a pixi task)."""

import hashlib
import sys
import urllib.request
from pathlib import Path


def main(url: str, sha256: str, destination: Path) -> None:
    """Fetch ``url`` to ``destination`` (atomically) when missing; fail on a digest mismatch."""
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        staged: Path = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, staged)
        staged.replace(destination)
    digest: str = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{destination}: sha256 {digest}, expected {sha256}")
    print(f"ok {destination}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
