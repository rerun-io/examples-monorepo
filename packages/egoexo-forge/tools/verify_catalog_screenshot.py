#!/usr/bin/env python3
"""Verify the egoexo-forge catalog spike end-to-end and save a screenshot.

Steps:

1. Resolve 5 RRDs per source from HuggingFace via the hub cache.
2. Mount the catalog server (in-process); query each Dataset's
   ``segment_table()`` to confirm the expected number of segments.
3. Spawn a Rerun viewer subprocess pointed at one cached ``.rrd`` from
   each source (``rerun --port 9876 --hide-welcome-screen <rrd>...``).
   Concrete proof the RRD blobs resolved from HF are intact and renderable.
4. Save a PNG screenshot via the experimental ``ViewerClient`` API
   (mirrors https://github.com/rerun-io/rerun/blob/0.29.0/docs/snippets/all/howto/screenshot.py).

Output: ``data/egoexo-forge/screenshots/catalog_spike.png`` (relative to
repo root). Exits 0 on success, non-zero on any verification failure.

Requires a working display + GPU/wgpu surface — wgpu can't render through
``xvfb-run`` without DRI3, so the screenshot step is no-op in pure-headless
sandboxes (the catalog/segment-count steps still verify there).
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import rerun as rr
import tyro
from rerun.experimental import ViewerClient

from egoexo_forge.catalog import mount_catalog, resolve_hf_rrds
from egoexo_forge.catalog.hf_resolve import ALL_SOURCES, SourceName

EXPECTED_PER_SOURCE: int = 5
VIEWER_PORT: int = 9876
VIEWER_STARTUP_SECONDS: float = 8.0
SCREENSHOT_FLUSH_SECONDS: float = 5.0


@dataclass
class VerifyConfig:
    out_path: Path = Path("../../data/egoexo-forge/screenshots/catalog_spike.png")
    """Where to save the screenshot, relative to ``packages/egoexo-forge`` (cwd of the pixi task)."""

    catalog_port: int = 9988
    """Port for the in-process catalog server (different from the spawned viewer's 9876)."""


def _uri_to_path(uri: str) -> Path:
    """Convert a ``file://`` URI back to a local ``Path``."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"expected file:// URI, got {uri!r}")
    return Path(unquote(parsed.path))


def _verify_catalog_segments(server: rr.server.Server, sources: tuple[SourceName, ...]) -> None:
    """Assert each Dataset has ``EXPECTED_PER_SOURCE`` segments registered."""
    client = server.client()
    for src in sources:
        dataset = client.get_dataset(src)
        df = dataset.segment_table().to_pandas()  # type: ignore[attr-defined]
        n_segments: int = len(df)
        print(f"  catalog[{src}]: {n_segments} segments")
        if n_segments != EXPECTED_PER_SOURCE:
            raise AssertionError(
                f"expected {EXPECTED_PER_SOURCE} segments in {src}, got {n_segments}",
            )


def main(cfg: VerifyConfig) -> None:
    sources: tuple[SourceName, ...] = ALL_SOURCES

    print(f"[1/4] Resolving {EXPECTED_PER_SOURCE} RRDs per source from HuggingFace…")
    uris_by_source: dict[SourceName, list[str]] = resolve_hf_rrds(
        sources=sources,
        max_per_source=EXPECTED_PER_SOURCE,
    )
    for src in sources:
        if len(uris_by_source[src]) != EXPECTED_PER_SOURCE:
            raise AssertionError(
                f"expected {EXPECTED_PER_SOURCE} URIs for {src}, got {len(uris_by_source[src])}",
            )

    print(f"\n[2/4] Mounting catalog on port {cfg.catalog_port}…")
    with mount_catalog(uris_by_source, port=cfg.catalog_port) as server:
        print(f"  catalog URL: {server.url()}")
        _verify_catalog_segments(server, sources)

        rrd_paths: list[Path] = [_uri_to_path(uris_by_source[s][0]) for s in sources]
        print(f"\n[3/4] Spawning Rerun viewer on port {VIEWER_PORT} with {len(rrd_paths)} RRDs…")
        for src, rrd_path in zip(sources, rrd_paths, strict=True):
            print(f"  {src}: {rrd_path.name}")
        viewer_proc = subprocess.Popen(
            [
                "rerun",
                "--port",
                str(VIEWER_PORT),
                "--hide-welcome-screen",
                *(str(p) for p in rrd_paths),
            ],
        )
        try:
            time.sleep(VIEWER_STARTUP_SECONDS)

            print(f"\n[4/4] Saving screenshot to {cfg.out_path}…")
            cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
            viewer = ViewerClient(addr=f"127.0.0.1:{VIEWER_PORT}")
            viewer.save_screenshot(str(cfg.out_path.resolve()))

            # save_screenshot is async (gRPC message to the viewer); give the
            # viewer time to render and write the PNG before the size check.
            time.sleep(SCREENSHOT_FLUSH_SECONDS)
            if not cfg.out_path.exists() or cfg.out_path.stat().st_size < 1024:
                raise AssertionError(
                    f"screenshot at {cfg.out_path} is missing or too small (<1KB)",
                )
            size_kb: float = cfg.out_path.stat().st_size / 1024
            print(f"  ✓ screenshot saved ({size_kb:.1f} KB)")
        finally:
            viewer_proc.terminate()
            try:
                viewer_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer_proc.kill()

    print("\nPASS: catalog spike verified end-to-end.")


if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_cyan")
    try:
        main(tyro.cli(VerifyConfig, description="Verify catalog spike + save a viewer screenshot."))
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
