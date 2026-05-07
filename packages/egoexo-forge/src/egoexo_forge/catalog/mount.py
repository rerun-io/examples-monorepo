"""Mount egoexo-forge ``.rrd`` URIs as a Rerun catalog server.

One catalog ``Dataset`` per source (``hocap``, ``egodex``, ``assembly101``);
each source's ``.rrd`` files register under ``layer_name="base"``. Egoexo
RRDs are self-contained per-sequence (not layered like slam-evals), so each
file becomes its own segment.

The URIs are typically ``file://`` paths into the HuggingFace hub cache —
see ``egoexo_forge.catalog.hf_resolve.resolve_hf_rrds``.
"""

from __future__ import annotations

import rerun as rr
from rerun.catalog import OnDuplicateSegmentLayer
from tqdm import tqdm

from egoexo_forge.catalog.hf_resolve import SourceName


def mount_catalog(
    uris_by_source: dict[SourceName, list[str]],
    *,
    port: int | None = None,
    show_progress: bool = True,
) -> rr.server.Server:
    """Spin up an in-process gRPC catalog with one Dataset per source.

    Parameters
    ----------
    uris_by_source:
        Mapping of source name to the list of ``.rrd`` URIs to register
        into that source's Dataset. Each URI must be a scheme-qualified
        string (``file://...`` or an object-store URI). Empty lists are
        skipped (no Dataset created).
    port:
        gRPC port for the catalog server. ``None`` picks a free random port.
    show_progress:
        Render a tqdm bar over the per-source registration loop.

    Returns
    -------
    rr.server.Server
        The running catalog server. Use as a context manager so it shuts
        down cleanly on exit.
    """
    sources: list[SourceName] = sorted(s for s, uris in uris_by_source.items() if uris)
    if not sources:
        raise ValueError("uris_by_source is empty (no non-empty sources)")

    total_files: int = sum(len(uris_by_source[s]) for s in sources)
    print(
        f"Mounting catalog ({total_files} RRDs across {len(sources)} sources: {', '.join(sources)})…",
        flush=True,
    )

    server: rr.server.Server = rr.server.Server(datasets={s: [] for s in sources}, port=port)
    client = server.client()

    iterator = tqdm(sources, desc="register", unit="source", disable=not show_progress)
    for src in iterator:
        uris: list[str] = uris_by_source[src]
        iterator.set_postfix_str(f"{src} ({len(uris)} files)")
        dataset = client.get_dataset(src)
        # pyrefly can't see ``register`` because we accept ``dataset`` as the
        # untyped client return value (DatasetEntry-like). Same trick as
        # slam_evals/catalog/mount.py.
        dataset.register(uris, layer_name="base", on_duplicate=OnDuplicateSegmentLayer.ERROR).wait()  # type: ignore[attr-defined]

    return server
