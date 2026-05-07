"""Resolve egoexo-forge ``.rrd`` files on HuggingFace to local cached paths.

The Rerun catalog server only consumes ``file://`` and object-store URIs (not
``https://`` or ``hf://``), so to mount HF-hosted recordings without
pre-downloading the entire repo we route each blob through
``huggingface_hub.hf_hub_download`` — cache-first, fetch on miss — and hand
the catalog the resulting local cache path as a ``file://`` URI.

For the Phase 1 spike we cap at ``max_per_source`` files per source so the
cold mount stays cheap (a handful of GB instead of the full corpus).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

SourceName = Literal["hocap", "egodex", "assembly101"]
ALL_SOURCES: tuple[SourceName, ...] = get_args(SourceName)


def resolve_hf_rrds(
    *,
    sources: tuple[SourceName, ...] = ALL_SOURCES,
    owner: str = "pablovela5620",
    repo_prefix: str = "egoexo-forge",
    max_per_source: int | None = 5,
    show_progress: bool = True,
) -> dict[SourceName, list[str]]:
    """Return ``{source: [file:// URI, ...]}`` backed by the HF hub cache.

    For each source, lists the dataset repo's ``.rrd`` files via
    ``HfApi.list_repo_files``, sorts lexically, truncates to
    ``max_per_source`` (``None`` = all), and resolves each to a local cache
    path with ``hf_hub_download``. Files already in cache are not re-fetched.

    Parameters
    ----------
    sources:
        Which sources to resolve. Repo id is ``f"{owner}/{repo_prefix}-{src}"``.
    owner:
        HF user/org owning the dataset repos. Defaults to the existing
        egoexo-forge upload owner.
    repo_prefix:
        Repo name prefix; combined with ``src`` to form the repo id.
    max_per_source:
        Cap files per source. ``None`` = no cap (full corpus). Defaults to
        ``5`` to keep the spike cheap.
    show_progress:
        Render a tqdm bar over the per-file ``hf_hub_download`` loop.
    """
    api: HfApi = HfApi()
    out: dict[SourceName, list[str]] = {}

    for source in sources:
        repo_id: str = f"{owner}/{repo_prefix}-{source}"
        all_files: list[str] = api.list_repo_files(repo_id, repo_type="dataset")
        rrd_files: list[str] = sorted(f for f in all_files if f.endswith(".rrd"))
        if max_per_source is not None:
            rrd_files = rrd_files[:max_per_source]

        uris: list[str] = []
        iterator = tqdm(
            rrd_files,
            desc=f"resolve {source}",
            unit="rrd",
            disable=not show_progress,
            leave=False,
        )
        for filename in iterator:
            local_path_str: str = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            uris.append(Path(local_path_str).resolve().as_uri())
        out[source] = uris

    return out
