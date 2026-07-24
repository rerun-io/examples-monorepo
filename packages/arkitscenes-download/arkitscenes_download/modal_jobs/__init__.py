"""Reusable core for the ARKitScenes Modal jobs (abc-130k-conversion pattern, pixi env inside).

Two shared Modal artifacts:

- **the container image** (`arkitscenes_image`) — the monorepo's own pixi env, installed
  from the workspace ``pixi.toml`` + ``pixi.lock`` with ``--frozen``: byte-identical deps
  to the local run, including the conda-forge ffmpeg (av1_nvenc + libsvtav1). Workers
  subprocess into that env's python exactly like ``pipeline.py`` does locally; Modal's
  primary interpreter only orchestrates and uploads.
- **the secret** (`hf_credentials`) — the caller's own HuggingFace token shipped as an
  ephemeral per-run secret; nothing is stored on Modal.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HF_REPO_ID: str = os.getenv("ARKITSCENES_HF_REPO", "pablovela5620/arkitscenes-rrd")
"""Destination dataset repo, layer-first: ``<layer>/<video_id>.rrd``."""

ENV_BIN = "/workspace/.pixi/envs/arkitscenes-download/bin"
"""The pixi env's binaries inside the container (python, ffmpeg, rerun, curl)."""

PACKAGE_DIR = "/workspace/packages/arkitscenes-download"
"""Container path of this package — cwd for the download/ingest tool subprocesses."""

_PIXI_VERSION = "v0.73.0"  # matches the version that installs this lock locally
# parents[4]: modal_jobs → arkitscenes_download → <package> → packages → workspace root.
# In-container (source mounted at /root) that depth doesn't exist — guard on is_local.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[4] if modal.is_local() else Path("/workspace")
_IGNORE = ["**/.git", "**/__pycache__", "**/.pixi", "**/data", "**/*.rrd", "**/.pytest_cache"]

arkitscenes_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates")
    .run_commands(f"curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION={_PIXI_VERSION} PIXI_HOME=/usr/local bash")
    # The primary (Modal-runtime) interpreter needs only the upload client.
    .uv_pip_install("huggingface_hub[hf_xet]")
    # Workspace manifest + lock + the two editable path deps this env installs.
    .add_local_file(_WORKSPACE_ROOT / "pixi.toml", "/workspace/pixi.toml", copy=True)
    .add_local_file(_WORKSPACE_ROOT / "pixi.lock", "/workspace/pixi.lock", copy=True)
    .add_local_dir(_WORKSPACE_ROOT / "packages" / "simplecv", "/workspace/packages/simplecv", copy=True, ignore=_IGNORE)
    .add_local_dir(_WORKSPACE_ROOT / "packages" / "arkitscenes-download", PACKAGE_DIR, copy=True, ignore=_IGNORE)
    .run_commands(
        "cd /workspace && pixi install --frozen -e arkitscenes-download",
        # Fail the build early if the env's ffmpeg lacks either encoder.
        f"{ENV_BIN}/ffmpeg -hide_banner -encoders | grep av1_nvenc",
        f"{ENV_BIN}/ffmpeg -hide_banner -encoders | grep libsvtav1",
        f"{ENV_BIN}/rerun analytics disable",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    # Runtime mount (last, no copy): lets the container's primary interpreter import
    # this module tree without invalidating the pixi-install layer on code edits.
    .add_local_python_source("arkitscenes_download")
)


def _local_hf_token() -> str:
    """The caller's HuggingFace token, from ``$HF_TOKEN`` or the ``hf auth login`` cache."""
    from huggingface_hub import get_token

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise SystemExit("No HuggingFace token found; run `hf auth login` or set $HF_TOKEN.")
    return token


# Built at launch time from the caller's token; empty inside the container (the
# runtime injects the materialized secret as $HF_TOKEN there instead).
hf_credentials = modal.Secret.from_dict({"HF_TOKEN": _local_hf_token()} if modal.is_local() else {})

# Staging volume between the GPU converters and the single HF uploader: HF throttles
# per-repo commits hard (429s at even 8 concurrent per-sequence commits), so workers
# never upload to HF — they drop results here and one drain process batch-uploads.
staging_volume = modal.Volume.from_name("arkitscenes-rrd-staging", create_if_missing=True)
STAGING_MOUNT = "/staging"
