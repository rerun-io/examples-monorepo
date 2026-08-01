"""
Copy converted ARKitScenes RRDs from the HF dataset repo to rerun's AWS scratch bucket.

Pure byte transfer — no GPU, no pixi env, no RRD processing. Discovery makes a single
`list_repo_files` call and hands each worker one sequence's exact filenames, so workers
never touch the HF metadata API (the rate-limited surface). Both sides are layer-first
(the HF repo migrated to the layer-first layout on 2026-07-23):

    HF:  <layer>/<video_id>.rrd
    S3:  arkitscenes.<date>/<layer>/<video_id>.rrd

Layer-first keys make hub registration one `register_prefix` call per layer instead of
per-file operations. AWS access is Modal's OIDC identity exchanged for temporary STS
credentials (abc-130k pattern) — no keys stored anywhere.

Run:  modal run -m arkitscenes_download.modal_jobs.transfer_to_s3 --limit 5   # smoke test
      modal run --detach -m arkitscenes_download.modal_jobs.transfer_to_s3 --limit 0
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

from arkitscenes_download.modal_jobs import hf_credentials

HF_REPO_ID = "pablovela5620/arkitscenes-rrd"
# Mirrors ingest.layers.LAYER_NAMES — deliberately not imported: this module runs in
# a container interpreter that can't import the ingest package (rerun/rich-heavy).
KNOWN_LAYERS = ("base", "calibration", "depth", "gt", "imu", "video_ultrawide", "video_wide")

AWS_ROLE_ARN = "arn:aws:iam::069742552781:role/modal-oidc-role"
BUCKET = "rerun-datasets-scratch-446437544659-us-east-1-an"
REGION = "us-east-1"
DATASET_PREFIX = "arkitscenes.2026.07.22"

# HF rate-limits the per-file xet-read-token endpoint (~1 metadata call per download).
# 32 workers (~28 req/s) drew sustained 429s on the full run; 16 stays under it, and
# the retry ladders below absorb transient bursts.
MAX_CONTAINERS = 16

transfer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("boto3", "huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("arkitscenes-transfer-to-s3", image=transfer_image)


def _scratch_s3_client() -> Any:
    """Boto3 S3 client on the scratch bucket via Modal-OIDC → STS (no stored keys)."""
    import boto3

    sts = boto3.client("sts", region_name=REGION)
    resp = sts.assume_role_with_web_identity(
        RoleArn=AWS_ROLE_ARN,
        RoleSessionName="arkitscenes-transfer",
        WebIdentityToken=os.environ["MODAL_IDENTITY_TOKEN"],
    )
    creds = resp["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=REGION,
    ).client("s3")


def _s3_exists(s3: Any, key: str) -> bool:
    """Whether an object exists (403/404 both mean absent, per abc-130k)."""
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=BUCKET, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("403", "404", "NoSuchKey"):
            return False
        raise
    return True


@app.function(
    timeout=30 * 60,
    cpu=2,
    memory=2048,
    region=REGION,
    secrets=[hf_credentials],
    max_containers=MAX_CONTAINERS,
    retries=modal.Retries(max_retries=5, initial_delay=30.0, backoff_coefficient=2.0),
)
def transfer_sequence(video_id: str, filenames: list[str], overwrite: bool = False) -> dict[str, Any]:
    """Copy one sequence's layer files HF → scratch bucket, skipping keys that already exist."""
    import random

    from huggingface_hub import hf_hub_download

    def _download_with_backoff(filename: str, tmp: str) -> str:
        # HF 429s on the per-file token endpoint are transient; back off in-worker
        # before escalating to a (much more expensive) whole-sequence Modal retry.
        for attempt in range(5):
            try:
                return hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename, local_dir=tmp)
            except Exception as exc:
                if "429" not in str(exc) or attempt == 4:
                    raise
                time.sleep(2**attempt * 10 + random.uniform(0, 5))
        raise AssertionError("unreachable")

    s3 = _scratch_s3_client()
    t0 = time.monotonic()
    copied, skipped, total_bytes = [], [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for filename in filenames:
            layer = filename.split("/")[0]
            assert layer in KNOWN_LAYERS, f"unexpected HF path shape: {filename}"
            dest_key = f"{DATASET_PREFIX}/{layer}/{video_id}.rrd"
            if not overwrite and _s3_exists(s3, dest_key):
                skipped.append(layer)
                continue
            local = _download_with_backoff(filename, tmp)
            s3.upload_file(local, BUCKET, dest_key)
            total_bytes += Path(local).stat().st_size
            Path(local).unlink()  # keep the tmpdir footprint at one file
            copied.append(layer)
    return {
        "video_id": video_id,
        "copied": copied,
        "skipped": skipped,
        "bytes": total_bytes,
        "secs": round(time.monotonic() - t0, 1),
    }


def _discover(limit: int, layer: str | None = None) -> dict[str, list[str]]:
    """One `list_repo_files` call → {video_id: [its layer filenames]} (workers make no Hub calls)."""
    from huggingface_hub import HfApi

    files: list[str] = []
    for attempt in range(6):
        try:
            files = HfApi().list_repo_files(HF_REPO_ID, repo_type="dataset")
            break
        except Exception as exc:
            if "429" not in str(exc) or attempt == 5:
                raise
            wait = 120 * (attempt + 1)
            print(f"discovery rate-limited (429), retrying in {wait}s...")
            time.sleep(wait)

    sequences: dict[str, list[str]] = {}
    for f in files:
        parts = f.split("/")
        if len(parts) == 2 and parts[0] in KNOWN_LAYERS and f.endswith(".rrd"):
            if layer is not None and parts[0] != layer:
                continue
            sequences.setdefault(Path(f).stem, []).append(f)
    ordered = dict(sorted(sequences.items()))
    if limit > 0:
        ordered = dict(list(ordered.items())[:limit])
    return ordered


@app.local_entrypoint()
def main(limit: int = 5, overwrite: bool = False, layer: str | None = None, expect: int = 0) -> None:
    """Discover sequences on HF and fan the copy out across Modal workers."""
    sequences = _discover(limit, layer)
    n_files = sum(len(v) for v in sequences.values())
    if expect > 0 and n_files != expect:
        raise SystemExit(f"discovery found {n_files} files, expected {expect} — aborting before any transfer")
    print(f"transferring {len(sequences)} sequences ({n_files} files) → s3://{BUCKET}/{DATASET_PREFIX}/")

    ids = list(sequences.keys())
    file_lists = list(sequences.values())
    n_copied = n_skipped = n_bytes = n_done = 0
    failures: list[str] = []
    for i, result in enumerate(transfer_sequence.map(ids, file_lists, kwargs={"overwrite": overwrite}, return_exceptions=True)):
        if isinstance(result, Exception):
            failures.append(str(result)[:200])
            print(f"  FAILED (input #{i}): {str(result)[:200]}")
            continue
        n_done += 1
        n_copied += len(result["copied"])
        n_skipped += len(result["skipped"])
        n_bytes += result["bytes"]
        if n_done % 100 == 0 or result["copied"]:
            print(
                f"  [{n_done}/{len(ids)}] {result['video_id']}: copied {len(result['copied'])}, "
                f"skipped {len(result['skipped'])}, {result['bytes'] / 1e6:.0f} MB in {result['secs']}s"
            )
    print(f"done: {n_done}/{len(ids)} sequences, {n_copied} files copied, {n_skipped} skipped, {n_bytes / 1e9:.2f} GB moved")
    if failures:
        raise SystemExit(f"FAILURES: {len(failures)} sequences — rerun this command to retry (existing keys skip).")
