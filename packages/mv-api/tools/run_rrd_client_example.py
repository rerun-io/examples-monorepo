"""Call the Gradio RRD pipeline locally and report progress."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import tyro
from gradio_client import Client, handle_file
from gradio_client.client import Job
from gradio_client.utils import StatusUpdate

DEFAULT_GRADIO_URL: str = "http://127.0.0.1:7860/"
DEFAULT_RRD_PATH: Path = Path("/mnt/8tb/data/exoego-self-collected/gus/statisOrangePNP_av1.rrd")


@dataclass
class RRDClientConfig:
    """Configuration for the RRD client smoke test."""

    gradio_url: str = DEFAULT_GRADIO_URL
    """Base URL for the Gradio server to target."""

    rrd_path: Path = DEFAULT_RRD_PATH
    """Absolute path to the RRD file that will be uploaded."""

    max_frames: int = 10
    """Maximum number of frames to process (use 0 or negative for all)."""

    poll_interval_seconds: float = 2.0
    """Time to wait between queue status checks."""

    bearer_token: str | None = None
    """Optional bearer token to attach as `Authorization: Bearer …`."""

    http_timeout_seconds: float | None = None
    """Override the HTTPX timeout in seconds when targeting slow endpoints."""


def main(config: RRDClientConfig) -> None:
    """Submit the pipeline job and stream queue/progress updates."""
    bearer_token: str | None = config.bearer_token or os.environ.get("MV_API_LIGHTNING_TOKEN")

    timeout_seconds: float | None = config.http_timeout_seconds
    if timeout_seconds is None:
        timeout_env: str | None = os.environ.get("MV_API_HTTP_TIMEOUT")
        if timeout_env:
            try:
                timeout_seconds = float(timeout_env)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid MV_API_HTTP_TIMEOUT value: {timeout_env}"
                ) from exc

    if timeout_seconds is not None and timeout_seconds <= 0:
        timeout_seconds = None

    headers: dict[str, str] | None = None
    if bearer_token:
        headers = {"Authorization": f"Bearer {bearer_token}"}

    httpx_kwargs: dict[str, object] | None = None
    if timeout_seconds is not None:
        timeout: httpx.Timeout = httpx.Timeout(timeout_seconds, connect=timeout_seconds)
        httpx_kwargs = {"timeout": timeout}

    client: Client = Client(config.gradio_url, headers=headers, httpx_kwargs=httpx_kwargs)
    job: Job = client.submit(
        rrd_input=handle_file(str(config.rrd_path)),
        max_frames=None if config.max_frames <= 0 else config.max_frames,
        api_name="/run_rrd_pipeline",
    )
    print("Submitted job; polling queue status…")

    while not job.done():
        status: StatusUpdate = job.status()
        rank: int | None = status.rank
        queue_size: int | None = status.queue_size
        eta_seconds: float | None = status.eta
        eta_display: str = f"{eta_seconds:.1f}s" if eta_seconds is not None else "n/a"
        print(f"  position {rank}/{queue_size}, estimated wait {eta_display}")
        time.sleep(config.poll_interval_seconds)

    result = job.result()
    print("Pipeline completed.")
    print(f"Output RRD: {result}")


if __name__ == "__main__":
    main(tyro.cli(RRDClientConfig))
