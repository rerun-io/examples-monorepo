"""Opt-in NVDEC must preserve the frozen feed's timestamps and image bytes."""

import hashlib
import subprocess
from itertools import islice
from pathlib import Path
from shutil import which

import pytest

from slam_rs.apis.bench_track import Framesets, load_framesets
from slam_rs.catalog_feed import LocalSegment, open_segment
from slam_rs.reference import SMOKE_SEGMENTS, ImuParameters, load_manifest


@pytest.mark.slow
@pytest.mark.nvdec
def test_nvdec_matches_first_twenty_mio10_framesets() -> None:
    """Compare the public feed on the smallest available MIO10 NAS recording."""
    if which("nvidia-smi") is None:
        pytest.skip("No NVIDIA driver tools are installed")
    probe: subprocess.CompletedProcess[str] = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, check=False
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        pytest.skip("No NVIDIA device is visible")
    path: Path = Path("/mnt/nas/datasets/msd-rrd/base") / f"{SMOKE_SEGMENTS[1]}.rrd"
    if not path.exists():
        pytest.skip("MIO10 NAS clip is not mounted")
    parameters: ImuParameters = next(segment for segment in load_manifest().segments if segment.segment_id == SMOKE_SEGMENTS[1]).imu
    digests: list[list[tuple[int, tuple[str, ...]]]] = []
    for decoder in ("dav1d", "nvdec"):
        with open_segment(LocalSegment(path), parameters, decoder=decoder) as feed:
            digests.append([(frame.t_ns, frame.image_digests()) for frame in islice(feed.framesets(stop_ns=feed.stop_ns_after(20)), 20)])
    assert len(digests[0]) == 20
    assert digests[0] == digests[1]
    benchmark: Framesets = load_framesets(path, 20, decoder="nvdec")
    assert benchmark.t_ns.tolist() == [stamp for stamp, _ in digests[0]]
    assert [tuple(hashlib.sha256(image).hexdigest() for image in frame) for frame in benchmark.images] == [hashes for _, hashes in digests[0]]
