"""Dump catalog inputs for optional offline Rust replay and isolated timing.

The directory holds clip metadata, calibration JSON, IMU CSV, frame hashes,
gray8 PGMs and integer timestamps in clip metadata. The optional NPZ and calibration pickle
feed bench_track without decoding in its timing loop. NPZ holds frames in
memory, so bound benchmark dumps with max_framesets.

Catalog calibration uses f32 statics; fixture calibration uses the checked-in
JSON doubles. Keeping this explicit permits sensitivity measurements.
A full MIO07 stereo dump is about 7.5 GB and is not committed.
"""

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs import _core
from slam_rs.catalog_feed import CatalogSegment, Frameset, open_segment
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest, resolved_flow_config

FIXTURES: Path = Path(__file__).resolve().parents[2] / "crates/slam-rs/tests/fixtures"
"""Directory holding MSD calibration JSON files."""

DEVICE_CALIBRATION: dict[str, str] = {"msd-index": "msdmi_calib.json", "msd-g2": "msdmg_calib.json"}
"""Catalog dataset to its calibration JSON with double-precision values."""


@dataclass(slots=True)
class Config:
    """Dump one reference segment's pixels, inertial samples and calibration."""

    segment: str
    """Segment id from ``gate.toml``."""
    output: Path
    """Directory the clip is written to; created if missing."""
    calibration: Literal["catalog", "fixture"] = "catalog"
    """``catalog`` writes float32 statics; ``fixture`` writes calibration-file doubles."""
    max_framesets: int | None = None
    """Stop after this many framesets; None dumps the whole segment."""
    npz: bool = False
    """Also write ``clip.npz`` and ``clip.npz.calib.pkl``, which is what ``bench_track`` replays."""
    window_s: float = 60.0
    """Longest time window fetched in one round trip."""


def write_pgm(path: Path, image: UInt8[ndarray, "h w"]) -> None:
    """Write one grayscale image in the binary PGM the Rust lanes read.

    Args:
        path: Destination file.
        image: C-contiguous 8-bit raster.
    """
    header: bytes = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + image.tobytes())


def main(config: Config) -> None:
    """Dump the segment named by the config.

    Args:
        config: Parsed CLI options.

    Raises:
        ValueError: If ``--npz`` was asked for with no frameset to bundle, if the
            segment is not in the manifest, or if fixture mode is selected and its dataset has no fixture
            calibration.
    """
    # Before the manifest, the catalog and the output directory: zero framesets
    # was accepted, every non-NPZ side file was written, and then `np.stack` on
    # the empty image list raised out of NumPy — leaving a directory that reads
    # as a clip and holds no frameset (S25 review).
    if config.npz and config.max_framesets == 0:
        raise ValueError("--max-framesets 0 with --npz has no frameset to bundle; the bench dump needs at least one")
    manifest: ReferenceManifest = load_manifest()
    segment: ReferenceSegment = manifest.by_id(config.segment)

    config.output.mkdir(parents=True, exist_ok=True)
    bundle_images: list[UInt8[ndarray, "n_cameras h w"]] = []
    bundle_imu_counts: list[int] = []
    bundle_imu_t: list[Int64[ndarray, " n_samples"]] = []
    bundle_imu_gyro: list[Float64[ndarray, "n_samples 3"]] = []
    bundle_imu_accel: list[Float64[ndarray, "n_samples 3"]] = []
    imu_lines: list[str] = ["#t_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z"]
    frame_lines: list[str] = ["# t_ns,cam_index,sha256 of the gray8 HxW pixels fed to the tracker"]
    frame_t_ns: list[int] = []
    dumped: int = 0

    with open_segment(
        CatalogSegment(manifest.catalog_url, segment.dataset_name, segment.segment_id),
        manifest.dataset(segment.dataset_name).imu,
        window_s=config.window_s,
    ) as feed:
        calibration_text: str
        if config.calibration == "fixture":
            if segment.dataset_name not in DEVICE_CALIBRATION:
                raise ValueError(f"{segment.dataset_name} has no fixture calibration file")
            calibration_text = json.dumps(json.loads((FIXTURES / DEVICE_CALIBRATION[segment.dataset_name]).read_text()), indent=2)
        else:
            calibration_text = _core.Calibration.from_catalog(feed.cameras, feed.imu).to_json()
        (config.output / "calib.json").write_text(calibration_text + "\n")
        frameset: Frameset
        for frameset in feed.framesets():
            if config.max_framesets is not None and dumped >= config.max_framesets:
                break
            for index, image in enumerate(frameset.images):
                write_pgm(config.output / f"frame_{dumped:03d}_cam{index}.pgm", image)
            if config.npz:
                bundle_images.append(np.stack(frameset.images))
                bundle_imu_counts.append(len(frameset.imu))
                bundle_imu_t.append(frameset.imu.t_ns)
                bundle_imu_gyro.append(frameset.imu.gyro_rad_s)
                bundle_imu_accel.append(frameset.imu.accel_m_s2)
            absolute_ns: int = frameset.t_ns + feed.capture_start_time_ns
            frame_lines.extend(f"{absolute_ns},{index},{digest}" for index, digest in enumerate(frameset.image_digests()))
            for t_ns, gyro, accel in zip(frameset.imu.t_ns.tolist(), frameset.imu.gyro_rad_s.tolist(), frameset.imu.accel_m_s2.tolist(), strict=True):
                imu_lines.append(f"{t_ns}," + ",".join(repr(value) for value in [*gyro, *accel]))
            frame_t_ns.append(frameset.t_ns)
            dumped += 1
            if dumped % 200 == 0:
                print(f"{dumped} framesets", flush=True)

        clip: dict[str, Any] = {
            "segment_id": feed.segment_id,
            "dataset_name": segment.dataset_name,
            "capture_start_time_ns": feed.capture_start_time_ns,
            "calibration_source": config.calibration,
            "num_cameras": len(feed.cameras),
            "resolution_wh": [[camera.width, camera.height] for camera in feed.cameras],
            "framesets": dumped,
            "frame_t_ns": frame_t_ns,
            "imu_samples": len(imu_lines) - 1,
            "decode_path": segment.decode_path,
        }
    (config.output / "imu.csv").write_text("\n".join(imu_lines) + "\n")
    (config.output / "frames.sha256").write_text("\n".join(frame_lines) + "\n")
    (config.output / "clip.json").write_text(json.dumps(clip, indent=2) + "\n")
    if config.npz:
        bundle: Path = config.output / "clip.npz"
        np.savez(
            bundle,
            images=np.stack(bundle_images),
            t_ns=np.array(frame_t_ns, dtype=np.int64),
            imu_counts=np.array(bundle_imu_counts, dtype=np.int64),
            imu_t=np.concatenate(bundle_imu_t),
            imu_g=np.concatenate(bundle_imu_gyro),
            imu_a=np.concatenate(bundle_imu_accel),
            safe_radius=np.int64(resolved_flow_config(manifest, segment)[0].optical_flow_image_safe_radius),
        )
        # The dataclasses the feed built, because `bench_track` compares lanes on
        # the calibration the reference ran and not on a second derivation of it.
        # Pickle rather than JSON for the same reason: these two values are the
        # feed's own, and a hand-written schema here would be a third account of
        # what a calibration is. Producer and consumer therefore ship together.
        with bundle.with_suffix(bundle.suffix + ".calib.pkl").open("wb") as handle:
            pickle.dump({"cameras": feed.cameras, "imu": feed.imu}, handle)
        print(f"clip.npz {np.stack(bundle_images).shape} + calib.pkl -> {bundle}")
    digest: str = hashlib.sha256((config.output / "frames.sha256").read_bytes()).hexdigest()
    print(f"{dumped} framesets, {clip['imu_samples']} inertial samples -> {config.output}")
    print(f"frames.sha256 {digest}")

