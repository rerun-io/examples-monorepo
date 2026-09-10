"""Dump a whole reference segment to disk so the Rust lanes can replay it offline.

`crates/slam-rs/tests/full_clip.rs` needs three things the committed fixtures only
carry for sixty framesets: every frameset's pixels, every inertial sample, and the
calibration the C++ reference was actually handed. This writes all three into one
directory:

```
<clip>/clip.json              segment, clock, frameset timestamps, camera shapes
<clip>/calib.json             basalt-shaped calibration, from the catalog statics
<clip>/imu.csv                t_ns,gx,gy,gz,ax,ay,az on the video_time clock
<clip>/frames.sha256          the C++ reference's own format, absolute timestamps
<clip>/frame_<NNN>_cam<C>.pgm tools/dump_flow.cpp's layout, gray8
<clip>/timestamps.txt         the same layout's frameset clock, one per line
<clip>/imu.json               the same samples as imu.csv, in the fork's shape
<clip>/clip.npz               `--npz`: the same framesets as one array bundle
<clip>/clip.npz.calib.pkl     `--npz`: the feed's own `CameraCalib`/`ImuCalib`
```

The last pair is what `slam_rs.apis.bench_track` replays: that harness times
`Vio.track` with no decoder and no Rerun, so it needs the pixels as one array
and the calibration as the dataclasses the feed built — this is the tool that
writes them, and `--npz` holds every frame in memory, so a bench dump wants
`--max-framesets`.

The last two are what the fork's `basalt_vio_oracle <frames-dir> <calib.json>
<config.json> <out.json> [n]` reads, so one dump feeds both the port's own lane
and the C++ oracle that isolates the backend from the frontend.

The pixels come off the frozen `cpu_gray8_dav1d_1thread` decode path, so
`frames.sha256` is comparable line by line with the reference run's file — which
is the check that both implementations are fed the same clip. `--calibration
fixture` swaps in the fork's `data/msd/msd*_calib.json` doubles instead of the
catalog's float32-stored values, which is how the port's sensitivity to that
difference is measured.

A whole segment is large: MIO07 is 4,095 framesets of 960x960 stereo, 7.5 GB.
Nothing here is committed and the directory is meant to be deleted afterwards.
"""

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro
from jaxtyping import Float64, Int64, UInt8
from numpy import ndarray

from slam_rs.catalog_feed import CameraCalib, Frameset, LocalSegment, SegmentFeed, open_segment
from slam_rs.reference import ReferenceManifest, ReferenceSegment, load_manifest, resolved_flow_config

FIXTURES: Path = Path(__file__).resolve().parents[2] / "crates/slam-rs/tests/fixtures"
"""Where the fork's own MSD calibration files sit in this repository."""

DEVICE_CALIBRATION: dict[str, str] = {"msd-index": "msdmi_calib.json", "msd-g2": "msdmg_calib.json"}
"""Catalog dataset to the fork calibration file that carries the same rig in doubles."""


@dataclass(slots=True)
class Config:
    """Dump one reference segment's pixels, inertial samples and calibration."""

    segment: str
    """Segment id from ``reference_segments.toml``."""
    output: Path
    """Directory the clip is written to; created if missing."""
    calibration: Literal["catalog", "fixture"] = "catalog"
    """``catalog`` writes the float32-stored values the C++ reference was pushed; ``fixture`` writes the fork file's doubles."""
    max_framesets: int | None = None
    """Stop after this many framesets; None dumps the whole segment."""
    npz: bool = False
    """Also write ``clip.npz`` and ``clip.npz.calib.pkl``, which is what ``bench_track`` replays."""
    window_s: float = 60.0
    """Longest time window fetched in one round trip."""


def eigen_quaternion_xyzw(rotation: Float64[ndarray, "3 3"]) -> Float64[ndarray, " 4"]:
    """Eigen's ``Quaterniond(Matrix3d)``, term for term.

    ``vit_tracker.cpp:288`` builds the camera extrinsic this way from the 4x4 the
    driver pushes, and the matrix it is given is not exactly orthonormal — the
    catalog stores float32. Different conversions disagree in the last bits on
    such a matrix, so this reproduces the one the reference actually ran rather
    than calling a library.

    Args:
        rotation: Rotation block of ``imu_T_cam``.

    Returns:
        The quaternion as ``[qx, qy, qz, qw]``, unnormalised exactly as Eigen leaves it.
    """
    quaternion: Float64[ndarray, " 4"] = np.zeros(4, dtype=np.float64)
    trace: float = float(rotation.trace())
    if trace > 0.0:
        root: float = float(np.sqrt(trace + 1.0))
        quaternion[3] = 0.5 * root
        root = 0.5 / root
        quaternion[0] = (rotation[2, 1] - rotation[1, 2]) * root
        quaternion[1] = (rotation[0, 2] - rotation[2, 0]) * root
        quaternion[2] = (rotation[1, 0] - rotation[0, 1]) * root
        return quaternion
    i: int = 0
    if rotation[1, 1] > rotation[0, 0]:
        i = 1
    if rotation[2, 2] > rotation[i, i]:
        i = 2
    j: int = (i + 1) % 3
    k: int = (j + 1) % 3
    root = float(np.sqrt(rotation[i, i] - rotation[j, j] - rotation[k, k] + 1.0))
    quaternion[i] = 0.5 * root
    root = 0.5 / root
    quaternion[3] = (rotation[k, j] - rotation[j, k]) * root
    quaternion[j] = (rotation[j, i] + rotation[i, j]) * root
    quaternion[k] = (rotation[k, i] + rotation[i, k]) * root
    return quaternion


def catalog_calibration(feed: SegmentFeed, fork_file: dict[str, Any]) -> dict[str, Any]:
    """A basalt calibration file holding the catalog's own geometry.

    The intrinsics, resolutions and extrinsics are the values the C++ reference
    was handed through the VIT C API; everything else — the noise model, the
    update rate, the time offset — is not on the recording and stays as the fork
    file has it.

    Args:
        feed: Open segment feed, whose cameras carry the catalog geometry.
        fork_file: The fork's ``msd*_calib.json``, already unwrapped from ``value0``.

    Returns:
        The ``value0`` body of a basalt calibration file.
    """
    calibration: dict[str, Any] = dict(fork_file)
    poses: list[dict[str, float]] = []
    intrinsics: list[dict[str, Any]] = []
    resolution: list[list[int]] = []
    camera: CameraCalib
    for camera in feed.cameras:
        quaternion: Float64[ndarray, " 4"] = eigen_quaternion_xyzw(camera.imu_T_cam[:3, :3])
        poses.append(
            {
                "px": float(camera.imu_T_cam[0, 3]),
                "py": float(camera.imu_T_cam[1, 3]),
                "pz": float(camera.imu_T_cam[2, 3]),
                "qx": float(quaternion[0]),
                "qy": float(quaternion[1]),
                "qz": float(quaternion[2]),
                "qw": float(quaternion[3]),
            }
        )
        terms: dict[str, float] = {"fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy}
        if camera.model == "kb4":
            terms |= dict(zip(["k1", "k2", "k3", "k4"], camera.distortion.tolist(), strict=True))
            intrinsics.append({"camera_type": "kb4", "intrinsics": terms})
        else:
            terms |= dict(zip(["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"], camera.distortion.tolist(), strict=True))
            if camera.distortion_valid_radius is None:
                raise ValueError(f"cam_{camera.index:02d} is radtan8 without a valid radius; basalt needs rpmax")
            terms["rpmax"] = camera.distortion_valid_radius
            intrinsics.append({"camera_type": "pinhole-radtan8", "intrinsics": terms})
        resolution.append([camera.width, camera.height])
    calibration["T_imu_cam"] = poses
    calibration["intrinsics"] = intrinsics
    calibration["resolution"] = resolution
    return calibration


def write_pgm(path: Path, image: UInt8[ndarray, "h w"]) -> None:
    """Write one grayscale image in the binary PGM the Rust lanes read.

    Args:
        path: Destination file.
        image: C-contiguous 8-bit raster.
    """
    header: bytes = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    path.write_bytes(header + image.tobytes())


def write_oracle_inputs(output: Path, frame_t_ns: list[int], imu_lines: list[str]) -> None:
    """Write the two files `basalt_vio_oracle` reads beside the PGMs.

    The tool takes the frameset clock as one integer per line and the inertial
    window as JSON. Both are the values already written to ``clip.json`` and
    ``imu.csv``, in the shape ``tools/vio_oracle.cpp`` parses.

    Args:
        output: Clip directory.
        frame_t_ns: Frameset timestamps, in order.
        imu_lines: The data rows of ``imu.csv``, without the header.
    """
    (output / "timestamps.txt").write_text("".join(f"{t_ns}\n" for t_ns in frame_t_ns))
    samples: list[str] = []
    for line in imu_lines:
        fields: list[str] = line.split(",")
        samples.append(f'    {{"t_ns": {fields[0]}, "gyro": [{", ".join(fields[1:4])}], "accel": [{", ".join(fields[4:7])}]}}')
    (output / "imu.json").write_text('{"imu": [\n' + ",\n".join(samples) + "\n]}\n")


def main(config: Config) -> None:
    """Dump the segment named by the config.

    Args:
        config: Parsed CLI options.

    Raises:
        ValueError: If ``--npz`` was asked for with no frameset to bundle, if the
            segment is not in the manifest, or if its dataset has no fork
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
    if segment.dataset_name not in DEVICE_CALIBRATION:
        raise ValueError(f"{segment.dataset_name} has no fork calibration file")
    fork_file: dict[str, Any] = json.loads((FIXTURES / DEVICE_CALIBRATION[segment.dataset_name]).read_text())["value0"]

    config.output.mkdir(parents=True, exist_ok=True)
    bundle_images: list[UInt8[ndarray, "n_cameras h w"]] = []
    bundle_imu_counts: list[int] = []
    bundle_imu_t: list[Int64[ndarray, " n_samples"]] = []
    bundle_imu_gyro: list[Float64[ndarray, "n_samples 3"]] = []
    bundle_imu_accel: list[Float64[ndarray, "n_samples 3"]] = []
    imu_lines: list[str] = ["#t_ns,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z"]
    frame_lines: list[str] = ["# t_ns,cam_index,sha256 of the gray8 HxW pixels pushed to basalt"]
    frame_t_ns: list[int] = []
    dumped: int = 0

    with open_segment(
        LocalSegment(base_rrd=segment.base_path, gt_rrd=None),
        segment.imu,
        window_s=config.window_s,
    ) as feed:
        calibration: dict[str, Any] = catalog_calibration(feed, fork_file) if config.calibration == "catalog" else fork_file
        (config.output / "calib.json").write_text(json.dumps({"value0": calibration}, indent=2) + "\n")
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
    write_oracle_inputs(config.output, frame_t_ns, imu_lines[1:])
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


if __name__ == "__main__":
    main(tyro.cli(Config))
