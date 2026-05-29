"""Cut synced ExoEgo sequences into individual episodes based on episode_info.json bounds."""

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from serde import serde
from serde.json import from_json
from tqdm.auto import tqdm


@serde
class SessionBounds:
    """Time bounds for the entire session."""

    start_time_s: int | float
    """Start time in seconds."""
    end_time_s: int | float
    """End time in seconds."""


@serde
class Episode:
    """A single episode within a session."""

    episode_number: int
    """1-indexed episode number."""
    start_time_s: int | float
    """Start time in seconds."""
    end_time_s: int | float
    """End time in seconds."""


@serde
class EpisodeInfo:
    """Episode information for a synced sequence."""

    version: str
    """Schema version."""
    session_name: str
    """UUID of the session."""
    created_by: str
    """Email of the creator."""
    created_at: datetime
    """ISO 8601 timestamp of creation."""
    video_source: str
    """Relative path to the source video used for cutting."""
    session_bounds: SessionBounds
    """Overall session time bounds."""
    episodes: list[Episode]
    """List of episodes in the session."""


@dataclass
class Config:
    """Configuration for cutting synced sequences."""

    synced_dir: Path = Path(
        "/mnt/8tb/data/exoego-self-collected/quest+oak+exo/qwen-examples/cut/a8e0434b-bac5-4ffa-a5b2-5e679a0167bb/synced"
    )
    """Path to the synced directory containing ego/exo/quest subdirectories."""

    episode_info_path: Path = Path(
        "/mnt/8tb/data/exoego-self-collected/quest+oak+exo/qwen-examples/cut/a8e0434b-bac5-4ffa-a5b2-5e679a0167bb/episode_info.json"
    )
    """Path to the episode_info.json file."""

    output_dir: Path = Path("/mnt/8tb/data/exoego-self-collected/quest+oak+exo/qwen-examples/cut")
    """Local directory to write cut episodes."""

    episodes: list[int] | None = None
    """Specific episode numbers to cut (1-indexed). If None, cut all episodes."""

    dry_run: bool = False
    """If True, print what would be done without actually cutting."""

    parallel_workers: int = 4
    """Number of parallel FFmpeg processes for video encoding. NVENC supports 3-4 concurrent sessions."""



def cut_video_ffmpeg_nvenc(
    input_path: Path,
    output_path: Path,
    start_s: int | float,
    end_s: int | float,
) -> None:
    """Cut video using FFmpeg with NVENC for GPU-accelerated AV1 re-encoding.

    Args:
        input_path: Path to the input video file.
        output_path: Path to write the cut video to.
        start_s: Start time in seconds.
        end_s: End time in seconds.

    Note:
        Uses NVIDIA NVENC for fast AV1 encoding. Requires NVIDIA GPU with AV1 support
        (RTX 40 series or newer).
        
        Frame-accurate cutting is achieved by:
        1. Using -ss AFTER -i (decode from start, precise frame seeking)
        2. Full re-encoding with av1_nvenc (not stream copy)
    """
    import subprocess

    # -ss after -i = frame-accurate seeking (decodes from start)
    # -t for duration instead of -to for precise frame count
    duration: float = end_s - start_s
    cmd: list[str] = [
        "ffmpeg",
        "-hwaccel", "cuda",  # Use CUDA for hardware decoding
        "-i", str(input_path),
        "-ss", str(start_s),  # Seek AFTER input = frame-accurate
        "-t", str(duration),  # Duration instead of end time
        "-c:v", "av1_nvenc",  # NVENC AV1 encoder (full re-encode = frame accurate)
        "-c:a", "copy",  # Copy audio without re-encoding
        "-y",  # Overwrite output
        str(output_path),
    ]
    result: subprocess.CompletedProcess[str] = subprocess.run(
        cmd, check=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")


def cut_csv_by_timestamp(
    input_path: Path,
    output_path: Path,
    start_s: int | float,
    end_s: int | float,
) -> None:
    """Filter CSV rows by timestamp range and normalize to start at 0.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path to write the filtered CSV to.
        start_s: Start time in seconds.
        end_s: End time in seconds.
    """
    df: pd.DataFrame = pd.read_csv(input_path)

    # Convert bounds to nanoseconds
    start_ns: int = int(start_s * 1e9)
    end_ns: int = int(end_s * 1e9)

    # Check if ts_ns column exists
    if "ts_ns" not in df.columns:
        raise ValueError(f"CSV file {input_path} does not have 'ts_ns' column")

    # Handle ego CSVs with wrap issue - detect wrap point where timestamp decreases
    ts_ns_values: np.ndarray = df["ts_ns"].to_numpy()
    diffs: np.ndarray = np.diff(ts_ns_values, prepend=ts_ns_values[0])

    # Find indices where timestamp wraps (decreases significantly)
    wrap_indices: list[int] = [int(i) for i in np.where(diffs < -1e9)[0]]
    if wrap_indices:
        # Only use data before wrap
        first_wrap_idx: int = wrap_indices[0]
        df = df.iloc[:first_wrap_idx].copy()

    # Filter by timestamp range
    mask = (df["ts_ns"] >= start_ns) & (df["ts_ns"] <= end_ns)
    df_cut: pd.DataFrame = df[mask].copy()

    # Normalize timestamps to start at 0
    if len(df_cut) > 0:
        first_ts: int = int(df_cut["ts_ns"].iloc[0])
        df_cut["ts_ns"] = df_cut["ts_ns"] - first_ts

    df_cut.to_csv(output_path, index=False)


def cut_episode(
    synced_dir: Path,
    output_dir: Path,
    session_id: str,
    episode: Episode,
    dry_run: bool = False,
    parallel_workers: int = 4,
) -> Path:
    """Cut all files for a single episode.

    Args:
        synced_dir: Path to the source synced directory.
        output_dir: Base output directory for cut episodes.
        session_id: UUID of the session (used for directory structure).
        episode: Episode metadata with time bounds.
        dry_run: If True, print what would be done without actually cutting.
        parallel_workers: Number of parallel FFmpeg processes for video encoding.

    Returns:
        Path to the created episode directory.
    """
    start_s: int | float = episode.start_time_s
    end_s: int | float = episode.end_time_s
    ep_name: str = f"episode-{episode.episode_number:03d}"

    ep_output: Path = output_dir / session_id / "episodes" / ep_name

    if dry_run:
        print(f"[DRY RUN] Would create: {ep_output}")
        print(f"[DRY RUN] Episode {episode.episode_number}: {start_s:.2f}s - {end_s:.2f}s ({end_s - start_s:.2f}s)")
        return output_dir / session_id / "episodes" / ep_name

    ep_output.mkdir(parents=True, exist_ok=True)

    # Collect all video cut tasks: (input_path, output_path, start_s, end_s)
    video_tasks: list[tuple[Path, Path, int | float, int | float]] = []
    exo_cameras: list[Path] = []  # Track for calibration copy later

    # === Setup directories and collect video tasks ===

    # Ego
    ego_in: Path = synced_dir / "ego"
    ego_out: Path = ep_output / "ego"
    if ego_in.exists():
        ego_out.mkdir(exist_ok=True)
        for video in ["left.mp4", "right.mp4", "rgb.mp4"]:
            video_in: Path = ego_in / video
            video_out: Path = ego_out / video
            if video_in.exists():
                video_tasks.append((video_in, video_out, start_s, end_s))

    # Exo
    exo_in: Path = synced_dir / "exo"
    exo_out: Path = ep_output / "exo"
    if exo_in.exists():
        exo_out.mkdir(exist_ok=True)
        exo_cameras = [d for d in exo_in.iterdir() if d.is_dir()]
        for cam_dir in exo_cameras:
            cam_name: str = cam_dir.name
            cam_out: Path = exo_out / cam_name
            cam_out.mkdir(exist_ok=True)
            video_files: list[Path] = list(cam_dir.glob("*.mp4"))
            for video_in in video_files:
                video_out: Path = cam_out / f"{cam_name}.mp4"
                video_tasks.append((video_in, video_out, start_s, end_s))

    # Quest
    quest_in: Path = synced_dir / "quest"
    quest_out: Path = ep_output / "quest"
    if quest_in.exists():
        quest_out.mkdir(exist_ok=True)
        for video in ["left.mp4", "right.mp4"]:
            video_in: Path = quest_in / video
            video_out: Path = quest_out / video
            if video_in.exists():
                video_tasks.append((video_in, video_out, start_s, end_s))

    # === Execute video cuts in parallel ===
    def _cut_video_task(task: tuple[Path, Path, int | float, int | float]) -> str:
        """Worker function for parallel video cutting."""
        input_path, output_path, start, end = task
        cut_video_ffmpeg_nvenc(input_path, output_path, start, end)
        return output_path.name

    if video_tasks:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {executor.submit(_cut_video_task, task): task for task in video_tasks}
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"  Cutting videos ({ep_name})",
                leave=False,
            ):
                future.result()  # Raise any exceptions

    # === Sequential operations (fast) ===

    # Cut ego CSVs and copy calibration
    if ego_in.exists():
        ego_csvs: list[str] = ["left.csv", "right.csv", "rgb.csv", "imu.csv"]
        for csv in ego_csvs:
            csv_in: Path = ego_in / csv
            csv_out: Path = ego_out / csv
            if csv_in.exists():
                cut_csv_by_timestamp(csv_in, csv_out, start_s, end_s)

        for cal_file in ["calibration.json", "recording_stats.json"]:
            cal_in: Path = ego_in / cal_file
            cal_out: Path = ego_out / cal_file
            if cal_in.exists():
                shutil.copy(cal_in, cal_out)

    # Copy exo calibration files
    if exo_in.exists():
        for cam_dir in exo_cameras:
            cam_name = cam_dir.name
            cam_out = exo_out / cam_name
            cal_in: Path = cam_dir / "calibration.json"
            cal_out: Path = cam_out / "calibration.json"
            if cal_in.exists():
                shutil.copy(cal_in, cal_out)

    # Cut quest CSVs and copy calibration
    if quest_in.exists():
        quest_csvs: list[str] = [
            "head_pose.csv",
            "body_poses.csv",
            "left_hand_poses.csv",
            "right_hand_poses.csv",
        ]
        for csv in quest_csvs:
            csv_in: Path = quest_in / csv
            csv_out: Path = quest_out / csv
            if csv_in.exists():
                cut_csv_by_timestamp(csv_in, csv_out, start_s, end_s)

        cal_in: Path = quest_in / "calibration.json"
        cal_out: Path = quest_out / "calibration.json"
        if cal_in.exists():
            shutil.copy(cal_in, cal_out)

    return output_dir / session_id / "episodes" / ep_name


def main(config: Config) -> None:
    """Main entry point for cutting synced sequences into episodes.

    Args:
        config: Configuration with paths and episode selection.
    """
    # Validate paths
    if not config.synced_dir.exists():
        raise FileNotFoundError(f"Synced directory not found: {config.synced_dir}")

    if not config.episode_info_path.exists():
        raise FileNotFoundError(f"Episode info not found: {config.episode_info_path}")

    # Parse episode info
    episode_info: EpisodeInfo = from_json(EpisodeInfo, config.episode_info_path.read_text())

    print(f"Session: {episode_info.session_name}")
    print(f"Created by: {episode_info.created_by}")
    print(
        f"Session bounds: {episode_info.session_bounds.start_time_s}s - {episode_info.session_bounds.end_time_s}s"
    )
    print(f"Total episodes: {len(episode_info.episodes)}")

    # Filter episodes if specified
    episodes_to_cut: list[Episode] = episode_info.episodes
    if config.episodes is not None:
        episodes_to_cut = [ep for ep in episode_info.episodes if ep.episode_number in config.episodes]
        print(f"Cutting episodes: {config.episodes}")
    else:
        print("Cutting all episodes")

    if not episodes_to_cut:
        print("No episodes to cut!")
        return

    # Cut each episode
    for episode in tqdm(episodes_to_cut, desc="Cutting episodes"):
        duration: float = episode.end_time_s - episode.start_time_s
        print(f"\nEpisode {episode.episode_number}: {episode.start_time_s:.2f}s - {episode.end_time_s:.2f}s ({duration:.2f}s)")

        output_path: Path = cut_episode(
            synced_dir=config.synced_dir,
            output_dir=config.output_dir,
            session_id=episode_info.session_name,
            episode=episode,
            dry_run=config.dry_run,
            parallel_workers=config.parallel_workers,
        )

        print(f"  Output: {output_path}")

    print("\nDone!")
