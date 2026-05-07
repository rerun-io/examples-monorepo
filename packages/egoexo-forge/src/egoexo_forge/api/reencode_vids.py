from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

from icecream import ic
from natsort import natsorted
from simplecv.video_utils import reencode_video
from tqdm import tqdm


@dataclass
class EncodeConfig:
    input_dir: Path
    output_dir: Path
    num_sequences_to_process: int | None = None  # None means process all sequences


def reencode_videos(config: EncodeConfig):
    start_time: float = timer()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Re-encoding videos from {config.input_dir} to {config.output_dir}")
    sequence_paths: list[Path] = natsorted([d for d in config.input_dir.glob("*") if d.is_dir()])

    # Limit the number of sequences if specified
    if config.num_sequences_to_process is not None:
        sequence_paths = sequence_paths[: config.num_sequences_to_process]
        print(f"Processing only the first {config.num_sequences_to_process} sequences")

    # should be a total of 362 sequences, more info here https://github.com/assembly-101/assembly101-download-scripts
    ic(len(sequence_paths), "sequences found")
    bad_sequences: list[Path] = []
    total_original_size = 0
    total_reencoded_size = 0
    videos_processed = 0
    try:
        for sequence_path in tqdm(sequence_paths, desc="Sequences"):
            exo_videos: list[Path] = natsorted(sequence_path.glob("C*.mp4"))
            ego_videos: list[Path] = natsorted(sequence_path.glob("HMC_*.mp4"))
            if len(exo_videos) != 8 or len(ego_videos) != 4:
                bad_sequences.append(sequence_path)
                continue

            all_videos = exo_videos + ego_videos
            output_sequence_dir = config.output_dir / sequence_path.name

            for video_path in tqdm(all_videos, desc=f"Videos in {sequence_path.name}", leave=False):
                output_video_path = output_sequence_dir / video_path.name
                if output_video_path.exists():
                    continue

                # Create the subdirectory for the sequence if it doesn't exist.
                output_video_path.parent.mkdir(parents=True, exist_ok=True)
                # check if the video already exists
                new_video_path: Path = output_sequence_dir / f"{video_path.stem}_optimal{video_path.suffix}"
                if new_video_path.exists():
                    print(f"Video {video_path} already exists, skipping.")
                    continue

                # Get original file size
                original_size = video_path.stat().st_size

                resize = "720p" if video_path.name.startswith("C") else None
                reencode_video(
                    input_video_path=video_path,
                    resize=resize,
                    quality="low",
                    save_file=True,
                    output_directory=output_sequence_dir,
                )

                # Check if the new video was created and get its size
                if new_video_path.exists():
                    new_size = new_video_path.stat().st_size
                    size_reduction_mb = (original_size - new_size) / (1024 * 1024)
                    size_reduction_percent = ((original_size - new_size) / original_size) * 100

                    print(
                        f"Video {video_path.name}: {original_size / (1024 * 1024):.1f}MB -> {new_size / (1024 * 1024):.1f}MB "
                        f"(reduced by {size_reduction_mb:.1f}MB, {size_reduction_percent:.1f}%)"
                    )

                    total_original_size += original_size
                    total_reencoded_size += new_size
                    videos_processed += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user. The process will stop after the current sequence completes.")

    ic(bad_sequences)
    ic(len(bad_sequences), "bad sequences found")

    # Print size comparison summary
    if videos_processed > 0:
        total_reduction_mb = (total_original_size - total_reencoded_size) / (1024 * 1024)
        total_reduction_percent = ((total_original_size - total_reencoded_size) / total_original_size) * 100
        print("\nSize reduction summary:")
        print(f"Videos processed: {videos_processed}")
        print(f"Total original size: {total_original_size / (1024 * 1024):.1f}MB")
        print(f"Total reencoded size: {total_reencoded_size / (1024 * 1024):.1f}MB")
        print(f"Total reduction: {total_reduction_mb:.1f}MB ({total_reduction_percent:.1f}%)")

    end_time: float = timer()
    # For now, we just simulate the process with a print statement
    print(f"Re-encoded videos in {end_time - start_time:.2f} seconds.")
