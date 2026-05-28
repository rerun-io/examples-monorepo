"""Patch RRD metadata with updated data from local metadata.json files.

This tool appends updated metadata to existing RRD files using binary append.
This is the same technique used by mv-api for extrinsics propagation.

The approach works because RRD files are append-only log streams, and using
the same app_id + recording_id ensures data goes to the same recording.

Environment names are anonymized using the built-in mapping.
"""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rerun as rr
import rerun.experimental as rre
import tyro
from serde.json import from_json, to_json
from tqdm.auto import tqdm

from simplecv.apis.ingest_exoego_recording import (
    PARTICIPANT_INFO,
    RawSessionMetadata,
    RecordingMetadata,
)

# Environment mapping (PII name → anonymized ID)
ENVIRONMENT_MAPPING: dict[str, str] = {
    "ADIL_1": "ENV_001",
    "ADIL_2": "ENV_002",
    "AMINA_1": "ENV_003",
    "AMINA_2": "ENV_004",
    "AMINA_3": "ENV_005",
}


@dataclass
class PatchConfig:
    """Configuration for patching RRD metadata."""

    cut_dir: Path
    """Root directory containing all sequences (e.g., /mnt/8tb/.../cut)."""
    dry_run: bool = False
    """If True, show what would be patched without modifying files."""
    sequence_id: str = ""
    """Optional: patch only this sequence ID (for testing)."""


def load_local_metadata(sequence_dir: Path) -> RawSessionMetadata | None:
    """Load metadata.json from local filesystem."""
    metadata_path: Path = sequence_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        return from_json(RawSessionMetadata, metadata_path.read_text())
    except Exception:
        return None


def anonymize_environment(raw_env: str) -> str | None:
    """Convert PII environment name to anonymized ID."""
    if not raw_env:
        return None
    return ENVIRONMENT_MAPPING.get(raw_env)


def create_recording_metadata(raw: RawSessionMetadata) -> RecordingMetadata:
    """Transform raw metadata into RecordingMetadata for logging."""
    collector_name: str = raw.collector_name.lower()
    participant_info: dict[str, str | float | None] = PARTICIPANT_INFO.get(
        collector_name, {}
    )

    sex_value: str | float | None = participant_info.get("sex")
    participant_sex: str | None = sex_value if isinstance(sex_value, str) else None
    height_value: str | float | None = participant_info.get("height_cm")
    participant_height_cm: float | None = (
        height_value if isinstance(height_value, float) else None
    )

    return RecordingMetadata(
        time_collected=raw.time_collected,
        task=raw.task,
        participant_sex=participant_sex,
        participant_height_cm=participant_height_cm,
        environment=anonymize_environment(raw.environment),
    )


def patch_rrd_metadata(rrd_path: Path, metadata: RecordingMetadata) -> bool:
    """Append updated metadata to an existing RRD file using binary append.

    This technique creates a small patch RRD with the same app_id/recording_id
    and appends it to the original file. Works because RRD files are append-only.
    """
    try:
        # Load original recording to get app_id and recording_id
        reader: Any = rre.RrdReader(rrd_path)
        recordings: list[Any] = list(reader.recordings())
        if not recordings:
            raise ValueError(f"No recordings found in {rrd_path}")

        store_entry: Any = recordings[0]
        app_id: str = store_entry.application_id
        rec_id: str = store_entry.recording_id

        # Create temporary patch RRD
        with tempfile.NamedTemporaryFile(
            prefix="metadata_patch_", suffix=".rrd", delete=False
        ) as f:
            patch_path: Path = Path(f.name)

        # Create recording stream with same IDs
        rec: rr.RecordingStream = rr.RecordingStream(
            application_id=app_id, recording_id=rec_id
        )
        rec.save(str(patch_path))

        # Log updated metadata
        metadata_json: str = to_json(metadata, indent=2)
        rec.log(
            "/metadata",
            rr.TextDocument(metadata_json, media_type="application/json"),
            static=True,
        )
        del rec  # Flush

        # Binary append patch to original RRD
        with open(rrd_path, "ab") as orig_file, open(patch_path, "rb") as patch_file:
            shutil.copyfileobj(patch_file, orig_file)

        patch_path.unlink()
        return True

    except Exception as e:
        print(f"    Error patching {rrd_path.name}: {e}")
        return False


def patch_sequence(sequence_dir: Path, dry_run: bool = False) -> tuple[int, int]:
    """Patch all episodes in a sequence with metadata from local file."""
    sequence_id: str = sequence_dir.name

    raw_metadata: RawSessionMetadata | None = load_local_metadata(sequence_dir)
    if raw_metadata is None:
        print(f"  No metadata.json for {sequence_id[:8]}")
        return 0, 1

    metadata: RecordingMetadata = create_recording_metadata(raw_metadata)

    if raw_metadata.environment and metadata.environment is None:
        print(f"  Warning: Unknown environment '{raw_metadata.environment}'")

    episodes_dir: Path = sequence_dir / "episodes"
    if not episodes_dir.exists():
        return 0, 0

    rrd_files: list[Path] = sorted(episodes_dir.glob("*/episode-*.rrd"))
    if not rrd_files:
        return 0, 0

    success_count: int = 0
    error_count: int = 0

    for rrd_path in rrd_files:
        if dry_run:
            print(f"    [DRY RUN] Would patch: {rrd_path.name} (env={metadata.environment})")
            success_count += 1
        else:
            if patch_rrd_metadata(rrd_path, metadata):
                success_count += 1
            else:
                error_count += 1

    return success_count, error_count


def discover_sequences(cut_dir: Path) -> list[Path]:
    """Find all sequence directories under cut root."""
    sequences: list[Path] = []
    for date_dir in sorted(cut_dir.iterdir()):
        if not date_dir.is_dir() or date_dir.name.startswith("."):
            continue
        for sequence_dir in sorted(date_dir.iterdir()):
            if not sequence_dir.is_dir():
                continue
            if (sequence_dir / "episodes").exists():
                sequences.append(sequence_dir)
    return sequences


def main(config: PatchConfig) -> None:
    """Main entry point for patching RRD metadata."""
    print(f"Cut directory: {config.cut_dir}")
    print(f"Environment mapping: {len(ENVIRONMENT_MAPPING)} entries")
    if config.dry_run:
        print("[DRY RUN MODE]")

    if config.sequence_id:
        sequences: list[Path] = []
        for date_dir in config.cut_dir.iterdir():
            if date_dir.is_dir():
                seq_dir: Path = date_dir / config.sequence_id
                if seq_dir.exists():
                    sequences.append(seq_dir)
                    break
        if not sequences:
            print(f"Sequence {config.sequence_id} not found")
            return
    else:
        sequences = discover_sequences(config.cut_dir)

    print(f"Found {len(sequences)} sequences")

    total_success: int = 0
    total_errors: int = 0

    for seq_dir in tqdm(sequences, desc="Patching sequences"):
        seq_id: str = seq_dir.name
        print(f"\n{seq_id[:8]}:")

        success, errors = patch_sequence(seq_dir, config.dry_run)
        total_success += success
        total_errors += errors

        if success > 0:
            print(f"  ✓ Patched {success} episodes")
        if errors > 0:
            print(f"  ✗ {errors} errors")

    print(f"\n{'='*50}")
    print(f"Total: {total_success} patched, {total_errors} errors")


if __name__ == "__main__":
    config: PatchConfig = tyro.cli(PatchConfig)
    main(config)
