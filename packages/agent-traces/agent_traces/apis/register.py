"""Register converted session recordings in a Rerun catalog."""

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse
from uuid import uuid4

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle

from agent_traces.blueprint import catalog_blueprint
from agent_traces.catalog_layout import save_table_blueprint
from agent_traces.manifest import Manifest, load_manifest
from agent_traces.writing import atomic_write


@dataclass(frozen=True, slots=True)
class Config:
    """Catalog registration arguments."""

    catalog_url: str
    """Required URL of the destination catalog."""
    out: tuple[Path, ...]
    """One or more convert-all output roots, in priority order."""
    profile: str | None = None
    """Register only this profile; by default discover all manifests."""
    dataset: str | None = None
    """Override the default agent-traces-<profile> dataset name."""
    replace: bool = False
    """Replace registered layers instead of skipping duplicates.

    A rebuilt rrd is a new file at a registered path. SKIP leaves the server
    serving the old registration, so use --replace after rebuilding recordings.
    """

    def __post_init__(self) -> None:
        """Require at least one root for registration and blueprint placement."""
        if not self.out:
            raise ValueError("At least one output root is required")


@dataclass(frozen=True, slots=True)
class Summary:
    """Catalog registration counts across all selected profiles."""

    registered: int
    """Successful new or replaced registrations."""
    skipped_duplicates: int
    """Already registered segments or later copies across output roots."""
    errors: int
    """Failed registrations."""
    missing: int
    """Manifest recordings missing from disk."""

    @property
    def exit_code(self) -> int:
        """Fail the process for missing files or failed registrations."""
        return int(self.errors > 0 or self.missing > 0)


def owned_blueprint_files(storage_urls: list[list[str]], profile: Path) -> list[Path]:
    """Select retired local stamped blueprints in matching profile directories across output roots."""
    owned: list[Path] = []
    for urls in storage_urls:
        for uri in urls:
            parsed = urlparse(uri)
            if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
                continue
            path: Path = Path(unquote(parsed.path)).resolve()
            if path.parent.name == profile.name and re.fullmatch(r"agent-traces-(?:table-)?[0-9a-fA-F]{32}\.rbl", path.name):
                owned.append(path)
    return owned


def profile_sessions(paths: list[Path]) -> tuple[dict[str, str], int, int]:
    """Gather a profile's first session copies and count duplicates and missing files."""
    sessions_by_uri: dict[str, str] = {}
    seen: set[str] = set()
    skipped_duplicates: int = 0
    missing: int = 0
    for directory in paths:
        manifest: Manifest = load_manifest(directory / "manifest.json")
        for session_id, session in manifest.sessions.items():
            if session_id in seen:
                skipped_duplicates += 1
                continue
            seen.add(session_id)
            path: Path = directory / session.rrd
            if not path.is_file():
                print(f"MISSING {path}")
                missing += 1
                continue
            sessions_by_uri[path.resolve().as_uri()] = session_id
    return sessions_by_uri, skipped_duplicates, missing


def main(config: Config) -> Summary:
    """Register one batch per profile and verify catalog visibility.

    Args:
        config: Catalog destination, converted output roots, and selection.
    """
    profiles: dict[str, list[Path]] = {}
    for root in config.out:
        out: Path = root.expanduser()
        candidates: list[Path] = [out / config.profile] if config.profile is not None else sorted(out.iterdir())
        for profile in candidates:
            if profile.is_dir() and (profile / "manifest.json").is_file():
                profiles.setdefault(profile.name, []).append(profile)
    if config.profile is not None and not profiles:
        raise ValueError(f"Missing manifest: {config.out[0].expanduser() / config.profile / 'manifest.json'}; run convert-all first")
    total: Summary = Summary(0, 0, 0, 0)
    for profile_paths in profiles.values():
        profile: Path = profile_paths[0]
        sessions_by_uri: dict[str, str]
        skipped_duplicates: int
        missing: int
        sessions_by_uri, skipped_duplicates, missing = profile_sessions(profile_paths)
        name: str = config.dataset if config.dataset is not None else f"agent-traces-{profile.name}"
        client: CatalogClient = CatalogClient(config.catalog_url)
        entry: DatasetEntry = client.get_dataset(name) if name in client.dataset_names() else client.create_dataset(name)
        existing: set[str] = set(entry.segment_ids())
        registered: int = 0
        errors: int = 0
        failed_uris: set[str] = set()
        policy: OnDuplicateSegmentLayer = OnDuplicateSegmentLayer.REPLACE if config.replace else OnDuplicateSegmentLayer.SKIP
        if sessions_by_uri:
            handle: RegistrationHandle = entry.register(list(sessions_by_uri), layer_name="base", on_duplicate=policy)
            wait_error: ValueError | None = None
            try:
                handle.wait()
            except ValueError as error:
                wait_error = error
            for result in handle.iter_results():
                if result.error is not None:
                    errors += 1
                    failed_uris.add(result.uri)
                    print(f"ERROR {result.uri}: {result.error}")
                elif not config.replace and result.segment_id in existing:
                    skipped_duplicates += 1
                else:
                    registered += 1
            if wait_error is not None and errors == 0:
                raise wait_error
        # Add fresh defaults before retiring old registrations; live files are never overwritten.
        previous = entry.blueprint_dataset()
        retiring: list[str] = []
        retired_files: list[Path] = []
        if previous is not None:
            table = previous.segment_table().select("rerun_segment_id", "rerun_storage_urls").to_arrow_table()
            retiring = table["rerun_segment_id"].to_pylist()
            retired_files = owned_blueprint_files(table["rerun_storage_urls"].to_pylist(), profile)
        stamp: str = uuid4().hex
        recording_path: Path = (profile / f"agent-traces-{stamp}.rbl").resolve()
        with atomic_write(recording_path) as temporary:
            catalog_blueprint().save("agent_traces", temporary)
        table_path: Path = (profile / f"agent-traces-table-{stamp}.rbl").resolve()
        save_table_blueprint(table_path, entry.segment_table().schema().names)
        entry.register_blueprint(recording_path.as_uri(), set_default=True)
        entry.register_blueprint(table_path.as_uri(), set_default=True, segment_table=True)
        if previous is not None and retiring:
            previous.unregister(segments_to_drop=retiring, layers_to_drop=[]).wait()
            for path in retired_files:
                path.unlink(missing_ok=True)
        segment_ids: set[str] = set(entry.segment_ids())
        print(
            f"registered={registered} skipped_duplicates={skipped_duplicates} errors={errors} missing={missing} "
            f"dataset={name} segments={len(segment_ids)} url={config.catalog_url}"
        )
        for uri, session_id in sessions_by_uri.items():
            if uri not in failed_uris and session_id not in segment_ids:
                raise RuntimeError(f"Session {session_id} is absent from dataset {name} after registration")
        total = Summary(total.registered + registered, total.skipped_duplicates + skipped_duplicates, total.errors + errors, total.missing + missing)
    return total
