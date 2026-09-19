"""Register converted session recordings in a Rerun catalog."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rerun.catalog import CatalogClient, DatasetEntry, OnDuplicateSegmentLayer, RegistrationHandle

from agent_traces.blueprint import session_blueprint
from agent_traces.manifest import Manifest, load_manifest


@dataclass(frozen=True, slots=True)
class Config:
    """Catalog registration arguments."""

    catalog_url: str
    """Required URL of the destination catalog."""
    out: Path
    """Convert-all output root containing profile directories."""
    profile: str | None = None
    """Register only this profile; by default discover all manifests."""
    dataset: str | None = None
    """Override the default agent-traces-<profile> dataset name."""
    replace: bool = False
    """Replace registered layers instead of skipping duplicates.

    A rebuilt rrd is a new file at a registered path. SKIP leaves the server
    serving the old registration, so use --replace after rebuilding recordings.
    """


def main(config: Config) -> None:
    """Register one batch per profile and verify catalog visibility.

    Args:
        config: Catalog destination, converted output root, and selection.
    """
    out: Path = config.out.expanduser()
    profiles: list[Path] = (
        [out / config.profile]
        if config.profile is not None
        else sorted(path for path in out.iterdir() if path.is_dir() and (path / "manifest.json").is_file())
    )
    for profile in profiles:
        manifest_path: Path = profile / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing manifest: {manifest_path}; run convert-all first")
        manifest: Manifest = load_manifest(manifest_path)
        sessions_by_uri: dict[str, str] = {}
        missing: int = 0
        for session_id, session in manifest.sessions.items():
            path: Path = profile / session.rrd
            if not path.is_file():
                print(f"MISSING {path}")
                missing += 1
                continue
            sessions_by_uri[path.resolve().as_uri()] = session_id
        name: str = config.dataset if config.dataset is not None else f"agent-traces-{profile.name}"
        client: CatalogClient = CatalogClient(config.catalog_url)
        entry: DatasetEntry = client.create_dataset(name, exist_ok=True)
        existing: set[str] = set(entry.segment_ids())
        registered: int = 0
        skipped_duplicates: int = 0
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
        # Blueprints register once: each call adds a NEW entry to the catalog's
        # blueprint list, cluttering the viewer selector. To refresh, delete the
        # dataset and re-register. Never truncate an .rbl a live server holds open.
        if entry.default_blueprint() is None:
            blueprint_path: Path = profile / "agent-traces.rbl"
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=profile, suffix=".rbl", delete=False) as stream:
                    temporary = Path(stream.name)
                session_blueprint().save("agent_traces", temporary)
                os.replace(temporary, blueprint_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            entry.register_blueprint(blueprint_path.resolve().as_uri(), set_default=True)
        segment_ids: set[str] = set(entry.segment_ids())
        print(
            f"registered={registered} skipped_duplicates={skipped_duplicates} errors={errors} missing={missing} "
            f"dataset={name} segments={len(segment_ids)} url={config.catalog_url}"
        )
        for uri, session_id in sessions_by_uri.items():
            if uri not in failed_uris and session_id not in segment_ids:
                raise RuntimeError(f"Session {session_id} is absent from dataset {name} after registration")
