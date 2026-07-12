"""Behavior checks for the resumable overnight pipeline."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from arkitscenes_download.pipeline import (
    Config,
    Destination,
    build_manifest,
    compute_state_counts,
    compute_status,
    requeue_failed,
    run_pipeline,
    ship_verify_and_cleanup,
    wait_for_disk_space,
)


class PipelineTest(unittest.TestCase):
    """Check pipeline planning and safety behavior."""

    def test_manifest_preserves_split_stable_order_and_excludes_terminal_ids(self) -> None:
        """Manifest orders Training before Validation and resumes around terminal ids."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            metadata_path: Path = root / "metadata.csv"
            metadata_path.write_text("video_id,fold\n22,Validation\n11,Training\n21,Validation\n12,Training\n13,Training\n")

            chunks: list[list[str]] = build_manifest(
                metadata_path=metadata_path,
                done_ids={"12"},
                failed_ids={"22"},
                chunk_size=2,
                max_chunks=None,
                video_ids=None,
            )

            self.assertEqual(chunks, [["11", "13"], ["21"]])

    def test_manifest_override_still_uses_metadata_order_and_max_chunks(self) -> None:
        """An override selects IDs without replacing deterministic metadata ordering."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path: Path = Path(temporary_directory) / "metadata.csv"
            metadata_path.write_text("video_id,fold\n21,Validation\n11,Training\n12,Training\n")

            chunks: list[list[str]] = build_manifest(metadata_path, set(), set(), 1, 2, ["21", "12", "11"])

            self.assertEqual(chunks, [["11"], ["12"]])

    def test_disk_guard_waits_until_free_space_reaches_threshold(self) -> None:
        """Downloads wait at the threshold gate and resume after space is freed."""
        disk_usage: Mock = Mock(side_effect=[SimpleNamespace(free=9), SimpleNamespace(free=10)])
        sleep: Mock = Mock()
        notices: list[str] = []

        wait_for_disk_space(Path("data"), min_free_bytes=10, disk_usage=disk_usage, sleep=sleep, write=notices.append)

        sleep.assert_called_once_with(30.0)
        self.assertEqual(len(notices), 1)

    def test_destination_parses_local_and_ssh_forms(self) -> None:
        """Only user-at-host absolute targets select SSH transport."""
        local: Destination = Destination.parse("data/published")
        local_with_colon: Destination = Destination.parse("data/archive:published")
        ssh: Destination = Destination.parse("user@example:/srv/rrd")

        self.assertEqual(local.local_root, Path("data/published"))
        self.assertEqual(local.read_mount, Path("data/published"))
        self.assertEqual(local_with_colon.local_root, Path("data/archive:published"))
        self.assertEqual((ssh.ssh_host, ssh.remote_root), ("user@example", "/srv/rrd"))
        self.assertIsNone(ssh.local_root)
        self.assertIsNone(ssh.read_mount)

    def test_matching_checksums_allow_cleanup(self) -> None:
        """Identical local and NAS trees are cleaned up after a completion callback."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            local_rrd: Path = root / "local-rrd"
            local_raw: Path = root / "local-raw"
            destination_root: Path = root / "published"
            local_rrd.mkdir()
            local_raw.mkdir()
            (local_rrd / "base.rrd").write_text("rrd")
            (local_raw / "raw.bin").write_text("raw")
            completions: list[str] = []

            verified: bool = ship_verify_and_cleanup(
                local_rrd,
                local_raw,
                Destination.parse(str(destination_root)),
                before_cleanup=lambda: completions.append("done"),
            )

            self.assertTrue(verified)
            self.assertEqual(completions, ["done"])
            self.assertFalse(local_rrd.exists())
            self.assertFalse(local_raw.exists())
            self.assertEqual((destination_root / "local-rrd" / "base.rrd").read_text(), "rrd")

    def test_ssh_destination_builds_ship_and_remote_verify_commands(self) -> None:
        """SSH publication uses tar transport and remote sha256 without network access."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source: Path = Path(temporary_directory) / "sequence"
            source.mkdir()
            (source / "base.rrd").write_text("rrd")
            digest: str = __import__("hashlib").sha256(b"rrd").hexdigest()
            replies: list[subprocess.CompletedProcess[str]] = [
                subprocess.CompletedProcess([], 0, "", ""),
                subprocess.CompletedProcess([], 0, f"{digest}  base.rrd\n", ""),
            ]

            def draining_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                """Consume the tar stream like a real ssh child would."""
                stdin = kwargs.get("stdin")
                if stdin is not None:
                    stdin.read()  # type: ignore[union-attr]
                return replies.pop(0)

            runner: Mock = Mock(side_effect=draining_runner)

            verified: bool = Destination.parse("user@example:/srv/rrd").ship_and_verify(source, runner)

            self.assertTrue(verified)
            ship_argv: list[str] = runner.call_args_list[0].args[0]
            self.assertEqual(ship_argv[:2], ["ssh", "user@example"])
            self.assertEqual(ship_argv[2], "mkdir -p /srv/rrd && tar -C /srv/rrd -xf -")
            self.assertEqual(runner.call_args_list[1].args[0], ["ssh", "user@example", "cd /srv/rrd/sequence && sha256sum -- *"])

    def test_hostile_destination_strings_never_select_ssh_transport(self) -> None:
        """Shell metacharacters in a destination cannot reach any shell: they parse as local paths."""
        for hostile in ("user@host$(rm -rf /tmp/x):/srv", "user@host;id:/srv", "user@`whoami`:/srv", "user@-oProxyCommand=evil:/srv"):
            destination: Destination = Destination.parse(hostile)
            self.assertIsNone(destination.ssh_host, hostile)
            self.assertEqual(destination.local_root, Path(hostile), hostile)

    def test_quoted_remote_paths_survive_awkward_directories(self) -> None:
        """Remote roots with spaces are shell-quoted in both ship and verify commands."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source: Path = Path(temporary_directory) / "42"
            source.mkdir()
            (source / "base.rrd").write_text("rrd")

            def denying_runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                """Consume the tar stream, then refuse."""
                stdin = kwargs.get("stdin")
                if stdin is not None:
                    stdin.read()  # type: ignore[union-attr]
                return subprocess.CompletedProcess([], 1, "", "denied")

            runner: Mock = Mock(side_effect=denying_runner)

            Destination.parse("user@example:/srv/my rrd").ship_and_verify(source, runner)

            self.assertIn("mkdir -p '/srv/my rrd'", runner.call_args_list[0].args[0][2])

    def test_manifest_rejects_malformed_video_ids(self) -> None:
        """Non-numeric ids never become path segments or remote command text."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            metadata_path: Path = Path(temporary_directory) / "metadata.csv"
            metadata_path.write_text("video_id,fold\n41125135,Training\nevil;id,Training\n")

            with self.assertRaises(ValueError):
                build_manifest(metadata_path, set(), set(), 1, None, None)

    def test_retry_failed_requeues_and_clears_state(self) -> None:
        """requeue_failed returns the failed ids and removes the state file."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir: Path = Path(temporary_directory)
            (state_dir / "failed.txt").write_text("41125135\tingest failed\n40753679\tdownload failed\n")

            requeued: set[str] = requeue_failed(state_dir)

            self.assertEqual(requeued, {"41125135", "40753679"})
            self.assertFalse((state_dir / "failed.txt").exists())
            self.assertEqual(requeue_failed(state_dir), set())

    def test_ssh_registration_requires_read_mount(self) -> None:
        """An SSH destination without a readable view cannot register files."""
        destination: Destination = Destination.parse("user@example:/srv/rrd")
        mounted: Destination = Destination.parse("user@example:/srv/rrd", Path("/readable/rrd"))

        self.assertIsNone(destination.registration_root)
        self.assertEqual(mounted.registration_root, Path("/readable/rrd"))

    def test_status_counts_terminal_and_remaining_ids(self) -> None:
        """Status combines state files with the metadata population."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            metadata_path: Path = root / "metadata.csv"
            metadata_path.write_text("video_id,fold\n11,Training\n12,Training\n21,Validation\n")
            state_dir: Path = root / "state"
            state_dir.mkdir()
            (state_dir / "done.txt").write_text("11\n")
            (state_dir / "failed.txt").write_text("21\tmissing trajectory\n")
            published: Path = root / "published" / "11"
            published.mkdir(parents=True)
            (published / "base.rrd").write_bytes(b"1234")

            counts = compute_state_counts(metadata_path, state_dir)
            status = compute_status(metadata_path, state_dir, root / "published")

            self.assertEqual(counts[:3], (1, 1, 1))
            self.assertEqual((status.done, status.failed, status.remaining), (1, 1, 1))
            self.assertEqual(status.failed_reasons, {"21": "missing trajectory"})
            self.assertEqual(status.destination_bytes, 4)

    def test_status_without_read_mount_reports_unknown_size(self) -> None:
        """Remote status remains available without walking a destination tree."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            metadata_path: Path = root / "metadata.csv"
            metadata_path.write_text("video_id,fold\n11,Training\n")
            state_dir: Path = root / "state"
            state_dir.mkdir()

            status = compute_status(metadata_path, state_dir, None)

            self.assertIsNone(status.destination_bytes)

    def test_plain_pipeline_uses_state_only_status_path(self) -> None:
        """A no-work plain run never asks for recursive destination size."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            root: Path = Path(temporary_directory)
            data_dir: Path = root / "data"
            raw_dir: Path = data_dir / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "metadata.csv").write_text("video_id,fold\n11,Training\n")
            config = Config(
                data_dir=data_dir,
                destination=str(root / "published"),
                state_dir=root / "state",
                max_chunks=0,
                register=False,
            )

            with patch("arkitscenes_download.pipeline.directory_size") as size:
                run_pipeline(config)

            size.assert_not_called()


if __name__ == "__main__":
    unittest.main()
