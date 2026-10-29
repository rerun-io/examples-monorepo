"""The CLI resolves explicit episode keys and writes only its output directory."""

from pathlib import Path

import pytest
from rrd_fixtures import synthetic_served

from selfcap_curate.apis.patch_base import Config, main
from selfcap_curate.records import write_sync


@pytest.mark.parametrize("sync_name", ["selfcap__{key}.json", "{key}.json"])
def test_patch_cli_reads_a_key_file_and_writes_one_new_base(tmp_path: Path, sync_name: str) -> None:
    served, raw, sync, _ = synthetic_served(tmp_path)
    raw_root = tmp_path / "cut"
    episode = raw_root / sync.date / sync.session / "episodes" / "episode-002"
    episode.parent.mkdir(parents=True)
    raw.rename(episode)
    served_root = tmp_path / "served"
    served_root.mkdir()
    filename = f"selfcap__{sync.key}.rrd"
    served.rename(served_root / filename)
    sync_dir = tmp_path / "sync"
    sync_dir.mkdir()
    write_sync(sync_dir / sync_name.format(key=sync.key), sync)
    key_file = tmp_path / "keys.txt"
    key_file.write_text(f"selfcap__{sync.key}\n")
    output = tmp_path / "outputs"
    main(Config(output_dir=output, key_file=key_file, served_dir=served_root, raw_root=raw_root, sync_dir=sync_dir, sync_name=sync_name))
    assert list(output.iterdir()) == [output / filename]
    assert (served_root / filename).is_file()
