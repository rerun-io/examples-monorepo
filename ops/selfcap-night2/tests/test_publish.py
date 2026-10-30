import json
from types import SimpleNamespace

import pyarrow as pa
import stages


def test_publication_streams_hash_and_reuses_only_verified_base(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "ROOT", tmp_path)
    stages.root("test").mkdir(parents=True)
    path = tmp_path / "base.rrd"
    path.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    uri = f"file://{stages.REMOTE}/base/test-{digest[:16]}.rrd"
    registrations = []
    commands = []
    registered_uri = [uri]

    def table():
        return pa.table({"rerun_layer_names": [["base"]], "rerun_storage_urls": [[registered_uri[0]]]})

    catalog = SimpleNamespace(
        register=lambda urls, **kwargs: registrations.append(urls) or SimpleNamespace(wait=lambda: None),
        filter_segments=lambda segment: SimpleNamespace(
            segment_table=lambda: SimpleNamespace(select=lambda *args: SimpleNamespace(to_arrow_table=table))
        ),
    )
    monkeypatch.setattr(stages, "CatalogClient", lambda url: SimpleNamespace(get_dataset=lambda **kwargs: catalog))
    monkeypatch.setattr(stages.subprocess, "run", lambda args, **kwargs: commands.append(args))

    def no_read_bytes(self):
        raise AssertionError("publication must stream the file")

    monkeypatch.setattr(type(path), "read_bytes", no_read_bytes)
    stages.publish("test", {"base": path})
    assert len(commands) == 2
    assert registrations == [[uri]]
    assert json.loads((stages.root("test") / "publish-receipts.json").read_text())["base"]["sha256"] == digest
    commands.clear()
    registrations.clear()
    stages.publish("test", {"base": path})
    assert commands == [] and registrations == []
    registered_uri[0] = "file:///changed.rrd"
    stages.publish("test", {"base": path})
    assert len(commands) == 2
    assert registrations == [[uri]]
