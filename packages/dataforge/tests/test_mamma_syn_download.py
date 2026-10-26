"""MammaSyn download: the NAS copy against a pinned Hub listing, and the fetch proof through a fake Hub transport."""

from pathlib import Path

import pytest

from dataforge import transports
from dataforge.datasets.mamma_syn_download import fetch_proof, verify_syn
from dataforge.transports import HfFileInfo

RENDER: str = "MammaSyn/MammaSyn-Hands/interhand_2-6_NC_200_00/be_test_seq_000001"
BEDLAM: str = "MammaSyn/MammaSyn-Singles/BEDLAM_MASKS_WD/20221010_3-10_500_batch01hand_zoom_suburb_d"


def listed(path: str, payload: bytes) -> HfFileInfo:
    return HfFileInfo(path.removeprefix("MammaSyn/"), len(payload), "0" * 64, "0" * 40)


def write(root: Path, path: str, payload: bytes) -> HfFileInfo:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_bytes(payload)
    return listed(path, payload)


def test_verify_compares_every_listed_shard_of_the_family(tmp_path: Path) -> None:
    listing: list[HfFileInfo] = [
        write(tmp_path, f"{RENDER}/000000.tar", b"shard zero"),
        write(tmp_path, f"{RENDER}/metadata.json", b"{}"),
        write(tmp_path, f"{RENDER}/000001.tar", b"truncated"),
        listed(f"{RENDER}/000002.tar", b"never downloaded"),
        listed(f"{BEDLAM}/000000.tar", b"the other family"),
        listed("MammaSyn-Hands/interhand_2-6_NC_200_00/train_data.txt", b"a training list, not read"),
    ]
    (tmp_path / f"{RENDER}/000001.tar").write_bytes(b"trunc")
    files, problems = verify_syn(tmp_path, "syn", listing)
    assert files == 4
    assert problems == [f"{tmp_path / RENDER}/000001.tar: 5 bytes, the Hub lists 9", f"{tmp_path / RENDER}/000002.tar: missing"]


def test_the_fetch_proof_compares_every_fetched_file_with_the_release_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root: Path = tmp_path / "nas"
    write(root, f"{RENDER}/000000.tar", b"shard zero")
    write(root, f"{RENDER}/metadata.json", b"{}")
    served: dict[str, bytes] = {f"{RENDER.removeprefix('MammaSyn/')}/000000.tar": b"shard zero", f"{RENDER.removeprefix('MammaSyn/')}/metadata.json": b"{ }"}
    fetched: list[str] = []

    def fake_fetch(repo_id: str, paths: list[str], *, local_dir: Path, revision: str, workers: int = 16, repo_type: str = "dataset") -> Path:
        assert (repo_id, revision, repo_type) == ("Intelligent-Systems/MammaSyn", "01ecdc887819a91e992b58fdb2c1684335543558", "dataset")
        for path in paths:
            fetched.append(path)
            (local_dir / path).parent.mkdir(parents=True, exist_ok=True)
            (local_dir / path).write_bytes(served[path])
        return local_dir

    monkeypatch.setattr(transports, "hf_fetch_files", fake_fetch)
    mismatches: list[str] = fetch_proof(root, root / RENDER, ["000000.tar"], tmp_path / "proof")
    assert sorted(fetched) == sorted(served)
    assert mismatches == [f"{RENDER.removeprefix('MammaSyn/')}/metadata.json: differs from the release copy"]
