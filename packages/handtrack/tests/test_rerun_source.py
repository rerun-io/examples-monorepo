"""``rerun_producer`` fails a pass that yields no samples instead of starting the next one forever."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("rerun.experimental.dataloader", reason="rerun.experimental.dataloader is required by this test module")

from handtrack.data import rerun_source  # noqa: E402


def test_an_endless_producer_raises_on_the_first_pass_that_yields_no_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    first = object()
    passes = iter([[first], []])
    seeds: list[int] = []

    def one_pass(dataset: rerun_source.LabelledFrames, **_loader_options: object) -> list[object]:
        seeds.append(dataset.epoch)
        return next(passes)

    monkeypatch.setattr(rerun_source, "DataLoader", one_pass)
    monkeypatch.setattr(torch.multiprocessing, "set_sharing_strategy", lambda _strategy: None)
    produce = rerun_source.rerun_producer(rerun_source.source_spec("gsi-six-view", 0), ["seg-a", "seg-b"], catalog_url="rerun+http://unused",
                                          curation_path=Path("unused.parquet"), workers=1, device=torch.device("cpu"), endless=True)
    stream = iter(produce(3))
    assert next(stream) is first
    with pytest.raises(RuntimeError, match="gsi-six-view: a pass over 2 segments yielded no samples"):
        next(stream)
    assert seeds == [3 * rerun_source.PASS_STRIDE, 3 * rerun_source.PASS_STRIDE + 1]
