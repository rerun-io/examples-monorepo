import numpy as np
import pytest
import torch
from beartype.roar import BeartypeException

pytest.importorskip("asmk", reason="Retrieval requires ASMK")
pytest.importorskip("mast3r", reason="Retrieval requires MASt3R")

from asmk.asmk_method import ASMKMethod, IvfBuilder  # noqa: E402
from asmk.codebook import Codebook  # noqa: E402
from asmk.kernel import ASMKKernel  # noqa: E402

from mast3r_slam.retrieval_database import RetrievalDatabase  # noqa: E402


@pytest.fixture
def database() -> RetrievalDatabase:
    """Use a real CPU inverted file with synthetic descriptors and no backbone."""
    database = RetrievalDatabase.__new__(RetrievalDatabase)
    database.query_device = "cpu"
    database.query_dtype = torch.float32
    database.centroids = torch.zeros((1, 2), dtype=torch.float32)
    codebook = Codebook(None, size=1)
    codebook.centroids = database.centroids.numpy()
    kernel = ASMKKernel(codebook, binary=False)
    database.ivf_builder = IvfBuilder({"ivf": {"use_idf": False}}, codebook, kernel, cache_path=None)
    database.ivf_builder.ivf.add(
        np.eye(2, dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    )
    database.asmk = ASMKMethod(
        metadata={},
        codebook=codebook,
        params={"query_ivf": {
            "quantize": {"multiple_assignment": 1},
            "aggregate": {},
            "search": {"topk": 2},
            "similarity": {"alpha": 3.0, "similarity_threshold": 0.0},
        }},
    )
    return database


def test_query_returns_integer_ranks_and_float64_scores(database: RetrievalDatabase) -> None:
    ranks, scores, codes = database.query(
        np.array([[1.0, 0.0]], dtype=np.float32), np.array([2], dtype=np.int64)
    )

    np.testing.assert_array_equal(ranks, [[0, 1]])
    np.testing.assert_allclose(scores, [[1.0, 0.0]], atol=1e-5)
    np.testing.assert_array_equal(codes, [[0]])
    assert ranks.dtype == codes.dtype == np.int64
    assert scores.dtype == np.float64


def test_query_rejects_float32_scores(database: RetrievalDatabase, monkeypatch: pytest.MonkeyPatch) -> None:
    """An external search result must satisfy the float64 score contract."""
    monkeypatch.setattr(
        database.ivf_builder.ivf,
        "search",
        lambda *_args, **_kwargs: (np.array([0, 1], dtype=np.int64), np.array([1.0, 0.0], dtype=np.float32)),
    )
    with pytest.raises(BeartypeException):
        database.query(np.array([[1.0, 0.0]], dtype=np.float32), np.array([2], dtype=np.int64))
