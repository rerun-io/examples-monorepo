"""The training package imports, and the network comes from monopriors (no local model copy)."""

from pathlib import Path


def test_no_local_model_copy() -> None:
    pkg = Path(__file__).resolve().parents[1] / "zipdepth"
    assert not (pkg / "model").exists() and not list(pkg.rglob("model_utils.py"))


def test_training_stack_imports_and_builds() -> None:
    from monopriors.third_party.zipdepth.architecture import create_model

    from zipdepth.data.dataset import LargeScaleDepthDataset  # noqa: F401
    from zipdepth.loss.depth_loss import ZipDepthLoss
    from zipdepth.training.trainer import ZipDepthTrainer  # noqa: F401

    assert sum(p.numel() for p in create_model(variant="base").parameters()) > 5_000_000
    assert ZipDepthLoss is not None
