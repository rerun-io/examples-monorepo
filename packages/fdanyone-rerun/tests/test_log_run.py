"""``fdanyone-log-run --train`` checks the fused SSIM kernel before it logs or mattes anything."""

from pathlib import Path

import pytest


def test_training_without_the_ssim_kernel_fails_before_logging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from simplecv.rerun_log_utils import RerunTyroConfig

    from fdanyone_rerun.apis import log_run

    logged: list[Path] = []

    def missing_kernel() -> None:
        raise RuntimeError("hexels_mojo.fused_ssim does not import")

    monkeypatch.setattr(log_run, "require_fused_ssim", missing_kernel)
    monkeypatch.setattr(log_run, "log_run", lambda output_dir, *_: logged.append(output_dir))
    config = log_run.Config(rr_config=RerunTyroConfig(headless=True), output_dir=tmp_path, train=True)
    with pytest.raises(RuntimeError, match="fused_ssim"):
        log_run.main(config)
    assert not logged
