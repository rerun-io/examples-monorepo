"""Production autocast and the serialized CLI boundary."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip('torch', reason='requires handtrack environment')
from serde.json import from_json, to_json

from handtrack.apis.train import Config, StreamSettings, build_source


def test_config_roundtrip_and_factory_arguments(tmp_path: Path) -> None:
    config = Config(run_dir=tmp_path, stream=StreamSettings(segment_ids=('fixed-segment',)))
    assert from_json(Config, to_json(config)) == config
    # Both are rejected before any catalog or GPU access.
    with pytest.raises(ValueError, match='unknown split'):
        build_source(config.stream, 'holdout', config.nets)
    with pytest.raises(ValueError, match='unknown datasets'):
        build_source(StreamSettings(datasets=('hot3d',)), config.stream.train_split, config.nets)


def test_production_cpu_bf16(tmp_path: Path) -> None:
    script = '''
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_train_loop import DETNET_SGD, KEYNET_SGD, FakeSource
from handtrack.train.loop import Trainer, LoopSettings

torch.set_num_threads(2)
trainer = Trainer('both', DETNET_SGD, KEYNET_SGD, LoopSettings(epochs=1, bf16=True), Path(sys.argv[1]), 'cpu')
state = trainer.run(FakeSource(1, 1), FakeSource(1, 1))
assert state.step == 2
assert all(torch.isfinite(p).all() for model in trainer.models.values() for p in model.parameters())
'''
    environment = dict(os.environ)
    environment.pop('PIXI_DEV_MODE', None)
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=Path(__file__).resolve().parents[1], env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
