from __future__ import annotations

from pathlib import Path

from simplecv.apis.batch_raw_to_rrd import BatchConvertConfig, main
from simplecv.data.exoego.epfl_smart_kitchen import EpflSmartKitchenConfig
from simplecv.data.exoego.sequence_identity import SequenceIdentity


class _CountingSequence:
    init_count: int = 0

    def __init__(self, cfg: EpflSmartKitchenConfig) -> None:
        type(self).init_count += 1
        self.config: EpflSmartKitchenConfig = cfg
        self.ego_sequence = None
        self.exo_sequence = None

    def __len__(self) -> int:
        return 1

    @property
    def sequence_identity(self) -> SequenceIdentity:
        return SequenceIdentity(dataset="dummy", parts=("sequence",))

    def iter_dataset(self):
        yield from self.__class__.iter_episode_sequences(self.config)

    def num_sequences(self) -> int:
        return self.__class__.num_sequences_for_config(self.config)

    @classmethod
    def iter_episode_sequences(cls, cfg: EpflSmartKitchenConfig):
        yield cls(cfg)

    @classmethod
    def num_sequences_for_config(cls, cfg: EpflSmartKitchenConfig) -> int:
        return 1


def test_batch_raw_to_rrd_does_not_instantiate_discovery_sequence_for_counting() -> None:
    _CountingSequence.init_count = 0
    cfg = BatchConvertConfig(
        dataset=EpflSmartKitchenConfig(_target=_CountingSequence, root_directory=Path(), load_labels=True),
        dry_run=True,
        max_conversions=1,
    )

    main(cfg)

    assert _CountingSequence.init_count == 1
