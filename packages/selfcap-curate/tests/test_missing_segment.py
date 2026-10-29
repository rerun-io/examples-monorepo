"""Missing catalog segments report registration and borrowing context."""

from unittest.mock import Mock

import pytest
from rerun.catalog import DatasetEntry
from test_oak_calibration import factory

from selfcap_curate.catalog_io import Segment, open_segment
from selfcap_curate.factory import read_quest_block
from selfcap_curate.records import SelfcapLayout


def test_missing_segment_names_dataset_before_reading_layout() -> None:
    dataset: Mock = Mock(spec=DatasetEntry)
    dataset.name = "dataforge-selfcap"
    dataset.segment_ids = Mock(return_value=["selfcap__2025-12-11__0ec1e39a__ep001"])
    donor: str = "selfcap__2025-12-08__cc808233__ep001"

    with pytest.raises(ValueError, match=f"segment {donor} is not registered in dataset dataforge-selfcap"):
        open_segment(dataset, donor)

    dataset.filter_segments.assert_not_called()


def test_missing_camera50_donor_names_borrowing_session() -> None:
    dataset: Mock = Mock(spec=DatasetEntry)
    dataset.name = "dataforge-selfcap"
    target: str = "selfcap__2025-12-11__0ec1e39a__ep001"
    dataset.segment_ids = Mock(return_value=[target])
    segment: Segment = Segment(dataset, target, SelfcapLayout(3, 4, ("a", "b", "c")))

    with pytest.raises(
        ValueError,
        match="session 2025-12-11__0ec1e39a borrows camera 50 from selfcap__2025-12-08__cc808233__ep001; register that segment",
    ) as raised:
        read_quest_block(segment, factory())

    assert isinstance(raised.value.__cause__, ValueError)
    assert "is not registered in dataset dataforge-selfcap" in str(raised.value.__cause__)
