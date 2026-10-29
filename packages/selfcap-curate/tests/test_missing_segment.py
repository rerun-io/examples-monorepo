"""Missing catalog segments report registration and borrowing context."""

from unittest.mock import Mock

import pytest
from rerun.catalog import DatasetEntry

from selfcap_curate.catalog_io import open_segment


def test_missing_segment_names_dataset_before_reading_layout() -> None:
    dataset: Mock = Mock(spec=DatasetEntry)
    dataset.name = "dataforge-selfcap"
    dataset.segment_ids = Mock(return_value=["selfcap__2025-12-11__0ec1e39a__ep001"])
    donor: str = "selfcap__2025-12-08__cc808233__ep001"

    with pytest.raises(ValueError, match=f"segment {donor} is not registered in dataset dataforge-selfcap"):
        open_segment(dataset, donor)

    dataset.filter_segments.assert_not_called()


