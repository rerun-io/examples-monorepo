"""Behavior checks for the GT-clean sub-dataset selection."""

import pandas as pd

from arkitscenes_download.ingest.subdataset import gt_clean_mask


def test_gt_clean_mask_requires_gt_and_bounded_interior_gaps() -> None:
    """Only gt-covered segments within the gap tolerance are selected; missing GT never passes."""
    table = pd.DataFrame(
        {
            "rerun_segment_id": ["clean", "gappy", "boundary", "no_gt"],
            "property:gt:provenance": [["ca1m-v1"], ["ca1m-v1"], ["ca1m-v1"], None],
            "property:gt:max_interior_gap_s": [[0.117], [30.5], [1.0], None],
        }
    )

    mask = gt_clean_mask(table, max_interior_gap_s=1.0)

    assert list(table[mask]["rerun_segment_id"]) == ["clean", "boundary"]
