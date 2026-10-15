"""The verify adapter fails closed on a case it cannot honour (unit lane: it refuses before any model or catalog)."""

import pytest
import rerun as rr
from dagkit.verify import Case

from exo_calib.verify import run


def test_run_refuses_a_windowed_case() -> None:
    case = Case("window", "short", "gt", dataset="synthetic", segment="ring", window_s=(1.0, 2.0), params={"frame_budget": 10})
    with pytest.raises(ValueError, match="window"):
        run(case, "post", rr.RecordingStream("exo-calib-test", send_properties=False))
