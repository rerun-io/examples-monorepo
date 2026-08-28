from pathlib import Path

import pytest

from dataforge.identity import SequenceIdentity
from dataforge.paths import rrd_path


def test_recording_id_joins_dataset_and_parts() -> None:
    identity: SequenceIdentity = SequenceIdentity(dataset="robocap", parts=("f408193e6447b3b0", "s1", "seg1"))
    assert identity.recording_id == "robocap__f408193e6447b3b0__s1__seg1"


def test_sequence_key_is_slash_joined() -> None:
    identity: SequenceIdentity = SequenceIdentity(dataset="hocap", parts=("subject_1", "seq01"))
    assert identity.sequence_key == "subject_1/seq01"


def test_parts_reject_double_underscore() -> None:
    with pytest.raises(ValueError, match="__"):
        SequenceIdentity(dataset="robocap", parts=("bad__part",))


def test_parts_reject_traversal() -> None:
    with pytest.raises(ValueError, match="Invalid"):
        SequenceIdentity(dataset="robocap", parts=("..",))


def test_dataset_must_be_single_part() -> None:
    with pytest.raises(ValueError, match="one path part"):
        SequenceIdentity(dataset="a/b", parts=("x",))


def test_rrd_path_is_layer_major_with_recording_id_filename() -> None:
    identity: SequenceIdentity = SequenceIdentity(dataset="robocap", parts=("f408193e6447b3b0", "s1", "seg1"))
    path: Path = rrd_path(Path("/out"), layer="base", identity=identity)
    assert path == Path("/out/base/robocap__f408193e6447b3b0__s1__seg1.rrd")
