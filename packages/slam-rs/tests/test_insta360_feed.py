"""The feed recognises an Insta360 recording by its ``insta360`` property and reads the factory calibration from it."""

import pyarrow as pa
import pytest

from slam_rs.catalog_feed import Insta360Factory, RigProfile, read_insta360_factory


def properties(**columns: pa.Array) -> pa.Table:
    return pa.table({name: pa.array([value]) for name, value in columns.items()})


def test_a_recording_without_the_insta360_property_is_not_an_insta360_one() -> None:
    assert read_insta360_factory(properties(**{"property:capture:start_time_ns": 5})) is None


def test_the_factory_string_crop_and_readout_become_the_vio_argument() -> None:
    table: pa.Table = properties(
        **{
            "property:insta360:offset_v6": ["2_2.31_x"],
            "property:insta360:window_crop": [[7680, 7680, 7424, 7424, 0, 0]],
            "property:insta360:rolling_shutter_readout_s": [0.015883],
        }
    )
    factory: Insta360Factory | None = read_insta360_factory(table)
    assert factory is not None
    assert factory == Insta360Factory(lens_string="2_2.31_x", window_crop=(7680, 7680, 7424, 7424, 0, 0), readout_s=0.015883)
    assert factory.vio_argument() == ("2_2.31_x", (7680, 7680, 7424, 7424, 0, 0), 0.015883)


def test_an_old_conversion_without_the_readout_is_refused() -> None:
    with pytest.raises(ValueError, match="rolling_shutter_readout_s"):
        read_insta360_factory(properties(**{"property:insta360:offset_v6": ["2_2.31_x"]}))


def test_decode_threads_must_be_positive() -> None:
    assert RigProfile(decode_threads=4).decode_threads == 4
    with pytest.raises(ValueError, match="decode_threads"):
        RigProfile(decode_threads=0)
