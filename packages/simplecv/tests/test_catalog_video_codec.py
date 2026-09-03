import pytest

from simplecv.catalog_video_codec import catalog_codec_name


def test_catalog_codec_name_maps_catalog_codecs() -> None:
    assert catalog_codec_name(int.from_bytes(b"av01", "big")) == "av1"
    assert catalog_codec_name(int.from_bytes(b"avc1", "big")) == "h264"
    assert catalog_codec_name(int.from_bytes(b"hev1", "big")) == "hevc"


def test_catalog_codec_name_rejects_unknown_and_undecodable_codecs() -> None:
    with pytest.raises(ValueError):  # not a Rerun VideoCodec at all
        catalog_codec_name(int.from_bytes(b"mp4v", "big"))
    with pytest.raises(ValueError, match="unsupported catalog video codec VP9"):
        catalog_codec_name(int.from_bytes(b"vp09", "big"))
