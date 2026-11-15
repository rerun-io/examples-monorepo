"""The vendored cuSFM schema must build usable message classes."""

from __future__ import annotations

import pytest

from colsfm.schema import CusfmSchema, load_schema


@pytest.fixture(scope="module")
def schema() -> CusfmSchema:
    """The vendored cuSFM schema, loaded once."""
    return load_schema()


def test_schema_carries_every_extracted_proto_file(schema: CusfmSchema) -> None:
    """All 31 descriptors the binaries embed are present, not just the extractor's 19."""
    assert len(schema.proto_file_names) == 31
    assert "protos/visual/cusfm/pose_graph.proto" in schema.proto_file_names
    assert "protos/visual/cusfm/vision_mapping_config.proto" in schema.proto_file_names
    assert "protos/visual/geometry/bundle_adjustment_config.proto" in schema.proto_file_names
    assert "protos/common/ransac/ransac_config.proto" in schema.proto_file_names


def test_message_class_lookup_covers_the_pipeline_messages(schema: CusfmSchema) -> None:
    """Every message the open pipeline reads or writes resolves by full name."""
    for full_name in (
        "protos.visual.general.Keyframe",
        "protos.visual.general.KeyframesMetadataCollection",
        "protos.visual.cusfm.MapKeyPoints",
        "protos.visual.cusfm.PoseGraphConfig",
        "protos.visual.cusfm.MatchPairSelectConfig",
        "protos.visual.cusfm.MatchingTaskWorkerConfig",
        "protos.visual.cusfm.VisionMappingConfig",
        "protos.visual.geometry.BundleAdjustmentConfig",
        "protos.common.image.KeypointCreationParams",
        "protos.common.image.KeypointMatchingParams",
    ):
        message_class = schema.message_class(full_name)
        assert message_class.DESCRIPTOR.full_name == full_name


def test_unknown_message_name_is_rejected(schema: CusfmSchema) -> None:
    """A typo in a message name fails loudly rather than returning None."""
    with pytest.raises(KeyError):
        schema.message_class("protos.visual.general.NoSuchMessage")


