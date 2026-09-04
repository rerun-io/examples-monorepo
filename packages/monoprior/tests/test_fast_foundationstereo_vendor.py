"""Import smoke test for the unowned Fast-FoundationStereo vendor boundary."""

from monopriors.third_party.fast_foundationstereo.distill_block import ForwardHelper
from monopriors.third_party.fast_foundationstereo.extractor import Feature
from monopriors.third_party.fast_foundationstereo.foundation_stereo import FastFoundationStereo
from monopriors.third_party.fast_foundationstereo.geometry import Combined_Geo_Encoding_Volume
from monopriors.third_party.fast_foundationstereo.submodule import build_gwc_volume_optimized_pytorch1
from monopriors.third_party.fast_foundationstereo.update import BasicSelectiveMultiUpdateBlock
from monopriors.third_party.fast_foundationstereo.utils import InputPadder


def test_fast_foundationstereo_inference_subset_imports() -> None:
    """The vendored inference modules import without upstream path hacks or optional TRT/ONNX dependencies."""
    assert all(
        symbol is not None
        for symbol in (
            ForwardHelper,
            Feature,
            FastFoundationStereo,
            Combined_Geo_Encoding_Volume,
            build_gwc_volume_optimized_pytorch1,
            BasicSelectiveMultiUpdateBlock,
            InputPadder,
        )
    )
