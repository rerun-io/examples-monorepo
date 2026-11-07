"""Unit checks for choosing the ffmpeg decode/rotate chain that feeds the AV1 encoder."""

from arkitscenes_download.ingest.mov import DecodeChain, cpu_decode_chain, cuda_decode_chain, parse_cuda_filters

CUDA_INPUT: tuple[str, ...] = ("-hwaccel", "cuda", "-hwaccel_output_format", "cuda")
HWACCELS_WITH_CUDA: str = "Hardware acceleration methods:\ncuda\nvaapi\nvulkan\n"
FILTER_LISTING: str = (
    "Filters:\n"
    "  T.. = Timeline support\n"
    "  ... afir              N->N       Apply Finite Impulse Response filter.\n"
    " .. scale_cuda        V->V       GPU accelerated video resizer\n"
    " .. transpose_cuda    V->V       Transpose input video using CUDA\n"
    " T. yadif_cuda        V->V       Deinterlace CUDA frames\n"
    " .. transpose         V->V       Transpose input video.\n"
)


def test_parse_cuda_filters_lists_only_cuda_filters() -> None:
    """The *_cuda filter names are collected when the build has the cuda hwaccel."""
    assert parse_cuda_filters(HWACCELS_WITH_CUDA, FILTER_LISTING) == frozenset({"scale_cuda", "transpose_cuda", "yadif_cuda"})


def test_parse_cuda_filters_without_cuda_hwaccel_is_none() -> None:
    """A build without the cuda hwaccel cannot decode on NVDEC at all."""
    assert parse_cuda_filters("Hardware acceleration methods:\nvaapi\n", FILTER_LISTING) is None


def test_cpu_chain_keeps_the_software_rotation_and_trim() -> None:
    """The CPU chain is the original software decode, rotate and PTS rebase."""
    assert cpu_decode_chain(0, 0) == DecodeChain((), "setpts=PTS-STARTPTS")
    assert cpu_decode_chain(1, 0) == DecodeChain((), "transpose=2,setpts=PTS-STARTPTS")
    assert cpu_decode_chain(2, 0) == DecodeChain((), "hflip,vflip,setpts=PTS-STARTPTS")
    assert cpu_decode_chain(7, 4) == DecodeChain((), "select='gte(n\\,4)',transpose=1,setpts=PTS-STARTPTS")


def test_cuda_chain_rotates_with_transpose_cuda() -> None:
    """Each quarter turn maps to the transpose_cuda direction of the CPU filter."""
    cuda_filters: frozenset[str] = frozenset({"transpose_cuda"})
    assert cuda_decode_chain(0, 0, cuda_filters) == DecodeChain(CUDA_INPUT, "setpts=PTS-STARTPTS")
    assert cuda_decode_chain(1, 0, cuda_filters) == DecodeChain(CUDA_INPUT, "transpose_cuda=dir=cclock,setpts=PTS-STARTPTS")
    assert cuda_decode_chain(2, 0, cuda_filters) == DecodeChain(CUDA_INPUT, "transpose_cuda=dir=reversal,setpts=PTS-STARTPTS")
    assert cuda_decode_chain(3, 0, cuda_filters) == DecodeChain(CUDA_INPUT, "transpose_cuda=dir=clock,setpts=PTS-STARTPTS")


def test_cuda_chain_trims_leading_frames_before_rotating() -> None:
    """The drop_leading select runs on CUDA frames ahead of the rotation."""
    chain: DecodeChain | None = cuda_decode_chain(5, 12, frozenset({"transpose_cuda"}))
    assert chain == DecodeChain(CUDA_INPUT, "select='gte(n\\,12)',transpose_cuda=dir=cclock,setpts=PTS-STARTPTS")


def test_cuda_chain_needs_the_filters_it_uses() -> None:
    """Without transpose_cuda only the unrotated track can stay on the GPU; without the hwaccel none can."""
    assert cuda_decode_chain(1, 0, frozenset({"scale_cuda"})) is None
    assert cuda_decode_chain(0, 3, frozenset()) == DecodeChain(CUDA_INPUT, "select='gte(n\\,3)',setpts=PTS-STARTPTS")
    assert cuda_decode_chain(0, 0, None) is None
