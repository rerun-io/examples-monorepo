"""NVENC scheduling and file transcode contracts without a GPU."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Literal

import pytest

from dataforge import video_encoding as video


@pytest.fixture(name="cuda_filter_available", autouse=True)
def fixture_cuda_filter_available(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setattr(video, "has_scale_cuda", lambda binary: True)


@pytest.mark.parametrize("value", ["0", "-1", "bad", "1.5"])
@pytest.mark.parametrize("entry", ["parser", "reservation"])
def test_invalid_slot_counts(value: str, entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", value)
    with pytest.raises(ValueError, match="DATAFORGE_NVENC_SLOTS"):
        if entry == "parser":
            video.nvenc_slot_count()
        else:
            with video.nvenc_slot():
                pytest.fail("invalid slot count accepted")


def test_slots_bound_concurrent_holders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "2")
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    entered = Event()
    waiting = Event()

    def third() -> None:
        waiting.set()
        with video.nvenc_slot():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with video.nvenc_slot():
            with video.nvenc_slot():
                future = pool.submit(third)
                assert waiting.wait(2)
                assert not entered.wait(0.3)
            assert entered.wait(2)
        future.result(timeout=2)


@pytest.mark.parametrize("error", ["OpenEncodeSessionEx failed", "incompatible client key", "No capable devices found", "bad input"])
def test_session_retry_only(error: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    waits = []
    monkeypatch.setattr(video.time, "sleep", waits.append)
    calls = []

    def run(fds: tuple[int, ...]) -> subprocess.CompletedProcess[str]:
        assert len(fds) == 1 and fds[0] >= 0
        calls.append(True)
        return subprocess.CompletedProcess([], 1, "", error)

    result = video.run_nvenc(run)
    assert result.stderr == error
    assert len(calls) == (1 if error in ("bad input", "No capable devices found") else 6)
    assert waits == ([] if error in ("bad input", "No capable devices found") else [2, 4, 8, 16, 32])


@pytest.mark.parametrize("decode", ["cpu", "cuda"])
def test_gray_transcode_command(
    decode: Literal["cpu", "cuda"], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAFORGE_FFMPEG", "/fake/ffmpeg")
    monkeypatch.setattr(video, "require_av1_nvenc", lambda binary: None)
    monkeypatch.setattr(video, "mp4_frame_count", lambda path: 4)
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path / "slots")
    commands = []

    def run(command: list[str], *, pass_fds: tuple[int, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        import os

        assert len(pass_fds) == 1
        assert os.fstat(pass_fds[0]).st_ino == (tmp_path / "slots/slot-0.lock").stat().st_ino
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=60, frames=4, gray=True, decode=decode)
    expected = ["/fake/ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if decode == "cuda":
        expected += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    expected += ["-r", "60", "-i", "in.mp4", "-map", "0:v:0", "-an"]
    if decode == "cpu":
        expected += ["-vf", "format=gray,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p"]
    else:  # NVENC's AV1 takes 4:2:0 only: a 4:4:4 source (MAMMA's HEVC) is converted on the GPU
        expected += ["-vf", "scale_cuda=format=nv12"]
    expected += [
        "-fps_mode",
        "passthrough",
        "-c:v",
        "av1_nvenc",
        "-preset",
        "p4",
        "-rc",
        "vbr",
        "-cq",
        "36",
        "-bf",
        "0",
        "-g",
        "60",
        "-movflags",
        "+faststart",
        "-frames:v",
        "4",
        "out.mp4",
    ]
    assert commands == [expected]


@pytest.fixture
def ffmpeg_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every command transcode_mp4 runs, captured instead of run (4-frame source)."""
    monkeypatch.setenv("DATAFORGE_FFMPEG", "/fake/ffmpeg")
    monkeypatch.setattr(video, "require_av1_nvenc", lambda binary: None)
    monkeypatch.setattr(video, "mp4_frame_count", lambda path: 4)
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path / "slots")
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(video.subprocess, "run", run)
    return commands


@pytest.mark.parametrize("decode", ["cpu", "cuda"])
def test_color_rescale_command(decode: Literal["cpu", "cuda"], ffmpeg_commands: list[list[str]]) -> None:
    """A color source keeps its chroma; the rescale runs on the GPU under CUDA decode and on the CPU otherwise."""
    video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=30, frames=4, gray=False, size=(1920, 1080), decode=decode)
    (command,) = ffmpeg_commands
    filters = command[command.index("-vf") + 1]
    if decode == "cuda":
        assert filters == "scale_cuda=1920:1080:format=nv12"
    else:
        assert filters == "scale=1920:1080,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p"
    assert "format=gray" not in filters


@pytest.mark.parametrize("decode", ["cpu", "cuda"])
def test_keep_selects_source_frames_before_any_other_filter(decode: Literal["cpu", "cuda"], ffmpeg_commands: list[list[str]]) -> None:
    keep = [range(4, 8, 3), range(13, 17, 3)]
    video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=30, frames=4, gray=True, keep=keep, decode=decode)
    (command,) = ffmpeg_commands
    filters = command[command.index("-vf") + 1]
    assert filters.startswith("select='between(n,4,7)*not(mod(n-4,3))+between(n,13,16)*not(mod(n-13,3))'")
    assert command[command.index("-frames:v") + 1] == "4"


def test_nvdec_failure_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(video, "require_av1_nvenc", lambda binary: None)
    monkeypatch.setattr(video, "mp4_frame_count", lambda path: 4)
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    commands = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, int("-hwaccel" in command), "", "decoder failed")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=60, frames=4, gray=True, decode="cuda")
    assert len(commands) == 2
    assert "-hwaccel" in commands[0] and commands[0][commands[0].index("-vf") + 1] == "scale_cuda=format=nv12"
    assert "-hwaccel" not in commands[1] and "format=gray" in commands[1][commands[1].index("-vf") + 1]
    assert "NVDEC path failed (decoder failed); CPU decode" in capsys.readouterr().out


def test_encoder_check_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    video.require_av1_nvenc.cache_clear()
    calls = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "av1_nvenc", "")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.require_av1_nvenc(Path("/fake/cache-test"))
    video.require_av1_nvenc(Path("/fake/cache-test"))
    assert len(calls) == 1
    video.require_av1_nvenc.cache_clear()


def test_pipe_encoder_holds_a_slot_and_does_not_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "ffmpeg"
    script.write_text("""#!/usr/bin/env python3
import sys
from pathlib import Path
if "-encoders" in sys.argv:
    print("av1_nvenc")
else:
    assert any(fd.resolve() == Path(__file__).parent / "slots/slot-0.lock" for fd in Path("/proc/self/fd").iterdir())
    sys.stdin.buffer.read()
    print("OpenEncodeSessionEx failed", file=sys.stderr)
    sys.exit(1)
""")
    script.chmod(0o700)
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path / "slots")
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "1")
    waits = []
    monkeypatch.setattr(video.time, "sleep", waits.append)
    with pytest.raises(RuntimeError, match="OpenEncodeSessionEx failed"):
        video.encode_frames_to_mp4(iter([b"a", b"b"]), tmp_path / "clip.mp4", source=video.FrameSource("png"), fps=30, ffmpeg=script)
    assert waits == []
    with video.nvenc_slot():  # released after the failure
        pass


@pytest.mark.integration
def test_cuda_gray_matches_cpu(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import av
    import numpy as np

    if not video.has_scale_cuda(nvenc_ffmpeg):
        pytest.skip("ffmpeg lacks scale_cuda; the NVDEC path is unavailable")

    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            str(nvenc_ffmpeg),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=256x160:rate=30,format=gray",
            "-frames:v",
            "12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    decoded = []
    for mode in ("cpu", "cuda"):
        target = tmp_path / f"{mode}.mp4"
        assert video.transcode_mp4(source, target, fps=30, gop=60, cq=36, frames=12, gray=True, decode=mode) == 12
        with av.open(str(target)) as container:
            decoded.append(np.stack([frame.to_ndarray(format="gray") for frame in container.decode(video=0)]).astype(np.float64))
    if "NVDEC path failed" in capsys.readouterr().out:
        pytest.skip("NVDEC unavailable; CPU fallback ran")
    assert decoded[0].shape == decoded[1].shape == (12, 160, 256)
    mse = np.mean((decoded[0] - decoded[1]) ** 2)
    assert mse == 0.0 or 10 * np.log10(255**2 / mse) >= 40.0


@pytest.mark.integration
@pytest.mark.parametrize("decode", ["cpu", "cuda"])
def test_keep_keeps_exactly_the_selected_frames(decode: Literal["cpu", "cuda"], tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import av
    import numpy as np

    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    source = tmp_path / "source.mp4"
    subprocess.run(
        [str(nvenc_ffmpeg), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=256x160:rate=30,format=gray", "-frames:v", "12",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    target = tmp_path / "keep.mp4"
    keep = [range(1, 5, 3), range(10, 11, 3)]  # frames 1, 4 and 10: the sample at 7 was dropped
    assert video.transcode_mp4(source, target, fps=30, gop=60, cq=30, frames=3, gray=True, keep=keep, decode=decode) == 3
    with av.open(str(source)) as container:
        originals = np.stack([frame.to_ndarray(format="gray") for frame in container.decode(video=0)]).astype(np.float64)
    with av.open(str(target)) as container:
        kept = np.stack([frame.to_ndarray(format="gray") for frame in container.decode(video=0)]).astype(np.float64)
    nearest = [int(np.argmin(((originals - frame) ** 2).mean(axis=(1, 2)))) for frame in kept]
    assert nearest == [1, 4, 10]


def test_session_failure_never_falls_back_to_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    monkeypatch.setattr(video, "require_av1_nvenc", lambda binary: None)
    monkeypatch.setattr(video.time, "sleep", lambda seconds: None)
    commands = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, "", "OpenEncodeSessionEx failed")

    monkeypatch.setattr(video.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="OpenEncodeSessionEx failed"):
        video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=60, frames=4, gray=True, decode="cuda")
    assert len(commands) == 6
    assert all("-hwaccel" in command and command[command.index("-vf") + 1] == "scale_cuda=format=nv12" for command in commands)


def test_slot_released_after_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "1")
    with pytest.raises(RuntimeError, match="encoder failed"), video.nvenc_slot():
        raise RuntimeError("encoder failed")
    with (tmp_path / "slot-0.lock").open("a+b") as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(slot, fcntl.LOCK_UN)


def test_child_keeps_slot_after_parent_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import sys

    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "1")
    release_read, release_write = os.pipe()
    parent_code = """
import os
import subprocess
import sys
from pathlib import Path
from dataforge import video_encoding as video
video.NVENC_SLOT_DIR = Path(sys.argv[1])
with video.nvenc_slot() as fds:
    subprocess.Popen(
        [sys.executable, "-c", "import os, sys; os.read(int(sys.argv[1]), 1); print('done', flush=True)", sys.argv[2]],
        pass_fds=(*fds, int(sys.argv[2])),
    )
    os._exit(0)
"""
    probe_code = """
import fcntl
import sys
with open(sys.argv[1], "a+b") as slot:
    try:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(1)
"""
    try:
        with subprocess.Popen(
            [sys.executable, "-c", parent_code, str(tmp_path), str(release_read)],
            pass_fds=(release_read,),
            stdout=subprocess.PIPE,
            text=True,
        ) as parent:
            try:
                assert parent.wait(timeout=2) == 0
                probe = subprocess.run([sys.executable, "-c", probe_code, str(tmp_path / "slot-0.lock")], timeout=2)
                assert probe.returncode == 1
            finally:
                os.write(release_write, b"x")
                output, _ = parent.communicate(timeout=2)
            assert output == "done\n"
        assert subprocess.run([sys.executable, "-c", probe_code, str(tmp_path / "slot-0.lock")], timeout=2).returncode == 0
    finally:
        os.close(release_read)
        os.close(release_write)


@pytest.mark.integration
def test_cuda_color_rescale_matches_cpu(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """A 4K H.264 source with B-frames (GoPro-like) comes out 1080p, every frame, with no reordered samples."""
    import av
    import numpy as np

    if not video.has_scale_cuda(nvenc_ffmpeg):
        pytest.skip("ffmpeg lacks scale_cuda; the NVDEC path is unavailable")

    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    source = tmp_path / "source.mp4"
    subprocess.run(
        [str(nvenc_ffmpeg), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=3840x2160:rate=30", "-frames:v", "12"]
        + ["-c:v", "libx264", "-bf", "2", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )
    decoded = []
    for mode in ("cpu", "cuda"):
        target = tmp_path / f"{mode}.mp4"
        assert video.transcode_mp4(source, target, fps=30, gop=60, cq=30, frames=12, gray=False, size=(1920, 1080), decode=mode) == 12
        with av.open(str(target)) as container:
            frames = list(container.decode(video=0))
            decoded.append(np.stack([frame.to_ndarray(format="rgb24") for frame in frames]).astype(np.float64))
        with av.open(str(target)) as container:
            pts = [packet.pts for packet in container.demux(video=0) if packet.pts is not None]
        assert pts == sorted(pts)
    if "NVDEC path failed" in capsys.readouterr().out:
        pytest.skip("NVDEC unavailable; CPU fallback ran")
    assert decoded[0].shape == decoded[1].shape == (12, 1080, 1920, 3)
    mse = np.mean((decoded[0] - decoded[1]) ** 2)
    assert 10 * np.log10(255**2 / mse) >= 30.0


def test_missing_cuda_filters_choose_cpu_without_fallback_noise(
    ffmpeg_commands: list[list[str]], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Builds without CUDA scaling skip NVDEC entirely, including for grayscale streams."""
    monkeypatch.setattr(video, "has_scale_cuda", lambda binary: False)
    for gray in (False, True):
        video.transcode_mp4(Path("in.mp4"), Path("out.mp4"), gop=60, cq=36, fps=30, frames=4, gray=gray, size=(1920, 1080), decode="cuda")
    assert len(ffmpeg_commands) == 2
    for command in ffmpeg_commands:
        assert "-hwaccel" not in command
        assert "scale=1920:1080" in command[command.index("-vf") + 1]
    assert "NVDEC path failed" not in capsys.readouterr().out


@pytest.mark.parametrize("slots,expected_inputs", [("4", 1), ("3", 2), ("1", 4)])
def test_mosaic_shares_decode_within_available_slots(
    slots: str, expected_inputs: int, ffmpeg_commands: list[list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", slots)
    video.transcode_crops(
        Path("mosaic.mp4"), [Path(f"cam{i}.mp4") for i in range(4)],
        crops=[(192, 160, i * 192, 0) for i in range(4)], fps=30, frames=4, gop=60, cq=36,
    )
    assert len(ffmpeg_commands) == expected_inputs
    assert sum(command.count("-i") for command in ffmpeg_commands) == expected_inputs
    assert sum(command.count("av1_nvenc") for command in ffmpeg_commands) == 4
    assert all(command.count("av1_nvenc") <= int(slots) for command in ffmpeg_commands)


@pytest.mark.integration
@pytest.mark.parametrize("slots", ["3", "4"])
def test_mosaic_packets_match_separate_crops(tmp_path: Path, nvenc_ffmpeg: Path, monkeypatch: pytest.MonkeyPatch, slots: str) -> None:
    import av

    monkeypatch.setenv("DATAFORGE_FFMPEG", str(nvenc_ffmpeg))
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", slots)
    source: Path = tmp_path / "mosaic.mp4"
    subprocess.run([
        str(nvenc_ffmpeg), "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=768x160:rate=30",
        "-frames:v", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    crops: list[tuple[int, int, int, int]] = [(192, 160, i * 192, 0) for i in range(4)]
    outputs: list[Path] = [tmp_path / f"shared{i}.mp4" for i in range(4)]
    video.transcode_crops(source, outputs, crops=crops, fps=30, frames=12, gop=60, cq=36)
    for index, output in enumerate(outputs):
        reference: Path = tmp_path / f"reference{index}.mp4"
        monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "1")
        video.transcode_crops(source, [reference], crops=[crops[index]], fps=30, frames=12, gop=60, cq=36)
        with av.open(str(output)) as shared, av.open(str(reference)) as separate:
            assert [bytes(packet) for packet in shared.demux(video=0) if packet.size] == [bytes(packet) for packet in separate.demux(video=0) if packet.size]


def test_shared_decode_reserves_every_encoder_slot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "2")
    monkeypatch.setenv("DATAFORGE_FFMPEG", "/fake/ffmpeg")
    monkeypatch.setattr(video, "NVENC_SLOT_DIR", tmp_path)
    monkeypatch.setattr(video, "require_av1_nvenc", lambda binary: None)
    monkeypatch.setattr(video, "mp4_frame_count", lambda path: 4)
    entered: Event = Event()
    waiting: Event = Event()

    def outsider() -> None:
        waiting.set()
        with video.nvenc_slot():
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            pool.submit(outsider)
            assert waiting.wait(2.0)
            assert not entered.wait(0.2)
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(video.subprocess, "run", run)
        video.transcode_crops(Path("mosaic.mp4"), [Path("a.mp4"), Path("b.mp4")], crops=[(192, 160, 0, 0), (192, 160, 192, 0)], fps=30, frames=4, gop=60, cq=36)
        assert entered.wait(2.0)


def test_mosaic_retries_refused_sessions(ffmpeg_commands: list[list[str]], monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared decode retries the whole batch after an external encoder takes a session."""
    monkeypatch.setenv("DATAFORGE_NVENC_SLOTS", "2")
    waits: list[float] = []
    monkeypatch.setattr(video.time, "sleep", waits.append)
    attempts: list[tuple[int, ...]] = []

    def run(command: list[str], *, pass_fds: tuple[int, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert len(pass_fds) == 2
        attempts.append(pass_fds)
        return subprocess.CompletedProcess(command, int(len(attempts) == 1), "", "OpenEncodeSessionEx failed" if len(attempts) == 1 else "")

    monkeypatch.setattr(video.subprocess, "run", run)
    video.transcode_crops(Path("mosaic.mp4"), [Path("a.mp4"), Path("b.mp4")], crops=[(192, 160, 0, 0), (192, 160, 192, 0)], fps=30, frames=4, gop=60, cq=36)
    assert len(attempts) == 2
    assert waits == [2.0]
