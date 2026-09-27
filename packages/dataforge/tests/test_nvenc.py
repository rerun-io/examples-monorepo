"""NVENC scheduling and file transcode contracts without a GPU."""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from dataforge import video_encoding as video


@pytest.mark.parametrize('value', ['0', '-1', 'bad', '1.5'])
def test_invalid_slot_counts(value, monkeypatch):
    monkeypatch.setenv('DATAFORGE_NVENC_SLOTS', value)
    with pytest.raises(ValueError, match='DATAFORGE_NVENC_SLOTS'), video.nvenc_slot():
        pytest.fail('invalid slot count accepted')


def test_slots_bound_concurrent_holders(tmp_path, monkeypatch):
    monkeypatch.setenv('DATAFORGE_NVENC_SLOTS', '2')
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path)
    entered = Event()
    waiting = Event()
    def third():
        waiting.set()
        with video.nvenc_slot():
            entered.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with video.nvenc_slot():
            with video.nvenc_slot():
                future = pool.submit(third)
                assert waiting.wait(2)
                assert not entered.wait(.3)
            assert entered.wait(2)
        future.result(timeout=2)


@pytest.mark.parametrize('error', ['OpenEncodeSessionEx failed', 'incompatible client key', 'No capable devices found', 'bad input'])
def test_session_retry_only(error, monkeypatch, tmp_path):
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path)
    waits = []
    monkeypatch.setattr(video.time, 'sleep', waits.append)
    calls = []
    def run():
        calls.append(True)
        return subprocess.CompletedProcess([], 1, '', error)
    result = video.run_nvenc(run)
    assert result.stderr == error
    assert len(calls) == (1 if error == 'bad input' else 6)
    assert waits == ([] if error == 'bad input' else [2, 4, 8, 16, 32])


@pytest.mark.parametrize(('decode', 'crop'), [('cpu', None), ('cuda', None), ('cuda', (32, 32, 0, 0))])
def test_gray_transcode_command(decode, crop, tmp_path, monkeypatch):
    monkeypatch.setenv('DATAFORGE_FFMPEG', '/fake/ffmpeg')
    monkeypatch.setattr(video, 'require_av1_nvenc', lambda binary: None)
    monkeypatch.setattr(video, 'mp4_frame_count', lambda path: 4)
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path / 'slots')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')
    monkeypatch.setattr(video.subprocess, 'run', run)
    video.transcode_mp4_gray(Path('in.mp4'), Path('out.mp4'), gop=60, cq=36, fps=60, frames=4, decode=decode, crop=crop)
    expected = ['/fake/ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    if decode == 'cuda' and crop is None:
        expected += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
    expected += ['-r', '60', '-i', 'in.mp4', '-map', '0:v:0', '-an']
    if decode == 'cpu' or crop is not None:
        expected += ['-vf', ('crop=32:32:0:0,' if crop else '') + 'format=gray,pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p']
    expected += ['-fps_mode', 'passthrough', '-c:v', 'av1_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '36', '-bf', '0', '-g', '60', '-movflags', '+faststart', '-frames:v', '4', 'out.mp4']
    assert commands == [expected]


def test_nvdec_failure_falls_back(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(video, 'require_av1_nvenc', lambda binary: None)
    monkeypatch.setattr(video, 'mp4_frame_count', lambda path: 4)
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path)
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, int('-hwaccel' in command), '', 'decoder failed')
    monkeypatch.setattr(video.subprocess, 'run', run)
    video.transcode_mp4_gray(Path('in.mp4'), Path('out.mp4'), gop=60, cq=36, fps=60, frames=4, decode='cuda')
    assert len(commands) == 2
    assert '-hwaccel' in commands[0] and '-vf' not in commands[0]
    assert '-hwaccel' not in commands[1] and '-vf' in commands[1]
    assert 'NVDEC path failed (decoder failed); CPU decode' in capsys.readouterr().out


def test_encoder_check_cached(monkeypatch):
    video.require_av1_nvenc.cache_clear()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, 'av1_nvenc', '')
    monkeypatch.setattr(video.subprocess, 'run', run)
    video.require_av1_nvenc(Path('/fake/cache-test'))
    video.require_av1_nvenc(Path('/fake/cache-test'))
    assert len(calls) == 1
    video.require_av1_nvenc.cache_clear()


def test_pipe_session_retry_replays_one_shot_frames(tmp_path, monkeypatch):
    script = tmp_path / 'ffmpeg'
    script.write_text('''#!/usr/bin/env python3
import pathlib, sys
if '-encoders' in sys.argv:
    print('av1_nvenc')
else:
    output = pathlib.Path(sys.argv[-1])
    marker = output.with_suffix('.attempt')
    if not marker.exists():
        marker.touch()
        sys.stdin.buffer.read(1)
        print('OpenEncodeSessionEx failed', file=sys.stderr)
        sys.exit(1)
    output.write_bytes(sys.stdin.buffer.read())
''')
    script.chmod(0o700)
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path / 'slots')
    waits = []
    monkeypatch.setattr(video.time, 'sleep', waits.append)
    monkeypatch.setattr(video, 'mp4_frame_count', lambda path: 2)
    output = tmp_path / 'clip.mp4'
    payload = [b'a' * 100000, b'b' * 100000]
    assert video.encode_frames_to_mp4(iter(payload), output, source=video.FrameSource('png'), fps=30, ffmpeg=script) == 2
    assert output.read_bytes() == b''.join(payload)
    assert waits == [2.0]


@pytest.mark.integration
def test_cuda_gray_matches_cpu(tmp_path, nvenc_ffmpeg, monkeypatch, capsys):
    import av
    import numpy as np
    monkeypatch.setenv('DATAFORGE_FFMPEG', str(nvenc_ffmpeg))
    source = tmp_path / 'source.mp4'
    subprocess.run([str(nvenc_ffmpeg), '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=256x160:rate=30,format=gray', '-frames:v', '12', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
    decoded = []
    for mode in ('cpu', 'cuda'):
        target = tmp_path / f'{mode}.mp4'
        assert video.transcode_mp4_gray(source, target, fps=30, gop=60, cq=36, frames=12, decode=mode) == 12
        with av.open(str(target)) as container:
            decoded.append(np.stack([frame.to_ndarray(format='gray') for frame in container.decode(video=0)]).astype(np.float64))
    if 'NVDEC path failed' in capsys.readouterr().out:
        pytest.skip('NVDEC unavailable; CPU fallback ran')
    assert decoded[0].shape == decoded[1].shape == (12, 160, 256)
    mse = np.mean((decoded[0] - decoded[1]) ** 2)
    assert mse == 0.0 or 10 * np.log10(255 ** 2 / mse) >= 40.0


def test_session_failure_never_falls_back_to_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path)
    monkeypatch.setattr(video, 'require_av1_nvenc', lambda binary: None)
    monkeypatch.setattr(video.time, 'sleep', lambda seconds: None)
    commands = []
    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1, '', 'OpenEncodeSessionEx failed')
    monkeypatch.setattr(video.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='OpenEncodeSessionEx failed'):
        video.transcode_mp4_gray(Path('in.mp4'), Path('out.mp4'), gop=60, cq=36, fps=60, frames=4, decode='cuda')
    assert len(commands) == 6
    assert all('-hwaccel' in command and '-vf' not in command for command in commands)


def test_slot_released_after_exception(tmp_path, monkeypatch):
    import fcntl
    monkeypatch.setattr(video, 'NVENC_SLOT_DIR', tmp_path)
    monkeypatch.setenv('DATAFORGE_NVENC_SLOTS', '1')
    with pytest.raises(RuntimeError, match='encoder failed'), video.nvenc_slot():
        raise RuntimeError('encoder failed')
    with (tmp_path / 'slot-0.lock').open('a+b') as slot:
        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(slot, fcntl.LOCK_UN)
