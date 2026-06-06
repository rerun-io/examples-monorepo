# Video Decode Format Tradeoffs

Current simplecv default:

```text
Store working multiview videos as AV1 Main yuv420p.
Decode with TorchCodec CUDA as RGB uint8 NCHW tensors.
Use chunk_size=32 and seek_mode=approximate unless a benchmark says otherwise.
Keep frames on GPU; convert to BGR HWC numpy only at legacy boundaries.
```

## Decision

AV1 Main `yuv420p` is the default because it gives the best storage/throughput
balance in the datasets we checked. H.264 `yuv420p` can decode faster when it is
really on the hardware path, but it costs more storage and needs fallback checks.

Use H.264 only when decode FPS is the bottleneck, `cpu_fallback` is false, and
the larger files are acceptable.

Avoid `yuv444p`, HEVC Rext, and other high-chroma or unsupported profiles for
the hot path. They are larger and were slow or unsupported in TorchCodec CUDA
tests.

## Benchmarks

Full-sequence TorchCodec CUDA runs, `chunk_size=32`, RGB `uint8` output.
FPS is `camera-fps`: decoded frames across all cameras per wall-clock second.

| Dataset | AV1 `yuv420p` | H.264 `yuv420p` | H.264 tradeoff |
| --- | ---: | ---: | --- |
| MAMMA, 32 x 748 | 176M, 2958 FPS | 254M, 3779 FPS | 1.28x faster, 44% larger |
| Assembly101, 8 x 16707 | 529M, 7673 FPS | 719M, 10244 FPS | 1.34x faster, 36% larger |
| HOCAP, 9 x 1085 | 24M, 10510 FPS | 26M, 12226 FPS | 1.16x faster, 8% larger |
| EPFL Kitchen, 9 x 83430 | 1.1G, 7658 FPS | 2.6G, 10387 FPS | 1.36x faster, 136% larger |

The earlier MAMMA result where AV1 looked roughly 7.8x faster was not a real
codec result. The default TorchCodec CUDA backend fell back for the MAMMA H.264
files. With the beta CUDA backend, H.264 stayed on hardware decode and was faster
than AV1 on that scratch set.

## TorchCodec Rules

The current reader returns one RGB tensor per video:

```python
list[UInt8[torch.Tensor, "b 3 h w"]]
```

Stack synchronized views only after verifying they have the same height and
width:

```python
UInt8[torch.Tensor, "n_views t 3 h w"]
```

For mixed-resolution captures, bucket by exact `(height, width)`, stack within
each bucket, and keep a short list across buckets.

TorchCodec output should be treated as RGB. The installed TorchCodec version does
not expose a BGR decode option, so BGR conversion is a legacy adapter step, not a
decoder mode.

## Fallback Check

Always verify fallback before trusting a speed number:

```python
from torchcodec.decoders import VideoDecoder, set_cuda_backend

with set_cuda_backend("beta"):
    decoder = VideoDecoder(
        video_path,
        device="cuda",
        dimension_order="NCHW",
        seek_mode="approximate",
        num_ffmpeg_threads=0,
    )
    frames = decoder.get_frames_in_range(0, 32).data
    print(decoder.cpu_fallback)
```

A fallback can still return a CUDA tensor after CPU decode plus transfer. Treat
that as a failed hot-path candidate.

## Encoding Targets

Default storage target:

```bash
ffmpeg -i input.mp4 -c:v av1_nvenc -preset p4 -cq 30 -pix_fmt yuv420p -an output.av1.mp4
```

Fast-decode candidate:

```bash
ffmpeg -i input.mp4 -c:v h264_nvenc -preset p4 -cq 24 -pix_fmt yuv420p -an output.h264.mp4
```

Check the result:

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,profile,pix_fmt,width,height,nb_frames,avg_frame_rate \
  -of default=nw=1 \
  output.mp4
```
