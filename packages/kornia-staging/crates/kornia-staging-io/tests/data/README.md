Synthetic constant 16x16 black, one frame (visible Y=16, limited range).
Generated with ffmpeg from `-f lavfi -i color=black:size=16x16:rate=1 -frames:v 1`:
- H.264: `-c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -qp 1 -threads 1 -f h264 black-16.h264`
- AV1: `-c:v libaom-av1 -cpu-used 8 -crf 0 -threads 1 -f obu black-16.obu`
These generated test fixtures contain no external content.

`black-16-full.obu` uses the same AV1 command with
`-vf setparams=range=full -color_range pc`. It retains Y=16 but signals full range,
so tests can distinguish range metadata from pixel
values. OpenH264's Rust API does not expose range metadata: the decoder reports
`Unknown` for the H.264 fixture rather than assuming that its Y=16 is full range.
