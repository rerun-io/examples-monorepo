# sam3-rerun

Standalone SAM3 segmentation package with Rerun visualization. Provides text-conditioned instance segmentation for images and video.

## Features

- **Single-image predictor**: `SAM3Predictor` wraps `facebook/sam3` for zero-shot instance segmentation
- **Video batch**: Full-video batch inference with `propagate_in_video_iterator`
- **Video chunk**: Memory-efficient chunk-based processing for longer videos
- **Video stream**: Frame-by-frame streaming for constant memory usage
- **Gradio UIs**: Rerun viewer integration and annotated image output

## Usage

```bash
# Install environment
pixi install -e sam3-rerun

# Run demos
pixi run sam3-video-batch --video-path path/to/video.mp4
pixi run sam3-video-chunk --video-path path/to/video.mp4
pixi run sam3-video-stream --video-path path/to/video.mp4

# Launch Gradio apps
pixi run sam3-rerun-app
pixi run sam3-annotated-app
```
