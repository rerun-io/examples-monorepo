# Fixed renderer camera paths

Each JSON file contains 300 pinhole `CameraSpec` records for one complete orbit.
The image size is 1920x1080. Poses are row-major camera-to-world transforms in
OpenCV right/down/forward coordinates; focal lengths and principal points are
in pixels. `--res 3840x2160` scales the intrinsics by two and leaves poses unchanged.
