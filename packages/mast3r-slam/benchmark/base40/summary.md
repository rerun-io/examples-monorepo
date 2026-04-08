# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 40
- Average frame time: 119.77 ms
- Average FPS: 8.35
- Average tracking time: 59.28 ms
- Average logging time: 30.24 ms
- Frame-time slope: -0.8727 ms/frame
- Logging slope vs keyframes: 0.0000 ms/keyframe

## Backend

- Tasks: 1
- Average task time: 149.32 ms
- Average add-factors time: 0.00 ms
- Average global-opt time: 0.00 ms
- Task-time slope: 0.0000 ms/keyframe

## Diagnosis

- Tracking cost itself grows over time; inspect matcher warm starts, optimizer iteration counts, or cached state growth.
