# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 504
- Average frame time: 126.98 ms
- Average FPS: 7.88
- Average tracking time: 82.58 ms
- Average logging time: 0.00 ms
- Frame-time slope: 0.0671 ms/frame
- Logging slope vs keyframes: 0.0000 ms/keyframe

## Backend

- Tasks: 87
- Average task time: 209.29 ms
- Average add-factors time: 139.13 ms
- Average global-opt time: 56.67 ms
- Task-time slope: 1.6675 ms/keyframe

## Diagnosis

- Tracking cost itself grows over time; inspect matcher warm starts, optimizer iteration counts, or cached state growth.
- Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck.
- Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time.
