# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 2523
- Average frame time: 117.53 ms
- Average FPS: 8.51
- Average tracking time: 90.06 ms
- Average logging time: 0.00 ms
- Frame-time slope: 0.0201 ms/frame
- Logging slope vs keyframes: 0.0000 ms/keyframe

## Backend

- Tasks: 97
- Average task time: 367.95 ms
- Average add-factors time: 181.44 ms
- Average global-opt time: 167.24 ms
- Task-time slope: 3.9996 ms/keyframe

## Diagnosis

- Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck.
- Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time.
