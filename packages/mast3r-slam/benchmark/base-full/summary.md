# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 2523
- Average frame time: 181.51 ms
- Average FPS: 5.51
- Average tracking time: 97.39 ms
- Average logging time: 54.72 ms
- Frame-time slope: 0.0442 ms/frame
- Logging slope vs keyframes: 0.4347 ms/keyframe

## Backend

- Tasks: 97
- Average task time: 379.51 ms
- Average add-factors time: 183.24 ms
- Average global-opt time: 176.85 ms
- Task-time slope: 4.4834 ms/keyframe

## Diagnosis

- Frontend logging cost increases with keyframe count; visualization overhead is a likely source of FPS decay.
- Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck.
- Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time.
