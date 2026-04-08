# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 504
- Average frame time: 181.45 ms
- Average FPS: 5.51
- Average tracking time: 80.29 ms
- Average logging time: 56.54 ms
- Frame-time slope: 0.2145 ms/frame
- Logging slope vs keyframes: 0.8613 ms/keyframe

## Backend

- Tasks: 88
- Average task time: 209.07 ms
- Average add-factors time: 135.10 ms
- Average global-opt time: 59.88 ms
- Task-time slope: 1.7850 ms/keyframe

## Diagnosis

- Frontend logging cost increases with keyframe count; visualization overhead is a likely source of FPS decay.
- Tracking cost itself grows over time; inspect matcher warm starts, optimizer iteration counts, or cached state growth.
- Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck.
- Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time.
