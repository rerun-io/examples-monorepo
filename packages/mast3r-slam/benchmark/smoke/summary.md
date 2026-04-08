# MASt3R-SLAM Benchmark Summary

## Frontend

- Frames: 20
- Average frame time: 139.36 ms
- Average FPS: 7.18
- Average tracking time: 68.10 ms
- Average logging time: 18.39 ms
- Frame-time slope: -2.2933 ms/frame
- Logging slope vs keyframes: 1.9989 ms/keyframe

## Backend

- Tasks: 3
- Average task time: 224.10 ms
- Average add-factors time: 146.87 ms
- Average global-opt time: 14.68 ms
- Task-time slope: 2.7937 ms/keyframe

## Diagnosis

- Frontend logging cost increases with keyframe count; visualization overhead is a likely source of FPS decay.
- Tracking cost itself grows over time; inspect matcher warm starts, optimizer iteration counts, or cached state growth.
- Backend pair construction dominates backend time; decoder plus dense matching remains the main backend bottleneck.
- Backend work grows noticeably with map size; retrieval and global optimization are not staying constant-time.
