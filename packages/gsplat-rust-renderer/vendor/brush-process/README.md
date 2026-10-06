# Brush process observer adapter

Temporary adapter from https://github.com/ArthurBrussee/brush, revision
`1388f74c6fe0236f68ee4915564bf00e9d2e3747`, path `crates/brush-process`.
Remove this vendor copy when Brush exposes these statistics upstream.

The observer patch adds existing `TrainStepStats` and step duration
to `TrainMessage::TrainStep`, and `RefineStats` plus refine duration to
`TrainMessage::RefineStep` so the native Rerun consumer
can log training metrics without enabling Brush's own logger. Brush retains
its five-step event cadence and always emits the final step. `observer.patch`
records the complete source delta; the manifest expands upstream workspace
dependencies for this workspace.
