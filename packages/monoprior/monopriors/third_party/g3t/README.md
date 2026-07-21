# Vendored G3T inference core

This directory contains the minimum feed-forward inference closure from
[`g3t-paper/g3t`](https://github.com/g3t-paper/g3t) at commit
`193ce19574b73f0778a475ee54aadeb848e86b88`.

Included:

- the G3T model and shared VGGT aggregator;
- the local/relative camera heads, depth head, and point head;
- transformer layers required by the aggregator and heads;
- camera-pose and quaternion decoding utilities.

Excluded:

- training code and losses;
- G3T-Long and all loop-closure dependencies;
- tracking heads and dependencies, which G3T does not instantiate;
- upstream image loading, CLI, serialization, and visualization code;
- examples and model weights.

Imports were rewritten from the upstream `vggt` package namespace to
`monopriors.third_party.g3t`. No model math was intentionally changed.
