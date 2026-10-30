# Ear-span and global-scale guards

The batch path rejects both ear targets in frames whose pair is missing or
outside 110–200 mm. Original targets remain in comparison recordings. Shape
selection uses the gated confidence. When fewer than three positive-confidence
ear pairs survive, it can select torso-only shape frames; the ear term is
disabled and soft Quest rotation plus torso fitting remain. This fallback is
flagged in the quality property. No Quest head-position term is introduced.

Global scale is projected to [0.85, 1.25] after each optimizer step, and seeds
are clamped to the same interval. Interior values are unchanged. The quality
property records surviving/rejected ear frames, the bounds, fallback use and
whether the final scale bound is active. No shoulder gating was added.

Four full-segment local runs used the existing RTMW batch configuration and
current v2 inputs. No catalog registration occurred. Distances below are mm;
arrows mean previous published fit → guarded local fit.

| Segment | Body median | Ears, all input | Ears, same surviving pairs | Global scale | Surviving pairs |
| --- | --- | --- | --- | --- | --- |
| d31bf1da | 18.65 → 18.66 | 1.93 → 1.91 | 1.93 → 1.91 | 1.0024 → 1.0009 | 4367/4367 |
| 819a09a4 | 43.70 → 35.99 | 46.29 → 122.61 | 30.78 → 8.64 | 1.0898 → 1.1390 | 711/7274 |
| e9dd8490 | 61.62 → 60.73 | 33.91 → 39.03 | 24.77 → 9.56 | 1.0978 → 1.1288 | 2990/6947 |
| 0b36cf91 | 56.45 → 38.50 | 47.75 → 37.65 | 142.07 → 6.42 | 1.5859 → 0.8846 | 1033/4245 |

d31 is within ±1 mm for body and ears; scale differs by 0.15%. Its actual
saved baseline scale is 1.0024, not the earlier approximate 1.10. None of these
four final scales sits on a bound, and none needs the torso-only fallback.
Quest-hand medians improve slightly in all four (5.54→5.52, 5.48→4.95,
6.60→6.24, 9.04→8.49 mm respectively).

The all-input ear score remains deliberately unfiltered for quality screening.
It worsens in 819 and e9 because it still scores rejected, impossible targets.
The matched surviving-pair score makes the improvement clear without changing
the evaluation subset between the old and new fit. These are training-input
residuals; they are not independent accuracy measurements. e9's 61 mm torso
error remains unresolved and all three bad sessions remain flagged.

Artifacts: `/tmp/soma-fit/anthropometry/`, with per-session full fits and local
layers under `selfcap-post-ep1-v2/`, plus `comparison.json`. Exo evidence uses
the nearest catalog video frame to 30 s and the nearest fitted mesh sample,
with the same calibrated pinhole; these are standalone local review recordings.

Pixel inspection at 30 s: `e9dd8490-exo02.png` has the fitted head overlapping
the real head. `819a09a4-exo02.png` has the head near the real head but the
torso remains forward of the subject. `0b36cf91-exo02.png` remains visibly
wrong: the torso leans forward and the head is below and beside the real head.
The guards do not repair absolute triangulation errors that retain a plausible
ear span. Do not treat the retained-ear residual improvement as proof of
image alignment. All three sessions still carry a quality flag.
