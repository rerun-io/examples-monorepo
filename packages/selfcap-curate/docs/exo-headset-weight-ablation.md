# Cached-label ablation with corrected coverage and physical camera IDs

Source: immutable original S1 RTMW recordings, original Sapiens samples, raw Quest tracks, and cached Sol marks. No new inference or catalog writes.

H/P/S/R/B = headset / phone / Sapiens hand / independent RTMW hand / body-transfer median pixel error. A = headset weight 1.0; C = 0.25; B = phones-only (weight 0.0). Other objective shares and all gate thresholds are unchanged.

| Session | Baseline H/P/S/R/B | A: decision; H/P/S/R/B | C: decision; H/P/S/R/B | B: decision; H/P/S/R/B |
|---|---|---|---|---|
| 027d8215 | 42.4/65.9/5.6/9.8/19.9 | reject; 2.0/1.1/7.3/10.8/17.0 | reject; 5.2/0.9/7.0/10.7/24.3 | reject; 41.0/1.8/6.0/9.9/33.5 |
| 12164e51 | 48.3/89.3/6.6/10.4/36.9 | reject; 3.2/1.8/8.9/12.3/17.0 | reject; 4.6/0.3/8.9/11.5/21.2 | reject; 52.7/2.0/7.5/10.6/47.7 |
| a717934e | 16.0/25.4/7.4/8.8/18.4 | reject; 2.0/1.5/8.2/9.5/21.8 | reject; 3.0/1.5/8.2/9.5/22.3 | reject; 18.3/1.4/8.0/9.6/23.1 |
| e72bf6b0 | 11.2/45.1/7.1/8.8/17.9 | reject; 1.8/2.0/8.8/10.1/23.9 | reject; 2.8/2.5/8.7/10.0/28.5 | reject; 20.4/0.5/8.2/9.6/30.4 |
| 426c5f35 | 33.5/59.3/6.3/8.6/18.0 | reject; 1.4/7.1/7.8/10.2/18.6 | reject; 2.9/1.7/7.6/10.0/20.0 | reject; 15.1/0.6/7.0/9.4/28.8 |
| bc4c0a67 | 25.6/31.4/6.6/9.6/21.0 | reject; 3.1/0.7/7.7/10.3/19.7 | reject; 1.8/0.7/7.7/10.3/19.4 | reject; 13.4/0.5/7.6/10.0/21.6 |
| eeba0fb3 | 21.8/42.0/7.6/9.1/18.1 | reject; 2.3/0.7/8.2/9.6/15.9 | reject; 2.8/0.7/8.0/9.4/16.9 | reject; 18.2/0.1/8.0/9.3/16.9 |
| 44ee4ceb | 27.8/88.6/6.8/9.0/22.8 | reject; 1.6/1.3/8.8/10.9/17.1 | reject; 3.3/0.7/8.4/10.6/16.1 | reject; 20.0/1.3/7.4/9.6/16.8 |
| b910c05f | 24.0/103.8/7.1/8.8/19.5 | reject; 1.7/0.2/9.7/10.7/21.7 | reject; 2.8/0.2/9.2/10.4/20.5 | reject; 28.0/0.1/7.3/8.7/21.3 |
| 44369a66 | 36.8/50.0/7.7/9.6/19.6 | reject; 5.1/2.2/8.7/10.4/22.2 | reject; 4.9/0.7/8.5/10.3/22.0 | reject; 27.5/1.0/8.5/10.0/23.4 |
| 2f65a76f | 33.6/90.3/7.3/10.2/18.9 | accept; 4.0/1.6/8.7/10.7/16.8 | reject; 3.8/1.3/8.5/10.4/16.7 | reject; 22.4/0.8/8.2/10.6/16.4 |
| 8be203b4 | 36.0/85.9/6.7/8.7/32.2 | reject; 2.1/0.6/8.4/9.1/17.9 | reject; 4.0/0.8/8.7/9.7/25.2 | reject; 28.2/2.9/8.7/10.2/74.2 |
| 20615ce6 | 17.7/80.2/6.6/8.9/15.8 | reject; 4.4/1.1/7.5/10.0/19.3 | reject; 5.5/1.1/7.4/9.8/17.2 | reject; 16.5/0.4/7.1/9.3/16.8 |
| faa525fa | 34.3/67.7/7.3/9.7/15.9 | accept; 1.1/1.2/8.4/10.4/12.6 | accept; 2.5/0.7/8.1/10.1/13.8 | accept; 17.4/0.5/7.9/9.9/15.6 |
| 34a9a78a | 40.6/52.6/5.6/9.4/24.2 | reject; 3.5/2.1/8.8/11.0/14.5 | reject; 6.4/1.7/8.4/10.5/14.0 | accept; 32.6/1.3/7.1/10.0/18.1 |
| 9ee28105 | 26.1/121.4/7.3/9.9/16.8 | reject; 2.4/1.5/9.6/10.4/15.9 | reject; 6.6/1.0/8.9/9.9/18.1 | reject; 23.5/0.8/8.1/10.0/17.3 |
| de80d0c2 | 25.4/109.3/7.2/10.3/18.6 | reject; 1.6/1.9/9.9/11.1/14.7 | reject; 3.7/0.6/9.3/10.8/14.5 | reject; 23.5/0.8/8.0/10.4/15.2 |
| 0dd4108b | 30.9/93.1/8.3/9.1/20.0 | reject; 3.0/0.8/9.2/10.5/15.3 | reject; 4.0/1.5/8.9/10.3/15.6 | reject; 19.3/0.7/8.2/9.4/18.0 |
| 4e77673b | 33.0/73.1/6.0/7.4/22.8 | reject; 3.7/0.9/7.5/8.5/17.7 | reject; 4.8/0.7/7.4/8.3/17.8 | accept; 17.7/1.1/7.2/8.0/20.1 |
| 9277c713 | 23.4/102.4/6.7/9.7/25.0 | reject; 3.0/0.5/8.2/10.6/17.7 | reject; 3.0/0.6/8.1/10.6/17.1 | accept; 15.2/0.6/7.6/10.3/17.7 |
| d2780560 | 26.7/112.5/6.4/8.8/20.1 | reject; 2.6/2.2/8.3/9.7/17.2 | reject; 3.6/1.7/7.9/9.3/18.6 | reject; 18.9/0.8/7.5/9.2/22.8 |
| 26e95da8 | 29.4/49.1/6.6/9.0/22.5 | reject; 2.5/1.0/8.7/9.3/24.5 | reject; 3.3/0.7/8.6/9.2/24.8 | reject; 11.5/0.7/8.2/9.1/25.8 |
| 998e7457 | 36.5/107.2/6.9/11.0/19.2 | reject; 2.2/0.5/9.5/12.2/17.6 | reject; 4.1/0.9/9.2/12.0/17.4 | accept; 30.2/0.5/8.1/11.7/17.7 |
| d31bf1da | 38.6/104.5/8.4/9.6/25.3 | accept; 4.1/0.7/9.6/10.0/15.4 | accept; 4.7/0.5/9.4/9.9/15.1 | accept; 33.1/0.5/8.8/10.4/15.1 |
| f4de787a | 25.6/102.0/8.4/11.3/20.1 | accept; 2.0/1.7/10.2/12.4/16.6 | accept; 3.2/0.6/9.8/12.2/18.9 | reject; 17.1/1.1/9.1/12.2/21.6 |
| 5cb1e14c | 20.7/196.8/10.2/13.0/24.3 | reject; 1.3/134.8/11.9/12.8/19.1 | reject; 2.3/0.8/22.2/23.5/127.2 | reject; 50.0/0.4/12.7/16.2/123.3 |
| fe05fe3e | 21.5/51.3/8.6/10.8/21.9 | reject; 3.4/1.1/10.4/11.4/15.9 | accept; 8.8/1.9/10.2/11.2/18.0 | reject; 16.2/0.9/10.0/11.0/23.5 |
| 1753192b | 10.9/119.0/12.2/14.3/18.6 | accept; 3.7/1.8/13.4/15.4/16.8 | accept; 3.8/1.4/13.3/15.3/17.4 | reject; 9.4/0.7/12.8/14.7/19.5 |
| 3455a1f1 | 25.0/89.1/9.0/13.0/31.1 | accept; 1.5/1.9/10.6/13.7/17.8 | reject; 3.6/1.4/10.8/13.6/15.8 | accept; 16.6/0.5/10.3/13.2/22.4 |
| 5adb9b92 | 16.0/146.3/11.1/17.0/56.8 | reject; 2.6/0.7/13.1/14.8/15.2 | reject; 2.4/1.0/12.7/14.6/16.3 | reject; 17.0/0.8/12.3/14.5/17.6 |
| 021d2e8b | 23.3/93.8/7.3/10.6/19.8 | reject; 2.1/1.3/9.0/11.7/16.7 | accept; 4.0/1.3/8.5/11.3/18.4 | reject; 21.2/0.5/7.9/10.7/21.5 |
| 429f94b5 | 23.7/45.8/5.7/8.7/23.6 | reject; 1.3/1.1/7.0/8.8/17.4 | reject; 2.2/1.0/6.9/8.6/16.9 | reject; 20.2/0.8/6.1/9.0/16.5 |

Complete: 32/32. Accept counts: A 6, C 6, B 7.

Decision flips relative to original pipeline: {'a': ['4e77673b', 'd2780560', 'f4de787a', '1753192b', '3455a1f1', '5adb9b92'], 'b': ['2f65a76f', '34a9a78a', '9277c713', 'd2780560', '998e7457', '3455a1f1', '5adb9b92'], 'c': ['2f65a76f', '4e77673b', 'd2780560', 'f4de787a', 'fe05fe3e', '1753192b', '5adb9b92', '021d2e8b']}

## Decision

Keep A (headset share 1.0) as the default. Across 32 frozen sessions A accepts
6, C accepts 6, and B accepts 7. C exchanges two acceptances for two rejections;
B gains four but loses three. Neither alternative clearly wins.

Median of per-session candidate medians, in pixels:

| Variant | Headset | Phones | Sapiens | RTMW | Body transfer |
|---|---:|---:|---:|---:|---:|
| A, headset 1.0 | 2.34 | 1.21 | 8.78 | 10.55 | 17.16 |
| C, headset 0.25 | 3.67 | 0.88 | 8.53 | 10.33 | 17.89 |
| B, phones only | 19.64 | 0.75 | 7.98 | 10.00 | 20.69 |

No acceptance thresholds changed: independent RTMW +1.5 px; Sapiens +2.0 px;
body must improve; headset/phone medians must improve by >=2 px and >=10%;
all solves must converge and deployed labels must stay within the existing
2 px fold tolerance. These checks remain independent of the objective shares.

## Corrected failures

The old 10-second split trained fe05fe3e fold 0 on only [30, 0, 0, 0]
hand correspondences per camera and 021d2e8b on [0, 0, 0, 0]. Cameras moved
122–306 mm and 465–1146 mm respectively to satisfy optical anchors without
hand constraints. In inspected fe05 frames, headset and phone marks had plausible
identities; this was not a gross phone swap.

The new split alternates usable hand frames separately per camera and checks
that both folds have observations. Dense hand scoring follows the nearest
sparse frame's held-out assignment. Body frames are deduplicated, then split
with explicit per-camera visibility coverage. Missing coverage fails closed.

Borrowed camera 50 is physically LEFT even when stored in slot 1. The new
physical-target mapping uses its resolved factory transform for quest0 and
omits the unavailable physical-right target. Crops use the resolved camera
orientation when picking front-facing frames. Quest/OAK transforms and hands
remain unchanged.

With corrected folds/identities, A yields:
- fe05fe3e: headset 3.39, phones 1.07, Sapiens 10.37, RTMW 11.42, body 15.91 px;
  old Sapiens/RTMW were 33.56/19.88. One fold still fails convergence (and the
  Sapiens guard narrowly fails). C accepts, at headset 8.82 px.
- 021d2e8b: headset 2.10, phones 1.29, Sapiens 9.03, RTMW 11.67, body 16.66 px;
  old Sapiens/RTMW/body were 89.85/29.38/73.22. One fold still fails convergence.
  C accepts, at headset 3.99 px.

The earlier replay against the current catalog RTMW layer was discarded:
S2 had replaced raw S1 observations with smoothed output. Final scores use
immutable original S1 RRDs (or exact raw S1 Parquet reconstruction), frozen under
/tmp/exo-headset-ablation-raw/frozen-rtmw. No inference or catalog writes occurred.
Full typed experiment records are /tmp/exo-headset-ablation-fixed/experiments.json.

## Layout and operational checks

b910c05f has exo rigs 0–2, OAK rig 3, Quest rig 4. Its 6,396 head poses were read
from the correct discovered Quest rig. There is no hardcoded four-phone layout
in this tool, and that session does not need relabelling for a rig-index bug.

1c094794 contains the Hammer key, but its camera value is null; the other three
camera poses exist. 12617e39 has no exo camera poses. Missing/null cameras now
receive no crops or constraints; remaining calibrated cameras retain their
original rig IDs in refinement output. Fewer than two calibrated cameras gives
`skipped`, with a reason and no calibration output. A cached-label d31 test with
rig 1 removed accepted and retained exactly rigs (0, 2, 3); 12617e39 skipped.

ef6b7189 failed with "Selected model is at capacity". Codex stderr is retained;
the tool now retries one transient service failure and then falls back. Permanent
errors do not retry. Tests cover both cases without invoking Sol.

Relabel ef6b7189 and 1c094794. Re-run 12617e39 to record skipped. Borrowed-cam50
sessions 021d2e8b and 429f94b5 need fresh packets to benefit from corrected physical
lens selection; healthy sessions can reuse cached labels when re-scoring folds.
