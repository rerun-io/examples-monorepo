# Benchmarks

Everything measured about slam-rs, with the commands that produced it. The
README shows the chart and the summary; this page holds the rows.

## Every Monado SLAM Dataset recording

Fast profile, GPU lane, RTX 5090, one pass, decode `cpu_gray8_dav1d_1thread`.
ATE is RMSE in centimetres against the catalog ground truth after rigid SE(3)
alignment with estimate-driven association and scale fixed at one. The Basalt
column is Table IV of [the MSD paper](https://arxiv.org/abs/2508.00088)
(Basalt, multi-camera build, causal), same units, its own alignment and its own
decode: a reference point, not a paired measurement. The tracker column is the
median `Vio.track` call inside the full replay, unpinned, with decode
interleaved; the isolated number is about a third lower (see below). Produced
by the gate's measurement over every catalog segment.

<!-- msd-sweep:start -->
Measured 2026-09-16 on `215ad203`, core `40c7ab243c22`, decode `cpu_gray8_dav1d_1thread`.

| dataset | recordings | slam-rs median ATE cm | Basalt (paper) median ATE cm | lost framesets | slam-rs lower on |
|---|---:|---:|---:|---:|---:|
| msd-index | 33 | 20.14 | 19.80 | 0 | 16 / 33 |
| msd-g2 | 15 | 8.53 | 7.00 | 0 | 8 / 15 |
| msd-odyssey | 16 | 7.57 | 6.05 | 0 | 10 / 16 |
| all | 64 | 10.97 | 11.20 | 0 | 34 / 64 |

<details>
<summary>Every recording: slam-rs ATE, the tracker call inside the replay, and the paper's Basalt ATE</summary>

#### msd-index (Valve Index, 2 cameras)

| recording | framesets | tracked / lost | slam-rs ATE cm | tracker ms | Basalt (paper) ATE cm |
|---|---:|---:|---:|---:|---:|
| MIO01_hand_puncher_1 | 7855 | 7855 / 0 | 74.06 | 2.84 | 62.0 |
| MIO02_hand_puncher_2 | 4706 | 4706 / 0 | 134.42 | 2.78 | 117.7 |
| MIO03_hand_shooter_easy | 6101 | 6101 / 0 | 9.77 | 2.84 | 9.5 |
| MIO04_hand_shooter_hard | 6119 | 6119 / 0 | 22.63 | 2.76 | 20.6 |
| MIO05_inspect_easy | 6613 | 6613 / 0 | 3.62 | 2.82 | 3.4 |
| MIO06_inspect_hard | 5123 | 5123 / 0 | 8.41 | 2.89 | 4.9 |
| MIO07_mapping_easy | 4095 | 4095 / 0 | 2.11 | 2.78 | 2.3 |
| MIO08_mapping_hard | 1517 | 1517 / 0 | 5.04 | 2.67 | 5.7 |
| MIO09_short_1_updown | 186 | 186 / 0 | 0.62 | 2.72 | 0.6 |
| MIO10_short_2_panorama | 412 | 412 / 0 | 1.55 | 1.94 | 1.5 |
| MIO11_short_3_backandforth | 590 | 590 / 0 | 2.75 | 2.37 | 2.4 |
| MIO12_moving_screens | 19163 | 19163 / 0 | 44.62 | 2.88 | 43.1 |
| MIO13_moving_person | 20227 | 20227 / 0 | 81.53 | 2.83 | 112.8 |
| MIO14_moving_props | 22117 | 22117 / 0 | 6.01 | 2.82 | 5.9 |
| MIO15_moving_person_props | 13545 | 13545 / 0 | 57.34 | 2.81 | 81.3 |
| MIO16_moving_screens_person_props | 14304 | 14304 / 0 | 49.72 | 2.86 | 53.8 |
| MIPB01_beatsaber_100bills_360_normal | 11764 | 11764 / 0 | 25.30 | 2.93 | 27.7 |
| MIPB02_beatsaber_crabrave_360_hard | 11945 | 11945 / 0 | 21.14 | 2.93 | 23.5 |
| MIPB03_beatsaber_countryrounds_360_expert | 20576 | 20576 / 0 | 20.89 | 2.92 | 19.1 |
| MIPB04_beatsaber_fitbeat_hard | 9899 | 9899 / 0 | 8.63 | 2.90 | 10.5 |
| MIPB05_beatsaber_fitbeat_360_expert | 9208 | 9208 / 0 | 5.17 | 2.94 | 4.4 |
| MIPB06_beatsaber_fitbeat_expertplus_1 | 8742 | 8742 / 0 | 6.03 | 2.87 | 4.8 |
| MIPB07_beatsaber_fitbeat_expertplus_2 | 8105 | 8105 / 0 | 4.92 | 2.70 | 6.2 |
| MIPB08_beatsaber_long_session_1 | 118279 | 118279 / 0 | 62.05 | 2.51 | 63.0 |
| MIPP01_pistolwhip_blackmagic_hard | 19057 | 19057 / 0 | 44.97 | 2.32 | 45.5 |
| MIPP02_pistolwhip_lilith_hard | 12772 | 12772 / 0 | 23.16 | 2.30 | 24.1 |
| MIPP03_pistolwhip_requiem_hard | 14555 | 14555 / 0 | 17.64 | 2.26 | 26.1 |
| MIPP04_pistolwhip_revelations_hard | 14287 | 14287 / 0 | 22.91 | 1.94 | 28.7 |
| MIPP05_pistolwhip_thefall_hard_2pistols | 11670 | 11670 / 0 | 20.12 | 1.99 | 18.3 |
| MIPP06_pistolwhip_thegrave_hard | 22183 | 22183 / 0 | 25.78 | 2.43 | 28.3 |
| MIPT01_thrillofthefight_setup | 19064 | 19064 / 0 | 11.52 | 2.83 | 10.7 |
| MIPT02_thrillofthefight_fight_1 | 29145 | 29145 / 0 | 20.14 | 2.84 | 19.8 |
| MIPT03_thrillofthefight_fight_2 | 31577 | 31577 / 0 | 39.27 | 2.84 | 40.0 |

#### msd-g2 (HP Reverb G2, 4 cameras)

| recording | framesets | tracked / lost | slam-rs ATE cm | tracker ms | Basalt (paper) ATE cm |
|---|---:|---:|---:|---:|---:|
| MGO01_low_light | 4255 | 4255 / 0 | 39.83 | 2.84 | 68.0 |
| MGO02_hand_puncher | 4724 | 4724 / 0 | 42.45 | 2.81 | 55.6 |
| MGO03_hand_shooter_easy | 4863 | 4863 / 0 | 13.49 | 2.87 | 14.5 |
| MGO04_hand_shooter_hard | 4363 | 4363 / 0 | 26.02 | 2.83 | 26.2 |
| MGO05_inspect_easy | 4086 | 4086 / 0 | 2.31 | 3.04 | 3.0 |
| MGO06_inspect_hard | 4045 | 4045 / 0 | 8.53 | 2.88 | 11.1 |
| MGO07_mapping_easy | 1596 | 1596 / 0 | 2.37 | 3.04 | 2.1 |
| MGO08_mapping_hard | 746 | 746 / 0 | 2.67 | 2.57 | 2.7 |
| MGO09_short_1_updown | 107 | 107 / 0 | 0.98 | 2.78 | 0.8 |
| MGO10_short_2_panorama | 400 | 400 / 0 | 0.85 | 2.66 | 0.8 |
| MGO11_short_3_backandforth | 539 | 539 / 0 | 2.30 | 2.64 | 1.7 |
| MGO12_freemovement_long_session | 76438 | 76438 / 0 | 65.36 | 2.88 | 61.1 |
| MGO13_sudden_movements | 3735 | 3735 / 0 | 77.17 | 2.87 | 68.3 |
| MGO14_flickering_light | 2887 | 2887 / 0 | 8.60 | 2.87 | 7.0 |
| MGO15_seated_screen | 23915 | 23915 / 0 | 1.99 | 2.63 | 5.5 |

#### msd-odyssey (Samsung Odyssey+, 2 cameras)

| recording | framesets | tracked / lost | slam-rs ATE cm | tracker ms | Basalt (paper) ATE cm |
|---|---:|---:|---:|---:|---:|
| MOO01_hand_puncher_1 | 4706 | 4706 / 0 | 29.46 | 1.56 | 28.1 |
| MOO02_hand_puncher_2 | 5404 | 5404 / 0 | 23.26 | 1.56 | 23.8 |
| MOO03_hand_shooter_easy | 4415 | 4415 / 0 | 16.65 | 1.57 | 17.6 |
| MOO04_hand_shooter_hard | 4406 | 4406 / 0 | 9.80 | 1.53 | 6.5 |
| MOO05_inspect_easy | 3014 | 3014 / 0 | 1.77 | 1.62 | 1.9 |
| MOO06_inspect_hard | 4171 | 4171 / 0 | 4.56 | 1.63 | 5.6 |
| MOO07_mapping_easy | 1237 | 1237 / 0 | 1.00 | 1.61 | 1.3 |
| MOO08_mapping_hard | 592 | 592 / 0 | 5.34 | 1.45 | 2.8 |
| MOO09_short_1_updown | 147 | 147 / 0 | 0.34 | 1.57 | 0.4 |
| MOO10_short_2_panorama | 274 | 274 / 0 | 1.36 | 1.43 | 1.0 |
| MOO11_short_3_backandforth | 405 | 405 / 0 | 1.80 | 1.41 | 1.9 |
| MOO12_freemovement_long_session | 72810 | 72810 / 0 | 65.10 | 1.60 | 67.4 |
| MOO13_sudden_movements | 4403 | 4403 / 0 | 50.37 | 1.52 | 50.1 |
| MOO14_flickering_light | 5026 | 5026 / 0 | 10.42 | 1.57 | 11.3 |
| MOO15_seated_screen | 19380 | 19380 / 0 | 273.64 | 1.33 | 81.5 |
| MOO16_still | 20082 | 20082 / 0 | 0.55 | 1.28 | 3.4 |

</details>
<!-- msd-sweep:end -->

## Where the time goes

A catalog replay is decode-bound: 73–75 % of samples in dav1d and the gray8 reformat,
17–18 % in the tracker, 6 % in Python glue (py-spy over `MIO07` and `MGO07`).

Inside `Vio.track` the fast profile is bimodal. Stage timers from `.npz` dumps, one
pinned core, three rounds pooled, first 60 framesets dropped, milliseconds:

| stage | MIO10 median / mean / p95 | MGO07 median / mean / p95 |
|---|---|---|
| `track` | 1.19 / 1.80 / 6.29 | 1.94 / 2.87 / 8.09 |
| `frontend_track` (temporal KLT, GPU round trip) | 0.61 / 0.66 / 0.85 | 0.87 / 1.00 / 1.37 |
| `frontend_stereo` | 0.23 / 0.23 / 0.58 | 0.48 / 0.55 / 1.05 |
| `frontend_pyramid` + `detect` + `imu` | 0.07 / 0.07 / 0.09 | 0.15 / 0.16 / 0.17 |
| `measure` (estimator) | 0.15 / 0.73 / 5.44 | 0.26 / 1.07 / 6.35 |
| of which `optimize` (joint solve, 14 % of framesets) | 0.04 / 0.62 / 5.29 | 0.08 / 0.86 / 5.92 |

The median is the frontend round trip: 86 % of framesets never solve the window, and
per frameset the Vulkan trace shows 1.8 semaphore waits (0.27 ms), 5.5 submits and
1.3 memory allocations. The mean and p95 are the keyframe solve on the CPU (`solver`
3.1–3.2 ms, `linearize` 1.5–2.0 ms at p95). The per-recording tracker column above is
the same call inside a full replay, unpinned with decode interleaved, and reads about
1.5× the isolated number.

## Fast versus reference profile

Tracker call on the 5090 GPU lane, median after the first 60 framesets, three
interleaved rounds. cuVSLAM is NVIDIA's tracker in offline Inertial mode on the same
frames; its four-camera mode runs without the IMU, so that cell is blank.

| clip | cameras | length | reference: ms / cm | fast: ms / cm | cuVSLAM: ms / cm |
|---|---:|---:|---|---|---|
| `MIO10_short_2_panorama` (the smoke segment) | 2 | 7.6 s | 5.12 / 1.50 | 1.38 / 1.55 | 1.20 / 4.00 |
| `MIO11_short_3_backandforth` | 2 | 11 s | 4.73 / 2.47 | 1.35 / 2.76 | 1.04 / 2.54 |
| `MIO07_mapping_easy` | 2 | 76 s | 5.7 / 2.08 | 1.39 / 2.10 | 1.00 / 1.77 |
| `MGO07_mapping_easy` | 4 | 53 s | 10.2 / 2.29 | 2.10 / 2.37 | — |

Over the whole catalog (S32, 2026-09-10) the fast profile is within its 10 % band of
the reference on 51 of 64 recordings, more accurate on 32, loses no frameset, and its
tracker call is 2.0–2.7× shorter at the median.

## Other hosts

Fast-profile tracker medians in milliseconds for `MIO10` / `MIO07` / `MGO07`, GPU
against CPU on the same host. The 5090 rows are the `benchmarks.toml` baselines; GB10 and
M4 are the median of three matched runs from S36. Different sessions, not a
cross-machine budget.

| device | backend | GPU vs CPU fast medians, ms | CPU/GPU | ≥1.2x, accuracy in band |
|---|---|---|---|---|
| RTX 5090, x86-64 | Vulkan | 2.02 / 2.70 / 3.08 vs 5.60 / 5.94 / 9.76 | 2.77x / 2.20x / 3.17x | pass; ten-clip ratios span 2.0–3.2x |
| GB10 (Spark), aarch64 | Vulkan | 2.88 / 3.07 / 4.75 vs 5.35 / 5.42 / 9.00 | 1.86x / 1.76x / 1.90x | pass |
| Apple M4 (Mac mini) | Metal | 4.59 / 4.76 / 5.97 vs 4.93 / 5.06 / 8.48 | 1.07x / 1.06x / 1.42x | MIO10 and MIO07 miss; MGO07 passes |
| RTX 3060, x86-64 | Vulkan | not re-run since the S32 tip; box needs a driver reboot | — | not measured |

The Metal lane's two sleeps are fixed (wgpu 30 replaces the 1 ms completion poll, the
CubeCL patch parks the idle worker); the Mac's two-camera clips still need about
0.5 ms to meet the 1.2× margin. See
[S36](design-notes.md#decision-references) and the
[host baseline decisions](design-notes.md#decision-references).
