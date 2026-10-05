# Checked-in test fixtures

Small binaries the tests need on **any** machine, GPU or not. Anything that has
to prove the encoder itself works stays an NVENC-gated encode in
`test_encoding.py`; these files exist so everything downstream of an mp4 — the
`log_video_stream` remux and its retiming — runs without one.

| file | how it was made |
| --- | --- |
| `av1_48f_192x160.mp4` | `ffmpeg -f lavfi -i "color=c=gray:s=192x160:r=24:d=2" -frames:v 48 -c:v av1_nvenc -bf 0 -g 12 -cq 40 -pix_fmt yuv420p -movflags +faststart av1_48f_192x160.mp4` — 48 samples, keyframes every 12, no B-frames (`rr.VideoStream` rejects reordered samples), 2.3 KB. |
| `msd/{index,g2,odyssey}-calibration.json` | Verbatim copies of `M_monado_datasets/<device>/extras/calibration.json` from the upstream HuggingFace dataset `collabora/monado-slam-datasets` (CC-BY 4.0). Nothing is stripped, so `load_calibration` is tested against what upstream really ships. |
| `aria/{gen1-hot3d-P0015_179e1b84,gen2-pilot-clean_0}-calib.json` | Verbatim `calib_json` VRS file tags of HOT3D Aria `P0015_179e1b84/recording.vrs` and Aria Gen2 Pilot `clean_0/video.vrs`. |
| `aria/*-projectaria.json` | What projectaria-tools 2.3.0 `device_calibration_from_json_string` made of each: every FISHEYE624 camera's size, parameters, valid radius, max solid angle and `T_Device_Camera`, every `T_Device_Imu`. Written while the SDK was still in the env; `test_aria_calibration.py` holds `dataforge.aria` to it exactly. |
| `aria/*-readers.json` | What projectaria-tools 2.3.0's VRS data provider read from HOT3D Aria `P0015_179e1b84`, HOT3D Quest 3 `P0003_cae067da` and `P0003_01e416d3`, Aria Gen2 Pilot `clean_0` and `eat_1`, and LaMAria `R_01_easy`: every camera's DEVICE-time frame clock (count, first, last, sha256), every IMU record's clock, valid flags, accel and gyro (sha256 each), and for LaMAria every decoded frame (blake2b-8 of the pixels and shape; sha256 of those hashes). Condensed from the SDK dumps taken while the SDK was still in the env; `conftest.assert_readers_match_projectaria` holds `dataforge.vrs` and `dataforge.aria` to them in asset-gated golden tests (`test_hot3d_real.py`, `test_aria_gen2_pilot_real.py`, `test_aria_vrs.py`). |
