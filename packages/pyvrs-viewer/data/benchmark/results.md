# pyvrs-viewer Benchmark Results

| Sequence | Device | VRS (MB) | Records | AV1 Time (s) | AV1 RRD (MB) | Compression | JPEG Time (s) | JPEG RRD (MB) | AV1 Speedup | AV1 Size Reduction | Status |
|----------|--------|----------|---------|--------------|--------------|-------------|---------------|---------------|-------------|-------------------|--------|
| P0002_1464cbdc | quest | 2717 | 7966 | 5.1 | 67 | 40x | 0.9 | 2733 | 0.2x | 41x smaller | OK |
| P0002_210fc0da | quest | 797 | 3720 | 2.3 | 31 | 25x | 0.3 | 800 | 0.1x | 26x smaller | OK |
| P0002_273c2819 | quest | 1526 | 7284 | 4.0 | 61 | 25x | 0.6 | 1534 | 0.2x | 25x smaller | OK |
| P0002_28bcd2d9 | quest | 1286 | 7314 | 4.0 | 61 | 21x | 0.6 | 1293 | 0.2x | 21x smaller | OK |
| P0002_2f137f83 | quest | 1470 | 7322 | 3.9 | 61 | 24x | 0.6 | 1478 | 0.2x | 24x smaller | OK |
| P0001_10a27bf7 | aria | 1658 | 237185 | 15.1 | 121 | 14x | 10.9 | 1638 | 0.7x | 14x smaller | OK |
| P0001_15c4300c | aria | 1774 | 249552 | 16.0 | 129 | 14x | 11.5 | 1753 | 0.7x | 14x smaller | OK |
| P0001_23fa0ee8 | aria | 1536 | 238798 | 15.0 | 122 | 13x | 10.8 | 1516 | 0.7x | 12x smaller | OK |
| P0001_4bf4e21a | aria | 820 | 116150 | 7.7 | 59 | 14x | 5.1 | 790 | 0.7x | 13x smaller | OK |
| P0001_550ea2ac | aria | 1723 | 248383 | 15.8 | 128 | 13x | 14.1 | 1696 | 0.9x | 13x smaller | OK |

## Summary

**AV1 Speedup** = JPEG time / AV1 time (how much slower AV1 is vs JPEG passthrough)
- Quest: AV1 is ~5-7x slower than JPEG passthrough (4s vs 0.6s), but JPEG produces unusable 1.5GB+ files
- Aria: AV1 is only ~1.1-1.5x slower than JPEG passthrough — IMU record processing dominates both modes

**AV1 Size Reduction** = JPEG RRD / AV1 RRD (how much smaller AV1 output is)
- Quest: **21-41x smaller** (61-67 MB vs 800-2733 MB)
- Aria: **12-14x smaller** (59-129 MB vs 790-1696 MB)

**Tradeoff**: A few extra seconds of processing gives 13-41x smaller files that load instantly in the Rerun viewer.
