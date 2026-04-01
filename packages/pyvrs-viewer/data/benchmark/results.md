# pyvrs-viewer Benchmark Results

| Sequence | Device | VRS (MB) | Records | AV1 Time (s) | AV1 RRD (MB) | Compression | JPEG Time (s) | JPEG RRD (MB) | AV1 Speedup | AV1 Size Reduction | Status |
|----------|--------|----------|---------|--------------|--------------|-------------|---------------|---------------|-------------|-------------------|--------|
| P0002_1464cbdc | quest | 2717 | 7966 | 5.2 | 67 | 40x | 0.9 | 2718 | 0.2x | 40x smaller | OK |
| P0002_210fc0da | quest | 797 | 3720 | 2.4 | 31 | 25x | 0.3 | 800 | 0.1x | 25x smaller | OK |
| P0002_273c2819 | quest | 1526 | 7284 | 4.2 | 61 | 25x | 0.6 | 1300 | 0.1x | 21x smaller | OK |
| P0002_28bcd2d9 | quest | 1286 | 7314 | 4.2 | 62 | 21x | 0.5 | 1292 | 0.1x | 21x smaller | OK |
| P0002_2f137f83 | quest | 1470 | 7322 | 4.1 | 61 | 24x | 0.6 | 1477 | 0.1x | 24x smaller | OK |
| P0001_10a27bf7 | aria | 1658 | 237185 | 7.8 | 104 | 16x | 11.1 | 1638 | 1.4x | 16x smaller | OK |
| P0001_15c4300c | aria | 1774 | 249552 | 8.7 | 110 | 16x | 11.3 | 1753 | 1.3x | 16x smaller | OK |
| P0001_23fa0ee8 | aria | 1536 | 238798 | 8.2 | 105 | 15x | 10.9 | 1513 | 1.3x | 14x smaller | OK |
| P0001_4bf4e21a | aria | 820 | 116150 | 4.5 | 51 | 16x | 5.3 | 790 | 1.2x | 15x smaller | OK |
| P0001_550ea2ac | aria | 1723 | 248383 | 8.9 | 109 | 16x | 11.4 | 1705 | 1.3x | 16x smaller | OK |

## Summary

**AV1 Speedup** = JPEG time / AV1 time (how much slower AV1 is vs JPEG passthrough)
**AV1 Size Reduction** = JPEG RRD / AV1 RRD (how much smaller AV1 output is)
