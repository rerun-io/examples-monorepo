// The clip under review: parsed from the URL hash, mapped onto the segment's 5 fps grid, and named on screen.

export interface ClipHash {
  sid: string | null;
  from: number | null;
  to: number | null;
}

/** `#sid=<segment_id>&from=20&to=30` (seconds from the segment's first grid time); every key is optional. */
export function parseHash(hash: string): ClipHash {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const num = (key: string): number | null => {
    const value = params.get(key);
    if (value === null || value.trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  };
  return { sid: params.get('sid') || null, from: num('from'), to: num('to') };
}

/** Inclusive grid indices [i0, i1] of the times within [from, to] seconds; the whole grid when none fall inside. */
export function clipRange(
  times_ns: number[],
  from: number | null,
  to: number | null
): { i0: number; i1: number } {
  const whole = { i0: 0, i1: Math.max(times_ns.length - 1, 0) };
  if (times_ns.length === 0) return whole;
  const lo = from ?? -Infinity;
  const hi = to ?? Infinity;
  const start = times_ns[0];
  // Half a grid period of tolerance: the grid drifts off round seconds (30.009 s on RoboCap), and a clip edge takes
  // the nearest grid time.
  const half = times_ns.length > 1 ? (times_ns[1] - start) / 2e9 : 0;
  let i0 = -1;
  let i1 = -1;
  times_ns.forEach((t, i) => {
    const s = (t - start) / 1e9;
    if (s >= lo - half && s < hi + half) {
      if (i0 < 0) i0 = i;
      i1 = i;
    }
  });
  return i0 < 0 ? whole : { i0, i1 };
}

/** Seconds of `t_ns` from the segment's first grid time. */
export function segmentSeconds(t_ns: number, start_ns: number): number {
  return (t_ns - start_ns) / 1e9;
}

export function fmtSeconds(s: number): string {
  return `${s.toFixed(1)} s`;
}

/** `robocap__fe6fede545c972fa__s00000013` -> `fe6fede5 · s13`; other ids are returned unchanged. */
export function shortSegment(sid: string): string {
  const match = /^.*__([0-9a-f]{8})[0-9a-f]*__s0*(\d+)$/.exec(sid);
  return match ? `${match[1]} · s${match[2]}` : sid;
}

export function camLabel(cam: number): string {
  return `cam_${String(cam).padStart(2, '0')}`;
}
