import { describe, expect, it } from 'vitest';
import { camLabel, clipRange, parseHash, shortSegment } from './clip';

const grid = (n: number, start = 5_000_000_000): number[] =>
  Array.from({ length: n }, (_, i) => start + i * 200_000_000);

describe('parseHash', () => {
  it('reads sid, from and to', () => {
    expect(parseHash('#sid=robocap__ab__s1&from=20&to=30')).toEqual({
      sid: 'robocap__ab__s1',
      from: 20,
      to: 30
    });
  });

  it('leaves missing or broken keys null', () => {
    expect(parseHash('')).toEqual({ sid: null, from: null, to: null });
    expect(parseHash('#from=abc&to=')).toEqual({ sid: null, from: null, to: null });
  });
});

describe('clipRange', () => {
  it('maps seconds from the first grid time to inclusive indices', () => {
    // 5 fps: 20.0 s is index 100, 30.0 s is index 150.
    expect(clipRange(grid(300), 20, 30)).toEqual({ i0: 100, i1: 150 });
  });

  it('takes the nearest grid time at each edge when the grid drifts', () => {
    // RoboCap: 20.006 s and 30.009 s are the grid times nearest 20 s and 30 s.
    const drift = Array.from({ length: 200 }, (_, i) => Math.round(i * 200_045_000));
    const { i0, i1 } = clipRange(drift, 20, 30);
    expect([i0, i1]).toEqual([100, 150]);
  });

  it('takes the whole segment without from/to', () => {
    expect(clipRange(grid(10), null, null)).toEqual({ i0: 0, i1: 9 });
  });

  it('takes the whole segment when the clip misses the grid', () => {
    expect(clipRange(grid(10), 100, 200)).toEqual({ i0: 0, i1: 9 });
  });
});

describe('names', () => {
  it('shortens a RoboCap segment id', () => {
    expect(shortSegment('robocap__fe6fede545c972fa__s00000013')).toBe('fe6fede5 · s13');
    expect(shortSegment('other')).toBe('other');
  });

  it('pads the camera index', () => {
    expect(camLabel(1)).toBe('cam_01');
  });
});
