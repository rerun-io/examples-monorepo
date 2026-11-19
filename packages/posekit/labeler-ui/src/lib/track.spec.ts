import { describe, expect, it } from 'vitest';
import type { TrackBox, TrackFrame, TrackOrigin } from './api';
import { acceptedPrefix, boxKind, fixTimes, isAccepted, nextChunk } from './track';

const frame = (t_ns: number, origins: TrackOrigin[]): TrackFrame => ({
  t_ns,
  boxes: origins.map((origin, i): TrackBox => ({
    cam: Math.floor(i / 2),
    side: i % 2 === 0 ? 'left' : 'right',
    box: origin === 'none' ? null : { x0: 0, y0: 0, x1: 10, y1: 10 },
    origin,
    score: null
  }))
});

describe('accepted prefix', () => {
  it('counts leading frames whose boxes are all reviewed or derived', () => {
    const frames = [
      frame(0, ['reviewed', 'derived', 'reviewed', 'reviewed']),
      frame(1, ['derived', 'derived']),
      frame(2, ['reviewed', 'detected']),
      frame(3, ['reviewed', 'reviewed'])
    ];
    expect(acceptedPrefix(frames)).toBe(2);
  });

  it('stops at an unloaded frame', () => {
    expect(acceptedPrefix([frame(0, ['reviewed']), undefined, frame(2, ['reviewed'])])).toBe(1);
  });

  it('does not accept a frame without boxes or with a no-box machine origin', () => {
    expect(isAccepted(frame(0, []))).toBe(false);
    expect(isAccepted(frame(0, ['reviewed', 'none']))).toBe(false);
  });

  it('is zero for an empty clip', () => {
    expect(acceptedPrefix([])).toBe(0);
  });
});

describe('box kind', () => {
  it('maps origins to the three box styles', () => {
    expect(boxKind('reviewed')).toBe('labeled');
    expect(boxKind('derived')).toBe('derived');
    expect(boxKind('sphere')).toBe('derived');
    expect(boxKind('detected')).toBe('machine');
    expect(boxKind('fused')).toBe('machine');
  });
});

describe('fix times', () => {
  it('keeps the times of interventions only', () => {
    const times = fixTimes([
      { t_ns: 1, kind: 'accept_range' },
      { t_ns: 2, kind: 'move_box' },
      { t_ns: 2, kind: 'draw_box' },
      { t_ns: 3, kind: 'accept_frame' },
      { t_ns: 4, kind: 'no_hand' }
    ]);
    expect([...times].sort()).toEqual([2, 4]);
  });
});

describe('next track chunk', () => {
  it('fetches the chunk under the playhead first, then the look-ahead', () => {
    expect(nextChunk(['missing', 'missing', 'missing', 'missing'], 25, 10, 10)).toBe(2);
    expect(nextChunk(['missing', 'missing', 'loaded', 'missing'], 25, 10, 10)).toBe(3);
  });

  it('then fills the clip from the start, and stops when all are loaded or loading', () => {
    expect(nextChunk(['missing', 'loading', 'loaded', 'loaded'], 25, 10, 10)).toBe(0);
    expect(nextChunk(['loaded', 'loading', 'loaded'], 0, 10, 10)).toBeNull();
  });
});
