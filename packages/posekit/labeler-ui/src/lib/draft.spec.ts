import { describe, expect, it } from 'vitest';
import type { SolveResult, TrackFrame } from './api';
import {
  drawSide,
  sideFromTrack,
  sideLabels,
  solveSummary,
  viewBoxes,
  withBox,
  withNoHand,
  withoutBox,
  type SideDraft
} from './draft';

const A = { x0: 10, y0: 20, x1: 110, y1: 120 };
const B = { x0: 500, y0: 500, x1: 600, y1: 640 };
const C = { x0: 1, y0: 2, x1: 3.04, y1: 4.06 };

const solve = (over: Partial<SolveResult> = {}): SolveResult => ({
  side: 'left',
  mode: 'triangulated',
  confidence: 0.99,
  radius_m: 0.074,
  labeled: [1, 2],
  boxes: [
    { cam: 0, box: B },
    { cam: 1, box: A },
    { cam: 2, box: A },
    { cam: 3, box: C },
    { cam: 4, box: null },
    { cam: 5, box: null }
  ],
  ...over
});

const frame: TrackFrame = {
  t_ns: 7,
  boxes: [
    { cam: 0, side: 'left', box: B, origin: 'detected', score: 0.9 },
    { cam: 0, side: 'right', box: A, origin: 'reviewed', score: null },
    { cam: 1, side: 'left', box: A, origin: 'reviewed', score: null },
    { cam: 1, side: 'right', box: null, origin: 'reviewed', score: null },
    { cam: 2, side: 'left', box: C, origin: 'derived', score: null }
  ]
};

describe('label post for a solve', () => {
  it('posts labeled views as corrected, removals as no_hand, the rest from the rig', () => {
    let s: SideDraft = { labeled: {}, deleted: [], solve: null };
    s = withBox(s, 1, A);
    s = withBox(s, 2, A);
    s = withNoHand(s, 5);
    s = { ...s, solve: solve() };
    expect(sideLabels('left', s, [0, 1, 2, 3, 4, 5], 'pablo')).toEqual([
      { cam: 0, side: 'left', reviewer: 'pablo', status: 'accepted', box: B, source: 'rig' },
      { cam: 1, side: 'left', reviewer: 'pablo', status: 'corrected', box: A, source: 'person' },
      { cam: 2, side: 'left', reviewer: 'pablo', status: 'corrected', box: A, source: 'person' },
      {
        cam: 3,
        side: 'left',
        reviewer: 'pablo',
        status: 'accepted',
        box: { x0: 1, y0: 2, x1: 3, y1: 4.1 },
        source: 'rig'
      },
      { cam: 4, side: 'left', reviewer: 'pablo', status: 'no_hand', box: null, source: 'rig' },
      { cam: 5, side: 'left', reviewer: 'pablo', status: 'no_hand', box: null, source: 'person' }
    ]);
  });

  it('posts only the person labels when there is no solve', () => {
    const s = withNoHand(withBox({ labeled: {}, deleted: [], solve: null }, 3, A), 4);
    expect(sideLabels('right', s, [0, 1, 2, 3, 4, 5], 'p').map((l) => [l.cam, l.status])).toEqual([
      [3, 'corrected'],
      [4, 'no_hand']
    ]);
  });
});

describe('side drafts', () => {
  it('seeds from the reviewed boxes of the track', () => {
    expect(sideFromTrack(frame, 'left')).toEqual({ labeled: { 1: A }, deleted: [], solve: null });
    expect(sideFromTrack(frame, 'right')).toEqual({ labeled: { 0: A }, deleted: [1], solve: null });
  });

  it('labeling a removed view clears the removal, and the reverse', () => {
    const s = withBox(withNoHand(sideFromTrack(frame, 'left'), 3), 3, B);
    expect(s.deleted).toEqual([]);
    expect(s.labeled[3]).toEqual(B);
    expect(withNoHand(s, 3).labeled[3]).toBeUndefined();
    expect(withoutBox(s, 3)).toEqual({ labeled: { 1: A }, deleted: [], solve: null });
  });
});

describe('view boxes', () => {
  it('shows the machine track with its origin style without a draft', () => {
    expect(viewBoxes(frame, null, 0)).toEqual([
      { side: 'left', box: B, kind: 'machine' },
      { side: 'right', box: A, kind: 'labeled' }
    ]);
    expect(viewBoxes(frame, null, 2)).toEqual([{ side: 'left', box: C, kind: 'derived' }]);
  });

  it('shows labeled, then solved, then tracked boxes for an edited side', () => {
    const left = { labeled: { 1: B }, deleted: [0], solve: solve() };
    const draft = { t_ns: 7, sides: { left } };
    expect(viewBoxes(frame, draft, 1)).toEqual([{ side: 'left', box: B, kind: 'labeled' }]);
    expect(viewBoxes(frame, draft, 0)).toEqual([{ side: 'right', box: A, kind: 'labeled' }]);
    expect(viewBoxes(frame, draft, 3)).toEqual([{ side: 'left', box: C, kind: 'derived' }]);
    expect(viewBoxes(frame, draft, 4)).toEqual([]);
  });

  it('keeps the tracked box while the solve is pending', () => {
    const draft = { t_ns: 7, sides: { left: { labeled: {}, deleted: [], solve: null } } };
    expect(viewBoxes(frame, draft, 2)).toEqual([{ side: 'left', box: C, kind: 'derived' }]);
  });
});

describe('draw side', () => {
  it('draws the side whose box is missing', () => {
    expect(drawSide(['left'], 'left')).toBe('right');
    expect(drawSide(['right'], 'right')).toBe('left');
  });

  it('draws the last chosen side when both or neither are present', () => {
    expect(drawSide([], 'right')).toBe('right');
    expect(drawSide(['left', 'right'], 'left')).toBe('left');
  });
});

describe('solve summary', () => {
  it('describes a triangulation', () => {
    expect(solveSummary(solve())).toEqual({
      text: 'triangulated from cam_01 + cam_02 · r 7.4 cm',
      confidence: 'depth confidence 0.99',
      warning: null
    });
  });

  it('asks for another view when the depth is weak', () => {
    expect(solveSummary(solve({ confidence: 0.3 })).warning).toBe('label a view on the other side');
    const prior = solveSummary(solve({ mode: 'prior', labeled: [1], confidence: 0.2 }));
    expect(prior.text).toBe('from cam_01, depth from the hand-size prior · r 7.4 cm');
    expect(prior.warning).toBe('label a second view');
  });
});
