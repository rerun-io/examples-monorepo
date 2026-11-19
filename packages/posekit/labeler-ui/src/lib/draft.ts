// The unsaved fix at one time: per hand side, the person's boxes, the views where the person removed the box, and
// the latest 3D solve. Plain immutable data: every edit returns a new value.

import type { Box, HandSide, SolveResult, TrackFrame, ViewLabel } from './api';
import { SIDES } from './api';
import { roundBox } from './box-edit';
import { camLabel } from './clip';
import { boxKind, type BoxKind } from './track';

export interface SideDraft {
  /** cam -> the person's box. */
  labeled: Record<number, Box>;
  /** Cams where the person removed this side's box (saved as `no_hand`). */
  deleted: number[];
  /** Latest `/api/solve` answer for the labeled boxes, or null before it returns. */
  solve: SolveResult | null;
}

export interface Draft {
  t_ns: number;
  /** Only the sides the person edited. */
  sides: Partial<Record<HandSide, SideDraft>>;
}

/** A side's draft seeded from the person's labels already in the track at that time. */
export function sideFromTrack(frame: TrackFrame | undefined, side: HandSide): SideDraft {
  const draft: SideDraft = { labeled: {}, deleted: [], solve: null };
  for (const b of frame?.boxes ?? []) {
    if (b.side !== side || b.origin !== 'reviewed') continue;
    if (b.box) draft.labeled[b.cam] = b.box;
    else draft.deleted.push(b.cam);
  }
  return draft;
}

export function withBox(s: SideDraft, cam: number, box: Box): SideDraft {
  return {
    ...s,
    labeled: { ...s.labeled, [cam]: box },
    deleted: s.deleted.filter((c) => c !== cam)
  };
}

/** The person removed the box: the hand is not in this view. */
export function withNoHand(s: SideDraft, cam: number): SideDraft {
  const labeled = { ...s.labeled };
  delete labeled[cam];
  return { ...s, labeled, deleted: s.deleted.includes(cam) ? s.deleted : [...s.deleted, cam] };
}

/** The box left this side (re-sided to the other hand): the view is neither labeled nor removed, the solve fills it. */
export function withoutBox(s: SideDraft, cam: number): SideDraft {
  const labeled = { ...s.labeled };
  delete labeled[cam];
  return { ...s, labeled, deleted: s.deleted.filter((c) => c !== cam) };
}

export function labeledViews(s: SideDraft): { cam: number; box: Box }[] {
  return Object.entries(s.labeled).map(([cam, box]) => ({ cam: Number(cam), box: roundBox(box) }));
}

export interface ViewBox {
  side: HandSide;
  box: Box;
  kind: BoxKind;
}

/** What one camera shows at the draft's time: the draft for edited sides, the machine track for the rest. */
export function viewBoxes(
  frame: TrackFrame | undefined,
  draft: Draft | null,
  cam: number
): ViewBox[] {
  const out: ViewBox[] = [];
  for (const side of SIDES) {
    const s = draft?.sides[side];
    const tracked = frame?.boxes.find((b) => b.cam === cam && b.side === side);
    if (s) {
      if (s.labeled[cam]) {
        out.push({ side, box: s.labeled[cam], kind: 'labeled' });
        continue;
      }
      if (s.deleted.includes(cam)) continue;
      const solved = s.solve?.boxes.find((b) => b.cam === cam);
      if (solved) {
        if (solved.box) out.push({ side, box: solved.box, kind: 'derived' });
        continue;
      }
    }
    if (tracked?.box) out.push({ side, box: tracked.box, kind: boxKind(tracked.origin) });
  }
  return out;
}

/** Side of a newly drawn box: the side whose box is missing in this view, else the last chosen side. */
export function drawSide(present: readonly HandSide[], last: HandSide): HandSide {
  const missing = SIDES.filter((s) => !present.includes(s));
  return missing.length === 1 ? missing[0] : last;
}

/**
 * The `/api/labels` entries of one edited side: the person's boxes (`corrected`), the person's removals (`no_hand`),
 * and every other camera from the solve (`accepted` from the rig, or `no_hand` where the sphere is out of view).
 */
export function sideLabels(
  side: HandSide,
  s: SideDraft,
  cams: readonly number[],
  reviewer: string
): ViewLabel[] {
  const labels: ViewLabel[] = [];
  for (const cam of cams) {
    const person = s.labeled[cam];
    const base = { cam, side, reviewer };
    if (person) {
      labels.push({ ...base, status: 'corrected', box: roundBox(person), source: 'person' });
    } else if (s.deleted.includes(cam)) {
      labels.push({ ...base, status: 'no_hand', box: null, source: 'person' });
    } else {
      const solved = s.solve?.boxes.find((b) => b.cam === cam);
      if (!solved) continue;
      labels.push(
        solved.box
          ? { ...base, status: 'accepted', box: roundBox(solved.box), source: 'rig' }
          : { ...base, status: 'no_hand', box: null, source: 'rig' }
      );
    }
  }
  return labels;
}

export interface SolveSummary {
  text: string;
  confidence: string;
  /** What to do when the depth is weak, or null. */
  warning: string | null;
}

export function solveSummary(r: SolveResult): SolveSummary {
  const views = r.labeled.map(camLabel).join(' + ');
  const radius = `r ${(r.radius_m * 100).toFixed(1)} cm`;
  const how =
    r.mode === 'triangulated'
      ? `triangulated from ${views}`
      : r.mode === 'proposals'
        ? `from ${views}, depth from EgoHand proposals`
        : `from ${views}, depth from the hand-size prior`;
  let warning: string | null = null;
  if (r.confidence < 0.5) {
    warning = r.mode === 'prior' ? 'label a second view' : 'label a view on the other side';
  }
  return {
    text: `${how} · ${radius}`,
    confidence: `depth confidence ${r.confidence.toFixed(2)}`,
    warning
  };
}
