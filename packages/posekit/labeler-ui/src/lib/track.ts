// The machine track of the clip: which times are accepted, how each box is drawn, which chunk to fetch next.

import type { ActionEvent, TrackFrame, TrackOrigin } from './api';

/** How a box is drawn: a person's (solid, thick), the 3D sphere's (dashed, `3D` tag) or the machine's (solid, thin). */
export type BoxKind = 'labeled' | 'derived' | 'machine';

export function boxKind(origin: TrackOrigin): BoxKind {
  if (origin === 'reviewed') return 'labeled';
  if (origin === 'derived' || origin === 'sphere') return 'derived';
  return 'machine';
}

/** A time is accepted when every (cam, side) box there is a person's or the 3D sphere's. */
export function isAccepted(frame: TrackFrame | undefined): boolean {
  if (!frame || frame.boxes.length === 0) return false;
  return frame.boxes.every((b) => b.origin === 'reviewed' || b.origin === 'derived');
}

/** Number of leading accepted frames; an unloaded frame ends the prefix. */
export function acceptedPrefix(frames: readonly (TrackFrame | undefined)[]): number {
  let n = 0;
  while (n < frames.length && isAccepted(frames[n])) n++;
  return n;
}

const PASSIVE_ACTIONS = new Set(['accept_range', 'accept_frame']);

/** Grid times with at least one intervention (a saved fix), from the segment's action log. */
export function fixTimes(actions: readonly ActionEvent[]): Set<number> {
  return new Set(actions.filter((a) => !PASSIVE_ACTIONS.has(a.kind)).map((a) => a.t_ns));
}

export type ChunkState = 'missing' | 'loading' | 'loaded';

/**
 * The next track chunk to fetch: first the chunks under the playhead and the next `lookahead` frames, then the
 * first missing one of the clip. `null` when nothing is missing.
 */
export function nextChunk(
  states: readonly ChunkState[],
  current: number,
  chunkSize: number,
  lookahead: number
): number | null {
  const first = Math.floor(current / chunkSize);
  const last = Math.min(Math.floor((current + lookahead) / chunkSize), states.length - 1);
  for (let c = first; c <= last; c++) if (states[c] === 'missing') return c;
  const any = states.indexOf('missing');
  return any < 0 ? null : any;
}
