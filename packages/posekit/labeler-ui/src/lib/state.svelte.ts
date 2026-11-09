// The labeler's one store: the clip, the playhead, the machine track, the unsaved fix and the actions on them.

import { api, frameUrl, SIDES } from './api';
import type { Box, HandSide, SegmentInfo, TrackFrame } from './api';
import { clipRange, fmtSeconds, parseHash, segmentSeconds } from './clip';
import {
  labeledViews,
  sideFromTrack,
  sideLabels,
  viewBoxes,
  withBox,
  withNoHand,
  type Draft,
  type SideDraft,
  type ViewBox
} from './draft';
import { prefetchFrames } from './frames';
import { acceptedPrefix, fixTimes, nextChunk, type ChunkState } from './track';

export const GRID_SIDE = 640;
export const FIX_SIDE = 1280;
const FRAME_MS = 200; // the 5 fps grid at real time
const LOOKAHEAD = 10; // frames prefetched (JPEGs and track) ahead of the playhead
const TRACK_CHUNK = 25; // grid times per /api/track request

export type Tone = 'info' | 'good' | 'warn' | 'bad';

class Labeler {
  reviewer = $state('');
  segment = $state.raw<SegmentInfo | null>(null);
  /** Inclusive grid indices of the clip. */
  i0 = $state(0);
  i1 = $state(0);
  /** Grid index of the playhead. */
  idx = $state(0);
  /** Machine track of the clip, by clip index; undefined until its chunk loads. */
  track = $state.raw<(TrackFrame | undefined)[]>([]);
  /** Grid times with a saved fix. */
  fixes = $state.raw<ReadonlySet<number>>(new Set());
  draft = $state.raw<Draft | null>(null);
  /** Camera enlarged in the fix view, or null in the grid. */
  fixCam = $state<number | null>(null);
  /** Side of the selected box in the fix view. */
  selected = $state<HandSide | null>(null);
  lastSide = $state<HandSide>('left');
  playing = $state(false);
  busy = $state(false);
  message = $state<{ text: string; tone: Tone } | null>(null);

  sid = $derived(this.segment?.segment_id ?? '');
  times = $derived(this.segment?.times_ns ?? []);
  t = $derived(this.times[this.idx] ?? 0);
  /** Number of leading accepted frames of the clip. */
  accepted = $derived(acceptedPrefix(this.track));
  clipFixes = $derived(
    [...this.fixes].filter((t) => t >= this.times[this.i0] && t <= this.times[this.i1]).length
  );

  private chunks: ChunkState[] = [];
  private trackGen = 0;
  private pumping = false;
  private localFixes = new Set<number>();
  private solveSeq: Record<HandSide, number> = { left: 0, right: 0 };
  private solving: Partial<Record<HandSide, Promise<void>>> = {};
  private timer: ReturnType<typeof setInterval> | undefined;

  seconds(t_ns: number): number {
    return segmentSeconds(t_ns, this.times[0] ?? 0);
  }

  say(text: string, tone: Tone = 'info'): void {
    this.message = { text, tone };
  }

  private fail(error: unknown): void {
    this.say(error instanceof Error ? error.message : String(error), 'bad');
  }

  async load(hash: string): Promise<void> {
    this.pause();
    this.segment = null;
    this.draft = null;
    this.fixCam = null;
    this.message = null;
    try {
      const config = await api.config();
      this.reviewer = localStorage.getItem('hl.reviewer') || config.reviewer;
      const clip = parseHash(hash);
      const sid = clip.sid ?? (await api.segments())[0]?.segment_id;
      if (!sid) {
        this.say('The server has no segments', 'bad');
        return;
      }
      const segment = await api.segment(sid);
      const { i0, i1 } = clipRange(segment.times_ns, clip.from, clip.to);
      this.i0 = i0;
      this.i1 = i1;
      this.idx = i0;
      this.track = new Array(segment.times_ns.length ? i1 - i0 + 1 : 0);
      this.chunks = new Array(Math.ceil(this.track.length / TRACK_CHUNK)).fill('missing');
      this.localFixes = new Set();
      this.fixes = new Set();
      this.segment = segment;
      this.pumpTrack();
      this.loadFixes();
    } catch (error) {
      this.fail(error);
    }
  }

  /** Fetch missing track chunks one at a time, the playhead's first. */
  private async pumpTrack(): Promise<void> {
    if (this.pumping) return;
    this.pumping = true;
    try {
      for (;;) {
        const chunks = this.chunks;
        const c = nextChunk(chunks, this.idx - this.i0, TRACK_CHUNK, LOOKAHEAD);
        if (c === null) break;
        const gen = this.trackGen;
        const lo = c * TRACK_CHUNK;
        const n = Math.min(TRACK_CHUNK, this.track.length - lo);
        chunks[c] = 'loading';
        let frames: TrackFrame[];
        try {
          frames = await api.track(this.sid, this.times[this.i0 + lo], n);
        } catch (error) {
          chunks[c] = 'missing';
          this.fail(error);
          break; // retried on the next playhead move
        }
        if (chunks !== this.chunks) continue; // another clip was loaded meanwhile
        const next = this.track.slice();
        for (const f of frames) {
          const i = this.times.indexOf(f.t_ns, this.i0 + lo) - this.i0;
          if (i >= 0 && i < next.length) next[i] = f;
        }
        this.track = next;
        chunks[c] = gen === this.trackGen ? 'loaded' : 'missing';
      }
    } finally {
      this.pumping = false;
    }
  }

  /** Refetch the track from a clip index on: labels change the machine track from that time forward. */
  private invalidateTrack(fromClipIdx: number): void {
    this.trackGen++;
    for (let c = Math.floor(Math.max(fromClipIdx, 0) / TRACK_CHUNK); c < this.chunks.length; c++) {
      this.chunks[c] = 'missing';
    }
    this.pumpTrack();
  }

  private loadFixes(): void {
    const sid = this.sid;
    api
      .actions(sid)
      .then((actions) => {
        if (sid === this.sid) this.fixes = new Set([...fixTimes(actions), ...this.localFixes]);
      })
      .catch(() => {}); // optional route: the fixes saved in this session still show
  }

  frameAt(idx: number): TrackFrame | undefined {
    return this.track[idx - this.i0];
  }

  trackAt(t_ns: number): TrackFrame | undefined {
    const idx = this.times.indexOf(t_ns, this.i0);
    return idx < 0 ? undefined : this.frameAt(idx);
  }

  /** Boxes one camera shows at a grid time: the unsaved fix there, else the machine track. */
  boxesAt(cam: number, t_ns: number): ViewBox[] {
    const draft = this.draft?.t_ns === t_ns ? this.draft : null;
    return viewBoxes(this.trackAt(t_ns), draft, cam);
  }

  prefetch(): void {
    if (!this.segment) return;
    const cams = this.fixCam === null ? this.segment.cams.map((c) => c.cam) : [this.fixCam];
    const side = this.fixCam === null ? GRID_SIDE : FIX_SIDE;
    const urls: string[] = [];
    for (let i = this.idx + 1; i <= Math.min(this.idx + LOOKAHEAD, this.i1); i++) {
      for (const cam of cams) urls.push(frameUrl(this.sid, cam, this.times[i], side));
    }
    prefetchFrames(urls);
  }

  // ---- time

  /** An unsaved fix pins the playhead: it must be saved (Enter) or discarded (Esc in the grid) first. */
  private pinned(): boolean {
    if (!this.draft) return false;
    this.say('Unsaved fix: Enter saves it, Esc in the grid discards it', 'warn');
    return true;
  }

  seek(idx: number): void {
    const next = Math.min(Math.max(idx, this.i0), this.i1);
    if (next === this.idx || this.pinned()) return;
    this.idx = next;
    this.selected = null;
    this.pumpTrack();
  }

  step(delta: number): void {
    this.pause();
    this.seek(this.idx + delta);
  }

  togglePlay(): void {
    if (this.playing) this.pause();
    else this.play();
  }

  play(): void {
    if (!this.segment || this.pinned()) return;
    if (this.idx >= this.i1) this.seek(this.i0);
    const startIdx = this.idx;
    const startMs = performance.now();
    this.playing = true;
    // Wall-clock frame index: when frames load late the playhead jumps ahead instead of slowing down.
    this.timer = setInterval(() => {
      const target = startIdx + Math.floor((performance.now() - startMs) / FRAME_MS);
      if (target >= this.i1) {
        this.seek(this.i1);
        this.pause();
      } else if (target !== this.idx) {
        this.seek(target);
      }
    }, FRAME_MS / 4);
  }

  pause(): void {
    clearInterval(this.timer);
    this.timer = undefined;
    this.playing = false;
  }

  // ---- views

  openFix(cam: number): void {
    if (!this.segment?.cams.some((c) => c.cam === cam)) return;
    this.fixCam = cam;
    this.selected = null;
  }

  back(): void {
    if (this.fixCam !== null) {
      this.fixCam = null;
      this.selected = null;
    } else if (this.draft) {
      this.draft = null;
      this.say('Discarded the unsaved fix');
    }
  }

  // ---- edits (fix view)

  private editSide(side: HandSide, edit: (s: SideDraft) => SideDraft): void {
    const draft = this.draft ?? { t_ns: this.t, sides: {} };
    const current = draft.sides[side] ?? sideFromTrack(this.frameAt(this.idx), side);
    this.draft = { ...draft, sides: { ...draft.sides, [side]: edit(current) } };
  }

  select(side: HandSide | null): void {
    this.selected = side;
  }

  /** The person drew, moved or resized this side's box in `cam`: it becomes labeled. */
  setBox(cam: number, side: HandSide, box: Box): void {
    this.pause();
    this.editSide(side, (s) => withBox(s, cam, box));
    this.selected = side;
  }

  /** Del: the selected box goes, the hand is not in this view. */
  deleteSelected(): void {
    const side = this.selected;
    if (this.fixCam === null || !side) return;
    const cam = this.fixCam;
    this.editSide(side, (s) => withNoHand(s, cam));
    this.selected = null;
    this.solve(side);
  }

  /** L/R: re-side the selected box (swapping with the other side's box in this view), else pick the draw side. */
  setSide(side: HandSide): void {
    this.lastSide = side;
    const from = this.selected;
    if (this.fixCam === null || !from || from === side) return;
    const cam = this.fixCam;
    const shown = this.boxesAt(cam, this.t);
    const moving = shown.find((b) => b.side === from)!.box;
    const other = shown.find((b) => b.side === side)?.box;
    this.editSide(from, (s) => (other ? withBox(s, cam, other) : withNoHand(s, cam)));
    this.editSide(side, (s) => withBox(s, cam, moving));
    this.selected = side;
    this.solve(from);
    this.solve(side);
  }

  /** Re-solve one side's 3D sphere from its labeled views (after each pointer-up). */
  solve(side: HandSide): Promise<void> {
    const draft = this.draft;
    const s = draft?.sides[side];
    if (!draft || !s) return Promise.resolve();
    const seq = ++this.solveSeq[side];
    const labeled = labeledViews(s);
    const apply = (solve: SideDraft['solve']): void => {
      const now = this.draft;
      const current = now?.sides[side];
      if (seq !== this.solveSeq[side] || !now || now.t_ns !== draft.t_ns || !current) return;
      this.draft = { ...now, sides: { ...now.sides, [side]: { ...current, solve } } };
    };
    if (labeled.length === 0) {
      apply(null);
      return Promise.resolve();
    }
    const pending = api
      .solve({ segment_id: this.sid, t_ns: draft.t_ns, side, labeled })
      .then(apply, (error) => this.fail(error));
    this.solving[side] = pending;
    return pending;
  }

  // ---- saving

  /** Enter: post every edited side (person boxes, removals, the solve for the other views), then refetch. */
  async save(): Promise<void> {
    if (this.busy) return;
    if (!this.draft) {
      this.say('Nothing to save');
      return;
    }
    if (!this.reviewer) {
      this.say('No reviewer name: set it in the server config or localStorage hl.reviewer', 'bad');
      return;
    }
    this.busy = true;
    try {
      await Promise.all(Object.values(this.solving));
      const draft = this.draft;
      const cams = this.segment!.cams.map((c) => c.cam);
      let stored = 0;
      for (const side of SIDES) {
        const s = draft.sides[side];
        const labels = s ? sideLabels(side, s, cams, this.reviewer) : [];
        if (labels.length === 0) continue;
        const r = await api.labels({ segment_id: this.sid, t_ns: draft.t_ns, labels });
        stored += r.stored;
      }
      this.draft = null;
      this.localFixes.add(draft.t_ns);
      this.fixes = new Set([...this.fixes, draft.t_ns]);
      this.invalidateTrack(this.times.indexOf(draft.t_ns, this.i0) - this.i0);
      this.loadFixes();
      this.say(`Saved ${stored} labels at ${fmtSeconds(this.seconds(draft.t_ns))}`, 'good');
    } catch (error) {
      this.fail(error);
    } finally {
      this.busy = false;
    }
  }

  /** A: accept the machine track from the first not-yet-accepted time up to the playhead. */
  async accept(): Promise<void> {
    this.pause();
    if (this.busy || this.pinned()) return;
    const first = this.i0 + this.accepted;
    if (first > this.idx) {
      this.say('Already accepted up to here');
      return;
    }
    this.busy = true;
    try {
      const r = await api.acceptRange({
        segment_id: this.sid,
        t0_ns: this.times[first],
        t1_ns: this.t,
        reviewer: this.reviewer
      });
      this.invalidateTrack(first - this.i0);
      const span = `${fmtSeconds(this.seconds(this.times[first]))} to ${fmtSeconds(this.seconds(this.t))}`;
      this.say(`Accepted ${span}: ${r.accepted} boxes, ${r.kept} kept`, 'good');
    } catch (error) {
      this.fail(error);
    } finally {
      this.busy = false;
    }
  }

  /** WRITE LAYER: commit the segment's labels as the hand_boxes layer. */
  async commit(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      const r = await api.commit(this.sid);
      this.say(`Wrote ${r.n_labels} labels to ${r.path}`, 'good');
    } catch (error) {
      this.fail(error);
    } finally {
      this.busy = false;
    }
  }
}

export const app = new Labeler();
