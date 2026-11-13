<script lang="ts">
  import type { HandSide } from '../lib/api';
  import { SIDES } from '../lib/api';
  import { camLabel } from '../lib/clip';
  import { solveSummary, type SolveSummary } from '../lib/draft';
  import { app } from '../lib/state.svelte';

  /** The side's 3D solve when the person edited it at this time, else what the track holds there. */
  function describe(side: HandSide): SolveSummary {
    const draft = app.draft?.t_ns === app.t ? app.draft.sides[side] : undefined;
    if (draft?.solve) return solveSummary(draft.solve);
    if (draft && Object.keys(draft.labeled).length > 0) {
      return { text: 'solving…', confidence: '', warning: null };
    }
    const frame = app.trackAt(app.t);
    if (!frame) return { text: 'loading the track', confidence: '', warning: null };
    const mine = frame.boxes.filter((b) => b.side === side && b.box);
    const labeled = mine.filter((b) => b.origin === 'reviewed').map((b) => camLabel(b.cam));
    const derived = mine.filter((b) => b.origin === 'derived').length;
    if (labeled.length > 0) {
      const rest = derived > 0 ? ` · ${derived} derived` : '';
      return { text: `labeled in ${labeled.join(' + ')}${rest}`, confidence: '', warning: null };
    }
    return mine.length > 0
      ? { text: `machine track in ${mine.length} views`, confidence: '', warning: null }
      : { text: 'no box in any view', confidence: '', warning: null };
  }
</script>

<div class="solve">
  {#each SIDES as side (side)}
    {@const s = describe(side)}
    <div class={side}>
      <b class="label">{side} hand</b>
      <span>{s.text}</span>
      {#if s.confidence}<em>· {s.confidence}</em>{/if}
      {#if s.warning}<strong>· {s.warning}</strong>{/if}
    </div>
  {/each}
</div>

<style>
  .solve {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--color-border);
    border-top: 1px solid var(--color-border);
  }

  .solve > div {
    background: var(--color-bg);
    padding: 10px 16px;
  }

  b {
    font-size: 12px;
    font-weight: 500;
    margin-right: 10px;
  }

  .left b {
    color: var(--hand-left);
  }

  .right b {
    color: var(--hand-right);
  }

  em {
    font-style: normal;
    color: var(--color-muted);
  }

  strong {
    font-weight: inherit;
    color: var(--color-warning);
  }

  @media (max-width: 800px) {
    .solve {
      grid-template-columns: 1fr;
    }
  }
</style>
