<script lang="ts">
  import { fmtSeconds } from '../lib/clip';
  import { app } from '../lib/state.svelte';

  let bar: HTMLDivElement | undefined = $state();

  const n = $derived(app.i1 - app.i0 + 1);
  const pct = (clipIdx: number): string => `${(clipIdx / n) * 100}%`;
  const fixMarks = $derived(
    [...app.fixes].map((t) => app.times.indexOf(t, app.i0) - app.i0).filter((i) => i >= 0 && i < n)
  );

  function seekTo(e: PointerEvent): void {
    const rect = bar!.getBoundingClientRect();
    const frac = Math.min(Math.max((e.clientX - rect.left) / rect.width, 0), 0.9999);
    app.pause();
    app.seek(app.i0 + Math.floor(frac * n));
  }

  function onpointerdown(e: PointerEvent): void {
    bar!.setPointerCapture(e.pointerId);
    seekTo(e);
  }

  function onpointermove(e: PointerEvent): void {
    if (bar!.hasPointerCapture(e.pointerId)) seekTo(e);
  }
</script>

<div class="bar">
  <div class="transport">
    <button class="btn" aria-label="step back" onclick={() => app.step(-1)}>⏮</button>
    <button class="btn" aria-label="play or pause" onclick={() => app.togglePlay()}>
      {app.playing ? '⏸' : '▶'}
    </button>
    <button class="btn" aria-label="step forward" onclick={() => app.step(1)}>⏭</button>
  </div>
  <div
    class="timeline"
    bind:this={bar}
    role="slider"
    tabindex="-1"
    aria-label="clip timeline"
    aria-valuemin={app.i0}
    aria-valuemax={app.i1}
    aria-valuenow={app.idx}
    {onpointerdown}
    {onpointermove}
  >
    <div class="acc" style:width={pct(app.accepted)}></div>
    {#each fixMarks as i (i)}
      <div class="fix" style:left={pct(i + 0.5)}></div>
    {/each}
    <div class="head" style:left={pct(app.idx - app.i0 + 0.5)}></div>
    <span class="t">
      {fmtSeconds(app.seconds(app.t))} / {fmtSeconds(app.seconds(app.times[app.i1] ?? 0))}
    </span>
  </div>
  <button class="btn primary" onclick={() => app.accept()} disabled={app.busy}>
    Accept up to here<kbd>A</kbd>
  </button>
  <button class="btn" onclick={() => app.save()} disabled={app.busy || !app.draft}>
    Save<kbd>Enter</kbd>
  </button>
</div>

<style>
  .bar {
    display: grid;
    grid-template-columns: auto 1fr auto auto;
    gap: 12px;
    align-items: center;
    padding: 12px 16px;
    border-top: 1px solid var(--color-border);
  }

  .transport {
    display: flex;
    gap: 4px;

    & .btn {
      width: 40px;
      padding: 9px 0;
      text-align: center;
    }
  }

  .timeline {
    position: relative;
    height: 36px;
    border-radius: var(--radius-control);
    background: var(--color-surface-2);
    overflow: hidden;
    cursor: pointer;
    touch-action: none;
  }

  .acc {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    background: color-mix(in oklch, var(--color-success) 30%, transparent);
  }

  .fix,
  .head {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    margin-left: -1px;
    background: var(--color-strong);
  }

  .head {
    box-shadow: 0 0 0 1px var(--color-bg);
  }

  .t {
    position: absolute;
    right: 10px;
    top: 9px;
    font: 500 12px var(--font-mono);
    color: var(--color-muted);
    font-variant-numeric: tabular-nums;
    pointer-events: none;
  }

  @media (max-width: 800px) {
    .bar {
      grid-template-columns: 1fr;
    }
  }
</style>
