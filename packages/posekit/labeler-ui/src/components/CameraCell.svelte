<script lang="ts">
  import type { CameraInfo } from '../lib/api';
  import { frameUrl } from '../lib/api';
  import { camLabel } from '../lib/clip';
  import { app, GRID_SIDE } from '../lib/state.svelte';
  import BoxLayer from './BoxLayer.svelte';
  import FrameImage from './FrameImage.svelte';

  interface Props {
    cam: CameraInfo;
  }

  let { cam }: Props = $props();
  let shownT = $state<number | null>(null);

  const boxes = $derived(shownT === null ? [] : app.boxesAt(cam.cam, shownT));
  const loaded = $derived(shownT !== null && app.trackAt(shownT) !== undefined);
  const kind = $derived(
    boxes.some((b) => b.kind === 'labeled')
      ? 'labeled'
      : boxes.some((b) => b.kind === 'derived')
        ? 'derived'
        : boxes.length === 0 && loaded
          ? 'none'
          : 'machine'
  );
  const name = $derived(`${camLabel(cam.cam)} · ${cam.name.replaceAll('_', ' ')}`);
</script>

<button
  class="cell {kind}"
  style:aspect-ratio="{cam.width} / {cam.height}"
  onclick={() => app.openFix(cam.cam)}
  aria-label="{name}: enlarge to fix"
>
  <FrameImage
    url={frameUrl(app.sid, cam.cam, app.t, GRID_SIDE)}
    t={app.t}
    alt="{name} at {app.seconds(app.t).toFixed(1)} s"
    bind:shownT
  />
  <BoxLayer width={cam.width} height={cam.height} {boxes} />
  <span class="name label">{name}</span>
  {#if kind === 'labeled'}
    <span class="kind label">Labeled</span>
  {:else if kind === 'derived'}
    <span class="kind label">Derived</span>
  {:else if kind === 'none'}
    <span class="kind label">Not in view</span>
  {/if}
</button>

<style>
  .cell {
    position: relative;
    display: block;
    width: 100%;
    padding: 0;
    border: 0;
    background: var(--color-bg);
    cursor: zoom-in;

    &:focus-visible {
      outline: 2px solid var(--color-strong);
      outline-offset: -2px;
    }

    &.none :global(img) {
      opacity: 0.55;
    }
  }

  .name,
  .kind {
    position: absolute;
    top: 6px;
    font-size: 11px;
    padding: 1px 6px;
    background: var(--color-scrim);
    border-radius: var(--radius-control);
  }

  .name {
    left: 6px;
    color: var(--color-strong);
  }

  .kind {
    right: 6px;
    color: var(--color-muted);
  }

  .labeled .kind {
    color: var(--color-strong);
    outline: 1px solid var(--color-strong);
  }
</style>
