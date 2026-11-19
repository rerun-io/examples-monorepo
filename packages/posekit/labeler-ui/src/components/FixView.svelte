<script lang="ts">
  import type { Box, CameraInfo, HandSide } from '../lib/api';
  import { frameUrl } from '../lib/api';
  import {
    boxFromPoints,
    dragBox,
    hitTest,
    isTiny,
    toNative,
    type Handle,
    type Point
  } from '../lib/box-edit';
  import { camLabel } from '../lib/clip';
  import { drawSide } from '../lib/draft';
  import { app, FIX_SIDE } from '../lib/state.svelte';
  import BoxLayer from './BoxLayer.svelte';
  import FrameImage from './FrameImage.svelte';

  interface Props {
    cam: CameraInfo;
  }

  let { cam }: Props = $props();
  let stage: HTMLDivElement | undefined = $state();
  let shownT = $state<number | null>(null);

  const boxes = $derived(shownT === null ? [] : app.boxesAt(cam.cam, shownT));
  const name = $derived(`${camLabel(cam.cam)} · ${cam.name.replaceAll('_', ' ')}`);

  /** Corner grab radius and the move threshold, in screen pixels. */
  const GRAB_PX = 14;
  const MOVE_PX = 3;

  interface Drag {
    side: HandSide;
    /** null: drawing a new box from `p0`. */
    handle: Handle | null;
    start: Box | null;
    p0: Point;
    moved: boolean;
  }
  let drag: Drag | null = null;

  function nativePoint(e: PointerEvent): { p: Point; pxPerNative: number } {
    const rect = stage!.getBoundingClientRect();
    return {
      p: toNative({ x: e.clientX, y: e.clientY }, rect, cam.width, cam.height),
      pxPerNative: rect.width / cam.width
    };
  }

  function onpointerdown(e: PointerEvent): void {
    if (e.button !== 0 || shownT !== app.t) return;
    app.pause();
    const { p, pxPerNative } = nativePoint(e);
    const hit = hitTest(
      boxes.map((b) => b.box),
      p,
      GRAB_PX / pxPerNative
    );
    stage!.setPointerCapture(e.pointerId);
    if (hit) {
      const b = boxes[hit.index];
      app.select(b.side);
      drag = { side: b.side, handle: hit.handle, start: b.box, p0: p, moved: false };
    } else {
      const side = drawSide(
        boxes.map((b) => b.side),
        app.lastSide
      );
      drag = { side, handle: null, start: null, p0: p, moved: false };
    }
  }

  function onpointermove(e: PointerEvent): void {
    if (!drag) return;
    const { p, pxPerNative } = nativePoint(e);
    const dx = p.x - drag.p0.x;
    const dy = p.y - drag.p0.y;
    if (!drag.moved && Math.hypot(dx, dy) * pxPerNative < MOVE_PX) return;
    if (drag.handle && drag.start) {
      drag.moved = true;
      app.setBox(
        cam.cam,
        drag.side,
        dragBox(drag.start, drag.handle, dx, dy, cam.width, cam.height)
      );
    } else {
      const box = boxFromPoints(drag.p0, p, cam.width, cam.height);
      if (!drag.moved && isTiny(box)) return;
      drag.moved = true;
      app.setBox(cam.cam, drag.side, box);
    }
  }

  function onpointerup(): void {
    const done = drag;
    drag = null;
    if (!done) return;
    if (done.moved) app.solve(done.side);
    else if (!done.handle) app.select(null); // a click on empty image clears the selection
  }
</script>

<section class="fix" aria-label="{name}: fix view">
  <div
    class="stage"
    bind:this={stage}
    style:aspect-ratio="{cam.width} / {cam.height}"
    style:width="min(100%, calc(66vh * {cam.width} / {cam.height}))"
    role="application"
    aria-label="{name}: drag to draw, drag a box to move, drag a corner to resize"
    {onpointerdown}
    {onpointermove}
    {onpointerup}
    onpointercancel={onpointerup}
  >
    <FrameImage
      url={frameUrl(app.sid, cam.cam, app.t, FIX_SIDE)}
      t={app.t}
      alt="{name} at {app.seconds(app.t).toFixed(1)} s"
      bind:shownT
    />
    <BoxLayer
      width={cam.width}
      height={cam.height}
      {boxes}
      handles
      selected={app.selected}
      weight={0.75}
    />
    <span class="name label">{name}</span>
  </div>
  <div class="tools">
    <span class="hint">Drag to draw · drag a box to move · drag a corner to resize</span>
    <button class="btn" onclick={() => app.setSide('left')}>L left</button>
    <button class="btn" onclick={() => app.setSide('right')}>R right</button>
    <button class="btn" onclick={() => app.deleteSelected()} disabled={!app.selected}>
      Del delete
    </button>
    <button class="btn" onclick={() => app.back()}>Esc back</button>
  </div>
</section>

<style>
  .fix {
    background: var(--color-bg);
  }

  .stage {
    position: relative;
    margin: 0 auto;
    cursor: crosshair;
    touch-action: none;
    user-select: none;
  }

  .name {
    position: absolute;
    top: 6px;
    left: 6px;
    font-size: 11px;
    padding: 1px 6px;
    background: var(--color-scrim);
    border-radius: var(--radius-control);
    color: var(--color-strong);
  }

  .tools {
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
    padding: 10px 16px;
    border-top: 1px solid var(--color-border);
  }

  .hint {
    color: var(--color-muted);
    margin-right: auto;
    font-size: 0.9rem;
  }
</style>
