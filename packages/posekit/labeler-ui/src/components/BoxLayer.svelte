<script lang="ts">
  import type { HandSide } from '../lib/api';
  import type { ViewBox } from '../lib/draft';

  interface Props {
    /** Native image size: the boxes are drawn in native pixels. */
    width: number;
    height: number;
    boxes: ViewBox[];
    /** Corner handles (the fix view). */
    handles?: boolean;
    selected?: HandSide | null;
    /** Stroke scale; the enlarged view draws slightly thinner lines. */
    weight?: number;
  }

  let { width, height, boxes, handles = false, selected = null, weight = 1 }: Props = $props();

  // Sizes are in native pixels of a 1920 px wide frame, so they look the same on every camera.
  const k = $derived(width / 1920);
  const stroke = (kind: ViewBox['kind']): number =>
    (kind === 'labeled' ? 9 : kind === 'derived' ? 6 : 4) * k * weight;
  const tag = (b: ViewBox): string =>
    (b.side === 'left' ? 'L' : 'R') + (b.kind === 'derived' ? ' · 3D' : '');
  const corners = (b: ViewBox): [number, number][] => [
    [b.box.x0, b.box.y0],
    [b.box.x1, b.box.y0],
    [b.box.x0, b.box.y1],
    [b.box.x1, b.box.y1]
  ];
</script>

<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-hidden="true">
  {#each boxes as b (b.side)}
    <g class={b.side}>
      <rect
        class="box"
        x={b.box.x0}
        y={b.box.y0}
        width={b.box.x1 - b.box.x0}
        height={b.box.y1 - b.box.y0}
        stroke-width={stroke(b.kind)}
        stroke-dasharray={b.kind === 'derived' ? `${22 * k} ${14 * k}` : undefined}
      />
      <rect
        class="tag"
        x={b.box.x0}
        y={Math.max(0, b.box.y0 - 50 * k)}
        width={(b.kind === 'derived' ? 120 : 44) * k}
        height={46 * k}
      />
      <text x={b.box.x0 + 10 * k} y={Math.max(0, b.box.y0 - 50 * k) + 34 * k} font-size={34 * k}
        >{tag(b)}</text
      >
      {#if handles}
        {#each corners(b) as [cx, cy], i (i)}
          <rect
            class="handle"
            class:selected={selected === b.side}
            x={cx - 13 * k}
            y={cy - 13 * k}
            width={26 * k}
            height={26 * k}
            stroke-width={5 * k}
          />
        {/each}
      {/if}
    </g>
  {/each}
</svg>

<style>
  svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  .left {
    --hand: var(--hand-left);
  }

  .right {
    --hand: var(--hand-right);
  }

  .box {
    fill: none;
    stroke: var(--hand);
  }

  .tag {
    fill: var(--hand);
  }

  text {
    fill: var(--color-on-strong);
    font-family: var(--font-mono);
    font-weight: 600;
  }

  .handle {
    fill: var(--color-bg);
    stroke: var(--hand);

    &.selected {
      fill: var(--hand);
    }
  }
</style>
