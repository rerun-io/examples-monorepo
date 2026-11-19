// Pointer geometry of the fix view, all in native image pixels: hit testing, move, corner resize and draw.

import type { Box } from './api';

export interface Point {
  x: number;
  y: number;
}

export type Handle = 'move' | 'nw' | 'ne' | 'sw' | 'se';

/** Boxes smaller than this (native px, either side) are a click, not a drawn box. */
export const MIN_BOX_PX = 6;

/** The native pixel under a client point, for an element that shows the whole `width` x `height` image. */
export function toNative(
  client: Point,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number
): Point {
  return {
    x: ((client.x - rect.left) / rect.width) * width,
    y: ((client.y - rect.top) / rect.height) * height
  };
}

function corners(box: Box): [Handle, Point][] {
  return [
    ['nw', { x: box.x0, y: box.y0 }],
    ['ne', { x: box.x1, y: box.y0 }],
    ['sw', { x: box.x0, y: box.y1 }],
    ['se', { x: box.x1, y: box.y1 }]
  ];
}

/**
 * The box and handle under `p`: a corner within `tolerance` px wins over a box interior; among interiors the smallest
 * box wins (so a small box inside a large one stays reachable). `null` on empty image.
 */
export function hitTest(
  boxes: readonly Box[],
  p: Point,
  tolerance: number
): { index: number; handle: Handle } | null {
  for (let i = 0; i < boxes.length; i++) {
    for (const [handle, c] of corners(boxes[i])) {
      if (Math.abs(p.x - c.x) <= tolerance && Math.abs(p.y - c.y) <= tolerance) {
        return { index: i, handle };
      }
    }
  }
  let best: { index: number; handle: Handle } | null = null;
  let bestArea = Infinity;
  boxes.forEach((b, index) => {
    const inside = p.x >= b.x0 && p.x <= b.x1 && p.y >= b.y0 && p.y <= b.y1;
    const area = (b.x1 - b.x0) * (b.y1 - b.y0);
    if (inside && area < bestArea) {
      best = { index, handle: 'move' };
      bestArea = area;
    }
  });
  return best;
}

const clamp = (v: number, lo: number, hi: number): number => Math.min(Math.max(v, lo), hi);

/** A box spanned by two points, ordered and clipped to the image. */
export function boxFromPoints(a: Point, b: Point, width: number, height: number): Box {
  return {
    x0: clamp(Math.min(a.x, b.x), 0, width),
    y0: clamp(Math.min(a.y, b.y), 0, height),
    x1: clamp(Math.max(a.x, b.x), 0, width),
    y1: clamp(Math.max(a.y, b.y), 0, height)
  };
}

/**
 * `start` dragged by (dx, dy) through `handle`. A move keeps the size and stays inside the image; a corner moves only
 * that corner, and the box flips cleanly when the corner crosses the opposite one.
 */
export function dragBox(
  start: Box,
  handle: Handle,
  dx: number,
  dy: number,
  width: number,
  height: number
): Box {
  if (handle === 'move') {
    const mx = clamp(dx, -start.x0, width - start.x1);
    const my = clamp(dy, -start.y0, height - start.y1);
    return { x0: start.x0 + mx, y0: start.y0 + my, x1: start.x1 + mx, y1: start.y1 + my };
  }
  const moved: Point = {
    x: (handle === 'nw' || handle === 'sw' ? start.x0 : start.x1) + dx,
    y: (handle === 'nw' || handle === 'ne' ? start.y0 : start.y1) + dy
  };
  const fixed: Point = {
    x: handle === 'nw' || handle === 'sw' ? start.x1 : start.x0,
    y: handle === 'nw' || handle === 'ne' ? start.y1 : start.y0
  };
  return boxFromPoints(moved, fixed, width, height);
}

export function isTiny(box: Box): boolean {
  return box.x1 - box.x0 < MIN_BOX_PX || box.y1 - box.y0 < MIN_BOX_PX;
}

/** Native pixels rounded to 0.1 px, so posted boxes stay readable. */
export function roundBox(box: Box): Box {
  const r = (v: number): number => Math.round(v * 10) / 10;
  return { x0: r(box.x0), y0: r(box.y0), x1: r(box.x1), y1: r(box.y1) };
}
