import { describe, expect, it } from 'vitest';
import { boxFromPoints, dragBox, hitTest, isTiny, toNative } from './box-edit';

const W = 1920;
const H = 1080;
const box = { x0: 100, y0: 200, x1: 300, y1: 400 };

describe('toNative', () => {
  it('maps a client point on a scaled image to native pixels', () => {
    const rect = { left: 10, top: 20, width: 960, height: 540 };
    expect(toNative({ x: 10 + 480, y: 20 + 270 }, rect, W, H)).toEqual({ x: 960, y: 540 });
  });
});

describe('hitTest', () => {
  it('finds a corner within the tolerance before the interior', () => {
    expect(hitTest([box], { x: 305, y: 395 }, 10)).toEqual({ index: 0, handle: 'se' });
    expect(hitTest([box], { x: 95, y: 205 }, 10)).toEqual({ index: 0, handle: 'nw' });
  });

  it('moves the smallest box that contains the point', () => {
    const inner = { x0: 150, y0: 250, x1: 200, y1: 300 };
    expect(hitTest([box, inner], { x: 170, y: 270 }, 5)).toEqual({ index: 1, handle: 'move' });
  });

  it('returns null on empty image', () => {
    expect(hitTest([box], { x: 900, y: 900 }, 10)).toBeNull();
  });
});

describe('dragBox', () => {
  it('moves a box by the drag, keeping its size', () => {
    expect(dragBox(box, 'move', 50, -20, W, H)).toEqual({ x0: 150, y0: 180, x1: 350, y1: 380 });
  });

  it('keeps a moved box inside the image', () => {
    expect(dragBox(box, 'move', -500, 5000, W, H)).toEqual({ x0: 0, y0: 880, x1: 200, y1: 1080 });
  });

  it('resizes from one corner and keeps the opposite one fixed', () => {
    expect(dragBox(box, 'se', 40, 60, W, H)).toEqual({ x0: 100, y0: 200, x1: 340, y1: 460 });
    expect(dragBox(box, 'nw', -40, 10, W, H)).toEqual({ x0: 60, y0: 210, x1: 300, y1: 400 });
  });

  it('flips when a corner crosses the opposite corner', () => {
    expect(dragBox(box, 'ne', -300, 0, W, H)).toEqual({ x0: 0, y0: 200, x1: 100, y1: 400 });
  });
});

describe('draw', () => {
  it('orders and clips the two points', () => {
    expect(boxFromPoints({ x: 2000, y: 50 }, { x: 1800, y: -10 }, W, H)).toEqual({
      x0: 1800,
      y0: 0,
      x1: 1920,
      y1: 50
    });
  });

  it('treats a box under the minimum size as a click', () => {
    expect(isTiny({ x0: 10, y0: 10, x1: 13, y1: 80 })).toBe(true);
    expect(isTiny(box)).toBe(false);
  });
});
