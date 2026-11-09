// Frame JPEGs as object URLs. The server sends frames with `Cache-Control: no-store`, so a plain <img> prefetch would
// be fetched again on use; this cache keeps the bytes instead. Frames on screen load at once; prefetches run a few at
// a time and a newer prefetch list replaces the queued rest.

const MAX_FRAMES = 300;
const PREFETCH_PARALLEL = 4;

const cache = new Map<string, Promise<string>>();
let queue: string[] = [];
let running = 0;

function start(url: string): Promise<string> {
  const promise = fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`${url}: ${r.status}`);
      return r.blob();
    })
    .then((blob) => URL.createObjectURL(blob));
  cache.set(url, promise);
  promise.catch(() => cache.delete(url));
  while (cache.size > MAX_FRAMES) {
    const [oldest, old] = cache.entries().next().value!;
    cache.delete(oldest);
    old.then(URL.revokeObjectURL, () => {});
  }
  return promise;
}

/** The object URL of a frame, fetched now unless cached. */
export function loadFrame(url: string): Promise<string> {
  const hit = cache.get(url);
  if (!hit) return start(url);
  cache.delete(url); // re-insert: Map order is the eviction order
  cache.set(url, hit);
  return hit;
}

function pump(): void {
  while (running < PREFETCH_PARALLEL && queue.length > 0) {
    const url = queue.shift()!;
    if (cache.has(url)) continue;
    running++;
    start(url)
      .catch(() => {})
      .finally(() => {
        running--;
        pump();
      });
  }
}

/** Fetch these frames in the background, in order, dropping any earlier prefetch that has not started. */
export function prefetchFrames(urls: readonly string[]): void {
  queue = urls.filter((u) => !cache.has(u));
  pump();
}
