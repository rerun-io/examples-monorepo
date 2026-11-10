// TEMPORARY: replace with gradio_rerun's set_time command once
// https://github.com/rerun-io/gradio-rerun-viewer/pull/66 is released (then delete this file).
//
// After a streamed run, put the hands app's viewer back at the start of the recording, paused.
// gradio_rerun 0.42 keeps its @rerun-io/web-viewer WebViewer in a Svelte closure, so page JS cannot
// reach it. The bundled class stores its #private fields in WeakMaps (the build down-levels them), so
// the constructor calls WeakMap.prototype.set(viewer, ...): this script wraps that call and keeps a
// weak reference to every WebViewer. It must run before the component mounts (launch(head=...)).
// If a later gradio_rerun build keeps native #private fields, no viewer is found and the rewind times
// out silently: the cursor then stays at the end, as without this script.
(() => {
  if (window.__rerunViewers) return;
  const refs = [];
  const set = WeakMap.prototype.set;
  WeakMap.prototype.set = function (key, value) {
    if (key !== null && typeof key === "object" && typeof key.set_time === "function"
        && typeof key.get_state === "function" && typeof key.open_channel === "function"
        && !refs.some((r) => r.deref() === key)) {
      refs.push(new WeakRef(key));
    }
    return set.call(this, key, value);
  };
  window.__rerunViewers = refs;

  // The live viewer whose canvas sits inside #elemId (stopped viewers have canvas == null).
  const viewerIn = (elemId) => {
    const host = document.getElementById(elemId);
    return refs.map((r) => r.deref()).find((v) => v && v.canvas && host && host.contains(v.canvas));
  };
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // Wait until the timeline's range stops growing (the component may still be fetching stream
  // segments after the Python generator returned), then pause the cursor at the range start.
  window.__rerunRewind = async (elemId, timeline, { stableMs = 1500, timeoutMs = 60000, waitStable = true } = {}) => {
    const t0 = performance.now();
    let last = null, since = performance.now();
    while (performance.now() - t0 < timeoutMs) {
      const v = viewerIn(elemId);
      if (v && v.ready) {
        const s = await v.get_state();
        const rec = s.recordings.find((r) => r.store_id === s.active_store_id) ?? s.recordings[0];
        const tl = rec?.timelines.find((t) => t.name === timeline);
        if (tl?.time_range) {
          // Every timeline's end: video_time is complete as soon as the video is logged, frame grows per chunk.
          const key = `${rec.store_id}:${rec.timelines.map((t) => t.time_range?.end).join(",")}`;
          if (key !== last) { last = key; since = performance.now(); }
          if (!waitStable || performance.now() - since >= stableMs) {
            const at = tl.time_type === "sequence" ? { sequence: tl.time_range.start } : tl.time_type === "timestamp"
              ? { timestamp_nanos: tl.time_range.start } : { duration_nanos: tl.time_range.start };
            await v.set_time({ recording: rec.store_id, timeline, play_state: "paused", ...at });
            return `rewound ${timeline} to ${tl.time_range.start}`;
          }
        }
      }
      await sleep(250);
    }
    return "rewind timed out";
  };
})();
