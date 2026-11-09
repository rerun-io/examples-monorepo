// hand-labeler frontend: review queue + 6-camera box editor + embedded Rerun web viewer.
// Plain ES module, no build step. JSON shapes are the pyserde records in posekit/labeler/schema.py.
// Boxes are kept in NATIVE image pixels (CameraInfo.width x height); canvases show downscaled JPEGs.
//
// MEgATrack loop (SPEC-v2-megatrack-loop.md): the default boxes are the machine track (/api/track); W plays the
// 5 fps grid, A accepts the watched run (/api/accept_range), and the reviewer edits + saves only the failure frame.

// Module URLs tried in order. jsdelivr serves the +esm bundle but answers 403 for the ~40 MB
// re_viewer_bg.wasm it then fetches (its per-file size limit), so unpkg (which serves the wasm) goes first.
// `?viewer_esm=<url>` on the page URL puts another candidate in front (e.g. a copy served by the backend).
const VIEWER_ESM = [
  new URLSearchParams(location.search).get("viewer_esm"),
  "https://unpkg.com/@rerun-io/web-viewer@0.42.0/index.js",
  "https://cdn.jsdelivr.net/npm/@rerun-io/web-viewer@0.42.0/+esm",
].filter(Boolean);
const TIMELINE = "video_time";
const SIDES = ["left", "right"];
const COLOR = { left: "#4f9bff", right: "#ff9a2e", label: "#3ddc84", sel: "#ffffff" };
const GRID_SIDE = 960; // max_side of grid thumbnails (spec)
const FOCUS_SIDE = 1920; // max_side of the focused camera (native on RoboCap)
const HANDLE_CSS_PX = 7; // hit tolerance around edges/corners, CSS px
const MIN_BOX_PX = 4; // smallest box side, native px
const WATCH_MS = 200; // watch plays the 5 fps grid in real time
const PREFETCH = 10; // grid frames fetched ahead while watching (frame sets, track, thumbnails)
const TRACK_CHUNK = 20; // grid times per /api/track request
const IMG_CACHE_MAX = 6 * 2 * (PREFETCH + 4); // thumbnails kept (cams x a few frames behind and ahead)

// The page is served at <root>/ and this file at <root>/static/app.js; resolve the API from here so a proxy
// prefix (or loading /static/index.html directly) keeps working.
const ROOT = new URL("../", import.meta.url);
const apiUrl = (path) => new URL(`api/${path}`, ROOT).href;

const $ = (id) => document.getElementById(id);
const S = {
  segments: [],
  sid: null,
  info: null, // SegmentInfo
  queue: [], // QueueItem[]
  t: null, // current t_ns (number; video_time is a duration, so < 2^53)
  flags: [],
  views: [], // per cam: {cam, info, lo, hi, boxes, proposals, rejected, touched, el, canvas}
  focus: null, // focused cam index or null
  sel: null, // selected working box object (on the focused cam)
  confirmed: false, // A pressed, saved, or loaded with a full stored decision set
  dirty: false,
  discardArmed: false,
  reviewed: new Set(), // t_ns known to have a decision for every (cam, side)
  loadSeq: 0,
  loading: false, // a frame set (or segment) is loading: S.t and S.views still show the previous one
  saving: false, // a POST /api/labels is in flight
  rev: 0, // edit revision, bumped by every edit; a save marks the frame saved only if it is unchanged
  drag: null,
  // machine track: t_ns -> TrackFrame. Dropped after every save/accept (the backend recomputes it from the labels).
  track: new Map(),
  trackGen: 0,
  trackPending: [], // in-flight /api/track requests: {lo, hi, p} grid-index ranges
  trackOk: true, // false when the backend has no /api/track: the old proposal defaults are used then
  fsCache: new Map(), // t_ns -> Promise<FrameState>, prefetched while watching; each entry is used once
  imgCache: new Map(), // "sid/cam/t/maxSide" -> Image (LRU)
  // The run the next A accepts: grid times anchor..watchedTo were on screen since the last accept or jump.
  anchor: null,
  watchedTo: null,
  watch: { on: false, gen: 0, how: null, shown: [] }, // how the last watch stopped: space | key | edit | moved | end | accept
  accepting: false,
  effort: null, // EffortSummary or null
};

// ---------------------------------------------------------------- small helpers
function status(msg, kind = "") {
  const el = $("status");
  el.textContent = msg;
  el.className = `status ${kind}`;
}

async function getJSON(path) {
  const r = await fetch(apiUrl(path), { headers: { Accept: "application/json" } });
  if (!r.ok) throw new Error(`GET ${path}: ${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
}

async function postJSON(path, body) {
  const r = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? "" : JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`POST ${path}: ${r.status} ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) : {};
}

function fmtTime(tNs) {
  const s = tNs / 1e9;
  const m = Math.floor(s / 60);
  return `${m}:${(s - 60 * m).toFixed(2).padStart(5, "0")}`;
}

const reviewer = () => $("reviewer").value.trim();

// While a frame set loads, the views on screen belong to the previous time: no edits or saves until it is shown.
function editBlocked() {
  if (!S.loading) return false;
  status("the next frame set is still loading: edits and saves wait until it is shown", "err");
  return true;
}

// While a save is in flight the frame set stays put, so the answer is applied to the frame it was sent for.
function navBlocked() {
  if (!S.saving) return false;
  status("saving …: wait for the save to finish before moving", "err");
  return true;
}
const round1 = (v) => Math.round(v * 10) / 10;
const enc = encodeURIComponent;
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const gridIndex = (t) => (S.info && t !== null ? S.info.times_ns.indexOf(t) : -1);
const isKeyframe = (t) => S.info?.keyframes_ns.includes(t);
const chipClass = (reason) =>
  reason === "keyframe" ? "keyframe" : ["range_gap", "decode_error", "no_proposal"].includes(reason) ? "bad" : "warn";

// ---------------------------------------------------------------- segments + queue
async function loadSegments() {
  S.segments = await getJSON("segments");
  const sel = $("segment");
  const keep = S.sid;
  sel.innerHTML = "";
  for (const seg of S.segments) {
    const o = document.createElement("option");
    o.value = seg.segment_id;
    o.textContent = `${seg.segment_id} · ${seg.split} · ${seg.n_reviewed_keyframes}/${seg.n_keyframes}`;
    sel.append(o);
  }
  if (keep) sel.value = keep;
  renderProgress();
}

function renderProgress() {
  const seg = S.segments.find((s) => s.segment_id === S.sid);
  const fill = $("progress").querySelector(".fill");
  const txt = $("progress").querySelector(".txt");
  if (!seg) {
    fill.style.width = "0";
    txt.textContent = "";
    return;
  }
  const frac = seg.n_keyframes ? seg.n_reviewed_keyframes / seg.n_keyframes : 0;
  fill.style.width = `${(100 * frac).toFixed(1)}%`;
  txt.textContent = `${seg.n_reviewed_keyframes}/${seg.n_keyframes} kf · ${seg.n_labels} lbl`;
}

async function loadQueue() {
  if (!S.sid) return;
  S.queue = await getJSON(`queue?sid=${encodeURIComponent(S.sid)}&limit=50`);
  renderQueue();
}

function renderQueue() {
  const ul = $("queue");
  ul.innerHTML = "";
  $("qcount").textContent = `· ${S.queue.length}`;
  for (const item of S.queue) {
    const li = document.createElement("li");
    if (item.t_ns === S.t) li.className = "cur";
    else if (S.reviewed.has(item.t_ns)) li.className = "done";
    const chips = item.reasons.map((r) => `<span class="chip ${chipClass(r)}">${r}</span>`).join("");
    li.innerHTML = `<span class="t">${fmtTime(item.t_ns)}</span><br>${chips}`;
    li.addEventListener("click", () => goto(item.t_ns));
    ul.append(li);
  }
  ul.querySelector("li.cur")?.scrollIntoView({ block: "nearest" });
}

async function selectSegment(sid, t = null) {
  if (S.watch.on) stopWatch("moved");
  if (navBlocked() || !guardDirty()) {
    $("segment").value = S.sid ?? "";
    return;
  }
  const seq = ++S.loadSeq;
  S.loading = true;
  let info;
  try {
    info = await getJSON(`segment/${encodeURIComponent(sid)}`);
  } catch (e) {
    if (seq === S.loadSeq) {
      S.loading = false;
      $("segment").value = S.sid ?? "";
      status(`${String(e.message ?? e)} · still on ${S.sid ?? "no segment"}`, "err");
    }
    return;
  }
  if (seq !== S.loadSeq) return; // a newer navigation won
  // No frame set is shown for the new segment until goto succeeds: S.t = null keeps save() from posting.
  S.sid = sid;
  S.info = info;
  S.t = null;
  S.loading = false; // nothing editable is on screen until goto publishes a frame set
  S.dirty = false;
  S.discardArmed = false;
  $("segment").value = sid;
  S.reviewed.clear();
  dropTrack();
  S.imgCache.clear();
  S.trackOk = true;
  S.anchor = null;
  S.watchedTo = null;
  S.watch.how = null;
  S.effort = null;
  renderWatch();
  loadEffort();
  const scrub = $("scrub");
  scrub.max = String(Math.max(0, S.info.times_ns.length - 1));
  renderProgress();
  buildCams();
  await loadQueue();
  viewerOpenSegment();
  const start = t ?? S.queue[0]?.t_ns ?? S.info.keyframes_ns[0] ?? S.info.times_ns[0];
  if (start !== undefined) await goto(start, { force: true });
}

// ---------------------------------------------------------------- frame set
function buildCams() {
  const root = $("cams");
  root.innerHTML = "";
  S.focus = null;
  S.sel = null;
  root.className = "cams grid";
  S.views = S.info.cams.map((ci) => {
    const el = document.createElement("div");
    el.className = "cam";
    el.dataset.cam = String(ci.cam);
    const canvas = document.createElement("canvas");
    const lab = document.createElement("span");
    lab.className = "vlabel";
    lab.textContent = `${ci.cam + 1} · cam_${String(ci.cam).padStart(2, "0")} ${ci.name}`;
    el.append(canvas, lab);
    root.append(el);
    const v = { cam: ci.cam, info: ci, lo: null, hi: null, boxes: [], proposals: [], rejected: false, touched: false, el, canvas };
    new ResizeObserver(() => draw(v)).observe(el);
    canvas.addEventListener("pointerdown", (ev) => onPointerDown(v, ev));
    canvas.addEventListener("pointermove", (ev) => onPointerMove(v, ev));
    canvas.addEventListener("pointerup", (ev) => onPointerUp(v, ev));
    canvas.addEventListener("pointercancel", (ev) => onPointerUp(v, ev));
    return v;
  });
  renderStates();
}

function guardDirty() {
  if (!S.dirty || S.discardArmed) {
    S.discardArmed = false;
    return true;
  }
  S.discardArmed = true;
  status("Unsaved decisions on this frame set: Enter saves them; repeat the move to discard them.", "err");
  return false;
}

// The new time and its views are published together, only once the frame set has loaded; until then the
// previous frame set stays on screen, read-only (editBlocked). A failed load leaves the previous one in place.
// watching = a step of the W playback (uses the prefetched frame set); quiet = keep the status line (a refresh).
async function goto(t, { force = false, watching = false, quiet = false } = {}) {
  if (!S.info) return;
  if (!watching && S.watch.on) stopWatch("moved");
  if (navBlocked()) return;
  if (!force && t !== S.t && !guardDirty()) return;
  const seq = ++S.loadSeq;
  const sid = S.sid;
  S.loading = true;
  S.drag = null;
  $("tlabel").textContent = `${fmtTime(t)} …`;
  let fs;
  let tr;
  try {
    [fs, tr] = await Promise.all([watching ? takeFrameset(t) : getJSON(`frameset/${enc(sid)}/${t}`), trackAt(t)]);
  } catch (e) {
    if (seq !== S.loadSeq) return; // a newer goto owns the loading state
    S.loading = false;
    if (S.t !== null) updateTimeUi();
    status(`${String(e.message ?? e)} · still on ${S.t === null ? "no frame set" : fmtTime(S.t)}`, "err");
    return;
  }
  if (seq !== S.loadSeq) return; // a newer goto won
  S.loading = false;
  S.t = t;
  S.rev += 1;
  updateTimeUi();
  S.flags = fs.flags;
  S.sel = null;
  S.dirty = false;
  S.discardArmed = false;
  let complete = true;
  for (const v of S.views) {
    const vs = fs.views.find((x) => x.cam === v.cam) ?? { proposals: [], labels: [] };
    initView(v, vs, tr);
    if (vs.labels.length < SIDES.length) complete = false;
    v.lo = loadImage(v, GRID_SIDE, seq, "lo", t);
    v.hi = null;
    if (S.focus === v.cam) v.hi = loadImage(v, FOCUS_SIDE, seq, "hi", t);
  }
  S.confirmed = complete;
  if (complete) S.reviewed.add(t);
  renderFlags();
  renderStates();
  renderQueue();
  S.views.forEach(draw);
  history.replaceState(null, "", `#sid=${enc(S.sid)}&t=${t}`);
  viewerFollow(t);
  noteWatched(t);
  if (watching) S.watch.shown.push(performance.now());
  if (!quiet) {
    const machine = tr ? " · machine track" : S.trackOk ? " · no machine track (proposals shown)" : " · backend has no /api/track: proposals shown";
    status(`${S.sid} @ ${fmtTime(t)}${isKeyframe(t) ? " (keyframe)" : ""}${complete ? " · already reviewed" : ""}${machine}`);
  }
}

// Thumbnails are cached (LRU) so W playback and prefetching reuse the same decoded image.
function cachedImage(cam, t, maxSide) {
  const key = `${S.sid}/${cam}/${t}/${maxSide}`;
  let img = S.imgCache.get(key);
  if (img) {
    S.imgCache.delete(key);
  } else {
    img = new Image();
    img.decoding = "async";
    img.addEventListener("error", () => S.imgCache.get(key) === img && S.imgCache.delete(key), { once: true });
    img.src = apiUrl(`frame/${enc(S.sid)}/${cam}/${t}.jpg?max_side=${maxSide}`);
  }
  S.imgCache.set(key, img);
  while (S.imgCache.size > IMG_CACHE_MAX) S.imgCache.delete(S.imgCache.keys().next().value);
  return img;
}

function loadImage(v, maxSide, seq, which, t) {
  const img = cachedImage(v.cam, t, maxSide);
  if (img.complete && img.naturalWidth) return img; // goto draws it right away
  img.addEventListener("load", () => {
    if (seq === S.loadSeq && v[which] === img) draw(v);
  }, { once: true });
  img.addEventListener("error", () => {
    if (seq === S.loadSeq && v[which] === img) status(`frame cam ${v.cam} failed to load`, "err");
  }, { once: true });
  return img;
}

// ---------------------------------------------------------------- machine track (/api/track)
function dropTrack() {
  S.trackGen += 1;
  S.track.clear();
  S.trackPending = [];
  S.fsCache.clear();
}

// Fetches the TRACK_CHUNK grid times from t0 unless a request in flight already covers t0.
function fetchTrack(t0) {
  const ts = S.info?.times_ns ?? [];
  const i = ts.indexOf(t0);
  if (i < 0 || !S.trackOk) return Promise.resolve();
  const pending = S.trackPending.find((e) => e.lo <= i && i <= e.hi);
  if (pending) return pending.p;
  const gen = S.trackGen;
  const sid = S.sid;
  const entry = { lo: i, hi: i + TRACK_CHUNK - 1, p: null };
  entry.p = getJSON(`track/${enc(sid)}?t0=${t0}&n=${TRACK_CHUNK}`)
    .then((frames) => {
      if (gen === S.trackGen && sid === S.sid) for (const f of frames) S.track.set(f.t_ns, f);
    })
    .catch((e) => {
      if (/: 404 /.test(String(e.message))) S.trackOk = false;
      else status(`track: ${e.message ?? e}`, "err");
    })
    .finally(() => {
      S.trackPending = S.trackPending.filter((x) => x !== entry);
    });
  S.trackPending.push(entry);
  return entry.p;
}

// The TrackFrame at t, or null (no track endpoint, or the fetch failed): then the proposals are the defaults.
async function trackAt(t) {
  for (let tries = 0; tries < 2 && S.trackOk && !S.track.has(t); tries++) await fetchTrack(t);
  return S.track.get(t) ?? null;
}

function takeFrameset(t) {
  const p = S.fsCache.get(t) ?? getJSON(`frameset/${enc(S.sid)}/${t}`);
  S.fsCache.delete(t);
  return p;
}

// Fetch the frame sets, track and thumbnails of the grid times after index i (watch playback).
function prefetch(i) {
  const ts = S.info.times_ns;
  for (let k = i; k < Math.min(ts.length, i + PREFETCH); k++) {
    const t = ts[k];
    if (!S.track.has(t)) fetchTrack(t);
    if (!S.fsCache.has(t)) {
      const p = getJSON(`frameset/${enc(S.sid)}/${t}`);
      p.catch(() => S.fsCache.get(t) === p && S.fsCache.delete(t));
      S.fsCache.set(t, p);
    }
    for (const v of S.views) {
      cachedImage(v.cam, t, GRID_SIDE);
      if (S.focus === v.cam) cachedImage(v.cam, t, FOCUS_SIDE);
    }
  }
}

// Resolves once the frame set, the track and every grid thumbnail of t are in hand.
async function frameReady(t) {
  const imgs = S.views.map((v) => cachedImage(v.cam, t, GRID_SIDE));
  await Promise.all([S.fsCache.get(t), trackAt(t), ...imgs.map((img) => img.decode().catch(() => {}))]);
}

// Working boxes: one per (cam, side). Stored labels win; then the machine track (TrackFrame) of that side;
// without a track, the best proposal of that side (EgoHand first, then the hands layer).
// An unedited box keeps its status/source when saved (machine boxes: accepted, source "tracker").
function initView(v, vs, tr = null) {
  v.proposals = vs.proposals;
  v.boxes = [];
  v.machineNone = new Set(); // sides the machine says "no hand": an untouched one posts no_hand from the tracker
  v.rejected = vs.labels.some((l) => l.status === "rejected");
  v.el.classList.toggle("rejected", v.rejected);
  v.touched = false;
  for (const side of SIDES) {
    const label = vs.labels.find((l) => l.side === side);
    if (label) {
      if (label.box) v.boxes.push({ side, origSide: side, box: { ...label.box }, status: label.status, source: label.source, edited: false, stored: true, origin: "reviewed" });
      continue;
    }
    if (tr) {
      const tb = tr.boxes.find((b) => b.cam === v.cam && b.side === side);
      if (tb?.box) v.boxes.push({ side, origSide: side, box: { ...tb.box }, status: "accepted", source: "tracker", edited: false, stored: false, origin: tb.origin, score: tb.score });
      else v.machineNone.add(side);
      continue;
    }
    const best = (model) =>
      vs.proposals.filter((p) => p.side === side && p.model === model).sort((a, b) => b.score - a.score)[0];
    const p = best("egohand") ?? best("hands_layer");
    if (p) v.boxes.push({ side, origSide: side, box: { ...p.box }, status: "accepted", source: p.model, edited: false, stored: false, proposal: p });
  }
}

// A box counts as edited when its geometry changed or its side differs from where it came from.
const changed = (b) => b.edited || b.side !== b.origSide;

function decision(v, side) {
  if (v.rejected) return { status: "rejected", box: null, source: "person" };
  const b = v.boxes.find((x) => x.side === side);
  if (!b) return { status: "no_hand", box: null, source: v.machineNone?.has(side) ? "tracker" : "person" };
  if (changed(b)) return { status: "corrected", box: b.box, source: "person" };
  return { status: b.status, box: b.box, source: b.source };
}

function frameLabels() {
  const who = reviewer();
  const labels = [];
  for (const v of S.views) {
    for (const side of SIDES) {
      const d = decision(v, side);
      // Edited boxes are rounded to 0.1 px; untouched ones go back bit-exact, so the backend sees them equal to
      // the machine box (or stored label) they came from and logs no intervention for them.
      const b = v.boxes.find((x) => x.side === side);
      const exact = d.box && b && !changed(b);
      const box = d.box ? (exact ? { ...d.box } : { x0: round1(d.box.x0), y0: round1(d.box.y0), x1: round1(d.box.x1), y1: round1(d.box.y1) }) : null;
      labels.push({ cam: v.cam, side, status: d.status, box, source: d.source, reviewer: who });
    }
  }
  return { segment_id: S.sid, t_ns: S.t, labels };
}

function renderFlags() {
  const parts = [];
  if (isKeyframe(S.t) && !S.flags.includes("keyframe")) parts.push(`<span class="chip keyframe">keyframe</span>`);
  for (const f of S.flags) parts.push(`<span class="chip ${chipClass(f)}">${f}</span>`);
  $("flags").innerHTML = parts.join("") || `<span class="muted">not flagged</span>`;
}

function renderStates() {
  const tbl = $("states");
  const head = S.views.map((v) => `<th>cam_${String(v.cam).padStart(2, "0")} <span class="muted">${v.info.name}</span></th>`).join("");
  const rows = SIDES.map((side) => {
    const cells = S.views
      .map((v) => {
        const d = decision(v, side);
        const pending = !(S.confirmed || v.touched || v.boxes.some((b) => b.stored)) ? " pending" : "";
        return `<td><span class="st ${d.status}${pending}" data-cam="${v.cam}" data-side="${side}">${d.status}</span></td>`;
      })
      .join("");
    return `<tr><td class="side">${side}</td>${cells}</tr>`;
  }).join("");
  tbl.innerHTML = `<tr><th class="side"></th>${head}</tr>${rows}`;
}

function touch(v) {
  v.touched = true;
  S.dirty = true;
  S.rev += 1;
  S.discardArmed = false;
  renderStates();
  draw(v);
}

// ---------------------------------------------------------------- drawing
function fit(v) {
  const c = v.canvas;
  const W = v.info.width;
  const H = v.info.height;
  const s = Math.min(c.width / W, c.height / H);
  return { s, ox: (c.width - W * s) / 2, oy: (c.height - H * s) / 2, W, H };
}

function draw(v) {
  const c = v.canvas;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(c.clientWidth * dpr));
  const h = Math.max(1, Math.round(c.clientHeight * dpr));
  if (c.width !== w || c.height !== h) {
    c.width = w;
    c.height = h;
  }
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  const f = fit(v);
  const img = v.hi?.complete && v.hi.naturalWidth ? v.hi : v.lo;
  if (img?.complete && img.naturalWidth) ctx.drawImage(img, f.ox, f.oy, f.W * f.s, f.H * f.s);
  const focused = S.focus === v.cam;
  const lw = (focused ? 2 : 1.5) * dpr;
  const font = `${Math.round((focused ? 13 : 11) * dpr)}px ui-monospace, Menlo, monospace`;
  const used = new Set(v.boxes.map((b) => b.proposal).filter(Boolean));

  // Proposals not in use: thin reference outlines (EgoHand solid, hands layer dashed), tagged only when focused.
  const near = (a, b) => Math.max(Math.abs(a.x0 - b.x0), Math.abs(a.y0 - b.y0), Math.abs(a.x1 - b.x1), Math.abs(a.y1 - b.y1)) < 0.5;
  for (const p of v.proposals) {
    if (used.has(p) || v.boxes.some((b) => near(b.box, p.box))) continue; // the machine box already is this proposal
    strokeBox(ctx, f, p.box, COLOR[p.side], lw * 0.6, p.model === "hands_layer", 0.6);
    if (focused) tag(ctx, f, p.box, `${p.model === "egohand" ? "EH" : "HL"} ${p.side[0].toUpperCase()} ${p.score.toFixed(2)}`, COLOR[p.side], font, 0.6);
  }
  // Working boxes: green when stored or edited; otherwise the side colour (machine track, or a proposal without
  // a track). Machine boxes: "T <origin>", held = dashed; stored labels thick with "*".
  for (const b of v.boxes) {
    const green = b.stored || changed(b);
    const color = green ? COLOR.label : COLOR[b.side];
    const dashed = !green && (b.source === "hands_layer" || b.origin === "held");
    strokeBox(ctx, f, b.box, color, lw * (b.stored ? 2.2 : 1.6), dashed, 1);
    const d = decision(v, b.side);
    const sc = b.proposal?.score ?? b.score;
    const score = typeof sc === "number" && !changed(b) && !b.stored ? ` ${sc.toFixed(2)}` : "";
    const who = green ? `${d.status}${b.stored ? "*" : ""}` : b.source === "tracker" ? `T ${b.origin}` : b.source === "egohand" ? "EH" : "HL";
    tag(ctx, f, b.box, `${b.side[0].toUpperCase()} ${who}${score}`, color, font, 1);
    if (focused && b === S.sel) drawHandles(ctx, f, b.box, dpr);
  }
}

function toCanvas(f, box) {
  return [f.ox + box.x0 * f.s, f.oy + box.y0 * f.s, (box.x1 - box.x0) * f.s, (box.y1 - box.y0) * f.s];
}

function strokeBox(ctx, f, box, color, lw, dashed, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.setLineDash(dashed ? [6 * lw, 4 * lw] : []);
  ctx.strokeRect(...toCanvas(f, box));
  ctx.restore();
}

function tag(ctx, f, box, text, color, font, alpha) {
  const [x, y] = toCanvas(f, box);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = font;
  const m = ctx.measureText(text);
  const th = parseInt(font, 10) + 4;
  const ty = y - th >= 0 ? y - th : y;
  ctx.fillStyle = "rgba(10,12,16,0.8)";
  ctx.fillRect(x, ty, m.width + 8, th);
  ctx.fillStyle = color;
  ctx.textBaseline = "top";
  ctx.fillText(text, x + 4, ty + 2);
  ctx.restore();
}

function drawHandles(ctx, f, box, dpr) {
  const [x, y, w, h] = toCanvas(f, box);
  ctx.save();
  ctx.setLineDash([4 * dpr, 3 * dpr]);
  ctx.strokeStyle = COLOR.sel;
  ctx.lineWidth = dpr;
  ctx.strokeRect(x - 2 * dpr, y - 2 * dpr, w + 4 * dpr, h + 4 * dpr);
  ctx.setLineDash([]);
  ctx.fillStyle = COLOR.sel;
  const r = 4 * dpr;
  for (const [hx, hy] of [[x, y], [x + w / 2, y], [x + w, y], [x, y + h / 2], [x + w, y + h / 2], [x, y + h], [x + w / 2, y + h], [x + w, y + h]]) {
    ctx.fillRect(hx - r, hy - r, 2 * r, 2 * r);
  }
  ctx.restore();
}

// ---------------------------------------------------------------- box editing (focused camera only)
function toNative(v, ev) {
  const rect = v.canvas.getBoundingClientRect();
  const dpr = v.canvas.width / rect.width;
  const f = fit(v);
  return { x: ((ev.clientX - rect.left) * dpr - f.ox) / f.s, y: ((ev.clientY - rect.top) * dpr - f.oy) / f.s, tol: (HANDLE_CSS_PX * dpr) / f.s, f };
}

function hitBox(box, x, y, tol) {
  const nearL = Math.abs(x - box.x0) <= tol;
  const nearR = Math.abs(x - box.x1) <= tol;
  const nearT = Math.abs(y - box.y0) <= tol;
  const nearB = Math.abs(y - box.y1) <= tol;
  const inX = x >= box.x0 - tol && x <= box.x1 + tol;
  const inY = y >= box.y0 - tol && y <= box.y1 + tol;
  if (!inX || !inY) return null;
  const v = nearT ? "n" : nearB ? "s" : "";
  const h = nearL ? "w" : nearR ? "e" : "";
  if (v || h) return v + h;
  return x > box.x0 && x < box.x1 && y > box.y0 && y < box.y1 ? "move" : null;
}

const CURSOR = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize", move: "move" };

function findHit(v, p) {
  const order = S.sel ? [S.sel, ...v.boxes.filter((b) => b !== S.sel)] : [...v.boxes].reverse();
  for (const b of order) {
    const h = hitBox(b.box, p.x, p.y, p.tol);
    if (h) return { box: b, mode: h };
  }
  const used = new Set(v.boxes.map((b) => b.proposal).filter(Boolean));
  for (const pr of v.proposals) {
    if (!used.has(pr) && hitBox(pr.box, p.x, p.y, 0) === "move") return { proposal: pr, mode: "adopt" };
  }
  return null;
}

function clampBox(v, b) {
  const W = v.info.width;
  const H = v.info.height;
  const x0 = Math.max(0, Math.min(b.x0, b.x1));
  const x1 = Math.min(W, Math.max(b.x0, b.x1));
  const y0 = Math.max(0, Math.min(b.y0, b.y1));
  const y1 = Math.min(H, Math.max(b.y0, b.y1));
  return { x0, y0, x1, y1 };
}

function onPointerDown(v, ev) {
  if (ev.button !== 0) return;
  if (S.watch.on) stopWatch("edit"); // a click while watching is a stop at a failure
  if (S.focus !== v.cam) {
    setFocus(v.cam);
    return;
  }
  if (editBlocked()) return;
  ev.preventDefault();
  v.canvas.setPointerCapture(ev.pointerId);
  const p = toNative(v, ev);
  const hit = findHit(v, p);
  if (hit?.mode === "adopt") {
    // Clicking an unused proposal makes it this side's box (replacing the current one).
    const pr = hit.proposal;
    v.boxes = v.boxes.filter((b) => b.side !== pr.side);
    v.machineNone?.delete(pr.side);
    const b = { side: pr.side, origSide: pr.side, box: { ...pr.box }, status: "accepted", source: pr.model, edited: false, stored: false, proposal: pr };
    v.boxes.push(b);
    S.sel = b;
    S.drag = { v, mode: "move", box: b, start: p, orig: { ...b.box }, moved: false };
    touch(v);
    return;
  }
  if (hit) {
    S.sel = hit.box;
    S.drag = { v, mode: hit.mode, box: hit.box, start: p, orig: { ...hit.box.box }, moved: false };
  } else {
    S.sel = null;
    S.drag = { v, mode: "new", box: null, start: p, orig: null, moved: false };
  }
  draw(v);
}

function onPointerMove(v, ev) {
  const p = toNative(v, ev);
  const d = S.drag;
  if (!d || d.v !== v) {
    if (S.focus === v.cam) {
      const hit = findHit(v, p);
      v.canvas.style.cursor = hit ? (hit.mode === "adopt" ? "pointer" : CURSOR[hit.mode]) : "crosshair";
    } else v.canvas.style.cursor = "pointer";
    return;
  }
  const dx = p.x - d.start.x;
  const dy = p.y - d.start.y;
  if (!d.moved && Math.hypot(dx, dy) < p.tol / 2) return;
  d.moved = true;
  if (d.mode === "new") {
    if (!d.box) {
      const taken = new Set(v.boxes.map((b) => b.side));
      const side = SIDES.find((s) => !taken.has(s)) ?? "right";
      v.boxes = v.boxes.filter((b) => b.side !== side);
      v.machineNone?.delete(side);
      d.box = { side, box: { x0: d.start.x, y0: d.start.y, x1: p.x, y1: p.y }, status: "corrected", source: "person", edited: true, stored: false };
      v.boxes.push(d.box);
      S.sel = d.box;
    }
    d.box.box = clampBox(v, { x0: d.start.x, y0: d.start.y, x1: p.x, y1: p.y });
  } else if (d.mode === "move") {
    const o = d.orig;
    const w = o.x1 - o.x0;
    const h = o.y1 - o.y0;
    const x0 = Math.max(0, Math.min(v.info.width - w, o.x0 + dx));
    const y0 = Math.max(0, Math.min(v.info.height - h, o.y0 + dy));
    d.box.box = { x0, y0, x1: x0 + w, y1: y0 + h };
  } else {
    const b = { ...d.orig };
    if (d.mode.includes("w")) b.x0 = d.orig.x0 + dx;
    if (d.mode.includes("e")) b.x1 = d.orig.x1 + dx;
    if (d.mode.includes("n")) b.y0 = d.orig.y0 + dy;
    if (d.mode.includes("s")) b.y1 = d.orig.y1 + dy;
    d.box.box = clampBox(v, b);
  }
  if (d.box) d.box.edited = true;
  touch(v);
}

function onPointerUp(v, ev) {
  const d = S.drag;
  if (!d || d.v !== v) return;
  S.drag = null;
  if (v.canvas.hasPointerCapture(ev.pointerId)) v.canvas.releasePointerCapture(ev.pointerId);
  if (d.box) {
    const b = d.box.box;
    if (b.x1 - b.x0 < MIN_BOX_PX || b.y1 - b.y0 < MIN_BOX_PX) {
      if (d.mode === "new") {
        v.boxes = v.boxes.filter((x) => x !== d.box);
        S.sel = null;
      } else d.box.box = d.orig;
    }
    if (d.moved) status(`cam ${v.cam} ${d.box.side}: ${fmtBox(d.box.box)} (native px)`);
  }
  touch(v);
}

const fmtBox = (b) => `[${Math.round(b.x0)}, ${Math.round(b.y0)}, ${Math.round(b.x1)}, ${Math.round(b.y1)}]`;

function setSide(side) {
  const v = S.views.find((x) => x.cam === S.focus);
  const b = S.sel;
  if (!v || !b || editBlocked()) return;
  if (b.side !== side) {
    const other = v.boxes.find((x) => x !== b && x.side === side);
    if (other) other.side = b.side;
    else v.machineNone?.delete(b.side); // that side is now empty by the reviewer's hand
    b.side = side;
  }
  touch(v);
  status(`cam ${v.cam}: box set to ${side}`);
}

function deleteSelected() {
  const v = S.views.find((x) => x.cam === S.focus);
  if (!v || !S.sel || editBlocked()) return;
  v.boxes = v.boxes.filter((b) => b !== S.sel);
  v.machineNone?.delete(S.sel.side);
  status(`cam ${v.cam}: ${S.sel.side} removed (no_hand)`);
  S.sel = null;
  touch(v);
}

function setFocus(cam) {
  const root = $("cams");
  S.focus = cam;
  S.sel = null;
  root.className = cam === null ? "cams grid" : "cams focus";
  for (const v of S.views) {
    v.el.classList.toggle("focused", v.cam === cam);
    if (v.cam === cam && !v.hi && S.t !== null) v.hi = loadImage(v, FOCUS_SIDE, S.loadSeq, "hi", S.t);
  }
  requestAnimationFrame(() => S.views.forEach(draw));
}

// ---------------------------------------------------------------- actions
function acceptAll() {
  if (editBlocked()) return;
  S.confirmed = true;
  S.dirty = true;
  S.rev += 1;
  S.discardArmed = false;
  let boxes = 0;
  for (const v of S.views) boxes += v.rejected ? 0 : v.boxes.length;
  renderStates();
  S.views.forEach(draw);
  status(`accepted as shown: ${boxes} boxes, ${S.views.length * 2 - boxes} no_hand/rejected · Enter saves`, "ok");
}

function rejectFocused() {
  if (editBlocked()) return;
  const v = S.views.find((x) => x.cam === S.focus);
  if (!v) {
    status("X rejects the focused camera: focus one first (click it or press 1-6)", "err");
    return;
  }
  v.rejected = !v.rejected;
  v.el.classList.toggle("rejected", v.rejected);
  touch(v);
  status(`cam ${v.cam} ${v.rejected ? "rejected" : "un-rejected"}`);
}

// Returns true when the frame set on screen is saved as it is now. Edits made while the request was in flight
// stay unsaved (dirty) and the answer only marks the frame set it was sent for.
async function save() {
  if (!S.sid || S.t === null || editBlocked()) return false;
  if (S.saving) {
    status("a save is already in flight: press Enter again once it is done", "err");
    return false;
  }
  if (!reviewer()) {
    status("Enter your name in the reviewer box first", "err");
    $("reviewer").focus();
    return false;
  }
  const body = frameLabels();
  const sent = { sid: S.sid, t: S.t, rev: S.rev };
  let res;
  S.saving = true;
  try {
    res = await postJSON("labels", body);
  } catch (e) {
    status(String(e.message ?? e), "err");
    return false;
  } finally {
    S.saving = false;
  }
  if (S.sid === sent.sid) S.reviewed.add(sent.t); // the server now holds a decision for every (cam, side)
  const current = S.sid === sent.sid && S.t === sent.t && S.rev === sent.rev;
  if (current) {
    S.dirty = false;
    S.discardArmed = false;
    S.confirmed = true;
    for (const v of S.views) for (const b of v.boxes) b.stored = true;
    renderStates();
    S.views.forEach(draw);
    status(`saved ${fmtTime(sent.t)}: stored ${res.stored ?? body.labels.length}`, "ok");
  } else {
    status(`saved ${fmtTime(sent.t)} as sent; edits made since are not saved yet: Enter saves them`, "err");
  }
  if (S.sid === sent.sid) await refreshAfterAction();
  return current;
}

// After a save or an accept the backend recomputes the machine track from the new labels: drop the cached track
// and frame sets, re-show the frame on screen when nothing on it is unsaved, and reload queue, progress, effort.
async function refreshAfterAction() {
  dropTrack();
  const tasks = [loadQueue(), loadSegments(), loadEffort()];
  if (S.t !== null) {
    const redraw = !S.dirty && !S.loading && !S.watch.on && !S.drag;
    tasks.push(redraw ? goto(S.t, { force: true, quiet: true }) : fetchTrack(S.t));
  }
  await Promise.all(tasks).catch((e) => status(String(e.message ?? e), "err"));
}

// ---------------------------------------------------------------- watch (W) + accept the watched run (A)
// Every frame set shown extends the run when it is at most one grid step past it; a jump starts a new run there.
function noteWatched(t) {
  const i = gridIndex(t);
  const a = gridIndex(S.anchor);
  const w = gridIndex(S.watchedTo);
  if (a >= 0 && i >= a && i <= w + 1) {
    if (i > w) S.watchedTo = t;
  } else {
    S.anchor = t;
    S.watchedTo = t;
  }
  renderWatch();
}

function startWatch() {
  if (!S.info || S.t === null) return;
  if (S.watch.on) {
    stopWatch("space");
    return;
  }
  if (S.dirty) {
    status("unsaved edits on this frame set: Enter saves them before watching", "err");
    return;
  }
  if (navBlocked()) return;
  S.watch.on = true;
  S.watch.how = null;
  S.watch.shown = [];
  S.watch.gen += 1;
  renderWatch();
  watchLoop(S.watch.gen);
}

function stopWatch(how) {
  if (!S.watch.on) return;
  S.watch.on = false;
  S.watch.how = how;
  S.fsCache.clear(); // prefetched frame sets are only trusted during playback
  renderWatch();
  if (["space", "key", "edit"].includes(how) && S.t !== null) {
    const prev = S.info.times_ns[gridIndex(S.t) - 1];
    const upTo = prev !== undefined && gridIndex(S.anchor) <= gridIndex(prev) ? `A accepts ${fmtTime(S.anchor)} → ${fmtTime(prev)}, ` : "";
    status(`stopped at ${fmtTime(S.t)}: ${upTo}fix this frame, Enter saves, W resumes (Shift+A: accept this frame too)`);
  }
}

async function watchLoop(gen) {
  const live = () => S.watch.on && S.watch.gen === gen;
  let last = performance.now();
  while (live()) {
    const ts = S.info.times_ns;
    const i = ts.indexOf(S.t);
    const next = ts[i + 1];
    if (next === undefined) {
      stopWatch("end");
      status(`end of ${S.sid}: A accepts the run up to ${fmtTime(S.t)}`, "ok");
      return;
    }
    prefetch(i + 1);
    try {
      await frameReady(next);
    } catch (e) {
      if (live()) {
        stopWatch("error");
        status(`watch stopped: ${e.message ?? e}`, "err");
      }
      return;
    }
    if (!live()) return;
    $("status").textContent = `watching ${S.sid} @ ${fmtTime(S.t)} · Space stops`;
    const wait = last + WATCH_MS - performance.now();
    if (wait > 0) await sleep(wait);
    if (!live()) return;
    last = performance.now();
    await goto(next, { watching: true, quiet: true });
    if (S.t !== next && live()) {
      stopWatch("error");
      return;
    }
  }
}

// A: accept the machine track over the watched run. While playing (or when playback hit the end) the run ends at
// the frame on screen; once the reviewer stopped (Space, a click, another key) the frame on screen is the failure,
// so the run ends one grid step before it. Shift+A always includes the frame on screen.
async function acceptRun(includeCurrent = false) {
  if (!S.info || S.t === null) return;
  if (S.saving || S.accepting) {
    status("a save or accept is in flight: press A again once it is done", "err");
    return;
  }
  if (!reviewer()) {
    status("Enter your name in the reviewer box first", "err");
    $("reviewer").focus();
    return;
  }
  const playing = S.watch.on;
  if (playing) stopWatch("accept");
  const ts = S.info.times_ns;
  const i = ts.indexOf(S.t);
  const a = gridIndex(S.anchor);
  const toCurrent = includeCurrent || playing || S.watch.how === "end";
  const j = Math.min(toCurrent ? i : i - 1, gridIndex(S.watchedTo));
  if (a < 0 || j < a) {
    status(`nothing watched to accept${a === i ? " before this frame" : ""}: W watches from here (Shift+A includes this frame)`, "err");
    return;
  }
  const body = { segment_id: S.sid, t0_ns: ts[a], t1_ns: ts[j], reviewer: reviewer() };
  const sid = S.sid;
  let res;
  S.accepting = true;
  try {
    res = await postJSON("accept_range", body);
  } catch (e) {
    status(String(e.message ?? e), "err");
    return;
  } finally {
    S.accepting = false;
  }
  if (S.sid !== sid) return;
  for (let k = a; k <= j; k++) S.reviewed.add(ts[k]);
  S.anchor = ts[j + 1] ?? null;
  S.watchedTo = S.anchor === null ? null : ts[j];
  renderWatch();
  const fix = j < i ? ` · ${fmtTime(S.t)} is next: fix it, Enter saves, W resumes` : "";
  status(`accepted ${fmtTime(body.t0_ns)} → ${fmtTime(body.t1_ns)} (${j - a + 1} frames): ${res.accepted} written, ${res.kept} kept${fix}`, "ok");
  await refreshAfterAction();
}

function renderWatch() {
  const el = $("watch");
  if (!el) return;
  const a = gridIndex(S.anchor);
  const w = gridIndex(S.watchedTo);
  const state = S.watch.on ? `<span class="chip play">watching</span>` : S.watch.how && S.watch.how !== "accept" ? `<span class="chip stop">stopped</span>` : "";
  const run = a < 0 ? "no run" : w >= a ? `run ${fmtTime(S.anchor)} → ${fmtTime(S.watchedTo)} · ${w - a + 1} fr` : `next run starts ${fmtTime(S.anchor)}`;
  el.innerHTML = `${state}<span class="mono">${run}</span>`;
}

async function loadEffort() {
  if (!S.sid) return;
  const sid = S.sid;
  let eff = null;
  try {
    eff = await getJSON(`effort/${enc(sid)}`);
  } catch {
    // older backend without /api/effort: no chip
  }
  if (sid !== S.sid) return;
  S.effort = eff;
  renderEffort();
}

function renderEffort() {
  const el = $("effort");
  const e = S.effort;
  el.hidden = !e;
  if (!e) return;
  const num = (x, d) => (typeof x === "number" ? x.toFixed(d) : "–");
  const pct = typeof e.share_corrected === "number" ? `${(100 * e.share_corrected).toFixed(1)}%` : "–";
  el.textContent = `${e.n_interventions} interventions · ${num(e.interventions_per_min, 2)}/min · ${pct} corrected`;
  el.title = Object.entries(e).map(([k, x]) => `${k}: ${typeof x === "number" && !Number.isInteger(x) ? x.toFixed(3) : x}`).join("\n");
}

async function nextQueue() {
  if (!S.queue.length) {
    status("queue is empty", "ok");
    return;
  }
  // In the queue: the item after this one. Saved (so dropped from the queue): the first later item in queue order.
  const open = (q) => q.t_ns !== S.t && !S.reviewed.has(q.t_ns);
  const i = S.queue.findIndex((q) => q.t_ns === S.t);
  const after = i >= 0 ? S.queue.slice(i + 1).find(open) : S.queue.find((q) => open(q) && q.t_ns > S.t);
  const item = after ?? S.queue.find(open) ?? S.queue[(i + 1) % S.queue.length];
  await goto(item.t_ns);
}

async function isReviewed(t) {
  if (S.reviewed.has(t)) return true;
  const fs = await getJSON(`frameset/${encodeURIComponent(S.sid)}/${t}`);
  const ok = fs.views.length > 0 && fs.views.every((vs) => vs.labels.length >= SIDES.length);
  if (ok) S.reviewed.add(t);
  return ok;
}

async function rangeFill() {
  if (!S.info || S.t === null || editBlocked()) return;
  if (!isKeyframe(S.t)) {
    status("Range fill runs between reviewed keyframes: go to a keyframe first", "err");
    return;
  }
  if ((S.dirty || !S.reviewed.has(S.t)) && !(await save())) return;
  const earlier = S.info.keyframes_ns.filter((k) => k < S.t).reverse().slice(0, 30);
  let t0 = null;
  try {
    for (const k of earlier) {
      if (await isReviewed(k)) {
        t0 = k;
        break;
      }
    }
    if (t0 === null) {
      status("no reviewed keyframe before this one (checked 30)", "err");
      return;
    }
    const res = await postJSON("range", { segment_id: S.sid, t0_ns: t0, t1_ns: S.t, reviewer: reviewer() });
    status(`range ${fmtTime(t0)} → ${fmtTime(S.t)}: filled ${res.filled}, queued ${res.queued.length}`, "ok");
    await Promise.all([loadQueue(), loadSegments()]);
  } catch (e) {
    status(String(e.message ?? e), "err");
  }
}

async function commit() {
  if (!S.sid) return;
  status(`committing ${S.sid} …`);
  try {
    const res = await postJSON(`commit/${encodeURIComponent(S.sid)}`);
    status(`committed ${res.n_labels} labels → ${res.path}`, "ok");
    await loadSegments();
  } catch (e) {
    status(String(e.message ?? e), "err");
  }
}

function step(delta) {
  if (!S.info) return;
  const ts = S.info.times_ns;
  let i = ts.indexOf(S.t);
  if (i < 0) i = ts.findIndex((x) => x > S.t) - (delta > 0 ? 1 : 0);
  const j = Math.max(0, Math.min(ts.length - 1, i + delta));
  if (ts[j] !== S.t) goto(ts[j]);
}

function updateTimeUi() {
  const ts = S.info.times_ns;
  const i = ts.indexOf(S.t);
  $("tlabel").textContent = `${fmtTime(S.t)}  ${i + 1}/${ts.length}${isKeyframe(S.t) ? " KF" : ""}`;
  if (i >= 0) $("scrub").value = String(i);
}

// ---------------------------------------------------------------- embedded Rerun web viewer
const V = { viewer: null, starting: null, url: null, warn: "", timeType: null, playing: false };

function viewerStatus(msg, err = false) {
  const el = $("viewer-status");
  el.textContent = msg;
  el.style.color = err ? "var(--red)" : "";
}

const viewerOpen = () => localStorage.getItem("hl.viewerOpen") !== "0";

function viewerWarnings(url) {
  const w = [];
  if (location.protocol === "https:" && /^rerun\+http:|^http:/.test(url)) w.push("https page + http catalog: mixed content may be blocked");
  const host = url.match(/^[a-z+]+:\/\/([^/:]+)/)?.[1];
  if (host && ["127.0.0.1", "localhost"].includes(host) && !["127.0.0.1", "localhost"].includes(location.hostname)) {
    w.push(`catalog at ${host} is the backend host's loopback, not this browser's`);
  }
  return w;
}

async function viewerOpenSegment() {
  if (!S.info) return;
  const url = S.info.viewer_url;
  if (!viewerOpen()) {
    viewerStatus("hidden");
    return;
  }
  if (url === V.url && (V.viewer || V.starting)) return;
  V.url = url;
  V.timeType = null;
  const warn = viewerWarnings(url);
  V.warn = warn.join("; ");
  try {
    if (!V.viewer) {
      viewerStatus("loading web viewer …");
      V.starting = (async () => {
        const errors = [];
        for (const src of VIEWER_ESM) {
          let viewer = null;
          try {
            viewerStatus(`loading web viewer from ${new URL(src).host} …`);
            const mod = await import(src);
            viewer = new mod.WebViewer();
            await viewer.start(url, $("viewer"), { hide_welcome_screen: true, width: "100%", height: "100%" });
            viewer.on("recording_open", () => viewerFollow(S.t));
            viewer.on("play", () => (V.playing = true));
            viewer.on("pause", () => (V.playing = false));
            V.viewer = viewer;
            return;
          } catch (e) {
            errors.push(`${new URL(src).host}: ${e?.message ?? e}`);
            try {
              viewer?.stop();
            } catch {
              // a viewer that never started has nothing to stop
            }
            $("viewer").replaceChildren();
          }
        }
        throw new Error(errors.join(" | "));
      })();
      await V.starting;
      V.starting = null;
    } else {
      await V.viewer.close("all").catch(() => {});
      await V.viewer.open(url);
    }
    viewerStatus(`${url}${warn.length ? ` · ${warn.join("; ")}` : ""}`, warn.length > 0);
    viewerFollow(S.t);
  } catch (e) {
    V.starting = null;
    viewerStatus(`viewer failed: ${e?.message ?? e}${warn.length ? ` (${warn.join("; ")})` : ""}`, true);
  }
}

async function viewerTimeType() {
  if (V.timeType) return V.timeType;
  const st = await V.viewer.get_state();
  for (const r of st.recordings ?? []) {
    const tl = (r.timelines ?? []).find((x) => x.name === TIMELINE);
    if (tl?.time_type) {
      V.timeType = tl.time_type;
      return V.timeType;
    }
  }
  return "duration";
}

let followTimer = null;
function viewerFollow(t) {
  if (!V.viewer || t === null || t === undefined) return;
  clearTimeout(followTimer);
  followTimer = setTimeout(async () => {
    try {
      const kind = await viewerTimeType();
      const value = BigInt(t);
      const opts = kind === "timestamp" ? { timestamp_nanos: value } : kind === "sequence" ? { sequence: value } : { duration_nanos: value };
      // Pause too: a catalog segment opens in "following" mode, which would pull the cursor back to the end.
      await V.viewer.set_time({ timeline: TIMELINE, play_state: "paused", ...opts });
      V.playing = false;
      viewerStatus(`${TIMELINE} → ${fmtTime(t)} · ${V.url}${V.warn ? ` · ${V.warn}` : ""}`, Boolean(V.warn));
    } catch (e) {
      // The recording may still be loading; recording_open retries.
      const msg = String(e?.message ?? e);
      viewerStatus(/no active recording/.test(msg) ? `no recording loaded yet from ${V.url} (still loading, or unreachable)` : `set_time: ${msg}`, true);
    }
  }, 120);
}

async function viewerTogglePlay() {
  if (!V.viewer) {
    status("viewer not running", "err");
    return;
  }
  try {
    const st = await V.viewer.get_state();
    const rec = (st.recordings ?? []).find((r) => r.store_id === st.active_store_id) ?? st.recordings?.[0];
    const playing = rec?.play_state ? rec.play_state !== "paused" : V.playing;
    await V.viewer.set_time({ timeline: TIMELINE, play_state: playing ? "paused" : "playing" });
    V.playing = !playing;
    status(`viewer ${V.playing ? "playing" : "paused"}`);
  } catch (e) {
    status(`viewer: ${e?.message ?? e}`, "err");
  }
}

function applyViewerPane() {
  $("viewer-pane").parentElement.classList.toggle("viewer-closed", !viewerOpen());
  requestAnimationFrame(() => S.views.forEach(draw));
}

// ---------------------------------------------------------------- keys + wiring
function onKey(ev) {
  const tag = ev.target?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
    if (ev.key === "Escape" || ev.key === "Enter") ev.target.blur();
    return;
  }
  if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const help = $("help");
  if (ev.key === "?") {
    help.hidden = !help.hidden;
    return;
  }
  if (!help.hidden) {
    if (ev.key === "Escape") help.hidden = true;
    return;
  }
  const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
  // Any other key while watching stops playback first (the reviewer saw a failure); Shift alone does nothing.
  if (S.watch.on && !["w", " ", "a", "Shift"].includes(k)) stopWatch("key");
  switch (k) {
    case "Escape":
      if (S.sel) {
        S.sel = null;
        S.views.forEach(draw);
      } else setFocus(null);
      break;
    case "1": case "2": case "3": case "4": case "5": case "6": {
      const cam = Number(k) - 1;
      if (S.views.some((v) => v.cam === cam)) setFocus(S.focus === cam ? null : cam);
      break;
    }
    case "l":
      setSide("left");
      break;
    case "r":
      if (ev.shiftKey || !S.sel) rangeFill();
      else setSide("right");
      break;
    case "Delete": case "Backspace":
      deleteSelected();
      break;
    case "w":
      startWatch();
      break;
    case "a":
      acceptRun(ev.shiftKey);
      break;
    case "f":
      acceptAll();
      break;
    case "x":
      rejectFocused();
      break;
    case "Enter":
      save();
      break;
    case "n":
      nextQueue();
      break;
    case "c":
      commit();
      break;
    case " ":
      if (S.watch.on) stopWatch("space");
      else viewerTogglePlay();
      break;
    case "ArrowLeft":
      step(-1);
      break;
    case "ArrowRight":
      step(1);
      break;
    default:
      return;
  }
  ev.preventDefault();
  ev.stopPropagation();
}

async function init() {
  const rv = $("reviewer");
  rv.value = localStorage.getItem("hl.reviewer") ?? "";
  // Backend defaults (reviewer name, dataset); localStorage overrides the reviewer. Optional route.
  try {
    const cfg = await getJSON("config");
    if (!rv.value && cfg.reviewer) rv.value = cfg.reviewer;
    if (cfg.dataset) {
      $("dataset").textContent = cfg.dataset;
      document.title = `hand-labeler · ${cfg.dataset}`;
    }
  } catch {
    // older backend without /api/config: keep the stored name
  }
  rv.addEventListener("input", () => localStorage.setItem("hl.reviewer", rv.value.trim()));
  $("segment").addEventListener("change", (e) => selectSegment(e.target.value));
  $("scrub").addEventListener("input", (e) => {
    const t = S.info?.times_ns[Number(e.target.value)];
    if (t !== undefined) $("tlabel").textContent = `${fmtTime(t)} …`;
  });
  $("scrub").addEventListener("change", (e) => {
    const t = S.info?.times_ns[Number(e.target.value)];
    if (t !== undefined) goto(t);
  });
  $("prev").addEventListener("click", () => step(-1));
  $("next").addEventListener("click", () => step(1));
  $("b-watch").addEventListener("click", startWatch);
  $("b-run").addEventListener("click", (e) => acceptRun(e.shiftKey));
  $("b-accept").addEventListener("click", acceptAll);
  $("b-save").addEventListener("click", save);
  $("b-next").addEventListener("click", nextQueue);
  $("b-range").addEventListener("click", rangeFill);
  $("b-commit").addEventListener("click", commit);
  $("b-help").addEventListener("click", () => ($("help").hidden = !$("help").hidden));
  $("help").addEventListener("click", (e) => {
    if (e.target === $("help")) $("help").hidden = true;
  });
  $("viewer-toggle").addEventListener("click", () => {
    localStorage.setItem("hl.viewerOpen", viewerOpen() ? "0" : "1");
    applyViewerPane();
    viewerOpenSegment();
  });
  // Buttons must not keep keyboard focus, or Space/Enter would click them again.
  for (const b of document.querySelectorAll("button")) b.addEventListener("mouseup", () => b.blur());
  // Window + capture phase: the Rerun viewer takes keyboard focus and its egui runner swallows key events.
  // The labeler keys win everywhere except in text fields (egui's own text input is an <input> too).
  window.addEventListener("keydown", onKey, true);
  window.addEventListener("beforeunload", (e) => {
    if (S.dirty || S.saving) e.preventDefault();
  });
  applyViewerPane();

  try {
    await loadSegments();
  } catch (e) {
    status(`backend: ${e.message ?? e}`, "err");
    return;
  }
  const hash = new URLSearchParams(location.hash.slice(1));
  const sid = hash.get("sid");
  const t = hash.get("t");
  const first = S.segments.find((s) => s.segment_id === sid) ?? S.segments[0];
  if (!first) {
    status("no segments", "err");
    return;
  }
  try {
    await selectSegment(first.segment_id, first.segment_id === sid && t ? Number(t) : null);
  } catch (e) {
    status(String(e.message ?? e), "err");
  }
}

window.__labeler = { S, V }; // for debugging from the console and for browser tests
init();
