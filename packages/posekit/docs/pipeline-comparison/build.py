#!/usr/bin/env python3
"""Build pipeline-comparison.html from the workflow result JSON + cropped figures.

v2 layout: visual-first — universal DAG as the TL;DR, chip matrix, headline
commonalities, tight paper cards with posekit rebuild one-liners, prose collapsed.
"""
import base64
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

ROLE_STYLE = {
    "detect":         ("#1f4230", "#4da884"),
    "track":          ("#173f42", "#4fa8ae"),
    "segment":        ("#173f42", "#4fa8ae"),
    "identity":       ("#3a2a4d", "#a583d1"),
    "crop/rewarp":    ("#2e3440", "#8a94a8"),
    "pose2d":         ("#1e3a5c", "#6ba3e8"),
    "dense-landmarks":("#1e3a5c", "#6ba3e8"),
    "lift3d":         ("#312a55", "#8f7fe8"),
    "triangulate":    ("#312a55", "#8f7fe8"),
    "calibrate":      ("#4d3a14", "#d1a94f"),
    "fit-model":      ("#4d2d14", "#e8914f"),
    "optimize":       ("#4d2d14", "#e8914f"),
    "render-verify":  ("#4d1f2e", "#e86a8f"),
    "temporal":       ("#3d3d29", "#b8b86a"),
    "other":          ("#26292f", "#6a7180"),
}
LEGEND_ROLES = [
    ("detect", "detect"), ("track", "track / segment"), ("identity", "identity"),
    ("crop/rewarp", "crop / rewarp"), ("pose2d", "2D estimation"),
    ("triangulate", "lift / triangulate"), ("calibrate", "calibrate"),
    ("fit-model", "fit / optimize"), ("render-verify", "render-verify"),
    ("temporal", "temporal"), ("other", "data / io"),
]


def dlbl(s: str, n: int = 16) -> str:
    s = (s or "").replace("\\", "").replace('"', r'\"')
    words, out, line = s.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > n and line:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return r"\n".join(out)


def _feedback_edges(nodes, edges):
    """Back edges via DFS (edge into a node on the current stack = feedback loop).
    List-order heuristics misclassify dependency edges like calib->lift."""
    adj = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
    ids = [n["id"] for n in nodes]
    color = dict.fromkeys(ids, 0)  # 0 white, 1 gray, 2 black
    back = set()

    def dfs(u):
        color[u] = 1
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == 1:
                back.add((u, v))
            elif color[v] == 0:
                dfs(v)
        color[u] = 2

    for nid in ids:
        if color[nid] == 0:
            dfs(nid)
    return back


def dag_svg(nodes, edges, prefix):
    def sid(x):
        return re.sub(r"\W", "_", x)

    L = ['digraph G { rankdir=LR; bgcolor="transparent"; ranksep=0.35; nodesep=0.3;']
    L.append('node [shape=box style="rounded,filled" fontname="Helvetica,Arial,sans-serif" '
             'fontsize=11 fontcolor="#eceef2" penwidth=1.2 margin="0.14,0.09"];')
    L.append('edge [fontname="Helvetica,Arial,sans-serif" fontsize=9 fontcolor="#9aa3b5" '
             'color="#5f6a80" arrowsize=0.7 penwidth=1.1];')
    ids = {n["id"] for n in nodes}
    back = _feedback_edges(nodes, edges)
    for n in nodes:
        fill, border = ROLE_STYLE.get(n.get("role", "other"), ROLE_STYLE["other"])
        L.append(f'"{sid(n["id"])}" [label="{dlbl(n["label"])}" fillcolor="{fill}" color="{border}"];')
    for e in edges:
        if e["from"] not in ids or e["to"] not in ids:
            continue
        fb = (e["from"], e["to"]) in back
        style = (' style=dashed constraint=false color="#8a8a5a" fontcolor="#b0b070"' if fb else "")
        L.append(f'"{sid(e["from"])}" -> "{sid(e["to"])}" [label="{dlbl(e.get("label", ""), 20)}"{style}];')
    L.append("}")
    svg = subprocess.run(["dot", "-Tsvg"], input="\n".join(L).encode(),
                         capture_output=True, check=True).stdout.decode()
    svg = svg[svg.index("<svg"):]
    m = re.search(r'viewBox="0(?:\.00)? 0(?:\.00)? ([0-9.]+) ([0-9.]+)"', svg)
    natural_w = float(m.group(1)) * 1.333 if m else 0.0
    # keep intrinsic width/height — the natural-size toggle relies on them
    svg = re.sub(r"<svg ", '<svg class="dag" ', svg, count=1)
    svg = svg.replace('id="', f'id="{sid(prefix)}-')
    return svg, natural_w


def dag_block(nodes, edges, prefix) -> str:
    svg, _ = dag_svg(nodes, edges, prefix)
    return f'<div class="dag-wrap" title="Click to view fullscreen">{svg}</div>'


def legend_html() -> str:
    sw = "".join(
        f'<span class="lg"><i style="background:{ROLE_STYLE[r][0]};border-color:{ROLE_STYLE[r][1]}"></i>{label}</span>'
        for r, label in LEGEND_ROLES)
    return f'<div class="legend">{sw}<span class="lg"><i class="dash"></i>feedback edge</span></div>'


data = json.load(open(ROOT / "json/result.json"))
papers = data["papers"]
tax = data["taxonomy"]

# papers added after the original 9-paper workflow (carry their own matrix_cells + rebuild)
for extra_path in sorted((ROOT / "json").glob("new-paper*.json")):
    extra = json.loads(extra_path.read_text())
    if extra["id"] not in {p["id"] for p in papers}:
        papers.append(extra)


def trim_preamble(md: str) -> str:
    i = md.find("# ")
    return md[i:] if i > 0 else md


inventory_md = trim_preamble(data["inventory"])
proposal_md = trim_preamble(data["proposal"])

PAPER_ORDER = ["megatrack", "umetrack", "kineo", "epfl-smart-kitchen", "assemblyhands",
               "assemblyx", "mamma", "egoexo-hands", "show3d", "hocap", "lookma"]
papers.sort(key=lambda p: PAPER_ORDER.index(p["id"]) if p["id"] in PAPER_ORDER else len(PAPER_ORDER))
N_WORD = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}
n_papers = len(papers)
n_word = N_WORD.get(n_papers, str(n_papers))

SHORT = {
    "megatrack": "MEgATrack", "umetrack": "UmeTrack", "kineo": "Kineo",
    "epfl-smart-kitchen": "EPFL Smart Kitchen", "assemblyhands": "AssemblyHands",
    "assemblyx": "AssemblyHands-X", "mamma": "MAMMA", "egoexo-hands": "EgoExo-Hands",
    "hocap": "HO-Cap", "lookma": "Look Ma, No Markers", "show3d": "SHOW3D",
}
LINKS = {
    "megatrack": [("paper", "https://research.facebook.com/file/977630383019036/MEgATrack-Monochrome-Egocentric-Articulated-Hand-Tracking-for-Virtual-Reality.pdf")],
    "umetrack": [("arXiv 2211.00099", "https://arxiv.org/abs/2211.00099"), ("code", "https://github.com/facebookresearch/UmeTrack")],
    "kineo": [("arXiv 2510.24464", "https://arxiv.org/abs/2510.24464"), ("code", "https://github.com/liris-xr/kineo")],
    "epfl-smart-kitchen": [("arXiv 2506.01608", "https://arxiv.org/abs/2506.01608")],
    "assemblyhands": [("arXiv 2304.12301", "https://arxiv.org/abs/2304.12301"), ("toolkit", "https://github.com/facebookresearch/assemblyhands-toolkit")],
    "assemblyx": [("arXiv 2509.23888", "https://arxiv.org/abs/2509.23888")],
    "mamma": [("arXiv 2506.13040", "https://arxiv.org/abs/2506.13040"), ("code", "https://github.com/cuevhv/mamma"), ("ours: packages/mamma", "")],
    "egoexo-hands": [("arXiv 2510.02601", "https://arxiv.org/abs/2510.02601")],
    "hocap": [("arXiv 2406.06843", "https://arxiv.org/abs/2406.06843"), ("code", "https://github.com/IRVLUTD/HO-Cap"), ("annotation", "https://github.com/IRVLUTD/HO-Cap-Annotation")],
    "lookma": [("arXiv 2410.11520", "https://arxiv.org/abs/2410.11520"), ("SIGGRAPH Asia 2024 · Microsoft", "")],
    "show3d": [("arXiv 2603.28760", "https://arxiv.org/abs/2603.28760"), ("follow-up to EgoExo-Hands ↑", "#egoexo-hands")],
}
FIG_NOTE = {
    "mamma": "No end-to-end pipeline figure exists in the paper; shown is Fig. 4 (MammaNet, the dense-landmark node). The full pipeline is the DAG below.",
}
KIND_LABEL = {
    "realtime-tracking": "realtime tracking",
    "offline-annotation": "offline annotation",
    "offline-capture": "offline capture",
    "hybrid": "hybrid",
}

# posekit rebuild one-liners, condensed from proposal §3
REBUILD = {
    "megatrack": 'PersonDetector(DetNet) → ops.crops[square+pad, R-mirror, +kpt channel] → TopDownPose2d(KeyNet: 2D + 1D rel-dist heatmaps) → glue[4-view reproj energy] → glue[fit: 26-DOF template LM + per-user scale φ] ⟳ glue[pose extrapolation → boxes + KeyNet input]',
    "umetrack": '(ROI from loop) → ops.crops[fisheye→virtual-pinhole ✦] → TopDownFeatureEncoder2d(ResNet) ✦ → glue[FTL + learnable multi-view fusion] → glue[Regressor-U scale] → glue[Regressor-K + SVD root + FK/LBS] ⟳ glue[RNN state + tracked pose → ROI]',
    "kineo": 'PersonDetector-via-pose-adapter (identity assumed; SAM2 re-ID only as optional multi-person pre-pass) → ops.crops[undistort images+kpts ✦] → TopDownPose2d(NLF / RTMPose / DWPose — swappable) → glue[weighted DLT] → glue[calibrate: essential graph → MST → 3-pass BA → metric scale] ⟳ glue[audio sync]',
    "epfl-smart-kitchen": 'PersonDetector-via-pose-adapter + glue[identity: Kinect-depth merge] → TopDownPose2d(RTMPose-x + RTMW-x) → glue[weighted SVD triangulation] → glue[fit: EasyMocap SMPL+MANO] → glue[verify: fit-vs-lift gate] ⟳ glue[temporal smoothing]',
    "assemblyhands": 'PersonDetector(body-kpt net) + glue[body-triangulate → virtual hand box] → ops.crops[ego fisheye→pinhole ✦] → TopDownFeatureEncoder2d(EfficientNet) ✦ → glue[feature volume → V2V → soft-argmax] → glue[SVEgoNet ego regressor] ⟳ glue[iterative re-crop]',
    "assemblyx": 'PromptableSegmenter(SAM — masks as fit target) → ops.crops[body/hand region] → TopDownPose2d(DWPose) + glue[median filter, edge-margin conf] → glue[weighted DLT + AssemblyHands hands] → glue[fit: SMPL-X] → glue[verify: diff-render vs mask inside fit loop]',
    "mamma": 'VideoSegmenter(SAM2 — masks as model input) + IdentityEncoder(CLIP) + glue[epipolar → Hungarian → cycle-consistency] → ops.crops[RGB+mask] → TopDownDenseLandmarks2d(MammaNet 512: μ/σ/vis/contact) → glue[ray-intersection init] → glue[fit: LBFGS SMPL-X] ⟳ glue[mask propagation]',
    "egoexo-hands": 'PersonDetector-via-TopDownPose2d-adapter(Sapiens-308) → ops.crops[fisheye→virtual-pinhole 256² ✦] → TopDownPose2d(Sapiens-42 ∥ InterNet — dual) → glue[RANSAC triangulation] → glue[calibrate: per-user LBS template] → glue[fit: IK] → glue[verify: overlay]',
    "hocap": 'VideoSegmenter(SAM2) + [external FoundationPose object branch] → TopDownPose2d(MediaPipe — no conf) → glue[triangulate-all-pairs, conf manufactured from reproj error, spline gap-fill] → glue[fit: MANO, then joint hand-object SDF] → glue[verify: render-to-views] ⟳ glue[object reproj → next-frame init]',
}

# headline compressions of taxonomy.commonalities / .divergences (full text in collapsibles)
COMMON_HEADLINES = [
    ("One boundary datatype", "(x, y, confidence) per (view, frame, instance) in a named skeleton + a camera model — shared by all; only HO-Cap's MediaPipe lacks the confidence channel."),
    ("Confidence-weighted everything", "gating, triangulation, BA and robust fits all consume it; HO-Cap, whose 2D net supplies none, selects triangulation candidates by reprojection error instead."),
    ("Crops are the currency", "the 2D net always sees a canonicalized instance-centered view; 5/11 need fisheye→virtual-pinhole rewarp."),
    ("The 2D estimator is a swappable slot", "KeyNet, RTMPose/DWPose/NLF, Sapiens/InterNet, MediaPipe, MammaNet — detector, tracker and pose mix freely."),
    ("The tail is fuse-then-fit", "triangulate or lift, then fit MANO / SMPL-X / a template hand — 9/11 emit a parametric model with a per-user shape prior."),
    ("Temporal info is near-universal", "realtime systems close feedback loops into detection and 2D; offline ones add feed-forward smoothing terms."),
    ("The localizer is an interface", "satisfied by a box net, a segmenter, a pose-net-as-detector, or the tracking loop itself."),
    ("Geometry glue never generalizes", "every paper hand-rolls its own triangulation / association / BA / fitting — the networks are the reusable part."),
]
DIVERGE_HEADLINES = [
    ("Learned vs classical 3D fusion", "UmeTrack and MVExoNet learn the lift from features; the rest use DLT/RANSAC; MEgATrack skips lift entirely (fit-only)."),
    ("Sparse joints vs dense landmarks", "MAMMA's 512 surface points with per-point uncertainty vs everyone else's 21/42/COCO skeletons — changes the fit contract."),
    ("Masks: inputs vs targets vs absent", "MAMMA/HO-Cap feed masks into the model; AssemblyX uses them only as a silhouette loss; the rest use none."),
    ("Calibration-free vs given", "only Kineo solves sync + extrinsics + intrinsics + scale from keypoints; everyone else assumes calibration."),
    ("Realtime loop vs offline batch", "feedback that rewrites earlier stages (MEgATrack, UmeTrack, HO-Cap objects) vs feed-forward smoothing."),
    ("Identity solved vs assumed", "MAMMA epipolar+Hungarian, EPFL depth-merge, HO-Cap SAM2 tracks; the rest fix identity by rig geometry."),
    ("Regress vs optimize the fit", "UmeTrack/SVEgoNet regress model params with a net; the rest run explicit LM / LBFGS / IK optimizers."),
    ("Metric scale", "free with calibration; Kineo needs a bone-length or MoGe prior; VR trackers solve a per-user scale."),
    ("Multi-modal vs pure RGB", "EPFL/HO-Cap/EgoExo-Hands fuse depth, ego streams or mocap; the others are RGB (+ masks) only."),
    ("Hand-object coupling", "only HO-Cap jointly optimizes hand + object 6DoF with an interpenetration SDF."),
]

STAGE_ROLE = {
    "capture": "other", "detect": "detect", "identity": "identity", "crop": "crop/rewarp",
    "pose2d": "pose2d", "lift3d": "triangulate", "calibrate": "calibrate",
    "fit": "fit-model", "verify": "render-verify", "feedback": "temporal",
}
STAGE_SHORT = {
    "capture": "capture", "detect": "detect / segment", "identity": "identity",
    "crop": "crop / rewarp", "pose2d": "2D estimation", "lift3d": "lift / triangulate",
    "calibrate": "calibrate", "fit": "parametric fit", "verify": "verify", "feedback": "temporal",
}


def b64img(pid: str) -> str:
    p = ROOT / "figures" / f"{pid}.png"
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


_ABBREV = ["Fig.", "Figs.", "Sec.", "Eq.", "et al.", "e.g.", "i.e.", "vs.", "approx."]


def sentences(s: str, n: int) -> str:
    s = s or ""
    for a in _ABBREV:
        s = s.replace(a, a.replace(".", "\x00"))
    parts = re.split(r"(?<=[.!?]) +", s)
    return " ".join(parts[:n]).replace("\x00", ".")


def caption_short(s: str) -> str:
    s = re.sub(r"^\s*Fig(?:ure)?\.?\s*\d+[.:]?\s*", "", s or "")
    out = sentences(s, 1)
    if len(out) < 30:  # degenerate lead like "Hand-tracking pipeline." — include the next sentence
        out = sentences(s, 2)
    return out


# ---------- universal DAG ----------
UNIVERSAL_NODES = [
    {"id": "capture", "label": "capture / calibration", "role": "other"},
    {"id": "detect", "label": "detect / segment / localize", "role": "detect"},
    {"id": "identity", "label": "cross-view + temporal identity", "role": "identity"},
    {"id": "crop", "label": "crop / rewarp (camera-aware)", "role": "crop/rewarp"},
    {"id": "pose2d", "label": "2D estimation (kpts + conf)", "role": "pose2d"},
    {"id": "lift", "label": "lift / triangulate", "role": "triangulate"},
    {"id": "calib", "label": "auto-calibration / scale", "role": "calibrate"},
    {"id": "fit", "label": "parametric fit (MANO / SMPL-X / template)", "role": "fit-model"},
    {"id": "verify", "label": "render-back verify", "role": "render-verify"},
]
UNIVERSAL_EDGES = [
    {"from": "capture", "to": "detect", "label": "synced views + camera model"},
    {"from": "detect", "to": "identity", "label": "boxes / masks (+track ids)"},
    {"from": "identity", "to": "crop", "label": "per-instance ROI"},
    {"from": "crop", "to": "pose2d", "label": "canonical crop + virtual cam"},
    {"from": "pose2d", "to": "lift", "label": "(x, y, conf) in skeleton convention"},
    {"from": "pose2d", "to": "calib", "label": "2D kpts + conf"},
    {"from": "calib", "to": "lift", "label": "extrinsics / scale"},
    {"from": "lift", "to": "fit", "label": "3D joints + weights"},
    {"from": "fit", "to": "verify", "label": "posed mesh"},
    {"from": "verify", "to": "fit", "label": "silhouette / reproj loss"},
    {"from": "fit", "to": "detect", "label": "pose t-1 extrapolation (realtime)"},
    {"from": "fit", "to": "pose2d", "label": "kpt-feature augmentation"},
]

# ---------- paper cards ----------
cards = []
for i, p in enumerate(papers, 1):
    pid = p["id"]
    links = " · ".join(
        f'<a href="{esc(u)}">{esc(t)}</a>' if u else f"<span>{esc(t)}</span>"
        for t, u in LINKS.get(pid, [])
    )
    nodes_rows = "".join(
        f'<tr><td>{esc(n["label"])}</td><td class="mono">{esc(n["role"])}</td>'
        f'<td>{esc(n.get("models", ""))}</td><td class="dim">{esc(n.get("desc", ""))}</td></tr>'
        for n in p["nodes"]
    )
    fignote = FIG_NOTE.get(pid, "")
    fignote_html = f'<p class="fignote">{esc(fignote)}</p>' if fignote else ""
    traits_top = p["traits"][:5]
    traits_rest = p["traits"][5:]
    traits_rest_html = (
        f'<h4>All traits</h4><div class="traits">{"".join(f"<span class=\"chip trait\">{esc(t)}</span>" for t in p["traits"])}</div>'
        if traits_rest else "")
    cards.append(f"""
<article class="paper" id="{pid}">
  <header>
    <h3><span class="num">{i}</span> {esc(SHORT.get(pid, p["name"].split(":")[0]))}</h3>
    <span class="chip kind">{esc(KIND_LABEL.get(p["kind"], p["kind"]))}</span>
    <span class="chip">{esc(str(p["year"]))}</span>
    <span class="links">{links}</span>
  </header>
  <div class="body">
    <p class="summary">{esc(sentences(p["summary"], 2))}</p>
    <p class="kvline"><b>in</b> {esc(p["rig"])} <b class="sep">out</b> {esc(p["outputs"])}</p>
    <div class="figwrap">
      <img src="{b64img(pid)}" alt="{esc(SHORT.get(pid, p["name"].split(":")[0]))} pipeline figure" loading="lazy" title="{esc(p["figure_caption"])}">
      <p class="figcap">{esc(caption_short(p["figure_caption"]))}</p>
      {fignote_html}
    </div>
    <div class="dagrow">
      {dag_block(p["nodes"], p["edges"], pid)}
    </div>
    <p class="rebuild"><b>as posekit</b> {esc(REBUILD.get(pid) or p.get("rebuild", ""))}</p>
    <div class="traits">{"".join(f'<span class="chip trait">{esc(t)}</span>' for t in traits_top)}{f'<span class="chip trait more">+{len(traits_rest)} more in details</span>' if traits_rest else ""}</div>
    <details>
      <summary>Details — node table ({len(p["nodes"])} nodes), full caption, posekit mapping</summary>
      <div class="inner">
        <p class="kv"><b>Figure caption</b> {esc(p["figure_caption"])}</p>
        <p class="kv"><b>posekit mapping</b> {esc(p.get("posekit_notes", ""))}</p>
        {traits_rest_html}
        <table class="nodes">
          <thead><tr><th>node</th><th>role</th><th>models / methods</th><th>what it does</th></tr></thead>
          <tbody>{nodes_rows}</tbody>
        </table>
      </div>
    </details>
  </div>
</article>""")

# ---------- matrix ----------
stages = tax["canonical_stages"]
mat_by_paper = {r["paper_id"]: {c["stage_id"]: c for c in r["cells"]} for r in tax["matrix"]}
for p in papers:  # post-workflow papers carry their own row
    if p["id"] not in mat_by_paper and "matrix_cells" in p:
        mat_by_paper[p["id"]] = {c["stage_id"]: c for c in p["matrix_cells"]}
head = "".join(
    f'<th title="{esc(s["desc"][:300])}" style="border-bottom: 2px solid {ROLE_STYLE[STAGE_ROLE.get(s["id"], "other")][1]}">'
    f'{esc(STAGE_SHORT.get(s["id"], s["name"]))}</th>'
    for s in stages
)
rows = []
for p in papers:
    cells = mat_by_paper.get(p["id"], {})
    tds = []
    for s in stages:
        c = cells.get(s["id"], {})
        fill = c.get("fill", "—")
        note = c.get("note", "")
        absent = fill.strip() in ("—", "-", "")
        tip = esc((fill + (" — " + note if note else "")).strip())
        tds.append(f'<td class="{"absent" if absent else ""}" title="{tip}"><span class="clamp">{esc(fill)}</span></td>')
    rows.append(f'<tr><th><a href="#{p["id"]}">{esc(SHORT.get(p["id"], p["name"].split(":")[0]))}</a></th>{"".join(tds)}</tr>')

matrix_html = f"""
<div class="matrix-wrap">
<table class="matrix">
<thead><tr><th>paper</th>{head}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
"""

common_full = "".join(f"<li>{esc(c)}</li>" for c in tax["commonalities"])
diverge_full = "".join(f"<li>{esc(d)}</li>" for d in tax["divergences"])
common_hl = "".join(f"<li><b>{esc(h)}</b> — {esc(t)}</li>" for h, t in COMMON_HEADLINES)
diverge_hl = "".join(f"<li><b>{esc(h)}</b> — {esc(t)}</li>" for h, t in DIVERGE_HEADLINES)

toc_papers = "".join(f'<a href="#{p["id"]}">{esc(SHORT.get(p["id"], p["name"].split(":")[0]))}</a>' for p in papers)

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keypoint Pipeline Comparison — {n_papers} papers vs posekit</title>
<style>
  :root {{
    --bg: oklch(18% .012 250); --surface: oklch(22% .014 250); --panel: oklch(25% .016 250);
    --line: oklch(32% .015 250); --line-strong: oklch(42% .02 250);
    --text: oklch(90% .01 250); --text-dim: oklch(72% .015 250); --text-faint: oklch(58% .015 250);
    --accent: oklch(72% .14 245); --accent-soft: oklch(35% .07 245);
    --mono: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
  body {{ margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans);
         font-size: 15.5px; line-height: 1.55; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  code, .mono {{ font-family: var(--mono); font-size: .87em; }}
  code {{ background: var(--panel); padding: 1px 5px; border-radius: 4px; border: 1px solid var(--line); }}
  pre {{ background: oklch(15% .01 250); border: 1px solid var(--line); border-radius: 8px;
        padding: 12px 14px; overflow-x: auto; font-family: var(--mono); font-size: 12.5px; }}

  header.page {{ padding: 30px clamp(18px,4vw,48px) 20px; border-bottom: 1px solid var(--line); }}
  header.page h1, header.page p.sub {{ max-width: 1244px; margin-left: auto; margin-right: auto; }}
  header.page h1 {{ margin: 0 auto 6px; font-size: 24px; text-wrap: balance; }}
  header.page p.sub {{ margin: 0 auto; color: var(--text-dim); }}

  nav.toc {{ position: sticky; top: 0; z-index: 30; display: flex; flex-wrap: wrap; gap: 3px 12px;
            justify-content: center; padding: 9px clamp(18px,4vw,48px);
            background: color-mix(in oklch, var(--bg) 88%, transparent);
            backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); font-size: 12.5px; }}
  nav.toc a {{ color: var(--text-dim); padding: 3px 6px; border-radius: 5px; }}
  nav.toc a:hover {{ color: var(--text); background: var(--panel); text-decoration: none; }}
  nav.toc a.hot {{ color: var(--accent); }}

  main {{ padding: 0 clamp(18px,4vw,48px) 90px; max-width: 1340px; margin: 0 auto; }}
  section {{ margin-top: 44px; }}
  h2 {{ font-size: 20px; margin: 0 0 4px; }}
  h2 + p.lead {{ margin: 0 0 14px; color: var(--text-dim); max-width: 95ch; }}
  h4 {{ margin: 14px 0 6px; font-size: 11.5px; text-transform: uppercase; letter-spacing: .07em;
       color: var(--text-faint); font-weight: 700; }}

  .chip {{ font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .04em;
          border: 1px solid var(--line-strong); border-radius: 999px; padding: 2px 9px; color: var(--text-dim);
          display: inline-block; }}
  .chip.kind {{ color: var(--accent); border-color: var(--accent-soft); }}
  .chip.trait {{ margin: 2px 3px 2px 0; letter-spacing: 0; text-transform: none; font-size: 11px; font-weight: 500; }}
  .chip.trait.more {{ color: var(--text-faint); border-style: dashed; }}

  /* hero */
  .hero {{ margin-top: 26px; }}
  .hero h2 {{ font-size: 22px; text-align: center; }}
  .hero p.claim {{ text-align: center; color: var(--text-dim); margin: 4px auto 14px; max-width: 90ch; }}
  .takeaways {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }}
  @media (max-width: 900px) {{ .takeaways {{ grid-template-columns: 1fr; }} }}
  .takeaways div {{ border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
                   padding: 12px 16px; font-size: 13.5px; color: var(--text-dim); }}
  .takeaways b {{ color: var(--text); display: block; margin-bottom: 3px; font-size: 14px; }}

  .legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 4px 0 10px; font-size: 12px;
            color: var(--text-dim); justify-content: center; }}
  .legend .lg {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend i {{ width: 14px; height: 11px; border-radius: 3px; border: 1px solid; display: inline-block; }}
  .legend i.dash {{ border: none; border-top: 2px dashed #8a8a5a; height: 0; width: 18px; border-radius: 0; }}

  .dag-wrap {{ overflow: hidden; border: 1px solid var(--line); border-radius: 8px;
              background: oklch(15% .01 250); padding: 12px 14px; cursor: zoom-in; }}
  .dag-wrap:hover {{ border-color: var(--line-strong); }}
  .dag-wrap svg.dag {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}

  /* DAG lightbox */
  dialog#dagbox {{ width: 100vw; height: 100vh; max-width: 100vw; max-height: 100vh;
    margin: 0; padding: 0; border: none; background: oklch(13% .01 250); overflow: auto;
    display: none; place-items: center; cursor: zoom-in; }}
  dialog#dagbox[open] {{ display: grid; }}
  dialog#dagbox::backdrop {{ background: rgba(0,0,0,.6); }}
  dialog#dagbox svg {{ max-width: 96vw; max-height: 90vh; width: auto; height: auto; display: block; }}
  dialog#dagbox.zoom {{ place-items: unset; cursor: zoom-out; }}
  dialog#dagbox.zoom svg {{ max-width: none; max-height: none; margin: 40px; }}
  dialog#dagbox .close-hint {{ position: fixed; top: 12px; right: 18px; z-index: 2;
    font-family: var(--mono); font-size: 12px; color: var(--text-faint);
    background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; }}

  /* headline lists */
  .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  @media (max-width: 980px) {{ .split {{ grid-template-columns: 1fr; }} }}
  .split ul.hl {{ margin: 6px 0 0; padding-left: 18px; }}
  .split ul.hl li {{ margin: 7px 0; font-size: 13.5px; color: var(--text-dim); }}
  .split ul.hl li b {{ color: var(--text); }}
  .split ul.hl li::marker {{ color: var(--accent); }}
  .split details ul {{ padding-left: 18px; }}
  .split details li {{ margin: 8px 0; font-size: 13px; color: var(--text-dim); }}

  /* matrix */
  .matrix-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 10px;
                 background: var(--surface); max-height: 78vh; }}
  table.matrix {{ border-collapse: collapse; font-size: 11.8px; min-width: 1420px; }}
  table.matrix th, table.matrix td {{ padding: 6px 9px; border-bottom: 1px solid var(--line);
    border-right: 1px solid var(--line); text-align: left; vertical-align: top; }}
  table.matrix thead th {{ position: sticky; top: 0; background: var(--panel); z-index: 2;
    font-size: 11px; min-width: 120px; }}
  table.matrix tbody th {{ position: sticky; left: 0; background: var(--panel); z-index: 1;
    font-size: 12.5px; white-space: nowrap; }}
  table.matrix thead th:first-child {{ left: 0; z-index: 3; }}
  table.matrix td .clamp {{ display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    overflow: hidden; color: var(--text-dim); }}
  table.matrix td.absent .clamp {{ color: var(--text-faint); }}

  /* paper cards */
  article.paper {{ border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
                  margin-top: 24px; overflow: hidden; }}
  article.paper > header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px;
    padding: 12px 20px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }}
  article.paper > header h3 {{ margin: 0; font-size: 17px; }}
  article.paper .num {{ color: var(--accent); font-family: var(--mono); margin-right: 4px; }}
  article.paper .links {{ margin-left: auto; font-size: 12.5px; color: var(--text-faint); }}
  article.paper .body {{ padding: 12px 20px 18px; }}
  .summary {{ margin: 2px 0 6px; color: var(--text-dim); max-width: 120ch; font-size: 14px; }}
  .kvline {{ margin: 0 0 12px; font-size: 12.5px; color: var(--text-faint); max-width: 130ch; }}
  .kvline b {{ color: var(--accent); font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; margin-right: 5px; }}
  .kvline b.sep {{ margin-left: 14px; }}
  .figwrap {{ background: #fff; border: 1px solid var(--line-strong); border-radius: 8px; padding: 10px; }}
  .figwrap img {{ width: 100%; height: auto; display: block; }}
  .figcap {{ margin: 7px 2px 0; font-size: 11.5px; color: #555; line-height: 1.4; }}
  .fignote {{ margin: 5px 2px 0; font-size: 11.5px; color: oklch(55% .13 85); font-style: italic; }}
  .dagrow {{ margin-top: 10px; }}
  .rebuild {{ margin: 10px 0 8px; font-family: var(--mono); font-size: 12px; line-height: 1.7;
             color: var(--text-dim); background: var(--panel); border: 1px solid var(--line);
             border-radius: 8px; padding: 9px 13px; }}
  .rebuild b {{ color: var(--accent); font-size: 10px; text-transform: uppercase; letter-spacing: .06em; margin-right: 8px; }}

  details {{ border: 1px solid var(--line); border-radius: 8px; margin-top: 12px; background: var(--panel); }}
  details summary {{ cursor: pointer; padding: 8px 14px; font-size: 12.5px; color: var(--text-dim); user-select: none; }}
  details summary:hover {{ color: var(--text); }}
  details[open] summary {{ border-bottom: 1px solid var(--line); }}
  details .inner {{ padding: 10px 14px 14px; overflow-x: auto; }}
  .kv {{ margin: 4px 0 10px; font-size: 13px; color: var(--text-dim); }}
  .kv b {{ color: var(--text); font-family: var(--mono); font-size: 10.5px; text-transform: uppercase; margin-right: 6px; }}
  table.nodes {{ border-collapse: collapse; font-size: 12.5px; width: 100%; }}
  table.nodes th, table.nodes td {{ text-align: left; padding: 6px 9px; border-bottom: 1px solid var(--line); vertical-align: top; }}
  table.nodes th {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-faint); }}
  table.nodes td.mono {{ color: var(--accent); white-space: nowrap; }}
  table.nodes td.dim {{ color: var(--text-dim); }}

  /* abstraction section */
  .amendments {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 12px; }}
  @media (max-width: 900px) {{ .amendments {{ grid-template-columns: 1fr; }} }}
  .amendments div {{ border: 1px solid var(--line); border-radius: 10px; background: var(--surface);
                    padding: 12px 16px; font-size: 13.5px; color: var(--text-dim); }}
  .amendments b {{ color: var(--text); display: block; margin-bottom: 3px; }}
  .amendments .n {{ color: var(--accent); font-family: var(--mono); margin-right: 6px; }}
  .verdict {{ border: 1px solid var(--accent-soft); border-radius: 10px; padding: 13px 18px;
             background: color-mix(in oklch, var(--accent-soft) 25%, var(--surface));
             font-size: 14.5px; max-width: 120ch; }}

  /* markdown containers */
  .md {{ max-width: 115ch; }}
  .md h1 {{ font-size: 20px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }}
  .md h2 {{ font-size: 17px; margin-top: 28px; }}
  .md h3 {{ font-size: 15px; margin-top: 20px; }}
  .md table {{ border-collapse: collapse; font-size: 13px; margin: 10px 0; display: block; overflow-x: auto; }}
  .md th, .md td {{ border: 1px solid var(--line); padding: 6px 10px; text-align: left; vertical-align: top; }}
  .md th {{ background: var(--panel); }}
  .md li {{ margin: 4px 0; }}
  .md blockquote {{ border-left: 3px solid var(--accent-soft); margin: 10px 0; padding: 2px 14px; color: var(--text-dim); }}

  @media print {{
    :root {{ --bg:#fff; --surface:#fff; --panel:#f4f4f0; --line:#ccc; --line-strong:#999;
      --text:#111; --text-dim:#333; --text-faint:#666; --accent:#1436c9; }}
    nav.toc {{ display: none; }}
    article.paper, .matrix-wrap {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<header class="page">
  <h1>Keypoint pipeline comparison — {n_papers} papers, one abstraction</h1>
  <p class="sub">MEgATrack · UmeTrack · Kineo · EPFL Smart Kitchen · AssemblyHands · AssemblyHands-X · MAMMA · EgoExo-Hands · HO-Cap · Look Ma, No Markers — decomposed into DAGs, compared stage by stage, and rebuilt as posekit roles.</p>
</header>

<nav class="toc" aria-label="Sections">
  <a href="#universal" class="hot">Universal shape</a>
  <a href="#matrix" class="hot">Stage matrix</a>
  <a href="#patterns">Patterns</a>
  {toc_papers}
  <a href="#abstraction" class="hot">The abstraction</a>
</nav>

<main>

<section class="hero" id="universal">
  <h2>All {n_word} pipelines are the same DAG</h2>
  <p class="claim">frames → detect / segment / localize → crop / rewarp → 2D keypoints + confidence → associate / lift → parametric fit. Solid = the common backbone; dashed = optional stages and feedback edges. The colors below are used in every diagram on this page.</p>
  {legend_html()}
  {dag_block(UNIVERSAL_NODES, UNIVERSAL_EDGES, "universal")}
  <div class="takeaways">
    <div><b>One boundary datatype</b> (x, y, confidence) per (view, frame, instance) in a named skeleton, plus a camera model — posekit's <code>Keypoints2d</code> already is this. All but HO-Cap carry per-keypoint confidence end-to-end; its MediaPipe lacks the channel and falls back to reprojection-error selection.</div>
    <div><b>Networks vs glue</b> The swappable networks are detect / segment / 2D-estimate. Triangulation, association, calibration and fitting are hand-rolled per paper — consumer glue, confirmed in every paper surveyed.</div>
    <div><b>design.md holds</b> The survey confirms posekit's role taxonomy, with 4 amendments — all in the geometry tail, none touching the network-role set. See <a href="#abstraction">the abstraction</a>.</div>
  </div>
</section>

<section id="matrix">
  <h2>Stage × paper matrix</h2>
  <p class="lead">What fills each canonical stage in each paper — same colors as the diagrams (column underlines). Hover any cell for the full text; "—" means the stage is absent.</p>
  {matrix_html}
</section>

<section id="patterns">
  <h2>What's shared, where they fork</h2>
  <div class="split">
    <div>
      <h4>Commonalities</h4>
      <ul class="hl">{common_hl}</ul>
      <details><summary>Full analysis</summary><div class="inner"><ul>{common_full}</ul></div></details>
    </div>
    <div>
      <h4>Divergences</h4>
      <ul class="hl">{diverge_hl}</ul>
      <details><summary>Full analysis</summary><div class="inner"><ul>{diverge_full}</ul></div></details>
    </div>
  </div>
</section>

<section id="papers">
  <h2>The {n_word} pipelines</h2>
  <p class="lead">Each card: the paper's own figure, our extracted DAG (click any diagram to view it fullscreen; click again for 100%), and the pipeline rewritten as posekit roles — <code>Role(model)</code> = posekit-owned, <code>glue[…]</code> = consumer, ✦ = capability posekit doesn't have yet, ⟳ = feedback loop.</p>
  {"".join(cards)}
</section>

<section id="abstraction">
  <h2>The abstraction</h2>
  <p class="verdict"><b>Verdict:</b> the survey confirms design.md's role taxonomy — the four inference paradigms hold, no fifth appears, and the boundary datatype is universal. Four amendments are needed, all in the geometry tail.</p>
  <div class="amendments">
    <div><b><span class="n">1</span>Promote crop / rewarp to a first-class node</b> 5/11 papers need fisheye→virtual-pinhole or undistort-with-keypoints. The crop must be camera-model-aware and return the virtual camera alongside the crop, so the 3D stage gets correct rays. Currently a "later" bullet; should be Phase 3.x.</div>
    <div><b><span class="n">2</span>New role: TopDownFeatureEncoder2d</b> UmeTrack and MVExoNet never emit keypoints — they emit per-view latents consumed by a learnable fusion. No current role covers this; decide whether posekit owns it or declares end-to-end fusion an exception.</div>
    <div><b><span class="n">3</span>Temporal feedback is an architectural seam</b> Realtime systems feed the fitted pose back into detection boxes and 2D estimation. The API must let consumers close this loop (pose-driven ROI, keypoint-feature augmentation), not treat it as an afterthought.</div>
    <div><b><span class="n">4</span>The fit stage forks: regress vs optimize</b> UmeTrack/SVEgoNet regress model parameters with a network; everyone else runs explicit optimizers. Both consume the same boundary datatype — the fork lives in glue, but the contract should name it.</div>
  </div>
  <h4>Top gaps by demand</h4>
  <ul class="hl">
    <li><b>Camera-aware crop</b> — needed by 5/11; analytic-grid variant of <code>ops.crops</code> taking a CameraModel, returning the virtual pinhole with the CropBatch.</li>
    <li><b>Skeleton registry growth + projection tables</b> — ~7/11 (incl. Look Ma's 1428/744/141-pt dense formats); add HALPE-26, H36M-17, BODY-25, SAPIENS-308, MAMMA-512 and coco133↔coco17 etc. as pure registry data.</li>
    <li><b>TopDownFeatureEncoder2d</b> — 2/11, research-path only; decide ownership first.</li>
    <li><b>InstancePose2d impls / ParametricPose3d</b> — 0–1/11 in the survey but committed design slots (RTMO port, WiLoR type alignment).</li>
  </ul>
  <details><summary>Full abstraction proposal (verdict, role set, per-paper rebuilds, ranked gaps, risks)</summary>
    <div class="inner md" id="proposal-md"></div>
  </details>
  <details><summary>posekit inventory — implemented roles, datatypes, glue locations (ground truth for the proposal)</summary>
    <div class="inner md" id="inventory-md"></div>
  </details>
</section>

<footer style="margin-top:60px; padding-top:16px; border-top:1px solid var(--line); color:var(--text-faint); font-size:12.5px;">
  Built from a 21-agent workflow (extract → adversarial verify → synthesize). Figures cropped from the papers' PDFs and visually verified; DAGs rendered with Graphviz from the verified node/edge extractions. Markdown sections render via CDN (marked); everything else is self-contained.
</footer>

</main>

<script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
<script>
  const INVENTORY_MD = {json.dumps(inventory_md)};
  const PROPOSAL_MD = {json.dumps(proposal_md)};
  if (window.marked) {{
    document.getElementById("inventory-md").innerHTML = marked.parse(INVENTORY_MD);
    document.getElementById("proposal-md").innerHTML = marked.parse(PROPOSAL_MD);
  }} else {{
    for (const [id, md] of [["inventory-md", INVENTORY_MD], ["proposal-md", PROPOSAL_MD]]) {{
      const pre = document.createElement("pre");
      pre.textContent = md;
      document.getElementById(id).appendChild(pre);
    }}
  }}

  // DAG lightbox: click a diagram -> fullscreen dialog (fit); click inside -> 100% + pan; Esc/backdrop closes
  const dagbox = document.createElement("dialog");
  dagbox.id = "dagbox";
  document.body.appendChild(dagbox);
  for (const wrap of document.querySelectorAll(".dag-wrap")) {{
    wrap.addEventListener("click", () => {{
      dagbox.innerHTML = '<span class="close-hint">click empty space or press Esc to close</span>';
      const clone = wrap.querySelector("svg").cloneNode(true);
      for (const el of clone.querySelectorAll("[id]")) el.removeAttribute("id");
      clone.removeAttribute("id");
      dagbox.appendChild(clone);
      dagbox.classList.remove("zoom");
      dagbox.showModal();
    }});
  }}
  dagbox.addEventListener("click", (e) => {{
    if (e.target.closest("svg")) dagbox.classList.toggle("zoom");
    else dagbox.close();
  }});
  dagbox.addEventListener("close", () => {{ dagbox.innerHTML = ""; }});
</script>
</body>
</html>
"""

out = ROOT / "pipeline-comparison.html"
out.write_text(page)
print(f"{out} : {len(page)/1e6:.1f} MB")
