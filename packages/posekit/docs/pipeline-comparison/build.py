#!/usr/bin/env python3
"""Build ``pipeline-comparison.html`` from the survey JSON and cropped paper figures.

Inputs (all relative to this directory):

- ``json/result.json`` — the original multi-agent survey output: paper records, the
  taxonomy (canonical stages, stage x paper matrix, commonalities/divergences), and
  the inventory/proposal markdown appendices.
- ``json/new-paper*.json`` — papers added after the original workflow. Each carries
  its own ``matrix_cells`` and ``rebuild`` fields (see README for the add-a-paper protocol).
- ``figures/<paper_id>.png`` — cropped pipeline figure per paper, embedded as base64.

Output: ``pipeline-comparison.html`` next to this script — fully self-contained except
the ``marked`` CDN script that renders the two markdown appendices client-side.

Requires Graphviz ``dot`` on PATH. Run as ``python build.py``.
"""

import base64
import html
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]
"""A decoded JSON object (paper record, taxonomy entry, DAG node/edge, ...)."""

ROOT: Path = Path(__file__).resolve().parent
"""Directory holding this script, the JSON inputs, the figures, and the output."""

ROLE_STYLE: dict[str, tuple[str, str]] = {
    "detect": ("#1f4230", "#4da884"),
    "track": ("#173f42", "#4fa8ae"),
    "segment": ("#173f42", "#4fa8ae"),
    "identity": ("#3a2a4d", "#a583d1"),
    "crop/rewarp": ("#2e3440", "#8a94a8"),
    "pose2d": ("#1e3a5c", "#6ba3e8"),
    "dense-landmarks": ("#1e3a5c", "#6ba3e8"),
    "lift3d": ("#312a55", "#8f7fe8"),
    "triangulate": ("#312a55", "#8f7fe8"),
    "calibrate": ("#4d3a14", "#d1a94f"),
    "fit-model": ("#4d2d14", "#e8914f"),
    "optimize": ("#4d2d14", "#e8914f"),
    "render-verify": ("#4d1f2e", "#e86a8f"),
    "temporal": ("#3d3d29", "#b8b86a"),
    "other": ("#26292f", "#6a7180"),
}
"""``(fill, border)`` hex colors per pipeline role — the one color code used by every diagram."""

LEGEND_ROLES: list[tuple[str, str]] = [
    ("detect", "detect"),
    ("track", "track / segment"),
    ("identity", "identity"),
    ("crop/rewarp", "crop / rewarp"),
    ("pose2d", "2D estimation"),
    ("triangulate", "lift / triangulate"),
    ("calibrate", "calibrate"),
    ("fit-model", "fit / optimize"),
    ("render-verify", "render-verify"),
    ("temporal", "temporal"),
    ("other", "data / io"),
]
"""``(ROLE_STYLE key, display label)`` legend entries, collapsing synonym roles into one swatch."""

PAPER_ORDER: list[str] = [
    "megatrack",
    "umetrack",
    "kineo",
    "epfl-smart-kitchen",
    "assemblyhands",
    "assemblyx",
    "mamma",
    "egoexo-hands",
    "show3d",
    "hocap",
    "lookma",
]
"""Curated card/matrix order; papers missing from this list sort to the end."""

COUNT_WORDS: dict[int, str] = {9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen"}
"""Spelled-out paper counts for the prose headings; unknown counts fall back to digits."""

SHORT_NAMES: dict[str, str] = {
    "megatrack": "MEgATrack",
    "umetrack": "UmeTrack",
    "kineo": "Kineo",
    "epfl-smart-kitchen": "EPFL Smart Kitchen",
    "assemblyhands": "AssemblyHands",
    "assemblyx": "AssemblyHands-X",
    "mamma": "MAMMA",
    "egoexo-hands": "EgoExo-Hands",
    "hocap": "HO-Cap",
    "lookma": "Look Ma, No Markers",
    "show3d": "SHOW3D",
}
"""Display name per paper id; papers absent here fall back to the record's ``name`` field."""

PAPER_LINKS: dict[str, list[tuple[str, str]]] = {
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
"""``(label, url)`` header links per paper; an empty url renders as plain text."""

FIGURE_NOTES: dict[str, str] = {
    "mamma": "No end-to-end pipeline figure exists in the paper; shown is Fig. 4 (MammaNet, the dense-landmark node). The full pipeline is the DAG below.",
}
"""Caveats rendered under a paper's figure when the figure is not the full pipeline."""

KIND_LABELS: dict[str, str] = {
    "realtime-tracking": "realtime tracking",
    "offline-annotation": "offline annotation",
    "offline-capture": "offline capture",
    "hybrid": "hybrid",
}
"""Display label per paper ``kind``; unknown kinds render verbatim."""

POSEKIT_REBUILDS: dict[str, str] = {
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
"""Pipeline-as-posekit one-liners condensed from proposal §3, overriding a paper's own ``rebuild``."""

COMMON_HEADLINES: list[tuple[str, str]] = [
    ("One boundary datatype", "(x, y, confidence) per (view, frame, instance) in a named skeleton + a camera model — shared by all; only HO-Cap's MediaPipe lacks the confidence channel."),
    ("Confidence-weighted everything", "gating, triangulation, BA and robust fits all consume it; HO-Cap, whose 2D net supplies none, selects triangulation candidates by reprojection error instead."),
    ("Crops are the currency", "the 2D net always sees a canonicalized instance-centered view; 5/11 need fisheye→virtual-pinhole rewarp."),
    ("The 2D estimator is a swappable slot", "KeyNet, RTMPose/DWPose/NLF, Sapiens/InterNet, MediaPipe, MammaNet — detector, tracker and pose mix freely."),
    ("The tail is fuse-then-fit", "triangulate or lift, then fit MANO / SMPL-X / a template hand — 9/11 emit a parametric model with a per-user shape prior."),
    ("Temporal info is near-universal", "realtime systems close feedback loops into detection and 2D; offline ones add feed-forward smoothing terms."),
    ("The localizer is an interface", "satisfied by a box net, a segmenter, a pose-net-as-detector, or the tracking loop itself."),
    ("Geometry glue never generalizes", "every paper hand-rolls its own triangulation / association / BA / fitting — the networks are the reusable part."),
]
"""``(headline, elaboration)`` compressions of ``taxonomy.commonalities`` (full text stays in a collapsible)."""

DIVERGE_HEADLINES: list[tuple[str, str]] = [
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
"""``(headline, elaboration)`` compressions of ``taxonomy.divergences`` (full text stays in a collapsible)."""

STAGE_ROLE: dict[str, str] = {
    "capture": "other",
    "detect": "detect",
    "identity": "identity",
    "crop": "crop/rewarp",
    "pose2d": "pose2d",
    "lift3d": "triangulate",
    "calibrate": "calibrate",
    "fit": "fit-model",
    "verify": "render-verify",
    "feedback": "temporal",
}
"""``ROLE_STYLE`` key per canonical stage id — colors the matrix column underlines."""

STAGE_SHORT: dict[str, str] = {
    "capture": "capture",
    "detect": "detect / segment",
    "identity": "identity",
    "crop": "crop / rewarp",
    "pose2d": "2D estimation",
    "lift3d": "lift / triangulate",
    "calibrate": "calibrate",
    "fit": "parametric fit",
    "verify": "verify",
    "feedback": "temporal",
}
"""Compact matrix column header per canonical stage id."""

UNIVERSAL_NODES: list[dict[str, str]] = [
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
"""Nodes of the hero "every pipeline is this DAG" diagram."""

UNIVERSAL_EDGES: list[dict[str, str]] = [
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
"""Edges of the hero DAG; the fit→detect / fit→pose2d / verify→fit edges render as feedback."""

ABBREVIATIONS: list[str] = ["Fig.", "Figs.", "Sec.", "Eq.", "et al.", "e.g.", "i.e.", "vs.", "approx."]
"""Dot-terminated tokens that must not end a sentence when splitting prose."""


@dataclass(slots=True)
class SurveyData:
    """Everything the page renders, loaded and normalized from the JSON inputs."""

    papers: list[JsonDict]
    """All paper records (original survey + ``new-paper*.json`` additions), in ``PAPER_ORDER``."""
    taxonomy: JsonDict
    """Canonical stages, the stage x paper matrix, and the commonality/divergence analyses."""
    inventory_md: str
    """posekit inventory markdown appendix (workflow preamble trimmed)."""
    proposal_md: str
    """Abstraction proposal markdown appendix (workflow preamble trimmed)."""


def load_survey() -> SurveyData:
    """Load ``result.json``, merge ``new-paper*.json`` additions, and order the papers.

    Returns:
        The normalized survey data, papers sorted by ``PAPER_ORDER``.
    """
    data: JsonDict = json.loads((ROOT / "json/result.json").read_text())
    papers: list[JsonDict] = data["papers"]
    known_ids: set[str] = {paper["id"] for paper in papers}
    for extra_path in sorted((ROOT / "json").glob("new-paper*.json")):
        extra: JsonDict = json.loads(extra_path.read_text())
        if extra["id"] not in known_ids:
            papers.append(extra)
            known_ids.add(extra["id"])
    papers.sort(key=lambda paper: PAPER_ORDER.index(paper["id"]) if paper["id"] in PAPER_ORDER else len(PAPER_ORDER))
    return SurveyData(
        papers=papers,
        taxonomy=data["taxonomy"],
        inventory_md=trim_preamble(data["inventory"]),
        proposal_md=trim_preamble(data["proposal"]),
    )


def trim_preamble(markdown: str) -> str:
    """Drop workflow chatter before the first markdown H1.

    Args:
        markdown: A markdown document that may start with non-document preamble.

    Returns:
        The document from its first ``# `` heading on (unchanged if none is found past index 0).
    """
    heading_index: int = markdown.find("# ")
    return markdown[heading_index:] if heading_index > 0 else markdown


def esc(text: str) -> str:
    """HTML-escape ``text`` (None-safe), quoting for attribute contexts too.

    Args:
        text: Raw text; ``None``/empty becomes the empty string.

    Returns:
        The escaped text.
    """
    return html.escape(text or "", quote=True)


def figure_data_uri(paper_id: str) -> str:
    """Embed the paper's cropped pipeline figure as a base64 data URI.

    Args:
        paper_id: Paper id naming ``figures/<paper_id>.png``.

    Returns:
        A ``data:image/png;base64,...`` URI.
    """
    figure_path: Path = ROOT / "figures" / f"{paper_id}.png"
    return "data:image/png;base64," + base64.b64encode(figure_path.read_bytes()).decode()


def first_sentences(text: str, count: int) -> str:
    """Return the first ``count`` sentences of ``text``.

    Dots inside known abbreviations (``ABBREVIATIONS``) are masked so ``Fig. 4``
    does not terminate a sentence.

    Args:
        text: Prose to split; ``None``/empty becomes the empty string.
        count: Number of leading sentences to keep.

    Returns:
        The leading sentences, re-joined with single spaces.
    """
    masked: str = text or ""
    for abbreviation in ABBREVIATIONS:
        masked = masked.replace(abbreviation, abbreviation.replace(".", "\x00"))
    parts: list[str] = re.split(r"(?<=[.!?]) +", masked)
    return " ".join(parts[:count]).replace("\x00", ".")


def shorten_caption(caption: str) -> str:
    """Compress a figure caption to its lead sentence for display under the figure.

    Drops a leading ``Fig. N.`` prefix. Degenerate one-liners like
    "Hand-tracking pipeline." (under 30 chars) get a second sentence.

    Args:
        caption: The full caption as extracted from the paper.

    Returns:
        The shortened caption.
    """
    stripped: str = re.sub(r"^\s*Fig(?:ure)?\.?\s*\d+[.:]?\s*", "", caption or "")
    short: str = first_sentences(stripped, 1)
    if len(short) < 30:
        short = first_sentences(stripped, 2)
    return short


def wrap_graphviz_label(text: str, max_chars: int = 16) -> str:
    """Escape and greedy word-wrap a label for Graphviz.

    Args:
        text: Raw label; backslashes are dropped and double quotes escaped.
        max_chars: Soft line-length limit for the greedy wrap.

    Returns:
        The label with Graphviz ``\\n`` line separators.
    """
    escaped: str = (text or "").replace("\\", "").replace('"', r"\"")
    lines: list[str] = []
    current_line: str = ""
    for word in escaped.split():
        if len(current_line) + len(word) + 1 > max_chars and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = (current_line + " " + word).strip()
    if current_line:
        lines.append(current_line)
    return r"\n".join(lines)


def find_feedback_edges(nodes: list[JsonDict], edges: list[JsonDict]) -> set[tuple[str, str]]:
    """Find cycle-closing edges via DFS back-edge detection.

    An edge into a node that is on the current DFS stack closes a cycle. List-order
    heuristics misclassify plain dependency edges like calibrate→lift, hence DFS.

    Args:
        nodes: DAG nodes; only ``id`` is read.
        edges: DAG edges with ``from``/``to`` node ids.

    Returns:
        The ``(from_id, to_id)`` pairs that close a cycle.
    """
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge["from"], []).append(edge["to"])
    node_ids: list[str] = [node["id"] for node in nodes]
    visit_state: dict[str, int] = dict.fromkeys(node_ids, 0)  # 0 = unvisited, 1 = on the DFS stack, 2 = done
    feedback: set[tuple[str, str]] = set()

    def visit(node_id: str) -> None:
        visit_state[node_id] = 1
        for successor in adjacency.get(node_id, []):
            if successor not in visit_state:
                continue
            if visit_state[successor] == 1:
                feedback.add((node_id, successor))
            elif visit_state[successor] == 0:
                visit(successor)
        visit_state[node_id] = 2

    for node_id in node_ids:
        if visit_state[node_id] == 0:
            visit(node_id)
    return feedback


def render_dag_svg(nodes: list[JsonDict], edges: list[JsonDict], id_prefix: str) -> str:
    """Render a pipeline DAG to inline SVG with Graphviz ``dot``.

    Args:
        nodes: Nodes with ``id``, ``label``, and optional ``role`` (colored via ``ROLE_STYLE``).
        edges: Edges with ``from``/``to`` and optional ``label``; edges referencing unknown
            nodes are skipped, feedback edges render dashed and unconstrained.
        id_prefix: Namespace prepended to every SVG element id so multiple DAGs coexist on one page.

    Returns:
        The ``<svg>`` markup with a ``dag`` class added. Intrinsic width/height attributes
        are kept — the fullscreen natural-size toggle relies on them.
    """

    def dot_id(raw: str) -> str:
        return re.sub(r"\W", "_", raw)

    dot_lines: list[str] = [
        'digraph G { rankdir=LR; bgcolor="transparent"; ranksep=0.35; nodesep=0.3;',
        'node [shape=box style="rounded,filled" fontname="Helvetica,Arial,sans-serif" fontsize=11 fontcolor="#eceef2" penwidth=1.2 margin="0.14,0.09"];',
        'edge [fontname="Helvetica,Arial,sans-serif" fontsize=9 fontcolor="#9aa3b5" color="#5f6a80" arrowsize=0.7 penwidth=1.1];',
    ]
    node_ids: set[str] = {node["id"] for node in nodes}
    feedback: set[tuple[str, str]] = find_feedback_edges(nodes, edges)
    for node in nodes:
        style: tuple[str, str] = ROLE_STYLE.get(node.get("role", "other"), ROLE_STYLE["other"])
        dot_lines.append(f'"{dot_id(node["id"])}" [label="{wrap_graphviz_label(node["label"])}" fillcolor="{style[0]}" color="{style[1]}"];')
    for edge in edges:
        if edge["from"] not in node_ids or edge["to"] not in node_ids:
            continue
        is_feedback: bool = (edge["from"], edge["to"]) in feedback
        edge_attrs: str = ' style=dashed constraint=false color="#8a8a5a" fontcolor="#b0b070"' if is_feedback else ""
        dot_lines.append(f'"{dot_id(edge["from"])}" -> "{dot_id(edge["to"])}" [label="{wrap_graphviz_label(edge.get("label", ""), 20)}"{edge_attrs}];')
    dot_lines.append("}")
    dot_result: subprocess.CompletedProcess[bytes] = subprocess.run(
        ["dot", "-Tsvg"], input="\n".join(dot_lines).encode(), capture_output=True, check=True
    )
    svg: str = dot_result.stdout.decode()
    svg = svg[svg.index("<svg") :]
    svg = re.sub(r"<svg ", '<svg class="dag" ', svg, count=1)
    svg = svg.replace('id="', f'id="{dot_id(id_prefix)}-')
    return svg


def render_dag(nodes: list[JsonDict], edges: list[JsonDict], id_prefix: str) -> str:
    """Wrap a rendered DAG SVG in the clickable fullscreen-lightbox container.

    Args:
        nodes: See :func:`render_dag_svg`.
        edges: See :func:`render_dag_svg`.
        id_prefix: See :func:`render_dag_svg`.

    Returns:
        The ``<div class="dag-wrap">`` markup.
    """
    svg: str = render_dag_svg(nodes, edges, id_prefix)
    return f'<div class="dag-wrap" title="Click to view fullscreen">{svg}</div>'


def render_legend() -> str:
    """Render the role color legend shared by every diagram on the page.

    Returns:
        The ``<div class="legend">`` markup, one swatch per ``LEGEND_ROLES`` entry
        plus the dashed feedback-edge sample.
    """
    swatches: str = "".join(
        f'<span class="lg"><i style="background:{ROLE_STYLE[role][0]};border-color:{ROLE_STYLE[role][1]}"></i>{label}</span>'
        for role, label in LEGEND_ROLES
    )
    return f'<div class="legend">{swatches}<span class="lg"><i class="dash"></i>feedback edge</span></div>'


def render_paper_card(index: int, paper: JsonDict) -> str:
    """Render one paper card: header, summary, figure, DAG, posekit rebuild, details.

    Args:
        index: 1-based position shown in the card header.
        paper: The paper record (nodes/edges, summary, traits, figure caption, ...).

    Returns:
        The ``<article class="paper">`` markup.
    """
    paper_id: str = paper["id"]
    display_name: str = SHORT_NAMES.get(paper_id, paper["name"].split(":")[0])
    links: str = " · ".join(
        f'<a href="{esc(url)}">{esc(label)}</a>' if url else f"<span>{esc(label)}</span>" for label, url in PAPER_LINKS.get(paper_id, [])
    )
    node_rows: str = "".join(
        f'<tr><td>{esc(node["label"])}</td><td class="mono">{esc(node["role"])}</td>'
        f'<td>{esc(node.get("models", ""))}</td><td class="dim">{esc(node.get("desc", ""))}</td></tr>'
        for node in paper["nodes"]
    )
    figure_note: str = FIGURE_NOTES.get(paper_id, "")
    figure_note_html: str = f'<p class="fignote">{esc(figure_note)}</p>' if figure_note else ""
    top_traits: list[str] = paper["traits"][:5]
    hidden_traits: list[str] = paper["traits"][5:]
    top_trait_chips: str = "".join(f'<span class="chip trait">{esc(trait)}</span>' for trait in top_traits)
    more_traits_chip: str = f'<span class="chip trait more">+{len(hidden_traits)} more in details</span>' if hidden_traits else ""
    all_trait_chips: str = "".join(f'<span class="chip trait">{esc(trait)}</span>' for trait in paper["traits"])
    all_traits_html: str = f'<h4>All traits</h4><div class="traits">{all_trait_chips}</div>' if hidden_traits else ""
    return f"""
<article class="paper" id="{paper_id}">
  <header>
    <h3><span class="num">{index}</span> {esc(display_name)}</h3>
    <span class="chip kind">{esc(KIND_LABELS.get(paper["kind"], paper["kind"]))}</span>
    <span class="chip">{esc(str(paper["year"]))}</span>
    <span class="links">{links}</span>
  </header>
  <div class="body">
    <p class="summary">{esc(first_sentences(paper["summary"], 2))}</p>
    <p class="kvline"><b>in</b> {esc(paper["rig"])} <b class="sep">out</b> {esc(paper["outputs"])}</p>
    <div class="figwrap">
      <img src="{figure_data_uri(paper_id)}" alt="{esc(display_name)} pipeline figure" loading="lazy" title="{esc(paper["figure_caption"])}">
      <p class="figcap">{esc(shorten_caption(paper["figure_caption"]))}</p>
      {figure_note_html}
    </div>
    <div class="dagrow">
      {render_dag(paper["nodes"], paper["edges"], paper_id)}
    </div>
    <p class="rebuild"><b>as posekit</b> {esc(POSEKIT_REBUILDS.get(paper_id) or paper.get("rebuild", ""))}</p>
    <div class="traits">{top_trait_chips}{more_traits_chip}</div>
    <details>
      <summary>Details — node table ({len(paper["nodes"])} nodes), full caption, posekit mapping</summary>
      <div class="inner">
        <p class="kv"><b>Figure caption</b> {esc(paper["figure_caption"])}</p>
        <p class="kv"><b>posekit mapping</b> {esc(paper.get("posekit_notes", ""))}</p>
        {all_traits_html}
        <table class="nodes">
          <thead><tr><th>node</th><th>role</th><th>models / methods</th><th>what it does</th></tr></thead>
          <tbody>{node_rows}</tbody>
        </table>
      </div>
    </details>
  </div>
</article>"""


def render_matrix(papers: list[JsonDict], taxonomy: JsonDict) -> str:
    """Render the stage x paper matrix with role-colored column underlines.

    Rows come from ``taxonomy.matrix``; papers added after the original workflow
    carry their own ``matrix_cells`` instead.

    Args:
        papers: All paper records, already in display order.
        taxonomy: The survey taxonomy with ``canonical_stages`` and ``matrix``.

    Returns:
        The scrollable ``<div class="matrix-wrap">`` markup.
    """
    stages: list[JsonDict] = taxonomy["canonical_stages"]
    cells_by_paper: dict[str, dict[str, JsonDict]] = {
        row["paper_id"]: {cell["stage_id"]: cell for cell in row["cells"]} for row in taxonomy["matrix"]
    }
    for paper in papers:
        if paper["id"] not in cells_by_paper and "matrix_cells" in paper:
            cells_by_paper[paper["id"]] = {cell["stage_id"]: cell for cell in paper["matrix_cells"]}
    header_cells: str = "".join(
        f'<th title="{esc(stage["desc"][:300])}" style="border-bottom: 2px solid {ROLE_STYLE[STAGE_ROLE.get(stage["id"], "other")][1]}">'
        f'{esc(STAGE_SHORT.get(stage["id"], stage["name"]))}</th>'
        for stage in stages
    )
    body_rows: list[str] = []
    for paper in papers:
        paper_cells: dict[str, JsonDict] = cells_by_paper.get(paper["id"], {})
        cell_tds: list[str] = []
        for stage in stages:
            cell: JsonDict = paper_cells.get(stage["id"], {})
            fill: str = cell.get("fill", "—")
            note: str = cell.get("note", "")
            is_absent: bool = fill.strip() in ("—", "-", "")
            tooltip: str = esc((fill + (" — " + note if note else "")).strip())
            cell_tds.append(f'<td class="{"absent" if is_absent else ""}" title="{tooltip}"><span class="clamp">{esc(fill)}</span></td>')
        display_name: str = SHORT_NAMES.get(paper["id"], paper["name"].split(":")[0])
        body_rows.append(f'<tr><th><a href="#{paper["id"]}">{esc(display_name)}</a></th>{"".join(cell_tds)}</tr>')
    return f"""
<div class="matrix-wrap">
<table class="matrix">
<thead><tr><th>paper</th>{header_cells}</tr></thead>
<tbody>{"".join(body_rows)}</tbody>
</table>
</div>
"""


def render_page(survey: SurveyData) -> str:
    """Assemble the full self-contained HTML page.

    Args:
        survey: The loaded survey data.

    Returns:
        The complete HTML document.
    """
    paper_count: int = len(survey.papers)
    paper_count_word: str = COUNT_WORDS.get(paper_count, str(paper_count))
    cards_html: str = "".join(render_paper_card(index, paper) for index, paper in enumerate(survey.papers, 1))
    matrix_html: str = render_matrix(survey.papers, survey.taxonomy)
    common_full: str = "".join(f"<li>{esc(item)}</li>" for item in survey.taxonomy["commonalities"])
    diverge_full: str = "".join(f"<li>{esc(item)}</li>" for item in survey.taxonomy["divergences"])
    common_headlines: str = "".join(f"<li><b>{esc(headline)}</b> — {esc(detail)}</li>" for headline, detail in COMMON_HEADLINES)
    diverge_headlines: str = "".join(f"<li><b>{esc(headline)}</b> — {esc(detail)}</li>" for headline, detail in DIVERGE_HEADLINES)
    toc_papers: str = "".join(
        f'<a href="#{paper["id"]}">{esc(SHORT_NAMES.get(paper["id"], paper["name"].split(":")[0]))}</a>' for paper in survey.papers
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Keypoint Pipeline Comparison — {paper_count} papers vs posekit</title>
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
  <h1>Keypoint pipeline comparison — {paper_count} papers, one abstraction</h1>
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
  <h2>All {paper_count_word} pipelines are the same DAG</h2>
  <p class="claim">frames → detect / segment / localize → crop / rewarp → 2D keypoints + confidence → associate / lift → parametric fit. Solid = the common backbone; dashed = optional stages and feedback edges. The colors below are used in every diagram on this page.</p>
  {render_legend()}
  {render_dag(UNIVERSAL_NODES, UNIVERSAL_EDGES, "universal")}
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
      <ul class="hl">{common_headlines}</ul>
      <details><summary>Full analysis</summary><div class="inner"><ul>{common_full}</ul></div></details>
    </div>
    <div>
      <h4>Divergences</h4>
      <ul class="hl">{diverge_headlines}</ul>
      <details><summary>Full analysis</summary><div class="inner"><ul>{diverge_full}</ul></div></details>
    </div>
  </div>
</section>

<section id="papers">
  <h2>The {paper_count_word} pipelines</h2>
  <p class="lead">Each card: the paper's own figure, our extracted DAG (click any diagram to view it fullscreen; click again for 100%), and the pipeline rewritten as posekit roles — <code>Role(model)</code> = posekit-owned, <code>glue[…]</code> = consumer, ✦ = capability posekit doesn't have yet, ⟳ = feedback loop.</p>
  {cards_html}
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
  const INVENTORY_MD = {json.dumps(survey.inventory_md)};
  const PROPOSAL_MD = {json.dumps(survey.proposal_md)};
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


def main() -> None:
    """Load the survey inputs, render the page, and write ``pipeline-comparison.html``."""
    survey: SurveyData = load_survey()
    page: str = render_page(survey)
    output_path: Path = ROOT / "pipeline-comparison.html"
    output_path.write_text(page)
    print(f"{output_path} : {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
