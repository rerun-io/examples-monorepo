"""Theme constants for the click-to-track Gradio app."""

DESCRIPTION: str = (
    "# posekit: Click to Track\n"
    "Click an object in the Rerun viewer, refine it on any frame, and propagate the mask through the whole clip. "
    "The confidence traces help you find frames that need another click."
)

VIEWER_HEIGHT: str = "calc(100vh - 7rem)"
"""Viewer frame height: the viewport minus the header/description band, so nothing scrolls."""

# HF Spaces render the default theme in dark mode; do the same here by forcing
# Gradio's ``__theme=dark`` URL switch before the app mounts (a <head> script —
# the launch ``js`` hook runs too late to redirect).
FORCE_DARK_HEAD: str = """
<script>
(() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "dark") {
        url.searchParams.set("__theme", "dark");
        window.location.replace(url.href);
    }
})();
</script>
"""

APP_CSS: str = """
html, body, gradio-app, .gradio-container {
    height: 100%;
    overflow: hidden;
}
.gradio-container {
    max-width: none !important;
    padding: 0.6rem 1rem !important;
}
#app-description { margin-bottom: 0.35rem; }
#app-description h1 { margin: 0 0 0.2rem; }
#app-description p { margin: 0; }
#main-row {
    height: calc(100vh - 6.4rem);
    min-height: 0;
    overflow: hidden;
}
#left-column, #viewer-column { min-height: 0; }
/* Only the controls scroll on short viewports; the viewer never moves. Reserve the
   scrollbar gutter so the controls do not shift horizontally when it appears. */
#left-column { height: 100%; max-height: 100%; overflow-y: auto; scrollbar-gutter: stable; flex-wrap: nowrap; }
#left-column > * { flex-shrink: 0; }
#source-video { height: auto !important; }
#source-video video { max-height: 225px !important; }
#examples { flex: none; }
/* Radio-based tab strip: Gradio 6.13 loops in Svelte when this app renders three
   native Tab components. The panels below are ordinary Columns. */
#control-tabs { overflow: visible; min-width: 0 !important; }
#control-tabs .wrap { display: flex; gap: 0; border-bottom: 1px solid var(--border-color-primary); }
#control-tabs label { flex: 1; justify-content: center; margin: 0; border: 0; border-bottom: 2px solid transparent; border-radius: 0; background: transparent; padding: 0.55rem 0.4rem; color: var(--body-text-color); cursor: pointer; }
#control-tabs label.selected { border-bottom-color: var(--color-accent); color: var(--color-accent); }
#control-tabs input { display: none; }
#control-tabs span { font-weight: 600; }
/* Click-mode radio as a segmented pill group. */
#click-mode .wrap { display: flex; gap: 0; border: 1px solid var(--border-color-primary); border-radius: var(--radius-lg); overflow: hidden; }
#click-mode label { flex: 1; justify-content: center; margin: 0; border-radius: 0; border: 0; background: var(--background-fill-secondary); padding: 0.45rem 0.4rem; cursor: pointer; }
#click-mode label + label { border-left: 1px solid var(--border-color-primary); }
#click-mode label.selected { background: var(--color-accent); color: white; }
#click-mode input { display: none; }
#click-mode span { font-weight: 600; }
/* Four small tiles per row with a pager, like the 4DAnyone Space. */
#examples .gallery { display: flex; flex-wrap: wrap; gap: 0.4rem; }
#examples .gallery button { flex: none; padding: 0; }
#examples video, #examples img { height: 64px !important; width: auto !important; max-width: 84px; object-fit: cover; }
/* The viewer's top edge lines up with the video card beside it. */
#viewer-column { padding-top: 0; }
#rerun-viewer { min-height: 0 !important; }
#run-status {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 4.5rem;
    overflow: visible !important;
    padding: 0.65rem 0.85rem;
    border-radius: var(--radius-lg);
    background: var(--background-fill-secondary);
}
#run-status p { font-size: 1.05rem; line-height: 1.4; margin: 0; }
footer { display: none !important; }
"""
