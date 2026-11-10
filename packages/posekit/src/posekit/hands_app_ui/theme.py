"""CSS of the 3D hands app: the 4DAnyone Space's look (default theme forced dark, viewer filling the page)."""

APP_CSS: str = """
/* The page never scrolls: the viewer fills the window, the run log scrolls inside itself, and the left
   column scrolls on its own only when a window is too short for it. */
html, body, gradio-app, .gradio-container { height: 100%; overflow: hidden; }
#main-row { height: calc(100vh - 9rem); min-height: 0; overflow: hidden; }
#left-column { height: 100%; max-height: 100%; overflow-y: auto; scrollbar-gutter: stable; flex-wrap: nowrap; }
#left-column > * { flex-shrink: 0; }
#run-log textarea { overflow-y: auto !important; }
/* The tab strip starts at the viewer's top edge and its underline meets the line under the viewer's top bar. */
#left-column .tab-wrapper { height: auto !important; }
#left-column .tab-container { height: 37px !important; margin-top: 0 !important; top: 0 !important; transform: none !important; }
#left-column .tab-container [role=tab] { height: 100% !important; }
.gradio-container { max-width: none !important; padding: 0.6rem 1.5rem !important; }
#app-description h1 { margin: 0 0 0.3rem; }
#app-description p { margin: 0 0 0.4rem; }
#main-row { min-height: 0; }
#source-video video { max-height: 320px !important; }
/* The examples side by side: four equal tiles in one row (clips of any shape are cropped to fill), with a pager. */
/* Gradio also gives each tile's inner box the class "gallery", so the row is the one holding the tiles. */
#examples .gallery:has(> .gallery-item) { display: grid !important; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.4rem; }
#examples .gallery-item { padding: 0; min-width: 0; width: 100%; }
#examples .gallery-item .container { width: 100% !important; height: 72px !important; }
#examples .gallery-item video, #examples .gallery-item img { width: 100% !important; height: 100% !important; max-width: none !important; object-fit: cover; }
#run-row button { font-size: 1.1rem; min-height: 3rem; }
#run-summary td:first-child, #run-summary th:first-child { white-space: nowrap; }
#run-status {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 4.5rem;
    padding: 0.65rem 0.85rem;
    border-radius: var(--radius-lg);
    background: var(--background-fill-secondary);
}
#run-status p { font-size: 1.05rem; line-height: 1.4; margin: 0; }
"""

VIEWER_HEIGHT: str = "calc(100vh - 9rem)"
"""Viewer frame height: the viewport minus the title band and the footer, the same as ``#main-row``."""
