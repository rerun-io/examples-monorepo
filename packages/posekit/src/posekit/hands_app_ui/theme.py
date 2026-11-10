"""CSS of the 3D hands app: the 4DAnyone Space's look (default theme forced dark, viewer filling the page)."""

APP_CSS: str = """
.gradio-container { max-width: none !important; padding: 0.6rem 1.5rem !important; }
#app-description h1 { margin: 0 0 0.3rem; }
#app-description p { margin: 0 0 0.4rem; }
#main-row { min-height: 0; }
#source-video video { max-height: 320px !important; }
/* Four small tiles per row with a pager. */
#examples .gallery { display: flex; flex-wrap: wrap; gap: 0.4rem; }
#examples .gallery button { flex: none; padding: 0; }
#examples video, #examples img { height: 96px !important; width: auto !important; max-width: 128px; object-fit: cover; }
#run-row button { font-size: 1.1rem; min-height: 3rem; }
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

VIEWER_HEIGHT: str = "calc(100vh - 11rem)"
"""Viewer frame height: the viewport minus the title band and the footer, so the viewer fits the page."""
