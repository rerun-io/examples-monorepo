"""Look of the 3D hands app: Rerun's brand colours and wordmark (https://rerun.io/media), the viewer filling the page."""

import base64
from pathlib import Path

VIEWER_HEIGHT: str = "calc(100vh - 10.5rem)"
"""Viewer frame height: the viewport minus the header band and the footer, the same as ``#main-row``."""

RERUN_LOGOS: dict[str, str] = {
    mode: "data:image/webp;base64," + base64.b64encode((Path(__file__).parent / f"rerun_wordmark_animated_{mode}.webp").read_bytes()).decode()
    for mode in ("light", "dark")
}
"""Rerun's animated wordmark from the media kit (https://rerun.io/media) on a transparent ground, plays once: black letters for
light mode, white for dark mode (the liquid keeps its colours). Inlined so the header needs no file route."""

APP_CSS: str = '@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap");\n' + f"#main-row {{ height: {VIEWER_HEIGHT}; min-height: 0; overflow: hidden; }}\n" + """
/* Rerun's brand base colours (https://rerun.io/media): #171616 ground, #F1EFEB type, #186ADD accent, #D72F21 red;
   the panels and borders use the viewer's own greys so the left column matches the embedded viewer. */
:root {
    --font: "Inter", ui-sans-serif, system-ui, sans-serif;
    --body-background-fill: #f1efeb;
    --background-fill-primary: #f1efeb;
    --background-fill-secondary: #ffffff;
    --block-background-fill: #ffffff;
    --panel-background-fill: #ffffff;
    --input-background-fill: #ffffff;
    --border-color-primary: #e0dfdf;
    --block-border-color: #e0dfdf;
    --body-text-color: #171616;
    --body-text-color-subdued: #646363;
    --block-label-text-color: #646363;
    --block-title-text-color: #171616;
    --color-accent: #186add;
    --color-accent-soft: #dce1ff;
    --link-text-color: #186add;
    --link-text-color-hover: #0044b2;
    --link-text-color-visited: #186add;
    --slider-color: #186add;
    --checkbox-background-color-selected: #186add;
    --checkbox-border-color-selected: #186add;
    --checkbox-border-color-focus: #186add;
    --input-border-color-focus: #186add;
    --button-primary-background-fill: #186add;
    --button-primary-background-fill-hover: #0062f9;
    --button-primary-border-color: #186add;
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #e6e6e5;
    --button-secondary-background-fill-hover: #e0dfdf;
    --button-secondary-text-color: #171616;
    --button-cancel-background-fill: #d72f21;
    --button-cancel-background-fill-hover: #e8473a;
    --button-cancel-border-color: #d72f21;
    --button-cancel-text-color: #ffffff;
    --header-strong: #000000;
    --header-text: #454545;
    --header-faint: #939090;
    --header-hover: #e6e6e5;
}
.dark {
    --font: "Inter", ui-sans-serif, system-ui, sans-serif;
    --body-background-fill: #171616;
    --background-fill-primary: #171616;
    --background-fill-secondary: #212121;
    --block-background-fill: #212121;
    --panel-background-fill: #212121;
    --input-background-fill: #171616;
    --border-color-primary: #353434;
    --block-border-color: #353434;
    --body-text-color: #f1efeb;
    --body-text-color-subdued: #9f9f9f;
    --block-label-text-color: #9f9f9f;
    --block-title-text-color: #f1efeb;
    --color-accent: #186add;
    --color-accent-soft: #001b53;
    --link-text-color: #7a95ff;
    --link-text-color-hover: #adbbff;
    --link-text-color-visited: #7a95ff;
    --slider-color: #186add;
    --checkbox-background-color-selected: #186add;
    --checkbox-border-color-selected: #186add;
    --checkbox-border-color-focus: #186add;
    --input-border-color-focus: #186add;
    --button-primary-background-fill: #186add;
    --button-primary-background-fill-hover: #0062f9;
    --button-primary-border-color: #186add;
    --button-primary-text-color: #ffffff;
    --button-secondary-background-fill: #212121;
    --button-secondary-background-fill-hover: #2c2b2b;
    --button-secondary-text-color: #f1efeb;
    --button-cancel-background-fill: #d72f21;
    --button-cancel-background-fill-hover: #e8473a;
    --button-cancel-border-color: #d72f21;
    --button-cancel-text-color: #ffffff;
    --header-strong: #ffffff;
    --header-text: #bfbfbf;
    --header-faint: #737070;
    --header-hover: #212121;
}
.gradio-container, .gradio-container button, .gradio-container input, .gradio-container textarea { font-family: "Inter", ui-sans-serif, system-ui, sans-serif !important; }
#left-column .tab-container [role=tab][aria-selected=true], #left-column .tab-container .selected { color: var(--header-strong) !important; }
#left-column .tab-container .selected::after { background: #186add !important; }
/* The page never scrolls: the viewer fills the window and the run log scrolls inside itself. */
html, body, gradio-app, .gradio-container { height: 100%; overflow: hidden; }
/* The left column never scrolls. The tabs keep their natural height and only shrink when the window is too short, so Run
   sits right under them; inside Outputs the run log is what gives way, scrolling by itself. Only the Input tab, on a
   window too short for its clip, settings and examples, scrolls inside its own panel. */
#left-column { height: 100%; max-height: 100%; overflow: hidden; flex-wrap: nowrap; }
#left-column > * { flex-shrink: 0; }
#left-column > .tabs { flex: 0 1 auto; min-height: 0; display: flex; flex-direction: column; }
#left-column .tabitem { flex: 0 1 auto; min-height: 0; flex-direction: column; overflow-y: auto; }
#left-column .tabitem > .column { flex: 0 1 auto; min-height: 0; flex-wrap: nowrap; }
#left-column .tabitem > .column > * { flex-shrink: 0; }
#left-column .tabitem > .column > .form:has(#run-log) { flex: 0 1 auto; min-height: 5rem; }
#run-log, #run-log > label, #run-log .input-container { display: flex; flex-direction: column; flex: 0 1 auto; min-height: 0; }
#run-log textarea { flex: 0 1 auto; min-height: 0; overflow-y: auto !important; }
/* The tab strip starts at the viewer's top edge and its underline meets the line under the viewer's top bar. */
#left-column .tab-wrapper { height: auto !important; }
#left-column .tab-container { height: 37px !important; margin-top: 0 !important; top: 0 !important; transform: none !important; }
#left-column .tab-container [role=tab] { height: 100% !important; }
.gradio-container { max-width: none !important; padding: 0.6rem 1.5rem !important; }
/* Header: the title on one line with Rerun's wordmark as a large link target, then the text in two short lines. */
#app-title { margin: 0 0 0.35rem; font-size: 1.9rem; line-height: 1.25; font-weight: 600; letter-spacing: -0.02em; color: var(--header-strong); }
#app-title .title-x { color: var(--header-faint); font-weight: 300; margin: 0 0.3em; }
/* The wordmark's letters fill rows 18-73 of the 96 px frame (the rest is the liquid's path): an image 1.24em tall makes
   them as tall as the title's capitals, top: 0.46em puts them on its baseline, and -0.169em cancels the 11 px left margin. */
#app-title a { display: inline-block; padding: 0.3em 0.35em; margin: -0.3em -0.35em; border-radius: 8px; transition: background 0.15s; }
#app-title a:hover { background: var(--header-hover); }
#app-title img { display: inline-block; height: 1.476em; width: auto; vertical-align: baseline; position: relative; top: 0.46em; margin-left: -0.169em; }
/* One selector per rule: Gradio prefixes .dark onto every selector of a list that mentions it. */
#app-title img.logo-dark { display: none; }
.dark #app-title img.logo-dark { display: inline-block; }
.dark #app-title img.logo-light { display: none; }
#app-description p { margin: 0 0 0.4rem; max-width: 72rem; font-size: 0.95rem; line-height: 1.45; color: var(--header-text); }
#app-description strong { color: var(--header-strong); }
#source-video video { max-height: 320px !important; }
/* The examples side by side: four equal tiles in one row (clips of any shape are cropped to fill), with a pager. */
/* Gradio also gives each tile's inner box the class "gallery", so the row is the one holding the tiles. */
#examples .gallery:has(> .gallery-item) { display: grid !important; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.4rem; }
#examples .gallery-item { padding: 0; min-width: 0; width: 100%; }
#examples .gallery-item .container { width: 100% !important; height: 72px !important; }
#examples .gallery-item video, #examples .gallery-item img { width: 100% !important; height: 100% !important; max-width: none !important; object-fit: cover; }
#run-row button { font-size: 1.1rem; min-height: 3rem; }
#run-summary td:first-child, #run-summary th:first-child { white-space: nowrap; }
#run-summary table { width: 100%; }
#run-summary th, #run-summary td { padding: 0.25rem 0.6rem !important; font-size: 0.9rem; }
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
