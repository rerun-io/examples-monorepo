"""
Demonstrates integrating Rerun visualization with Gradio.

Provides example implementations of data streaming, keypoint annotation, and dynamic
visualization across multiple Gradio tabs using Rerun's recording and visualization capabilities.
"""

import html
import json
from pathlib import Path
from typing import Literal

import gradio as gr
from huggingface_hub import HfApi

API = HfApi()  # thin REST wrapper
RERUN_WEB_VIEWER_VERSION = "0.31.4"

AVAILABLE_DATASETS = ["egoexo-forge-hocap", "egoexo-forge-egodex", "egoexo-forge-assembly101"]


def build_rerun_embed(rrd_url: str | None, *, height: int = 800) -> str:
    if not rrd_url:
        return "<div style='height: 800px; display: grid; place-items: center; border: 1px solid #2a2a2a; border-radius: 12px; color: #666;'>No recording selected.</div>"

    srcdoc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body, #viewer {{
        margin: 0;
        width: 100%;
        height: 100%;
        background: #0f1115;
      }}
    </style>
  </head>
  <body>
    <div id="viewer"></div>
    <script type="module">
      import {{ WebViewer }} from "https://esm.sh/@rerun-io/web-viewer@{RERUN_WEB_VIEWER_VERSION}?bundle";
      const viewer = new WebViewer();
      await viewer.start(
        {json.dumps(rrd_url)},
        document.getElementById("viewer"),
        {{
          hide_welcome_screen: true,
          allow_fullscreen: true,
          width: "",
          height: "",
        }},
      );
    </script>
  </body>
</html>"""
    escaped_srcdoc = html.escape(srcdoc, quote=True)
    return (
        f'<iframe srcdoc="{escaped_srcdoc}" '
        f'style="width: 100%; height: {height}px; border: 0; border-radius: 12px; background: #0f1115;" '
        'allow="fullscreen"></iframe>'
    )


def show_dataset(
    repo_name: Literal["egoexo-forge-hocap", "egoexo-forge-egodex", "egoexo-forge-assembly101"], episode_index: str
):
    episode_index = f"{int(episode_index):05d}"
    url_str = f"https://huggingface.co/datasets/pablovela5620/{repo_name}/resolve/main/{episode_index}.rrd"
    return build_rerun_embed(url_str)


def list_episodes(dataset: Literal["egoexo-forge-hocap"]) -> list[str]:
    """
    Return ["00000", "00001", ...] for the chosen task folder.
    """
    try:
        files = API.list_repo_files(f"pablovela5620/{dataset}", repo_type="dataset")
    except Exception:
        return []
    return sorted({Path(f).stem for f in files if f.endswith(".rrd")})


default_dataset = AVAILABLE_DATASETS[0]
initial_eps = list_episodes(default_dataset)


title = """# EgoExo Forge: All the Data + Utilities Needed for Egocentric and Exocentric Human Data"""
description1 = """
    <a title="Website" href="https://rerun.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/badge/Rerun-0.23.0-blue.svg?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzQ0MV8xMTAzOCkiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHJ4PSI4IiBmaWxsPSJibGFjayIvPgo8cGF0aCBkPSJNMy41OTcwMSA1Ljg5NTM0TDkuNTQyOTEgMi41MjM1OUw4Ljg3ODg2IDIuMTQ3MDVMMi45MzMgNS41MTg3NUwyLjkzMjk1IDExLjI5TDMuNTk2NDIgMTEuNjY2MkwzLjU5NzAxIDUuODk1MzRaTTUuMDExMjkgNi42OTc1NEw5LjU0NTc1IDQuMTI2MDlMOS41NDU4NCA0Ljk3NzA3TDUuNzYxNDMgNy4xMjI5OVYxMi44OTM4SDcuMDg5MzZMNi40MjU1MSAxMi41MTczVjExLjY2Nkw4LjU5MDY4IDEyLjg5MzhIOS45MTc5NUw2LjQyNTQxIDEwLjkxMzNWMTAuMDYyMUwxMS40MTkyIDEyLjg5MzhIMTIuNzQ2M0wxMC41ODQ5IDExLjY2ODJMMTMuMDM4MyAxMC4yNzY3VjQuNTA1NTlMMTIuMzc0OCA0LjEyOTQ0TDEyLjM3NDMgOS45MDAyOEw5LjkyMDkyIDExLjI5MTVMOS4xNzA0IDEwLjg2NTlMMTEuNjI0IDkuNDc0NTRWMy43MDM2OUwxMC45NjAyIDMuMzI3MjRMMTAuOTYwMSA5LjA5ODA2TDguNTA2MyAxMC40ODk0TDcuNzU2MDEgMTAuMDY0TDEwLjIwOTggOC42NzI1MlYyLjk5NjU2TDQuMzQ3MjMgNi4zMjEwOUw0LjM0NzE3IDEyLjA5Mkw1LjAxMDk0IDEyLjQ2ODNMNS4wMTEyOSA2LjY5NzU0Wk05LjU0NTc5IDUuNzMzNDFMOS41NDU4NCA4LjI5MjA2TDcuMDg4ODYgOS42ODU2NEw2LjQyNTQxIDkuMzA5NDJWNy41MDM0QzYuNzkwMzIgNy4yOTY0OSA5LjU0NTg4IDUuNzI3MTQgOS41NDU3OSA1LjczMzQxWiIgZmlsbD0id2hpdGUiLz4KPC9nPgo8ZGVmcz4KPGNsaXBQYXRoIGlkPSJjbGlwMF80NDFfMTEwMzgiPgo8cmVjdCB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIGZpbGw9IndoaXRlIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==">
    </a>
    <a title="Github" href="https://github.com/rerun-io/egoexo-forge" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/egoexo-forge?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
"""
description2 = "A comprehensive collection of datasets and tools for egocentric and exocentric human activity understanding, featuring hand-object interactions, manipulation tasks, and multi-view recordings."

dataset_description = """
    # Dataset Information
    
    ## Assembly101
    A procedural activity dataset with 4321 multi-view videos of people assembling and disassembling 101 take-apart toy vehicles, featuring rich variations in action ordering, mistakes, and corrections.
    **Note:** Assembly101 is very large and can take some time to fully load each episode.
    
    ## HoCap
    A dataset for 3D reconstruction and pose tracking of hands and objects in videos, featuring humans interacting with objects for various tasks including pick-and-place actions and handovers.
    
    ## EgoDex
    The largest and most diverse dataset of dexterous human manipulation with 829 hours of egocentric video and paired 3D hand tracking, covering 194 different tabletop tasks with everyday household objects.
    """

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)

    with gr.Tab("Hosted RRD"):
        with gr.Row():
            with gr.Column(scale=1):
                task_name = gr.Dropdown(
                    label="Dataset Name",
                    choices=AVAILABLE_DATASETS,
                    value=default_dataset,
                )
                episode_index = gr.Dropdown(
                    label="Episode Index",
                    choices=initial_eps,
                    value=initial_eps[0] if initial_eps else None,
                )

                def _update_eps(t):  # Gradio wants a fn
                    eps = list_episodes(t)
                    return gr.update(choices=eps, value=eps[0] if eps else None)

                task_name.change(_update_eps, inputs=task_name, outputs=episode_index)

                button = gr.Button("Show Dataset")
                gr.Markdown(dataset_description)
            with gr.Column(scale=4):
                viewer = gr.HTML(
                    value=build_rerun_embed(None),
                    show_label=False,
                    container=False,
                )
        button.click(fn=show_dataset, inputs=[task_name, episode_index], outputs=[viewer])
