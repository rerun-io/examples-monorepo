"""Standalone launcher for the PromptDA Gradio app."""

from argparse import ArgumentParser

import gradio as gr

from rerun_prompt_da.gradio_ui.prompt_da_ui import prompt_da_block

title = """# Prompt Depth Anything: Unofficial Demo of 4K Resolution Accurate Metric Depth Estimation"""
description1 = """
    <a title="Website" href="https://promptda.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
    </a>
    <a title="arXiv" href="https://arxiv.org/abs/2403.20309" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
    </a>
    <a title="Github" href="https://github.com/rerun-io/prompt-da" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/github/stars/rerun-io/prompt-da?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
    </a>
    <a title="Social" href="https://x.com/pablovelagomez1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
    </a>
"""
description2 = "Using Rerun to visualize the results of Prompt Depth Anything with Polycam Input"

with gr.Blocks() as demo:
    # Keep the outer app wrapper thin; the actual UI lives in prompt_da_block.
    gr.Markdown(title)
    gr.Markdown(description1)
    gr.Markdown(description2)
    with gr.Tab(label="Prompt Depth Anything"):
        prompt_da_block.render()

if __name__ == "__main__":
    # Support direct local execution as well as the pixi task wrapper.
    parser = ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    demo.launch(share=args.share)
