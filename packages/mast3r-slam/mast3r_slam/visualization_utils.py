import functools
import imgui
import matplotlib
import torch
import numpy as np
# from in3d.geometry import LineGeometry


@functools.cache
def get_colormap(colormap):
    colormap = matplotlib.colormaps[colormap]
    return colormap(np.linspace(0, 1, 256))[:, :3]


def depth2rgb(depth, min=None, max=None, colormap="turbo", add_alpha=False, alpha=1.0):
    # depth: HxW
    dmin = np.nanmin(depth) if min is None else min
    dmax = np.nanmax(depth) if max is None else max
    d = (depth - dmin) / np.maximum((dmax - dmin), 1e-8)
    d = np.clip(d * 255, 0, 255).astype(np.int32)
    img = get_colormap(colormap)[d].astype(np.float32)
    if add_alpha:
        img = np.concatenate([img, alpha * np.ones_like(img[..., :1])], axis=-1)
    return np.ascontiguousarray(img)


def image_with_text(img, size, text, same_line=False):
    # check if the img is too small to render
    if size[0] < 16:
        return
    text_cursor_pos = imgui.get_cursor_pos()
    imgui.image(img.texture.glo, *size)
    if same_line:
        imgui.same_line()
    next_cursor_pos = imgui.get_cursor_pos()
    imgui.set_cursor_pos(text_cursor_pos)
    imgui.text(text)
    imgui.set_cursor_pos(next_cursor_pos)
