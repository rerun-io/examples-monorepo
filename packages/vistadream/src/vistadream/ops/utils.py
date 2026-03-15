import os
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from omegaconf import OmegaConf
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


def gen_config(cfg_path):
    return OmegaConf.load(cfg_path)


def dpt2xyz(
    dpt: Float[np.ndarray, "H W"],
    intrinsic: Float[np.ndarray, "3 3"],
) -> Float[np.ndarray, "H W 3"]:
    # get grid
    height, width = dpt.shape[0:2]
    grid_u = np.arange(width)[None, :].repeat(height, axis=0)
    grid_v = np.arange(height)[:, None].repeat(width, axis=1)
    grid = np.concatenate([grid_u[:, :, None], grid_v[:, :, None], np.ones_like(grid_v)[:, :, None]], axis=-1)
    uvz = grid * dpt[:, :, None]
    # inv intrinsic
    inv_intrinsic = np.linalg.inv(intrinsic)
    xyz = np.einsum("ab,hwb->hwa", inv_intrinsic, uvz)
    return xyz


def transform_points(pts, transform):
    h, w = transform.shape
    if h == 3 and w == 3:
        return pts @ transform.T
    if h == 3 and w == 4:
        return pts @ transform[:, :3].T + transform[:, 3:].T
    elif h == 4 and w == 4:
        return pts @ transform[0:3, :3].T + transform[0:3, 3:].T
    else:
        raise NotImplementedError


def get_nml_from_quant(quant):
    """
    input N*4
    outut N*3
    follow https://arxiv.org/pdf/2404.17774
    """
    w = quant[:, 0]
    x = quant[:, 1]
    y = quant[:, 2]
    z = quant[:, 3]
    n0 = 2 * x * z + 2 * y * w
    n1 = 2 * y * z - 2 * x * w
    n2 = 1 - 2 * x * x - 2 * y * y
    nml = torch.cat((n0[:, None], n1[:, None], n2[:, None]), dim=1)
    return nml


def quaternion_from_matrix(M):
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]
    K = torch.zeros((len(M), 4, 4)).to(M)
    K[:, 0, 0] = m00 - m11 - m22
    K[:, 1, 0] = m01 + m10
    K[:, 1, 1] = m11 - m00 - m22
    K[:, 2, 0] = m02 + m20
    K[:, 2, 1] = m12 + m21
    K[:, 2, 2] = m22 - m00 - m11
    K[:, 3, 0] = m21 - m12
    K[:, 3, 1] = m02 - m20
    K[:, 3, 2] = m10 - m01
    K[:, 3, 3] = m00 + m11 + m22
    K = K / 3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = torch.linalg.eigh(K)
    q = V[torch.arange(len(V)), :, torch.argmax(w, dim=1)]
    q = q[:, [3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i, 0] < 0.0:
            q[i] = -q[i]
    return q


def numpy_quaternion_from_matrix(M):
    H, W = M.shape[0:2]
    M = M.reshape(-1, 3, 3)
    m00 = M[..., 0, 0]
    m01 = M[..., 0, 1]
    m02 = M[..., 0, 2]
    m10 = M[..., 1, 0]
    m11 = M[..., 1, 1]
    m12 = M[..., 1, 2]
    m20 = M[..., 2, 0]
    m21 = M[..., 2, 1]
    m22 = M[..., 2, 2]
    K = np.zeros((len(M), 4, 4))
    K[..., 0, 0] = m00 - m11 - m22
    K[..., 1, 0] = m01 + m10
    K[..., 1, 1] = m11 - m00 - m22
    K[..., 2, 0] = m02 + m20
    K[..., 2, 1] = m12 + m21
    K[..., 2, 2] = m22 - m00 - m11
    K[..., 3, 0] = m21 - m12
    K[..., 3, 1] = m02 - m20
    K[..., 3, 2] = m10 - m01
    K[..., 3, 3] = m00 + m11 + m22
    K = K / 3
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[np.arange(len(V)), :, np.argmax(w, axis=1)]
    q = q[..., [3, 0, 1, 2]]
    for i in range(len(q)):
        if q[i, 0] < 0.0:
            q[i] = -q[i]
    q = q.reshape(H, W, 4)
    return q


def numpy_normalize(input):
    input = input / (np.sqrt(np.sum(np.square(input), axis=-1, keepdims=True)) + 1e-5)
    return input


class suppress_stdout_stderr:
    """
    Avoid terminal output of diffusion processings!
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


def nei_delta(input, pad=2):
    if type(input) is not torch.Tensor:
        input = torch.from_numpy(input.astype(np.float32))
    if len(input.shape) < 3:
        input = input[:, :, None]
    h, w, c = input.shape
    # reshape
    input = input.permute(2, 0, 1)[None]
    input = F.pad(input, pad=(pad, pad, pad, pad), mode="replicate")
    kernel = 2 * pad + 1
    input = F.unfold(input, [kernel, kernel], padding=0)
    input = input.reshape(c, -1, h, w).permute(2, 3, 0, 1).squeeze()  # hw(3)*25
    return torch.amax(input, dim=-1), torch.amin(input, dim=-1), input


def inpaint_mask(render_dpt, render_rgb):
    # edge filter delta thres
    valid_dpt = render_dpt[render_dpt > 1e-3]
    valid_dpt = torch.sort(valid_dpt).values
    max = valid_dpt[int(0.85 * len(valid_dpt))]
    min = valid_dpt[int(0.15 * len(valid_dpt))]
    ths = (max - min) * 0.2
    # nei check
    nei_max, nei_min, _ = nei_delta(render_dpt, pad=1)
    edge_mask = (nei_max - nei_min) > ths
    # render hole
    hole_mask = render_dpt < 1e-3
    # whole mask -- original noise and sparse
    mask = edge_mask | hole_mask
    mask = mask.cpu().float().numpy()

    # modify rgb sightly for small holes : blur and sharpen
    render_rgb = render_rgb.detach().cpu().numpy()
    render_rgb = (render_rgb * 255).astype(np.uint8)
    render_rgb_blur = cv2.medianBlur(render_rgb, 5)
    render_rgb[mask > 0.5] = render_rgb_blur[mask > 0.5]  # blur and replace small holes
    render_rgb = torch.from_numpy((render_rgb / 255).astype(np.float32)).to(render_dpt)

    # slightly clean mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=7)
    mask = mask > 0.5

    return mask, render_rgb


def alpha_inpaint_mask(render_alpha):
    render_alpha = render_alpha.detach().squeeze().cpu().numpy()
    paint_mask = 1.0 - np.around(render_alpha)
    # slightly clean mask
    kernel = np.ones((5, 5), np.uint8)
    paint_mask = cv2.erode(paint_mask, kernel, iterations=1)
    paint_mask = cv2.dilate(paint_mask, kernel, iterations=3)
    paint_mask = paint_mask > 0.5
    return paint_mask


def edge_filter(metric_dpt, sky=None, times=0.1):
    sky = np.zeros_like(metric_dpt, bool) if sky is None else sky
    _max = np.percentile(metric_dpt[~sky], 95)
    _min = np.percentile(metric_dpt[~sky], 5)
    _range = _max - _min
    nei_max, nei_min, _ = nei_delta(metric_dpt)
    delta = (nei_max - nei_min).numpy()
    edge = delta > times * _range
    return edge


def fill_mask_with_nearest(imgs, mask):
    # mask and un-mask pixel coors
    mask_coords = np.column_stack(np.where(mask > 0.5))
    non_mask_coords = np.column_stack(np.where(mask < 0.5))
    # kd-tree on un-masked pixels
    tree = cKDTree(non_mask_coords)
    # nn search of masked pixels
    _, idxs = tree.query(mask_coords)
    # replace and fill
    for i, coord in enumerate(mask_coords):
        nearest_coord = non_mask_coords[idxs[i]]
        for img in imgs:
            img[coord[0], coord[1]] = img[nearest_coord[0], nearest_coord[1]]
    return imgs


def inpaint_tiny_holes(rgb, alpha, alpha_thres=0.9):
    ipaint_msk = alpha < alpha_thres
    output = deepcopy(rgb)
    for _i in range(5):
        output = cv2.blur(output, (10, 10))
        output[~ipaint_msk] = rgb[~ipaint_msk]
    output = cv2.blur(output, (5, 5))
    return output


def edge_rectify(metric_dpt, rgb, sky=None):
    edge = edge_filter(metric_dpt, sky)
    process_rgb = deepcopy(rgb)
    metric_dpt, process_rgb = fill_mask_with_nearest([metric_dpt, process_rgb], edge)
    return metric_dpt, process_rgb


def knn(x, K: int = 4):
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x)
    distances, _ = model.kneighbors(x)
    return distances
