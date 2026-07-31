from typing import *
from typing_extensions import Unpack
import math
from numbers import Number

import numpy as np
from numpy import ndarray


def sliding_window(
    x: ndarray, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    axis: Optional[Tuple[int,...]] = None
) -> ndarray:
    """
    Get a sliding window of the input array. Window axis(axes) will be appended as the last dimension(s).
    This function is a wrapper of `numpy.lib.stride_tricks.sliding_window_view` with additional support for padding and stride.

    ## Parameters
    - `x` (ndarray): Input array.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride of the sliding window. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (ndarray): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    # Process axis
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis, int):
        axis = (axis,)
    axis = [axis[i] % x.ndim for i in range(len(axis))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(axis)
    
    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(axis)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, int) for p in pad_size):
            pad_size = (pad_size,) * len(axis)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(axis)
            else:
                assert len(pad_size) == len(axis), f"pad_size {pad_size} must match the number of axes {len(axis)}"
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in axis else pad_size[axis.index(i)] for i in range(x.ndim)]
        if pad_mode == 'constant':
            x = np.pad(x, full_pad, mode=pad_mode, constant_values=pad_value)
        else:
            x = np.pad(x, full_pad, mode=pad_mode)
    
    # Apply sliding window
    x = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=axis)

    # Apply stride if needed
    if stride is not None:
        if isinstance(stride, int):
            stride = (stride,) * len(axis)
        stride_slice = tuple(slice(None) if i not in axis else slice(None, None, stride[axis.index(i)]) for i in range(x.ndim))
        x = x[stride_slice]

    return x


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: np.dtype = np.float32
) -> ndarray:
    """
    Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
    This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in uv space. Defaults to 0.
    - `left`: `float`, optional left boundary in uv space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    - `right`: `float`, optional right boundary in uv space. Defaults to 1.
    - `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to np.float32.

    ## Returns
    - `uv (ndarray)`: shape `(height, width, 2)`

    ## Example Usage

    >>> uv_map(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = np.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype)
    v = np.linspace(top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
    dtype: np.dtype = np.float32
) -> ndarray:
    """
    Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
            The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
        - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
            The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
    - `dtype`: `np.dtype`, optional data type of the output pixel coord map. Defaults to np.float32.

    ## Returns
        ndarray: shape (height, width, 2)
    
    >>> pixel_coord_map(10, 10, convention='integer-center', dtype=int):
    [[[0, 0], [1, 0], ..., [9, 0]],
     [[0, 1], [1, 1], ..., [9, 1]],
        ...      ...         ...
     [[0, 9], [1, 9], ..., [9, 9]]]

    >>> pixel_coord_map(10, 10, convention='integer-corner', dtype=np.float32):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = np.arange(left, left + width, dtype=dtype)
    v = np.arange(top, top + height, dtype=dtype)
    if convention == 'integer-corner':
        assert np.issubdtype(dtype, np.floating), "dtype should be a floating point type when convention is 'integer-corner'"
        u = u + 0.5
        v = v + 0.5
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)


def masked_nearest_resize(
    *image: ndarray,
    mask: ndarray, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Unpack[Tuple[ndarray, ...]], ndarray, Tuple[ndarray, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness. 

    ### Parameters
    - `*image`: Input image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: input mask of shape `(..., H, W)`, dtype=bool
    - `size`: target size `(H', W')`
    - `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
        Defaults to False.

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    - `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension.
    """
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention='integer-corner', dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, axis=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_centers - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.round(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    dist = np.square(target_window_pixels - target_centers[..., None])
    dist = dist[..., 0, :] + dist[..., 1, :]
    dist = np.where(target_window_mask, dist, np.inf)                                                   # (..., target_height, tgt_width, filter_size)
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)                                         # (..., target_height, tgt_width, 1)
    nearest_idx = np.take_along_axis(
        np.broadcast_to(target_window_indices, dist.shape),
        nearest_in_window, 
        axis=-1
    ).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    batch_indices = [np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    nearest_indices = (*batch_indices, nearest_i, nearest_j)
    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        return *outputs, target_mask, nearest_indices
    else:
        return *outputs, target_mask
