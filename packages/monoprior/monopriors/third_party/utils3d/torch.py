from functools import wraps
from itertools import chain
from numbers import Number
from typing import *
import inspect

import torch
from torch import Tensor
import torch.nn.functional as F


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper


P = ParamSpec("P")  
R = TypeVar("R")

def totensor(
    *args_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str, None], 
    _others: Union[torch.dtype, str] = None, 
    **kwargs_dtypes: Union[torch.dtype, Tuple[torch.dtype, torch.device], str]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that converts non-array arguments to array of specified default dtype.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dtypes_dict = {
            **dict(zip(argnames, args_dtypes)),
            **kwargs_dtypes
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            inputs = {
                **{argnames[i]: x for i, x in enumerate(args)},
                **kwargs
            }
            args = tuple(
                torch.tensor(x).to(inputs[dtype_device] if isinstance(dtype_device, str) else dtype_device)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype_device := dtypes_dict.get(argnames[i], _others)) is not None \
                else x
                for i, x in enumerate(args)
            )
            kwargs = {
                k: torch.tensor(x).to(inputs[dtype_device] if isinstance(dtype_device, str) else dtype_device)
                if isinstance(x, (Number, list, tuple)) \
                    and (dtype_device := dtypes_dict.get(k, _others)) is not None \
                else x
                for k, x in kwargs.items()
            }
            return func(*args, **kwargs)
        return wrapper
    return decorator


def batched(*args_dims: Union[int, None], _others: Union[int, None] = None, **kwargs_dims: Union[int, None]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator generator that extends a function's input and out batch dimensions.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        argnames = list(inspect.signature(func).parameters.keys())
        dims_dict = {
            **dict(zip(argnames, args_dims)),
            **kwargs_dims
        }
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            args = list(args)
            # Get arguments non-batch dimensions
            args_dim = tuple(dims_dict.get(argname, _others) for argname in argnames[:len(args)])
            kwargs_dim = {k: dims_dict.get(k, _others) for k in kwargs}
            # Find the common batch shape
            batch_shape = torch.broadcast_shapes(*(
                x.shape[:x.ndim - dim] 
                for x, dim in zip((*args, *kwargs.values()), (*args_dim, *kwargs_dim.values())) 
                if isinstance(x, Tensor) and dim is not None
            ))
            # Broadcast and flatten batch dimensions
            args = tuple(
                torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and dim is not None else x
                for x, dim in zip(args, args_dim)
            )
            kwargs = {
                k: torch.broadcast_to(x, (*batch_shape, *x.shape[x.ndim - dim:])).reshape((-1, *x.shape[x.ndim - dim:]))
                if isinstance(x, Tensor) and (dim := kwargs_dim[k]) is not None else x
                for k, x in kwargs.items()
            }
            # Call function
            result = func(*args, **kwargs)
            # Restore batch shape
            if isinstance(result, tuple):
                result = tuple(
                    x.reshape((*batch_shape, *x.shape[1:])) if isinstance(x, Tensor) else x
                    for x in result
                )
            elif isinstance(result, Tensor):
                result = result.reshape((*batch_shape, *result.shape[1:]))
            return result
        return wrapper
    return decorator


def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
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
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, int):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, int):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    if isinstance(stride, int):
        stride = (stride,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, int):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, int) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], window_size[i], stride[i])
    return x


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
    This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in uv space. Defaults to 0.
    - `left`: `float`, optional left boundary in uv space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    - `right`: `float`, optional right boundary in uv space. Defaults to 1.
    - `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to torch.float32.
    - `device`: `torch.device`, optional device of the output uv map. Defaults to None.

    ## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

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
    u = torch.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height, bottom - 0.5 / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


@totensor(_others=torch.float32)
@batched(_others=0)
def intrinsics_from_focal_center(
    fx: Union[float, Tensor],
    fy: Union[float, Tensor],
    cx: Union[float, Tensor],
    cy: Union[float, Tensor]
) -> Tensor:
    """
    Get OpenCV intrinsics matrix

    ## Parameters
        focal_x (float | Tensor): focal length in x axis
        focal_y (float | Tensor): focal length in y axis
        cx (float | Tensor): principal point in x axis
        cy (float | Tensor): principal point in y axis

    ## Returns
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    ret = torch.stack([
        fx, zeros, cx, 
        zeros, fy, cy, 
        zeros, zeros, ones
    ], dim=-1).unflatten(-1, (3, 3))
    return ret


def unproject_cv(
    uv: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor = None,
) -> Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    ## Parameters
        uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (Tensor): [..., N] depth value
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix

    ## Returns
        points (Tensor): [..., N, 3] 3d points
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    points = points[..., :3]
    return points


def depth_map_to_point_map(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor = None):
    height, width = depth.shape[-2:]
    uv = uv_map(height, width, dtype=depth.dtype, device=depth.device)
    pts = unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts
