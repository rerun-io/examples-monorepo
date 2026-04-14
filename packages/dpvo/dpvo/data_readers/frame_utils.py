"""Low-level file I/O utilities for optical flow, depth, and camera data.

Supports the following formats:

- **Middlebury ``.flo``** -- standard optical-flow binary format.
- **KITTI flow** -- 16-bit PNG with validity channel.
- **PFM** -- Portable Float Map (used by some stereo benchmarks).
- **DPT** -- tagged float-array depth format.
- **``.cam``** -- binary camera file (intrinsic + extrinsic matrices).
- Common image formats (PNG, JPEG, PPM) via PIL.
"""

import re
from os.path import splitext
from typing import IO

import cv2
import numpy as np
from jaxtyping import Float32, Float64
from numpy import ndarray
from PIL import Image
from scipy.spatial.transform import Rotation

cv2.setNumThreads(0)

# Magic number used by the Middlebury .flo and .dpt formats to validate files.
TAG_CHAR: Float32[ndarray, "1"] = np.array([202021.25], np.float32)


def readFlowKITTI(filename: str) -> tuple[Float32[ndarray, "h w 2"], Float32[ndarray, "h w"]]:
    """Read a KITTI-format optical flow file (16-bit PNG).

    The blue and green channels encode the horizontal and vertical flow
    (scaled by 64, offset by 2^15). The red channel is the validity mask.

    Args:
        filename: Path to the KITTI flow PNG file.

    Returns:
        Tuple of ``(flow, valid)`` where ``flow`` has shape ``(h, w, 2)``
        and ``valid`` has shape ``(h, w)`` with 1.0 for valid pixels.
    """
    flow_bgr_raw = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    assert flow_bgr_raw is not None, f"Failed to read flow file: {filename}"
    flow_bgr: Float32[ndarray, "h w 3"] = flow_bgr_raw
    flow_bgr = flow_bgr[:,:,::-1].astype(np.float32)
    flow: Float32[ndarray, "h w 2"] = flow_bgr[:, :, :2]
    valid: Float32[ndarray, "h w"] = flow_bgr[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readFlow(fn: str) -> Float32[ndarray, "h w 2"] | None:
    """Read a ``.flo`` optical-flow file in Middlebury format.

    The file starts with a float32 magic number (202021.25), followed by
    width and height as int32, and then ``2 * w * h`` float32 flow values.

    Note:
        Only works on little-endian architectures (e.g. Intel x86).

    Code adapted from:
        http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    Args:
        fn: Path to the ``.flo`` file.

    Returns:
        Flow array of shape ``(h, w, 2)`` or ``None`` if the magic number
        does not match.
    """
    with open(fn, 'rb') as f:
        magic: Float32[ndarray, "1"] = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w: ndarray = np.fromfile(f, np.int32, count=1)
            h: ndarray = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data: Float32[ndarray, "n"] = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file: str) -> Float32[ndarray, "..."] | Float32[ndarray, "h w 3"]:
    """Read a Portable Float Map (``.pfm``) file.

    Supports both single-channel (``Pf``) and three-channel (``PF``) formats.
    The image is flipped vertically to match the top-left origin convention.

    Args:
        file: Path to the ``.pfm`` file.

    Returns:
        The loaded data as a float32 array. Shape is ``(h, w)`` for
        greyscale or ``(h, w, 3)`` for colour.

    Raises:
        Exception: If the header is malformed or the magic bytes are invalid.
    """
    file_handle: IO[bytes] = open(file, 'rb')  # noqa: SIM115

    color: bool | None = None
    width: int | None = None
    height: int | None = None
    scale: float | None = None
    endian: str | None = None

    header: bytes = file_handle.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    try:
        dim_match: re.Match[bytes] | None = re.match(rb'^(\d+)\s(\d+)\s$', file_handle.readline())
    except Exception:
        dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file_handle.readline())

    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file_handle.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data: ndarray = np.fromfile(file_handle, endian + 'f')
    shape: tuple[int, ...] = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename: str, uv: Float32[ndarray, "h w 2"] | Float32[ndarray, "h w"], v: Float32[ndarray, "h w"] | None = None) -> None:
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands: int = 2

    u: Float32[ndarray, "h w"]
    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height: int = u.shape[0]
    width: int = u.shape[1]
    f: IO[bytes] = open(filename,'wb')  # noqa: SIM115
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp: Float64[ndarray, "h w2"] = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readDPT(filename: str) -> Float32[ndarray, "h w"]:
    """Read a ``.dpt`` depth file in tagged float-array format.

    The file starts with a float32 tag (202021.25), followed by int32
    width/height, then ``w * h`` float32 depth values.

    Args:
        filename: Path to the ``.dpt`` file.

    Returns:
        Depth array of shape ``(h, w)``.

    Raises:
        AssertionError: If the tag or dimensions are invalid.
    """
    f: IO[bytes] = open(filename,'rb')  # noqa: SIM115
    check: float = np.fromfile(f,dtype=np.float32,count=1)[0]
    TAG_FLOAT: float = 202021.25
    _TAG_CHAR_LOCAL: str = 'PIEH'
    assert check == TAG_FLOAT, f' depth_read:: Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check}). Big-endian machine? '
    width: int = np.fromfile(f,dtype=np.int32,count=1)[0]
    height: int = np.fromfile(f,dtype=np.int32,count=1)[0]
    size: int = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, f' depth_read:: Wrong input size (width = {width}, height = {height}).'
    depth: Float32[ndarray, "h w"] = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename: str) -> tuple[Float64[ndarray, "7"], Float64[ndarray, "4"]]:
    """Read a binary ``.cam`` camera file and return pose + intrinsics.

    The file encodes a 3×3 intrinsic matrix **M** and a 3×4 extrinsic
    matrix **N** such that ``x = M · N · X`` projects a homogeneous world
    point **X** to pixel coordinates **x**.

    The extrinsic rotation is converted to a unit quaternion and concatenated
    with the translation to form a 7-D pose vector.

    Args:
        filename: Path to the ``.cam`` binary file.

    Returns:
        Tuple of ``(pvec, kvec)`` where ``pvec`` is ``[tx, ty, tz, qx, qy,
        qz, qw]`` and ``kvec`` is ``[fx, fy, cx, cy]``.
    """
    f: IO[bytes] = open(filename,'rb')  # noqa: SIM115
    _check: float = np.fromfile(f,dtype=np.float32,count=1)[0]
    M: Float64[ndarray, "3 3"] = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N: Float64[ndarray, "3 4"] = np.fromfile(f,dtype='float64',count=12).reshape((3,4))

    E: Float64[ndarray, "4 4"] = np.eye(4)
    E[0:3,:] = N

    fx: float = M[0,0]
    fy: float = M[1,1]
    cx: float = M[0,2]
    cy: float = M[1,2]
    kvec: Float64[ndarray, "4"] = np.array([fx, fy, cx, cy])

    q: Float64[ndarray, "4"] = Rotation.from_matrix(E[:3,:3]).as_quat()
    pvec: Float64[ndarray, "7"] = np.concatenate([E[:3,3], q], 0)

    return pvec, kvec


def read_gen(file_name: str, _pil: bool = False) -> Image.Image | ndarray | Float32[ndarray, "..."] | tuple[Float64[ndarray, "7"], Float64[ndarray, "4"]] | list:
    """Generic file reader that dispatches on file extension.

    Supported extensions:

    - ``.png``, ``.jpeg``, ``.jpg``, ``.ppm`` -- image (via PIL).
    - ``.bin``, ``.raw`` -- numpy binary (via :func:`numpy.load`).
    - ``.flo`` -- Middlebury optical flow.
    - ``.pfm`` -- Portable Float Map.
    - ``.dpt`` -- tagged depth format.
    - ``.cam`` -- binary camera file.

    Args:
        file_name: Path to the file.
        pil: Unused; retained for API compatibility.

    Returns:
        The loaded data in a type appropriate to the format, or an empty
        list if the extension is unrecognised.
    """
    ext: str = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        flow = readFlow(file_name)
        assert flow is not None, f"Failed to read .flo file: {file_name}"
        return flow.astype(np.float32)
    elif ext == '.pfm':
        return readPFM(file_name).astype(np.float32)
    elif ext == '.dpt':
        return readDPT(file_name).astype(np.float32)
    elif ext == '.cam':
        return cam_read(file_name)
    return []
