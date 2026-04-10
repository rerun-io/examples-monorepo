import re
from os.path import *
from typing import IO

import cv2
import numpy as np
from jaxtyping import Float32, Float64
from PIL import Image
from scipy.spatial.transform import Rotation

cv2.setNumThreads(0)


TAG_CHAR: Float32[np.ndarray, "1"] = np.array([202021.25], np.float32)

def readFlowKITTI(filename: str) -> tuple[Float32[np.ndarray, "h w 2"], Float32[np.ndarray, "h w"]]:
    flow_bgr: Float32[np.ndarray, "h w 3"] = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    flow_bgr = flow_bgr[:,:,::-1].astype(np.float32)
    flow: Float32[np.ndarray, "h w 2"] = flow_bgr[:, :, :2]
    valid: Float32[np.ndarray, "h w"] = flow_bgr[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

def readFlow(fn: str) -> Float32[np.ndarray, "h w 2"] | None:
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic: Float32[np.ndarray, "1"] = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w: np.ndarray = np.fromfile(f, np.int32, count=1)
            h: np.ndarray = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data: Float32[np.ndarray, "n"] = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file: str) -> Float32[np.ndarray, "..."] | Float32[np.ndarray, "h w 3"]:
    file_handle: IO[bytes] = open(file, 'rb')

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
        dim_match: re.Match | None = re.match(rb'^(\d+)\s(\d+)\s$', file_handle.readline())
    except:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file_handle.readline())

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

    data: np.ndarray = np.fromfile(file_handle, endian + 'f')
    shape: tuple[int, ...] = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


def writeFlow(filename: str, uv: Float32[np.ndarray, "h w 2"] | Float32[np.ndarray, "h w"], v: Float32[np.ndarray, "h w"] | None = None) -> None:
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands: int = 2

    u: Float32[np.ndarray, "h w"]
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
    f: IO[bytes] = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp: Float64[np.ndarray, "h w2"] = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readDPT(filename: str) -> Float32[np.ndarray, "h w"]:
    """ Read depth data from file, return as numpy array. """
    f: IO[bytes] = open(filename,'rb')
    check: float = np.fromfile(f,dtype=np.float32,count=1)[0]
    TAG_FLOAT: float = 202021.25
    TAG_CHAR_LOCAL: str = 'PIEH'
    assert check == TAG_FLOAT, f' depth_read:: Wrong tag in flow file (should be: {TAG_FLOAT}, is: {check}). Big-endian machine? '
    width: int = np.fromfile(f,dtype=np.int32,count=1)[0]
    height: int = np.fromfile(f,dtype=np.int32,count=1)[0]
    size: int = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, f' depth_read:: Wrong input size (width = {width}, height = {height}).'
    depth: Float32[np.ndarray, "h w"] = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename: str) -> tuple[Float64[np.ndarray, "7"], Float64[np.ndarray, "4"]]:
    """ Read camera data, return (M,N) tuple.
    M is the intrinsic matrix, N is the extrinsic matrix, so that
    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates."""
    f: IO[bytes] = open(filename,'rb')
    check: float = np.fromfile(f,dtype=np.float32,count=1)[0]
    M: Float64[np.ndarray, "3 3"] = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N: Float64[np.ndarray, "3 4"] = np.fromfile(f,dtype='float64',count=12).reshape((3,4))

    E: Float64[np.ndarray, "4 4"] = np.eye(4)
    E[0:3,:] = N

    fx: float = M[0,0]
    fy: float = M[1,1]
    cx: float = M[0,2]
    cy: float = M[1,2]
    kvec: Float64[np.ndarray, "4"] = np.array([fx, fy, cx, cy])

    q: Float64[np.ndarray, "4"] = Rotation.from_matrix(E[:3,:3]).as_quat()
    pvec: Float64[np.ndarray, "7"] = np.concatenate([E[:3,3], q], 0)

    return pvec, kvec


def read_gen(file_name: str, pil: bool = False) -> Image.Image | np.ndarray | Float32[np.ndarray, "..."] | tuple[Float64[np.ndarray, "7"], Float64[np.ndarray, "4"]] | list:
    ext: str = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        return readPFM(file_name).astype(np.float32)
    elif ext == '.dpt':
        return readDPT(file_name).astype(np.float32)
    elif ext == '.cam':
        return cam_read(file_name)
    return []
