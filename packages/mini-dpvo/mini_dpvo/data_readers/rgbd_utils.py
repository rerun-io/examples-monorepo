import os.path as osp

import numpy as np
import torch
from jaxtyping import Float32, Float64
from lietorch import SE3
from scipy.spatial.transform import Rotation


def parse_list(filepath: str, skiprows: int = 0) -> np.ndarray:
    """ read list data """
    data: np.ndarray = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
    return data

def associate_frames(
    tstamp_image: Float64[np.ndarray, "n_img"],
    tstamp_depth: Float64[np.ndarray, "n_depth"],
    tstamp_pose: Float64[np.ndarray, "n_pose"] | None,
    max_dt: float = 1.0,
) -> list[tuple[int, ...]] :
    """ pair images, depths, and poses """
    associations: list[tuple[int, ...]] = []
    for i, t in enumerate(tstamp_image):
        if tstamp_pose is None:
            j: int = np.argmin(np.abs(tstamp_depth - t))
            if (np.abs(tstamp_depth[j] - t) < max_dt):
                associations.append((i, j))

        else:
            j = np.argmin(np.abs(tstamp_depth - t))
            k: int = np.argmin(np.abs(tstamp_pose - t))

            if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                    (np.abs(tstamp_pose[k] - t) < max_dt):
                associations.append((i, j, k))

    return associations

def loadtum(datapath: str, frame_rate: int = -1) -> tuple[
    list[str] | None,
    list[str] | None,
    list[Float64[np.ndarray, "7"]] | None,
    list[Float64[np.ndarray, "4"]] | None,
    list[float] | None,
]:
    """ read video data in tum-rgbd format """
    if osp.isfile(osp.join(datapath, 'groundtruth.txt')):
        pose_list: str = osp.join(datapath, 'groundtruth.txt')

    elif osp.isfile(osp.join(datapath, 'pose.txt')):
        pose_list = osp.join(datapath, 'pose.txt')

    else:
        return None, None, None, None, None

    image_list: str = osp.join(datapath, 'rgb.txt')
    depth_list: str = osp.join(datapath, 'depth.txt')

    calib_path: str = osp.join(datapath, 'calibration.txt')
    intrinsic: Float64[np.ndarray, "4"] | None = None
    if osp.isfile(calib_path):
        intrinsic = np.loadtxt(calib_path, delimiter=' ')
        intrinsic = intrinsic.astype(np.float64)

    image_data: np.ndarray = parse_list(image_list)
    depth_data: np.ndarray = parse_list(depth_list)
    pose_data: np.ndarray = parse_list(pose_list, skiprows=1)
    pose_vecs: Float64[np.ndarray, "n 7"] = pose_data[:,1:].astype(np.float64)

    tstamp_image: Float64[np.ndarray, "n_img"] = image_data[:,0].astype(np.float64)
    tstamp_depth: Float64[np.ndarray, "n_depth"] = depth_data[:,0].astype(np.float64)
    tstamp_pose: Float64[np.ndarray, "n_pose"] = pose_data[:,0].astype(np.float64)
    associations: list[tuple[int, ...]] = associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

    # print(len(tstamp_image))
    # print(len(associations))

    indicies: range = range(len(associations))[::5]

    # indicies = [ 0 ]
    # for i in range(1, len(associations)):
    #     t0 = tstamp_image[associations[indicies[-1]][0]]
    #     t1 = tstamp_image[associations[i][0]]
    #     if t1 - t0 > 1.0 / frame_rate:
    #         indicies += [ i ]

    images: list[str] = []
    poses: list[Float64[np.ndarray, "7"]] = []
    depths: list[str] = []
    intrinsics: list[Float64[np.ndarray, "4"]] = []
    tstamps: list[float] = []
    for ix in indicies:
        i: int
        j: int
        k: int
        (i, j, k) = associations[ix]
        images += [ osp.join(datapath, image_data[i,1]) ]
        depths += [ osp.join(datapath, depth_data[j,1]) ]
        poses += [ pose_vecs[k] ]
        tstamps += [ tstamp_image[i] ]

        if intrinsic is not None:
            intrinsics += [ intrinsic ]

    return images, depths, poses, intrinsics, tstamps


def all_pairs_distance_matrix(poses: list[Float64[np.ndarray, "7"]], beta: float = 2.5) -> Float32[np.ndarray, "n n"]:
    """ compute distance matrix between all pairs of poses """
    poses_np: Float32[np.ndarray, "n 7"] = np.array(poses, dtype=np.float32)
    poses_np[:,:3] *= beta # scale to balence rot + trans
    poses_se3: SE3 = SE3(torch.from_numpy(poses_np))

    r: torch.Tensor = (poses_se3[:,None].inv() * poses_se3[None,:]).log()
    return r.norm(dim=-1).cpu().numpy()

def pose_matrix_to_quaternion(pose: Float64[np.ndarray, "4 4"]) -> Float64[np.ndarray, "7"]:
    """ convert 4x4 pose matrix to (t, q) """
    q: Float64[np.ndarray, "4"] = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.concatenate([pose[:3, 3], q], axis=0)

def compute_distance_matrix_flow(
    poses: Float64[np.ndarray, "n 7"] | SE3,
    disps: Float64[np.ndarray, "n h w"] | Float32[torch.Tensor, "1 n h w"],
    intrinsics: Float64[np.ndarray, "n 4"] | Float32[torch.Tensor, "1 n 4"],
) -> Float32[np.ndarray, "n n"]:
    """ compute flow magnitude between all pairs of frames """
    if not isinstance(poses, SE3):
        poses = torch.from_numpy(poses).float().cuda()[None]
        poses = SE3(poses).inv()

        disps = torch.from_numpy(disps).float().cuda()[None]
        intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N: int = poses.shape[1]

    ii: torch.Tensor
    jj: torch.Tensor
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1).cuda()
    jj = jj.reshape(-1).cuda()

    MAX_FLOW: float = 100.0
    matrix: Float32[np.ndarray, "n n"] = np.zeros((N, N), dtype=np.float32)

    s: int = 2048
    for i in range(0, ii.shape[0], s):
        flow1: torch.Tensor
        val1: torch.Tensor
        flow1, val1 = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2: torch.Tensor
        val2: torch.Tensor
        flow2, val2 = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s])

        flow: torch.Tensor = torch.stack([flow1, flow2], dim=2)
        val: torch.Tensor = torch.stack([val1, val2], dim=2)

        mag: torch.Tensor = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.7] = np.inf

        i1: np.ndarray = ii[i:i+s].cpu().numpy()
        j1: np.ndarray = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix


def compute_distance_matrix_flow2(
    poses: SE3,
    disps: Float32[torch.Tensor, "1 n h w"],
    intrinsics: Float32[torch.Tensor, "1 n 4"],
    beta: float = 0.4,
) -> Float32[np.ndarray, "n n"]:
    """ compute flow magnitude between all pairs of frames """
    # if not isinstance(poses, SE3):
    #     poses = torch.from_numpy(poses).float().cuda()[None]
    #     poses = SE3(poses).inv()

    #     disps = torch.from_numpy(disps).float().cuda()[None]
    #     intrinsics = torch.from_numpy(intrinsics).float().cuda()[None]

    N: int = poses.shape[1]

    ii: torch.Tensor
    jj: torch.Tensor
    ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    MAX_FLOW: float = 128.0
    matrix: Float32[np.ndarray, "n n"] = np.zeros((N, N), dtype=np.float32)

    s: int = 2048
    for i in range(0, ii.shape[0], s):
        flow1a: torch.Tensor
        val1a: torch.Tensor
        flow1a, val1a = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s], tonly=True)
        flow1b: torch.Tensor
        val1b: torch.Tensor
        flow1b, val1b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])
        flow2a: torch.Tensor
        val2a: torch.Tensor
        flow2a, val2a = pops.induced_flow(poses, disps, intrinsics, jj[i:i+s], ii[i:i+s], tonly=True)
        flow2b: torch.Tensor
        val2b: torch.Tensor
        flow2b, val2b = pops.induced_flow(poses, disps, intrinsics, ii[i:i+s], jj[i:i+s])

        flow1: torch.Tensor = flow1a + beta * flow1b
        val1: torch.Tensor = val1a * val2b

        flow2: torch.Tensor = flow2a + beta * flow2b
        val2: torch.Tensor = val2a * val2b

        flow: torch.Tensor = torch.stack([flow1, flow2], dim=2)
        val: torch.Tensor = torch.stack([val1, val2], dim=2)

        mag: torch.Tensor = flow.norm(dim=-1).clamp(max=MAX_FLOW)
        mag = mag.view(mag.shape[1], -1)
        val = val.view(val.shape[1], -1)

        mag = (mag * val).mean(-1) / val.mean(-1)
        mag[val.mean(-1) < 0.8] = np.inf

        i1: np.ndarray = ii[i:i+s].cpu().numpy()
        j1: np.ndarray = jj[i:i+s].cpu().numpy()
        matrix[i1, j1] = mag.cpu().numpy()

    return matrix
