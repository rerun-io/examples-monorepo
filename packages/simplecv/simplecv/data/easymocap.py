import os
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import Float
from numpy import ndarray

from simplecv.camera_parameters import (
    BrownConradyDistortion,
    Extrinsics,
    Intrinsics,
    PinholeParameters,
)


class FileStorage:
    def __init__(self, filename):
        version = cv2.__version__
        self.major_version = int(version.split(".")[0])
        self.second_version = int(version.split(".")[1])

        assert os.path.exists(filename), filename
        self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = False

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def read(self, key, dt="mat"):
        if dt == "mat":
            output = self.fs.getNode(key).mat()
        elif dt == "list":
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == "":
                    val = str(int(n.at(i).real()))
                if val != "none":
                    results.append(val)
            output = results
        elif dt == "int":
            output = int(self.fs.getNode(key).real())
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read("names", dt="list")
    cameras = {}
    for key in camnames:
        cam = {}
        cam["K"] = intri.read(f"K_{key}")
        cam["invK"] = np.linalg.inv(cam["K"])
        cam["dist"] = intri.read(f"dist_{key}")
        cameras[key] = cam
    return cameras


def read_camera(intri_name: str, extri_name: str, cam_names: list[str] | None = None) -> dict:
    if cam_names is None:
        cam_names = []
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names: list[str] = intri.read("names", dt="list")
    for cam in cam_names:
        cams[cam] = {}
        cams[cam]["K"] = intri.read(f"K_{cam}")
        cams[cam]["invK"] = np.linalg.inv(cams[cam]["K"])
        H: int = intri.read(f"H_{cam}", dt="int")
        W: int = intri.read(f"W_{cam}", dt="int")
        if H is None or W is None:
            print(f"[camera] no H or W for {cam}")
            H, W = -1, -1
        cams[cam]["H"] = H
        cams[cam]["W"] = W
        Rvec = extri.read(f"R_{cam}")
        Tvec = extri.read(f"T_{cam}")
        assert Rvec is not None, cam
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]["RT"] = RT
        cams[cam]["R"] = R
        cams[cam]["Rvec"] = Rvec
        cams[cam]["T"] = Tvec
        cams[cam]["center"] = -Rvec.T @ Tvec
        P[cam] = cams[cam]["K"] @ cams[cam]["RT"]
        cams[cam]["P"] = P[cam]

        cams[cam]["dist"] = intri.read(f"dist_{cam}")
        if cams[cam]["dist"] is None:
            cams[cam]["dist"] = intri.read(f"D_{cam}")
            if cams[cam]["dist"] is None:
                print(f"[camera] no dist for {cam}")
    cams["basenames"] = cam_names
    return cams


def load_cameras(data_path: Path) -> list[PinholeParameters]:
    cameras = read_camera(str(data_path / "intri.yml"), str(data_path / "extri.yml"))
    cameras.pop("basenames")
    camera_list: list[PinholeParameters] = []
    for cam_name, cam in cameras.items():
        extri = Extrinsics(cam_R_world=cam["R"], cam_t_world=cam["T"].reshape(3))
        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=cam["K"][0, 0],
            fl_y=cam["K"][1, 1],
            cx=cam["K"][0, 2],
            cy=cam["K"][1, 2],
        )
        cam_distortion: Float[ndarray, "5"] = cam["dist"].squeeze()  # noqa UP307
        distortion = BrownConradyDistortion(
            k1=float(cam_distortion[0]),
            k2=float(cam_distortion[1]),
            p1=float(cam_distortion[2]),
            p2=float(cam_distortion[3]),
            k3=float(cam_distortion[4]),
        )
        pinhole_cam = PinholeParameters(name=cam_name, extrinsics=extri, intrinsics=intri, distortion=distortion)
        camera_list.append(pinhole_cam)
    return camera_list


def get_Pall(cameras, camnames):
    Pall = np.stack([cameras[cam]["K"] @ np.hstack((cameras[cam]["R"], cameras[cam]["T"])) for cam in camnames])
    return Pall
