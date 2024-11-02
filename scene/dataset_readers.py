#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import os.path as osp
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, depths_folder, test_cam_names_list):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = osp.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = osp.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readCamerasFromTransforms(path, transformsfile, depths_folder, white_background, is_test, extension=".png"):
    cam_infos = []

    with open(osp.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = osp.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = osp.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            depth_path = osp.join(depths_folder, f"{image_name}.png") if depths_folder != "" else ""

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image.size[0], height=image.size[1], depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):
    depths_folder=osp.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "points3d.ply")
    if not osp.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = osp.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = osp.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = osp.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None

    # not needed
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # not needed
    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
        else:
            with open(osp.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=osp.join(path, reading_dir), 
        depths_folder=osp.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = osp.join(path, "sparse/0/points3D.ply")
    bin_path = osp.join(path, "sparse/0/points3D.bin")
    txt_path = osp.join(path, "sparse/0/points3D.txt")
    if not osp.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info


class DriveStudioCamera:
    def __init__(self, uid, name, image_folder, extr_folder, intr_folder, lidarpose_folder, ends_with):
        self.uid = uid
        self.image_name = name

        self.ts, self.cam_id = name.split("_")

        self.image_path = osp.join(image_folder, self.image_name + ends_with)
        self.intr = self.read_intr(osp.join(intr_folder, self.cam_id + ".txt"))

        extr = self.read_pose(osp.join(extr_folder, name + ".txt"))
        self.lidarpose = self.read_pose(osp.join(lidarpose_folder, self.ts + ".txt"))

        # self.cam_pose = self.calc_cam_pose(self.lidarpose, extr)
        self.cam_pose = extr
    
    def calc_cam_pose(self, lidarpose, extr):
        print(lidarpose)
        print(np.linalg.inv(lidarpose))
        print(extr)
        raise
        return np.linalg.inv(lidarpose) @ extr

    def read_intr(self, path):
        data = np.loadtxt(path)
        if len(data) != 9:
            raise ValueError(f"line should be 9")
        return data

    def read_pose(self, path):
        data = np.loadtxt(path)

        if data.shape != (4, 4):
            raise ValueError(f"pose shape should be 4x4")
        
        return data

def readDriveStudioSceneInfo(path, eval, train_test_exp, llffhold=8, ends_with=".jpg", voxel_size = 0.1, plot=True):
    image_folder = osp.join(path, "images")
    extr_folder = osp.join(path, "extrinsics")
    intr_folder = osp.join(path, "intrinsics")
    lidarpose_folder = osp.join(path, "lidar_pose")
    bin_folder = osp.join(path, "lidar")

    images = [osp.basename(f).replace(ends_with, "") for f in os.listdir(image_folder)]

    drive_cameras = [DriveStudioCamera(i, f, image_folder, extr_folder, intr_folder, lidarpose_folder, ends_with) for i, f in enumerate(images)]

    # points
    lidar_points = []
    for i, cam in enumerate(drive_cameras):
        pointcloud = np.fromfile(osp.join(bin_folder, cam.ts + ".bin"), dtype=np.float32, count=-1).reshape([-1, 4])
        xyz = pointcloud[:, :3]
        xyz = np.hstack([xyz, np.ones((len(xyz), 1))])

        lidar_points.append((cam.lidarpose @ xyz.T).T[:, :3])

    lidar_pointssss = np.vstack(lidar_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pointssss)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_point_cloud = np.asarray(downsampled_pcd.points)
    
    point_mean = np.mean(downsampled_point_cloud, axis=0)
    # downsampled_point_cloud -= point_mean

    ply_path = osp.join(path, 'downsampled_point_cloud.ply')
    o3d.io.write_point_cloud(ply_path, downsampled_pcd)
    print(f"point cloud for initialization {ply_path}")

    # eval
    if eval:
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam.image_name for cam in drive_cameras]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
    else:
        test_cam_names_list = []

    # camera infos
    cam_infos_unsorted = []
    
    for i, cam in enumerate(drive_cameras):
        R = cam.cam_pose[:3, :3]
        T = cam.cam_pose[:3, 3]



        focal_length_x = cam.intr[0]
        focal_length_y = cam.intr[1]
        width = cam.intr[2] * 2
        height = cam.intr[3] * 2
        FovX = focal2fov(focal_length_x, width)
        FovY = focal2fov(focal_length_y, height)

        cam_info = CameraInfo(uid=cam.uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=None,
                                image_path=cam.image_path, image_name=cam.image_name, depth_path="",
                                width=width, height=height, is_test=cam.image_name in test_cam_names_list)
        cam_infos_unsorted.append(cam_info)

    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)


    if plot:
        # 将 numpy 数组转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(downsampled_point_cloud)

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 添加点云和线框坐标系到可视化窗口
        vis.add_geometry(pcd)

        # 创建一个线框坐标系来表示相机的位置和方向
        def create_rgb_frame(size=5.0):
            # 顶点
            vertices = [
                [0, 0, 0],  # 原点
                [size, 0, 0],  # X轴
                [0, size, 0],  # Y轴
                [0, 0, size]   # Z轴
            ]
            # 线段
            lines = [
                [0, 1],  # X轴
                [0, 2],  # Y轴
                [0, 3]   # Z轴
            ]
            # 颜色
            colors = [
                [1, 0, 0],  # X轴红色
                [0, 1, 0],  # Y轴绿色
                [0, 0, 1]   # Z轴蓝色
            ]

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(vertices)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            return line_set

        for cam in train_cam_infos + test_cam_infos:
            # 创建线框坐标系
            rgb_frame = create_rgb_frame(size=5.0)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = cam.R
            camera_pose[:3, 3] = cam.T
            rgb_frame.transform(camera_pose)

            vis.add_geometry(rgb_frame)

        # 运行可视化
        vis.run()

        # 关闭可视化窗口
        vis.destroy_window()

    num_points = len(downsampled_point_cloud)
    pcd = BasicPointCloud(points=downsampled_point_cloud, colors=np.ones((num_points, 3)), normals=np.ones((num_points, 3)))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DriveStudio": readDriveStudioSceneInfo
}