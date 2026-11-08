"""View the SfM result: sparse points + one pyramid per camera."""
import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TXT = os.path.join(ROOT, "data", "Ammonite", "dense", "sparse_txt")


def read_cameras(path):
    """cameras.txt -> {camera_id: (K, width, height)}   (C1a slides 42-46)"""
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        cam_id, model, w, h = int(p[0]), p[1], int(p[2]), int(p[3])
        fx, fy, cx, cy = map(float, p[4:8])          # PINHOLE after undistortion
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        cams[cam_id] = (K, w, h)
    return cams


def read_images(path):
    """images.txt -> list of (name, R, t, camera_id)   (C1a slides 48-53)"""
    lines = [l for l in open(path) if not l.startswith("#")]
    images = []
    for line in lines[0::2]:          # every image uses 2 lines; the 2nd lists 2D points
        p = line.split()
        if len(p) < 10:
            continue
        qw, qx, qy, qz = map(float, p[1:5])
        t = np.array(list(map(float, p[5:8])))
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()   # scipy wants w LAST
        images.append((p[9], R, t, int(p[8])))
    return images


def read_points(path):
    """points3D.txt -> (N,3) positions, (N,3) colors   (C3 slide 34)"""
    xyz, rgb = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        xyz.append(list(map(float, p[1:4])))
        rgb.append(list(map(int, p[4:7])))
    return np.array(xyz), np.array(rgb) / 255.0


cams = read_cameras(os.path.join(TXT, "cameras.txt"))
images = read_images(os.path.join(TXT, "images.txt"))
xyz, rgb = read_points(os.path.join(TXT, "points3D.txt"))
print(f"{len(cams)} camera models, {len(images)} images, {len(xyz)} points")

# Sparse point cloud
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
pcd.colors = o3d.utility.Vector3dVector(rgb)

# Pyramid size: a fraction of the scene size (scale is arbitrary, C3 slide 36)
extent = np.linalg.norm(np.percentile(xyz, 95, axis=0) - np.percentile(xyz, 5, axis=0))
scale = 0.05 * extent

geoms = [pcd]
for name, R, t, cam_id in images:
    K, w, h = cams[cam_id]
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R               # [R | t]: world -> camera
    extrinsic[:3, 3] = t
    frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, K, extrinsic, scale)
    frustum.paint_uniform_color([1, 0, 0])
    geoms.append(frustum)

    C = -R.T @ t                        # camera center
    print(f"{name:25s} center = {np.round(C, 3)}")

o3d.visualization.draw_geometries(geoms, window_name="SfM result")