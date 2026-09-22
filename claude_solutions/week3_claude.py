"""PIV PBClass 3 - 3D point cloud and camera calibration (numpy/scipy only, no OpenCV).

Usage:
    python week3.py 1      # Task 1: point cloud from camera 0 + depth
    python week3.py 2      # Task 2: calibrate camera 1 from matches (DLT + RQ)
    python week3.py 3      # Task 3: propagate RGB from image 1 to image 0 and the cloud
    python week3.py test   # self-check of the DLT/decomposition on synthetic data
"""

import argparse

import numpy as np
import open3d as o3d
import scipy.io
import scipy.linalg

K0_PATH = "K_0.mat"
DEPTH_PATH = "depth_0.mat"
IMAGE0_PATH = "image_0.png"
IMAGE1_PATH = "image_1.png"
MATCHES_PATH = "matches.mat"

# matches.mat stores pixel coords; set True if they came out of MATLAB 1-based.
MATCHES_ONE_BASED = False


# ---------------------------------------------------------------- loading ---
def load_image(path):
    """PNG -> float array in [0, 1]. Grayscale stays 2D, RGB stays 3D."""
    return np.asarray(o3d.io.read_image(path)).astype(np.float64) / 255.0


def load_scene():
    K0 = scipy.io.loadmat(K0_PATH)["K"].astype(np.float64)
    depth = scipy.io.loadmat(DEPTH_PATH)["depth_image"].astype(np.float64)
    matches = scipy.io.loadmat(MATCHES_PATH)
    pts0 = matches["pts0"].astype(np.float64)
    pts1 = matches["pts1"].astype(np.float64)
    if MATCHES_ONE_BASED:
        pts0 = pts0 - 1.0
        pts1 = pts1 - 1.0
    return K0, depth, load_image(IMAGE0_PATH), load_image(IMAGE1_PATH), pts0, pts1


# ------------------------------------------------------------- projection ---
def backproject(K, depth):
    """Depth map -> 3D points in camera-0 frame, one row per pixel (raster order)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
    rays = np.linalg.inv(K) @ pixels
    return rays.T * depth.reshape(-1, 1)


def project(P, X):
    """3D points (N x 3) through a 3x4 camera -> pixels (N x 2) and depths (N,)."""
    xh = _homogeneous(X) @ P.T
    return xh[:, :2] / xh[:, 2:3], xh[:, 2]


# ------------------------------------------------------------ calibration ---
def _homogeneous(pts):
    return np.hstack([pts, np.ones((len(pts), 1))])


def _normalizer(pts):
    """Hartley normalization: centroid at origin, mean distance sqrt(dim)."""
    dim = pts.shape[1]
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(dim) / d
    T = np.eye(dim + 1)
    T[:dim, :dim] *= s
    T[:dim, dim] = -s * c
    return T


def dlt(X, x):
    """Direct Linear Transform: 3D-2D correspondences -> 3x4 projection matrix."""
    if len(X) < 6:
        raise ValueError(f"DLT needs at least 6 correspondences, got {len(X)}")

    T3, T2 = _normalizer(X), _normalizer(x)
    Xn = _homogeneous(X) @ T3.T
    xn = _homogeneous(x) @ T2.T

    zero = np.zeros_like(Xn)
    A = np.vstack([
        np.hstack([zero, -Xn, xn[:, 1:2] * Xn]),
        np.hstack([Xn, zero, -xn[:, 0:1] * Xn]),
    ])

    _, _, Vt = np.linalg.svd(A)
    P = np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3

    # fix the arbitrary scale/sign: unit-norm rotation row, points in front
    P = P / np.linalg.norm(P[2, :3])
    if np.mean(_homogeneous(X) @ P[2]) < 0:
        P = -P
    return P


def decompose(P):
    """P = K [R | t] with K upper triangular, positive diagonal, K[2,2] = 1."""
    K, R = scipy.linalg.rq(P[:, :3])
    S = np.diag(np.sign(np.diag(K)))  # force positive focal lengths
    K, R = K @ S, S @ R
    if np.linalg.det(R) < 0:  # negating [R|t] leaves P unchanged
        R, P = -R, -P
    K = K / K[2, 2]
    return K, R, np.linalg.solve(K, P[:, 3])


def reprojection_error(P, X, x):
    proj, _ = project(P, X)
    return float(np.sqrt(np.mean(np.sum((proj - x) ** 2, axis=1))))


def calibrate_camera1():
    """Task 2 core: 3D points from camera 0 + their image-1 pixels -> P, K, R, t."""
    K0, depth, _, _, pts0, pts1 = load_scene()
    h, w = depth.shape

    cols = np.round(pts0[:, 0]).astype(int)
    rows = np.round(pts0[:, 1]).astype(int)
    inside = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    z = np.where(inside, depth[np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)], 0.0)
    good = inside & (z > 0)

    rays = np.linalg.inv(K0) @ _homogeneous(pts0[good]).T
    X = rays.T * z[good].reshape(-1, 1)
    x = pts1[good]

    P = dlt(X, x)
    K, R, t = decompose(P)
    return P, K, R, t, X, x, good


# ------------------------------------------------------------------ tasks ---
def task1():
    K0, depth, image0, _, _, _ = load_scene()
    points = backproject(K0, depth)

    gray = image0 if image0.ndim == 2 else image0[:, :, 0]
    colors = np.repeat(gray.reshape(-1, 1), 3, axis=1)

    valid = depth.reshape(-1) > 0
    print(f"Task 1: {valid.sum()} points out of {valid.size} pixels")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    o3d.visualization.draw_geometries([pcd])


def task2():
    P, K, R, t, X, x, good = calibrate_camera1()
    print(f"Task 2: calibrated on {good.sum()} of {len(good)} matches")
    print("\nP =\n", P)
    print("\nK =\n", K)
    print("\nR =\n", R)
    print("\nt =", t)
    print("\ncamera 1 centre in frame 0 =", -R.T @ t)
    print(f"\nRMS reprojection error = {reprojection_error(P, X, x):.3f} px")


def task3():
    P, _, _, _, X_match, x_match, _ = calibrate_camera1()
    print(f"RMS reprojection error = {reprojection_error(P, X_match, x_match):.3f} px")

    K0, depth, image0, image1, _, _ = load_scene()
    points = backproject(K0, depth)
    valid = depth.reshape(-1) > 0

    # ponytail: nearest-neighbour sampling, no occlusion test - points hidden in
    # view 1 still grab a colour. Add a z-buffer over the projected depths if it matters.
    uv, z = project(P, points)
    h1, w1 = image1.shape[:2]
    cols = np.round(uv[:, 0]).astype(int)
    rows = np.round(uv[:, 1]).astype(int)
    seen = valid & (z > 0) & (cols >= 0) & (cols < w1) & (rows >= 0) & (rows < h1)

    gray = image0 if image0.ndim == 2 else image0[:, :, 0]
    colors = np.repeat(gray.reshape(-1, 1), 3, axis=1)  # grey where unseen
    colors[seen] = image1[rows[seen], cols[seen], :3]
    print(f"Task 3: coloured {seen.sum()} of {valid.sum()} valid points")

    colored0 = (colors.reshape(*gray.shape, 3) * 255).astype(np.uint8)
    o3d.io.write_image("image_0_colored.png", o3d.geometry.Image(colored0))
    print("wrote image_0_colored.png")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[valid])
    pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    o3d.visualization.draw_geometries([pcd])


def selftest():
    """Synthetic scene: DLT + decomposition must recover the camera it was built from."""
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 0, 320.0], [0, 850.0, 240.0], [0, 0, 1.0]])
    R = scipy.linalg.expm(np.cross(np.eye(3), np.array([0.1, -0.2, 0.05])))
    t = np.array([0.3, -0.1, 4.0])
    P_true = K @ np.hstack([R, t.reshape(3, 1)])

    X = rng.uniform(-1, 1, size=(30, 3))
    x, z = project(P_true, X)
    assert np.all(z > 0), "test points must be in front of the camera"

    P = dlt(X, x)
    K_hat, R_hat, t_hat = decompose(P)

    assert reprojection_error(P, X, x) < 1e-6, "DLT did not fit the exact data"
    assert np.allclose(K_hat, K, atol=1e-6), f"K mismatch:\n{K_hat}"
    assert np.allclose(R_hat, R, atol=1e-8), f"R mismatch:\n{R_hat}"
    assert np.allclose(t_hat, t, atol=1e-8), f"t mismatch: {t_hat}"
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["1", "2", "3", "test"], help="which part to run")
    task = parser.parse_args().task
    {"1": task1, "2": task2, "3": task3, "test": selftest}[task]()


if __name__ == "__main__":
    main()
