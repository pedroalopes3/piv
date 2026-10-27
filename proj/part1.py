import pathlib
import glob, os
import numpy as np
import scipy.io
from scipy.io import loadmat, savemat

def compute_Hc(path_to_refdir):
    refdir = pathlib.Path(path_to_refdir)
    court = loadmat(refdir / "courtmodel.mat")
    pts_court = court["pts"]                                  # (N, 2) metres
    names = [str(n).strip() for n in np.ravel(court["names"])]
    anchor_file = refdir / "anchor.mat"

    if anchor_file.exists():
        # already clicked before -> just load
        pts_img = loadmat(anchor_file)["kps"]
    else:
        # click the points on the reference image
        import matplotlib.pyplot as plt

        img = plt.imread(refdir / "templateimg.jpg")
        pts_img = np.full((len(pts_court), 2), np.nan)        # NaN = skipped

        fig, (ax_img, ax_court) = plt.subplots(1, 2, figsize=(16, 6),
                                               gridspec_kw={"width_ratios": [3, 1]})
        ax_img.imshow(img)

        for k in range(len(pts_court)):
            ax_court.clear()
            ax_court.scatter(pts_court[:, 0], pts_court[:, 1], c="gray")
            ax_court.scatter(pts_court[k, 0], pts_court[k, 1], c="red", s=150)
            ax_court.set_aspect("equal")
            ax_court.invert_yaxis()
            ax_court.set_title("click this point")
            ax_img.set_title(f"{k}: {names[k]}   (Enter = skip)")
            plt.draw()

            click = plt.ginput(1, timeout=0)
            if len(click) == 0:
                continue
            pts_img[k] = click[0]
            ax_img.plot(click[0][0], click[0][1], "r+", markersize=12)
            ax_img.text(click[0][0] + 5, click[0][1] - 5, str(k), color="yellow")

        plt.close(fig)
        savemat(anchor_file, {"kps": pts_img})

    ok = ~np.isnan(pts_img[:, 0])
    if ok.sum() < 4:
        raise ValueError(f"Only {ok.sum()} points selected, need at least 4")

    H_C = homogrophy(pts_img[ok], pts_court[ok])              # pixels -> metres
    return H_C




def load_features(mat_path):
    kp = loadmat(mat_path)["kp"]    # (2+256, N)
    xy = kp[:2].T                    # (N, 2)
    D = kp[2:].T                     # (N, 256)
    return xy, D

def matching(path_feature_dir, index, xy_ref, D_ref):
    # list all feature files, sorted by number
    files = [f for f in sorted(glob.glob(os.path.join(path_feature_dir, "*.mat"))) if os.path.basename(f) != "templateimg.mat"]

    # features of frame 'index'
    xy, D = load_features(files[index])


    min_sim=0.75
    ratio=0.85

    """
    D1: (N1, 256) descriptors of the frame
    D2: (N2, 256) descriptors of the reference
    xy: (N1, 2) pixel coordinates of the frame
    xy_ref: (N2, 2) pixel coordinates of the reference
    min_sim: minimum similarity threshold (0-1)
    ratio: ratio test threshold (0-1)
    """
    # normalize each descriptor to length 1
    descriptor_frame = D / np.linalg.norm(D, axis=1, keepdims=True)
    descriptor_reference = D_ref / np.linalg.norm(D_ref, axis=1, keepdims=True)

    # similarity between every pair (1 = identical)
    S = descriptor_frame @ descriptor_reference.T                      # (N1, N2)

    # best match in each direction
    best12 = np.argmax(S, axis=1)      # for each point in 1, best in 2
    best21 = np.argmax(S, axis=0)      # for each point in 2, best in 1

    idx1 = []
    idx2 = []
    for i in range(len(descriptor_frame)):
        j = best12[i]

        # mutual check: j must also choose i
        if best21[j] != i:
            continue

        # similarity must be high enough
        if S[i, j] < min_sim:
            continue

        # ratio test: best must be clearly better than the second best
        row = np.sort(S[i])            # ascending
        d_best = np.sqrt(2 - 2 * row[-1])
        d_second = np.sqrt(2 - 2 * row[-2])
        if d_best > ratio * d_second:
            continue

        idx1.append(i)
        idx2.append(j)

    matches_frame = np.array(idx1, dtype=int)
    matches_ref = np.array(idx2, dtype=int)

    pts_frame = xy[matches_frame]      # (M, 2) points in the frame
    pts_ref = xy_ref[matches_ref]    # (M, 2) same points in the reference
    
    return pts_frame, pts_ref


def project(H, pts):
    # pts (N,2) -> apply H -> back to pixels
    ones = np.ones((len(pts), 1))
    p = np.hstack([pts, ones]) @ H.T
    return p[:, :2] / p[:, 2:3]


def collinear(p, min_area=50):
    # True if any 3 of the 4 points are almost on one line
    for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
        v1 = p[b] - p[a]
        v2 = p[c] - p[a]
        area = abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2
        if area < min_area:
            return True
    return False

def ransac(pts_frame, pts_ref, n_iter=1000, thresh=3.0):
    n = len(pts_frame)
    best_inliers = np.zeros(n, dtype=bool)
    if n < 4:
        return best_inliers

    for _ in range(n_iter):
        sample = np.random.choice(n, 4, replace=False)
        if collinear(pts_frame[sample]) or collinear(pts_ref[sample]):
            continue
        H_test = homogrophy(pts_frame[sample], pts_ref[sample])   # hypothesis
        err = np.linalg.norm(project(H_test, pts_frame) - pts_ref, axis=1)
        inliers = err < thresh
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    return best_inliers


def homogrophy(matches_0, matches_1):

        # normalizar pixeis com z-score em vez de hartley
        norm_matches_camera0 = []
        norm_matches_camera1 = []

        # matrizes de transformação:
        # T2D - pixel → pixel normalizado 3x3
        # T3D - ponto 3D → ponto 3D normalizado 4x4

        pixels_camera0 = np.array(matches_0)
        pixels_camera1 = np.array(matches_1)

        T2D0 = np.array([[1 / np.std(pixels_camera0, axis=0)[0], 0, -np.mean(pixels_camera0, axis=0)[0] / np.std(pixels_camera0, axis=0)[0]],
                        [0, 1 / np.std(pixels_camera0, axis=0)[1], -np.mean(pixels_camera0, axis=0)[1] / np.std(pixels_camera0, axis=0)[1]],
                          [0, 0, 1]])

        T2D1 = np.array([[1 / np.std(pixels_camera1, axis=0)[0], 0, -np.mean(pixels_camera1, axis=0)[0] / np.std(pixels_camera1, axis=0)[0]],
                        [0, 1 / np.std(pixels_camera1, axis=0)[1], -np.mean(pixels_camera1, axis=0)[1] / np.std(pixels_camera1, axis=0)[1]],
                        [0, 0, 1]])

        # normalizar os matches 2D usando as matrizes de transformação T2D
        for i in range(len(pixels_camera0)):
            u, v = pixels_camera0[i][0], pixels_camera0[i][1]
            pixel_homogeneous = np.array([u, v, 1])
            pixel_normalized = T2D0 @ pixel_homogeneous
            norm_matches_camera0.append(pixel_normalized[:2])  # Only take the first two coordinates

        for i in range(len(pixels_camera1)):
            u, v = pixels_camera1[i][0], pixels_camera1[i][1]
            pixel_homogeneous = np.array([u, v, 1])
            pixel_normalized = T2D1 @ pixel_homogeneous
            norm_matches_camera1.append(pixel_normalized[:2])  # Only take the first two coordinates

        # Construir matriz M (2*N x 9) com os matches normalizados
        M = np.zeros((2 * len(norm_matches_camera0), 9))
        for i in range(len(norm_matches_camera0)):
            u0_norm, v0_norm = norm_matches_camera0[i]
            u1_norm, v1_norm = norm_matches_camera1[i]
            vector1 = np.array([u0_norm, v0_norm, 1, 0, 0, 0, -u1_norm * u0_norm, -u1_norm * v0_norm, -u1_norm ])
            vector2 = np.array([0, 0, 0, u0_norm, v0_norm, 1, -v1_norm * u0_norm, -v1_norm * v0_norm, -v1_norm])


            # construir matrix M (2*N x 9)  N numero de matches
            M[2 * i, :] = vector1
            M[2 * i + 1, :] = vector2

        # Usar SVD para resolver M = U * S * V^T
        U, S, Vt = np.linalg.svd(M)

        # H_0 é o último vetor de V (ou a última linha de V^T)
        H_0 = Vt[-1].reshape(3, 3)

        # Desnormalizar H_0 para obter a matriz de projeção final
        H = np.linalg.inv(T2D1) @ H_0 @ T2D0

        #print("\nHomografia (H):\n", H)

        # normalizar para que h[2, 2] = 1
        scale = H[2, 2]
        H = H / scale

        #print("\nHomografia normalizada (H):\n", H)

        return H




def part1(path_to_refdir, path_images_dir, path_feature_dir, path_output_dir):

    out_dir = pathlib.Path(path_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in sorted(pathlib.Path(path_feature_dir).glob("*.mat")) if f.name != "templateimg.mat"]
    xy_ref, D_ref = load_features(pathlib.Path(path_feature_dir) / "templateimg.mat")

    H_C = compute_Hc(path_to_refdir)
    
    for i, f in enumerate(files):

        pts_frame, pts_ref = matching(path_feature_dir, i, xy_ref, D_ref)
        inliers = ransac(pts_frame, pts_ref)                       # outlier removal
        if inliers.sum() >= 15:
            H = homogrophy(pts_frame[inliers], pts_ref[inliers])  
            print("\nHomografia (H):\n", H) 
        else:
            H = np.full((3, 3), np.nan)                                                 
        
        Hc = H_C @ H

        print("\nHomografia mapping court (Hc):\n", Hc)  
        number = f.stem.split("_")[-1]
        savemat(out_dir / f"homography_{number}.mat", {"H": H, "Hc": Hc})