
import numpy as np
import open3d as o3d
import scipy.io        
import torch
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint, viz2d
from lightglue.utils import load_image, rbd


def load_images(image1_path, image2_path, device):
    image1 = load_image(image1_path).to(device)
    image2 = load_image(image2_path).to(device)
    return image1, image2

def feature_extraction(device,sp_params, image_pair):

    superpoint = SuperPoint(max_num_keypoints=sp_params["max_num_keypoints"]).eval().to(device)

    features = {}
    features["features0"] = superpoint.extract(image_pair[0])
    features["features1"] = superpoint.extract(image_pair[1])

    ## extract tambem converte para grayscale e dá resize internamente!
    #### structure of features
    # features0["keypoints"]        (1, N, 2)   pixel coordinates (x, y)
    # features0["keypoint_scores"]  (1, N)
    # features0["descriptors"]      (1, N, 256)
    
    return features


def feature_matching(device, lg_params, features):     

    lightglue = LightGlue(features="superpoint").eval().to(device)

    with torch.no_grad(): matches_both = lightglue({"image0": features["features0"], "image1": features["features1"]})


    features["features0"], features["features1"], matches_both = [rbd(x) for x in (features["features0"], features["features1"], matches_both)]

    matches = matches_both["matches"]    # (K, 2) indices into the two keypoint lists
    scores = matches_both["scores"]      # (K,) confidence per match

    matched_points = {}
    matched_points["matched_points0"] = features["features0"]["keypoints"][matches[:, 0]]   # (K, 2) in image A
    matched_points["matched_points1"] = features["features1"]["keypoints"][matches[:, 1]]   # (K, 2) in image B

    ## Filter matches by comfidence score
    keep = scores > lg_params["confidence_score_threshold"]
    matched_points["matches_cleanned0"] = matched_points["matched_points0"][keep].cpu().numpy()
    matched_points["matches_cleanned1"] = matched_points["matched_points1"][keep].cpu().numpy()
    matched_points["stop"] = matches_both["stop"]
    
    return  matched_points


def homogrophy(matches):
 
        matches_0 = matches['matches_cleanned0'] # load das pixels coordinates dos matches para camera 0
        matches_1 = matches['matches_cleanned1'] # load das pixels coordinates dos matches para camera 1
    

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

        print("\nHomografia (H):\n", H)

        # normalizar para que h[2, 2] = 1
        scale = H[2, 2]
        H = H / scale

        print("\nHomografia normalizada (H):\n", H)

        return H

def warp_homography(img, H, out_w, out_h):
    """Warp img onto an (out_h, out_w) canvas. H maps img pixels -> canvas pixels."""
    if img.ndim == 2:
        img = img[..., None]
    h, w, _ = img.shape

    # 1. every output pixel, in homogeneous coordinates (3 x N)
    xs, ys = np.meshgrid(np.arange(out_w), np.arange(out_h))
    dst = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])

    # 2. where it comes from in the source image (inverse mapping)
    src = np.linalg.inv(H) @ dst
    u = src[0] / src[2]
    v = src[1] / src[2]

    # 3. bilinear interpolation between the 4 neighbouring source pixels
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]

    valid = (u0 >= 0) & (v0 >= 0) & (u0 < w - 1) & (v0 < h - 1)
    u0 = np.clip(u0, 0, w - 2)
    v0 = np.clip(v0, 0, h - 2)

    out = ((1 - du) * (1 - dv) * img[v0,     u0    ] +
           du       * (1 - dv) * img[v0,     u0 + 1] +
           (1 - du) * dv       * img[v0 + 1, u0    ] +
           du       * dv       * img[v0 + 1, u0 + 1])

    out[~valid] = 0                       # pixels that fall outside the source -> black
    return out.reshape(out_h, out_w, -1).squeeze()


def plane_flat_projection(image0_path):

    image0 = plt.imread(image0_path)
    image0 = image0[..., :3] # remover alpha channel se existir
    if image0.dtype == np.uint8:
        image0 = image0 / 255.0  # jpg vem em uint8, interpolação precisa de float em [0, 1]

    # por agora fazer seleçao manual dos pontos do plano
    plt.figure()
    plt.imshow(image0)
    plt.title("click TL, TR, BR, BL")
    corners = np.array(plt.ginput(4, timeout=0))

    # ordenar cliques para TL, TR, BR, BL (ordem errada -> quadrilátero cruzado -> warp mostra o exterior do plano)
    d = corners - corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]   # sentido horário no ecrã (y para baixo)
    corners = np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)  # começar no TL (menor x+y)

    w = 500
    h = 500
    plane_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]])  # TL, TR, BR, BL

    # homografia imagem -> plano retificado 
    H_rect = homogrophy({"matches_cleanned0": corners, "matches_cleanned1": plane_corners})
    flat = warp_homography(image0, H_rect, w, h)

    plt.figure()
    plt.imshow(flat)
    plt.title("plano retificado")
    plt.show()
    return flat


def main():

    ko_path = 'K.mat'
    image0_path = 'parede1.jpg'
    image1_path = 'parede2.jpg'

    besta = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Superpoint parameters
    sp_params = {
        "max_num_keypoints": 2048,
    }

    # Lightglue parameters
    lg_params = {
        "confidence_score_threshold": 0.98,
    }

    image_pair = load_images(image0_path, image1_path, besta)

    print(f"Matching params: {sp_params} \n {lg_params}")

    features = feature_extraction(besta, sp_params, image_pair)
    matches = feature_matching(besta, lg_params, features)

    ## see the matches
    viz2d.plot_images([im.cpu() for im in image_pair], titles=["Image 0", "Image 1"])
    viz2d.plot_matches(matches["matches_cleanned0"], matches["matches_cleanned1"], color="lime", lw=0.2)
    viz2d.add_text(0, f"{len(matches['matches_cleanned0'])} matches, stopped after {matches['stop']} layers")
    plt.show(block=False)
    plt.pause(5)   



    H = homogrophy(matches)

    plane_flat_projection(image0_path)


if __name__ == "__main__":
    main()