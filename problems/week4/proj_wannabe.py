
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
    result = {
        "p": matched_points["matched_points0"][keep].cpu().numpy(),
        "p_prime": matched_points["matched_points1"][keep].cpu().numpy(),
        "stop": matches_both["stop"],
    }
    return result, matched_points


def homogrophy(depth_path, ko_path, matches_path):
 
        depth = scipy.io.loadmat(depth_path)
        ko = scipy.io.loadmat(ko_path)
        matches = scipy.io.loadmat(matches_path)
    
        depth_matrix = depth['depth_image']
        matrix_k0 = ko['K']
        matches_0 = matches['pts0'] # load das pixels coordinates dos matches para camera 0
        matches_1 = matches['pts1'] # load das pixels coordinates dos matches para camera 1
    
        # fazer a multiplicação da matriz K0 com a matriz de profundidade para cada entrada
        print("\n Arroz...\n")
    
        # Point i (X_i, Y_i, Z_i):  A = K_o ^ -1
        # X_i = (A11 * U_i + A12 * V_i + A13) * Z_i
        # Y_i = (A21 * U_i + A22 * V_i + A23) * Z_i
        # Z_i = (A31 * U_i + A32 * V_i + A33) * Z_i
    
        A = np.linalg.inv(matrix_k0)
        points_camera0 = []
        pixels_camera1 = []

        for i, match in enumerate(matches_0):
            u, v = match[0], match[1]
            Z = depth_matrix[round(v), round(u)]
            if Z <= 0:
                continue
    
            X_i = (A[0, 0] * u + A[0, 1] * v + A[0, 2]) * Z
            Y_i = (A[1, 0] * u + A[1, 1] * v + A[1, 2]) * Z
            Z_i = (A[2, 0] * u + A[2, 1] * v + A[2, 2]) * Z
            points_camera0.append([X_i, Y_i, Z_i])
            pixels_camera1.append(matches_1[i])

        # normalizar pixeis e coordenadas com z-score em vez de hartley
        norm_pts_camera0 = []
        norm_matches_camera1 = []

        # matrizes de transformação:
        # T2D - pixel → pixel normalizado 3x3
        # T3D - ponto 3D → ponto 3D normalizado 4x4

        T2D = np.array([[1 / np.std(pixels_camera1, axis=0)[0], 0, -np.mean(pixels_camera1, axis=0)[0] / np.std(pixels_camera1, axis=0)[0]],
                        [0, 1 / np.std(pixels_camera1, axis=0)[1], -np.mean(pixels_camera1, axis=0)[1] / np.std(pixels_camera1, axis=0)[1]], 
                          [0, 0, 1]])

        T3D = np.array([[1 / np.std(points_camera0, axis=0)[0], 0, 0, -np.mean(points_camera0, axis=0)[0] / np.std(points_camera0, axis=0)[0]],
                        [0, 1 / np.std(points_camera0, axis=0)[1], 0, -np.mean(points_camera0, axis=0)[1] / np.std(points_camera0, axis=0)[1]], 
                        [0, 0, 1 / np.std(points_camera0, axis=0)[2], -np.mean(points_camera0, axis=0)[2] / np.std(points_camera0, axis=0)[2]],
                        [0, 0, 0, 1]])

        # normalizar os pontos 3D e os matches 2D usando as matrizes de transformação T3D e T2D
        for i in range(len(points_camera0)):
            X_i, Y_i, Z_i = points_camera0[i]
            point_3D_homogeneous = np.array([X_i, Y_i, Z_i, 1])
            point_3D_normalized = T3D @ point_3D_homogeneous
            norm_pts_camera0.append(point_3D_normalized[:3])  # Only take the first three coordinates

            u, v = pixels_camera1[i][0], pixels_camera1[i][1]
            pixel_homogeneous = np.array([u, v, 1])
            pixel_normalized = T2D @ pixel_homogeneous
            norm_matches_camera1.append(pixel_normalized[:2])  # Only take the first two coordinates

        # Construir matriz M (2*N x 12) com os pontos normalizados e os matches normalizados  
        M = np.zeros((2 * len(pixels_camera1), 12))
        for i in range(len(points_camera0)):
            X_i_norm, Y_i_norm, Z_i_norm = norm_pts_camera0[i]
            u_norm, v_norm = norm_matches_camera1[i]

            vector1 = np.array([X_i_norm, Y_i_norm, Z_i_norm, 1, 0, 0, 0, 0, -u_norm * X_i_norm, -u_norm * Y_i_norm, -u_norm * Z_i_norm, -u_norm])
            vector2 = np.array([0, 0, 0, 0, X_i_norm, Y_i_norm, Z_i_norm, 1, -v_norm * X_i_norm, -v_norm * Y_i_norm, -v_norm * Z_i_norm, -v_norm])

            # construir matrix M (2*N x 12)  N numero de matches
            M[2 * i, :] = vector1
            M[2 * i + 1, :] = vector2
         
        # Usar SVD para resolver M = U * S * V^T 
        U, S, Vt = np.linalg.svd(M)
        
        # P é o último vetor de V (ou a última linha de V^T)
        P_0 = Vt[-1].reshape(3, 4)  

        # Desnormalizar P para obter a matriz de projeção final
        P = np.linalg.inv(T2D) @ P_0 @ T3D

        if np.linalg.det(P[:, :3]) < 0:
            P = -P

        # decomposiçao de P em K, R e t com RQ 
        Q_qr, R_qr = np.linalg.qr(np.linalg.inv(P[:, :3]))

        #    R = (Q')^T
        #    K = (R')^(-1)
        R = Q_qr.T
        K = np.linalg.inv(R_qr)

        print("\nMatriz de rotação da camera 1 (R):\n", R)
        print("\nMatriz de calibração da camera 1 (K):\n", K)

        # forçar K a ter valores positivos na diagonal e normalizar para que K[2, 2] = 1
        scale = K[2, 2]
        K = K / scale
        D = np.diag(np.sign(np.diag(K)))
        K = K @ D
        R = D @ R
        t = np.linalg.inv(K) @ P[:, 3] / scale


        print("\nMatriz de calibração da camera 1 normalizada(K):\n", K)
        print("\nMatriz de rotação da camera 1 normalizada(R):\n", R)
        print("\nVetor de translação da camera 1 (t):\n", t)

        # save K, R and t to a .mat file
        scipy.io.savemat('camera1_calibration.mat', {'K': K, 'R': R, 't': t})


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
        "confidence_score_threshold": 0.5,
    }

    image_pair = load_images(image0_path, image1_path, besta)
    features = feature_extraction(besta, sp_params, image_pair)
    match_results, matched_points = feature_matching(besta, lg_params, features)

    ## see the matches
    viz2d.plot_images(list(image_pair), titles=["Image 0", "Image 1"])
    viz2d.plot_matches(match_results["p"], match_results["p_prime"], color="lime", lw=0.2)
    viz2d.add_text(0, f"{len(match_results['p'])} matches, stopped after {match_results['stop']} layers")
    plt.show()


    #homogrophy(ko_path, matches_path)


if __name__ == "__main__":
    main()