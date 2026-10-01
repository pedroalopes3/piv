import numpy as np
import open3d as o3d
import scipy.io 
from PIL import Image

def task1(depth_path, image0_path, image1_path, ko_path):

    # Load the file into a dictionary
    depth = scipy.io.loadmat(depth_path)
    ko = scipy.io.loadmat(ko_path)
    #image0 = scipy.io.loadmat(image0_path)
    # load png grayscale image using open3d
    image0 = o3d.io.read_image(image0_path)
    image0 = np.asarray(image0)

    depth_matrix = depth['depth_image']
    matrix_k0 = ko['K']

    # Access a specific variable (returns a NumPy array
    #print(depth_matrix)
    print("\nmatrix_k0:\n", matrix_k0)

    # fazer a mltiplicação da matriz K0 com a matriz de profundidade para cada entrada
    print("\nComputing the 3d scene...\n")

    # Point i (X_i, Y_i, Z_i):  A = K_o ^ -1
    # X_i = (A11 * U_i + A12 * V_i + A13) * Z_i
    # Y_i = (A21 * U_i + A22 * V_i + A23) * Z_i
    # Z_i = (A31 * U_i + A32 * V_i + A33) * Z_i

    A = np.linalg.inv(matrix_k0)

    h, w = depth_matrix.shape
    gray = image0 if image0.ndim == 2 else image0[:, :, 0]

    point_cloud = []
    color = []
    for v in range(h):
        for u in range(w):
            Z = depth_matrix[v, u]
            if Z <= 0:
                continue

            X_i = (A[0, 0] * u + A[0, 1] * v + A[0, 2]) * Z
            Y_i = (A[1, 0] * u + A[1, 1] * v + A[1, 2]) * Z
            Z_i = (A[2, 0] * u + A[2, 1] * v + A[2, 2]) * Z
            point_cloud.append([X_i, Y_i, Z_i])

            # same pixel as the point, gray in [0,1] repeated as RGB
            c = gray[v, u] / 255.0
            color.append([c, c, c])


    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    pcd.colors = o3d.utility.Vector3dVector(np.array(color))
    o3d.visualization.draw_geometries([pcd])

def task2(depth_path, ko_path, matches_path, image0_path, image1_path):

        # sim eu sei muito bem o que é uma funçao mas epá nao me apetece
        depth = scipy.io.loadmat(depth_path)
        ko = scipy.io.loadmat(ko_path)
        matches = scipy.io.loadmat(matches_path)
    
        depth_matrix = depth['depth_image']
        matrix_k0 = ko['K']
        matches_0 = matches['pts0'] # load das pixels coordinates dos matches para camera 0
        matches_1 = matches['pts1'] # load das pixels coordinates dos matches para camera 1
    
        # fazer a mltiplicação da matriz K0 com a matriz de profundidade para cada entrada
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

def task3(depth_path, ko_path, matches_path, image0_path, image1_path, camera1_calibration_path):

    # Load the file into a dictionary
    depth = scipy.io.loadmat(depth_path)
    ko = scipy.io.loadmat(ko_path)
    camera1_calibration = scipy.io.loadmat(camera1_calibration_path)
    # load png image using open3d
    image0 = Image.open("image_0.png").convert("RGB")
    image1 = Image.open("image_1.png").convert("RGB")
    image0_pixels = image0.load()
    image1_pixels = image1.load()

    depth_matrix = depth['depth_image']
    matrix_k0 = ko['K']
    matrix_k1 = camera1_calibration['K']
    R1 = camera1_calibration['R']
    t1 = camera1_calibration['t']

    # calcular a matrix de projeçao de camera 1 P1 = K1 * [R1 | t1]
    P1 = matrix_k1 @ np.hstack((R1, t1.reshape(-1, 1))) 


    # Access a specific variable (returns a NumPy array
    #print(depth_matrix)
    print("\nmatrix_k0:\n", matrix_k0)

    # fazer a mltiplicação da matriz K0 com a matriz de profundidade para cada entrada
    print("\nComputing the 3d scene...\n")

    # Point i (X_i, Y_i, Z_i):  A = K_o ^ -1
    # X_i = (A11 * U_i + A12 * V_i + A13) * Z_i
    # Y_i = (A21 * U_i + A22 * V_i + A23) * Z_i
    # Z_i = (A31 * U_i + A32 * V_i + A33) * Z_i

    A = np.linalg.inv(matrix_k0)

    h, w = depth_matrix.shape


    for v0 in range(h):
        for u0 in range(w):
            Z = depth_matrix[v0, u0]
            if Z <= 0:
                continue

            # arranjar o ponto da imagem 0 pixel para coordenadas mundo
            X_i = (A[0, 0] * u0 + A[0, 1] * v0 + A[0, 2]) * Z
            Y_i = (A[1, 0] * u0 + A[1, 1] * v0 + A[1, 2]) * Z
            Z_i = (A[2, 0] * u0 + A[2, 1] * v0 + A[2, 2]) * Z
            v = np.array([X_i, Y_i, Z_i,1])

            # Calcular os pixeis homogeneos na imagem 1 com a matrix de projeçao
            homo_point = P1 @ np.transpose(v)
            u1 = int(round(homo_point[0] / homo_point[2]))
            v1 = int(round(homo_point[1] / homo_point[2])) 
            # did a funny ahahaha
            
            # ver se o ponto existe sequer na imagem, se sim transferir a cor do pixel para a imagem 0
            if 0 <= u1 < 415  and 0 <= v1 < 269 and homo_point[2] > 0:
                c1_r,c1_g, c1_b = image1_pixels[u1,v1]
                image0_pixels[u0, v0] = (c1_r, c1_g, c1_b)

    image0.save("image1_colorized.png")

    return  None

def main():

    k0_path = "K_0.mat"
    depth_path = "depth_0.mat"
    matches_path = "matches.mat"
    image0_path = "image_0.png"
    image1_path = "image_1.png"
    camera1_calibration_path = "camera1_calibration.mat"

    # depth_image0 = "MA_depth_0.mat"
    # depth_image1 = "MA_depth_1.mat"
    # image0_path = "MA_image_0.png"
    # image1_path = "MA_image_1.png"
    # ma_k0_path = "MA_K_0.mat"
    # matches_ma_path = ".mat"



    continue_running = True
    while continue_running:
        # perguntar ao user qual a task que quer executar
        task = input("\n----------------------------------------------------\nwhat task do you want to execute? (1, 2, 3)\n")
        if task == "1":
            task1(depth_path, image0_path, image1_path, k0_path)
        elif task == "2":
            task2(depth_path, k0_path, matches_path, image0_path, image1_path)
        elif task == "3":
            task3(depth_path, k0_path, matches_path, image0_path, image1_path, camera1_calibration_path)
        else:
            continue_running = False
            print("Invalid task number")
    


if __name__=="__main__":
    main()