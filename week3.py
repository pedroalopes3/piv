import numpy as np
import open3d as o3d
import scipy.io 

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
    v, u = np.mgrid[0:h, 0:w]
    # homogeneous pixels -> rays -> scaled by depth
    pixels = np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
    point_cloud = (A @ pixels).T * depth_matrix.reshape(-1, 1)

    # one RGB row per pixel, same raster order as point_cloud, in [0,1]
    gray = image0 if image0.ndim == 2 else image0[:, :, 0]
    color = np.repeat(gray.reshape(-1, 1) / 255.0, 3, axis=1)

    valid = depth_matrix.reshape(-1) > 0
    point_cloud = point_cloud[valid]
    color = color[valid]

    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])

def task2():

    return None


def task3():

    return None

def main():

    k0_path = "K_0.mat"
    depth_path = "depth_0.mat"
    image0_path = "image_0.png"
    image1_path = "image_1.png"

    # perguntar ao user qual a task que quer executar
    task = input("what task do you want to execute? (1, 2, 3)\n")
    if task == "1":
        task1(depth_path, image0_path, image1_path, k0_path)
    elif task == "2":
        task2()
    elif task == "3":
        task3()
    else:
        print("Invalid task number")
    


if __name__=="__main__":
    main()