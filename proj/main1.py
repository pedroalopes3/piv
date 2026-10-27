import pathlib
import sys
import cv2
import numpy as np
import torch
import scipy.io
import matplotlib.pyplot as plt
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from part1 import part1


def read_image(image_path, device): 
    return load_image(image_path).to(device)

def feature_extraction(device,sp_params, image, image_name, feature_dir, superpoint=None):

    features = superpoint.extract(image)

    kpts = features["keypoints"][0].cpu().numpy()      # (N, 2)
    desc = features["descriptors"][0].cpu().numpy()    # (N, 256)
    kp = np.vstack([kpts.T, desc.T])                   # (258, N)
    scipy.io.savemat(pathlib.Path(feature_dir) / image_name.replace(".jpg", ".mat"), {"kp": kp})
    

    ## extract tambem converte para grayscale e dá resize internamente!
    #### structure of features
    # features0["keypoints"]        (1, N, 2)   pixel coordinates (x, y)
    # features0["keypoint_scores"]  (1, N)
    # features0["descriptors"]      (1, N, 256)

    return None


# def feature_matching(device, lg_params, features):

#     lightglue = LightGlue(features="superpoint").eval().to(device)

#     with torch.no_grad(): matches_both = lightglue({"image0": features["features0"], "image1": features["features1"]})


#     features["features0"], features["features1"], matches_both = [rbd(x) for x in (features["features0"], features["features1"], matches_both)]

#     matches = matches_both["matches"]    # (K, 2) indices into the two keypoint lists
#     scores = matches_both["scores"]      # (K,) confidence per match

#     matched_points = {}
#     matched_points["matched_points0"] = features["features0"]["keypoints"][matches[:, 0]]   # (K, 2) in image A
#     matched_points["matched_points1"] = features["features1"]["keypoints"][matches[:, 1]]   # (K, 2) in image B

#     ## Filter matches by comfidence score
#     keep = scores > lg_params["confidence_score_threshold"]
#     matched_points["matches_cleanned0"] = matched_points["matched_points0"][keep].cpu().numpy()
#     matched_points["matches_cleanned1"] = matched_points["matched_points1"][keep].cpu().numpy()
#     matched_points["stop"] = matches_both["stop"]

#     return  matched_points




def main():
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)

    # cenas do stor:
    #   flowchart LR
    #   A[Frame i] --> B[Detect<br/>features]
    #   B --> C[Match to<br/>reference]
    #   C --> D[RANSAC]
    #   D --> E[DLT + LSE<br/>refinement]
    #   E --> F[H to reference<br/>+ court model]
    #   F --> B
    #
    # • path_to_refdir — directory with the reference data: templateimg.jpg, the reference image (normally the first frame of the sequence), and courtmodel.mat, 
    #                    a dictionary {"pts": pts, "names": names} where pts is an (N,2) numpy array with the coordinates of the court landmarks in metres on 
    #                    the court plane, and names labels them.
    # • path_images_dir — directory with the image sequence, jpg files named somename_NNNN.jpg where NNNN is the sequential number of the image.
    # • path_feature_dir — directory where image features are stored in .mat format, somename_NNNN.mat.
    # • path_output_dir — directory for the output results.


    besta = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Superpoint parameters
    sp_params = {
        "max_num_keypoints": 2048,
    }

    superpoint = SuperPoint(max_num_keypoints=sp_params["max_num_keypoints"]).eval().to(besta)


    refdir, images_dir, feature_dir, output_dir = sys.argv[1:5]

    reference_image_path = pathlib.Path(refdir) / "templateimg.jpg"

    image_ref = read_image(reference_image_path, besta)
    feature_extraction(besta,sp_params,image_ref,"templateimg.jpg", feature_dir, superpoint) 

    for image_path in sorted(pathlib.Path(images_dir).glob("*.jpg")):
        image = read_image(image_path, besta)
        feature_extraction(besta,sp_params,image,image_path.name, feature_dir, superpoint)
        
    part1(refdir, images_dir, feature_dir, output_dir)


if __name__ == "__main__":
    main()
