import numpy as np
import cv2 as cv
from pathlib import Path
import sys
import time
import glob

def extract_frames( video_path: str) -> int:
    # Extract frames from the video and store them in a dir "frames".
    
    step = 10
    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    out_path = Path("frames")
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % step == 0:
                frame_filename = out_path / f"frame_{saved_count:06d}.jpg"
                cv.imwrite(str(frame_filename), frame)
                saved_count += 1

            frame_count += 1
    finally:
        cap.release()

    return saved_count

def main():
    """
    Main function to run the camera calibration process.
    """
    video_path = 'video_s21_pedro.mp4'
    points_path = Path("points")
    points_path.mkdir(parents=True, exist_ok=True)

    frame_count = extract_frames(video_path)

    print("Extracted", frame_count, "frames from the video.")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    Cols = 6
    Rows = 4
    inner_corners = (Cols-1, Rows-1)
    objp = np.zeros((inner_corners[0] * inner_corners[1], 3), np.float32)

    objp[:,:2] = np.mgrid[0:inner_corners[0],0:inner_corners[1]].T.reshape(-1,2)
 
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
 
    frames = glob.glob('frames/*.jpg')
 
    for frame in frames:
        img = cv.imread(frame)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, inner_corners, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            refinement_window_size = (5, 5)  # para samsung s21+ pedro  
            corners2 = cv.cornerSubPix(gray,corners, refinement_window_size, (-1,-1), criteria)
            imgpoints.append(corners2)
    
            # Draw and display the corners
            cv.drawChessboardCorners(img, inner_corners, corners2, ret)
            #cv.imshow('img', img)
            cv.waitKey(500)
            frame_filename = points_path / f"frame_{frame_count:06d}.jpg"
            cv.imwrite(str(frame_filename), img)
            frame_count += 1
            
    
    #cv.destroyAllWindows() 

    # Calibrate the camera 

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Reprojection error

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)) )

if __name__=="__main__":
    main()