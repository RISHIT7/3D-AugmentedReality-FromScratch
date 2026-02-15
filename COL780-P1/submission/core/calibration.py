import cv2
import numpy as np
import os
import glob
from core.cv2_functions import CustomCV2
from tqdm import tqdm

def calibrate(source, grid_size, out_path):
    rows, cols = grid_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    img_size = None
    
    if os.path.isdir(source):
        images = sorted(glob.glob(os.path.join(source, '*.[jJ][pP][gG]')) + 
                        glob.glob(os.path.join(source, '*.[pP][nN][gG]')))
        if not images:
            print(f"Error: No images found in {source}")
            return

        for fname in tqdm(images, desc="Processing Images"):
            img = cv2.imread(fname)
            if img is None: continue
            gray = CustomCV2.cvtColor(img, CustomCV2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                objpoints.append(objp)
                corners2 = CustomCV2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                                (CustomCV2.TERM_CRITERIA_EPS + CustomCV2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                img_size = gray.shape[::-1]
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video {source}")
            return
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(frame_count), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret: break
            gray = CustomCV2.cvtColor(frame, CustomCV2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
            if ret:
                objpoints.append(objp)
                corners2 = CustomCV2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                                (CustomCV2.TERM_CRITERIA_EPS + CustomCV2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                imgpoints.append(corners2)
                img_size = gray.shape[::-1]
        cap.release()

    if not objpoints:
        print("Error: No checkerboard patterns found. Check grid size or input quality.")
        return

    print(f"Calibrating with {len(objpoints)} valid frames...")
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    if ret:
        np.save(out_path, {'camera_matrix': mtx, 'dist_coeffs': dist})
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], np.zeros(3), np.zeros(3), mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        print(f"\nCalibration Successful!\nReprojection Error: {mean_error/len(objpoints):.4f}")
        print(f"Saved to {out_path}")
    else:
        print("Error: Calibration calculation failed.")
