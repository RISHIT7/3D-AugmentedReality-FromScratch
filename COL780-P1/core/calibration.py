#!/usr/bin/env python3
"""
Camera Calibration Script for AR Tag Assignment
Task 3: 3D Augmented Reality

This script calibrates your webcam to obtain intrinsic parameters needed for 3D projection.
Uses a checkerboard pattern for calibration.

Author: Assignment Solution
Date: February 14, 2026
"""

import cv2
import numpy as np
import glob
import argparse
from pathlib import Path


def calibrate_camera_interactive(checkerboard_size=(9, 6), square_size=1.0, num_samples=20):
    """
    Interactive camera calibration using live webcam feed.
    
    Args:
        checkerboard_size: Internal corners of checkerboard (cols, rows)
        square_size: Size of checkerboard square in your chosen units (e.g., cm)
        num_samples: Number of different checkerboard poses to capture
        
    Returns:
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    print("="*70)
    print("INTERACTIVE CAMERA CALIBRATION")
    print("="*70)
    print(f"Checkerboard pattern: {checkerboard_size[0]}x{checkerboard_size[1]} internal corners")
    print(f"Square size: {square_size} units")
    print(f"Samples needed: {num_samples}")
    print()
    print("Instructions:")
    print("1. Print a checkerboard pattern (search 'checkerboard calibration pattern')")
    print("2. Hold the checkerboard in front of the camera")
    print("3. Press SPACE when corners are detected to capture a sample")
    print("4. Move checkerboard to different positions/angles for each sample")
    print("5. Press 'q' to finish calibration (need at least 10 samples)")
    print("="*70)
    
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return None, None, None, None
    
    # Get camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nCamera resolution: {width}x{height}")
    
    samples_captured = 0
    
    try:
        while samples_captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(
                gray, 
                checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Draw and display
            display_frame = frame.copy()
            
            if ret_corners:
                # Refine corner locations
                corners_refined = cv2.cornerSubPix(
                    gray, 
                    corners, 
                    (11, 11), 
                    (-1, -1), 
                    criteria
                )
                
                # Draw corners
                cv2.drawChessboardCorners(display_frame, checkerboard_size, corners_refined, ret_corners)
                
                # Add instructions
                cv2.putText(
                    display_frame,
                    f"Corners detected! Press SPACE to capture ({samples_captured}/{num_samples})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display_frame,
                    f"No corners detected. Position checkerboard ({samples_captured}/{num_samples})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            
            cv2.imshow('Camera Calibration', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and ret_corners:
                # Capture sample
                objpoints.append(objp)
                imgpoints.append(corners_refined)
                samples_captured += 1
                print(f"✓ Sample {samples_captured}/{num_samples} captured")
                
                # Visual feedback
                cv2.waitKey(200)
                
            elif key == ord('q'):
                if samples_captured >= 10:
                    print(f"\nFinishing with {samples_captured} samples")
                    break
                else:
                    print(f"\nNeed at least 10 samples, you have {samples_captured}")
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    if samples_captured < 10:
        print("Error: Not enough samples for calibration")
        return None, None, None, None
    
    print("\nPerforming calibration...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )
    
    if not ret:
        print("Error: Calibration failed")
        return None, None, None, None
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    print(f"Total samples used: {samples_captured}")
    print(f"Reprojection error: {mean_error:.4f} pixels")
    print(f"\nCamera Matrix (K):")
    print(camera_matrix)
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs)
    print("="*70)
    
    return camera_matrix, dist_coeffs, rvecs, tvecs


def calibrate_camera_from_images(image_dir, checkerboard_size=(9, 6), square_size=1.0):
    """
    Calibrate camera from saved checkerboard images.
    
    Args:
        image_dir: Directory containing calibration images
        checkerboard_size: Internal corners of checkerboard (cols, rows)
        square_size: Size of checkerboard square in your chosen units
        
    Returns:
        camera_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: Distortion coefficients
    """
    print("="*70)
    print("CAMERA CALIBRATION FROM IMAGES")
    print("="*70)
    
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store points
    objpoints = []
    imgpoints = []
    
    # Load images
    images = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")
    
    if not images:
        print(f"Error: No images found in {image_dir}")
        return None, None
    
    print(f"Found {len(images)} images")
    
    successful = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret:
            objpoints.append(objp)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners_refined)
            successful += 1
            print(f"✓ {fname}")
        else:
            print(f"✗ {fname} - corners not found")
    
    print(f"\nSuccessfully processed: {successful}/{len(images)} images")
    
    if successful < 10:
        print("Error: Need at least 10 successful images")
        return None, None
    
    # Calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None
    )
    
    if not ret:
        print("Error: Calibration failed")
        return None, None
    
    # Calculate error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)
    
    print("\n" + "="*70)
    print("CALIBRATION RESULTS")
    print("="*70)
    print(f"Reprojection error: {mean_error:.4f} pixels")
    print(f"\nCamera Matrix (K):")
    print(camera_matrix)
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs)
    print("="*70)
    
    return camera_matrix, dist_coeffs


def save_calibration(camera_matrix, dist_coeffs, output_file="camera_calibration.npz"):
    """Save calibration results to file."""
    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    print(f"\n✓ Calibration saved to: {output_file}")
    
    # Also save just the camera matrix for easy loading
    matrix_file = output_file.replace('.npz', '_matrix.npy')
    np.save(matrix_file, camera_matrix)
    print(f"✓ Camera matrix saved to: {matrix_file}")


def load_calibration(calibration_file="camera_calibration.npz"):
    """Load calibration from file."""
    data = np.load(calibration_file)
    return data['camera_matrix'], data['dist_coeffs']


def test_calibration(camera_matrix, dist_coeffs):
    """
    Test calibration by showing undistorted live feed.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    """
    print("\n" + "="*70)
    print("TESTING CALIBRATION")
    print("="*70)
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        1,
        (width, height)
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Undistort
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
            
            # Crop the image
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            
            # Show side-by-side
            comparison = np.hstack([
                cv2.resize(frame, (width//2, height//2)),
                cv2.resize(undistorted, (width//2, height//2))
            ])
            
            cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comparison, "Undistorted", (width//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Calibration Test', comparison)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Camera Calibration for AR Tag Assignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive calibration with webcam
  python camera_calibration.py --interactive
  
  # Calibrate from saved images
  python camera_calibration.py --images ./calibration_images
  
  # Test existing calibration
  python camera_calibration.py --test camera_calibration.npz
  
  # Custom checkerboard size
  python camera_calibration.py --interactive --board 7 5 --square 2.5
        """
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive calibration using live webcam'
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Directory containing calibration images'
    )
    parser.add_argument(
        '--test',
        type=str,
        help='Test existing calibration file'
    )
    parser.add_argument(
        '--board',
        type=int,
        nargs=2,
        default=[9, 6],
        metavar=('COLS', 'ROWS'),
        help='Checkerboard internal corners (default: 9 6)'
    )
    parser.add_argument(
        '--square',
        type=float,
        default=1.0,
        help='Size of checkerboard square in units (default: 1.0)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples for interactive calibration (default: 20)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='camera_calibration.npz',
        help='Output file for calibration (default: camera_calibration.npz)'
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Test existing calibration
        try:
            camera_matrix, dist_coeffs = load_calibration(args.test)
            print(f"✓ Loaded calibration from: {args.test}")
            test_calibration(camera_matrix, dist_coeffs)
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return
    
    camera_matrix = None
    dist_coeffs = None
    
    if args.interactive:
        # Interactive calibration
        camera_matrix, dist_coeffs, _, _ = calibrate_camera_interactive(
            checkerboard_size=tuple(args.board),
            square_size=args.square,
            num_samples=args.samples
        )
    elif args.images:
        # Calibrate from images
        camera_matrix, dist_coeffs = calibrate_camera_from_images(
            image_dir=args.images,
            checkerboard_size=tuple(args.board),
            square_size=args.square
        )
    else:
        parser.print_help()
        return
    
    if camera_matrix is not None and dist_coeffs is not None:
        save_calibration(camera_matrix, dist_coeffs, args.output)
        
        # Ask if user wants to test
        response = input("\nTest calibration? (y/n): ")
        if response.lower() == 'y':
            test_calibration(camera_matrix, dist_coeffs)
    else:
        print("\n✗ Calibration failed")


if __name__ == '__main__':
    main()