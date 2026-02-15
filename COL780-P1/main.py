import cv2
import argparse
import numpy as np
from core.utils import *
import core.calibration

def parser_tasks(args):
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    
    template_img = cv2.imread(args.template) if args.template else None
    
    obj_model = None
    if args.model:
        try:
            obj_model = OBJ(args.model, swapyz=True, swapxy=True)
        except Exception as e:
            print(f"Could not load 3D model: {e}")
    
    camera_matrix = None
    if args.calibration_file:
        try:
            if args.calibration_file.endswith('.npy'):
                camera_matrix = np.load(args.calibration_file)
            else:
                camera_matrix = np.load(args.calibration_file)['camera_matrix']
        except Exception as e:
            print(f"Could not load calibration: {e}")
    elif args.model:
        camera_matrix = np.eye(3, 3)
        print(f"Using Identity as Camera Matrix")

    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
    
    return cap, template_img, obj_model, camera_matrix

def calibrator(args):
    if not args.calibration_source:
        print("Error: --calibration_source is required for calibration mode.")
        return
    
    calibration.calibrate(args.calibration_source, tuple(args.grid_size), args.out)
    return

def main():
    parser = argparse.ArgumentParser(description="AR Tag detection")

    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    parser.add_argument("--calibration_file", type=str, help="Path to camera calibration matrix (.npy file).", default=None)
    parser.add_argument("--angle-granularity", type=str, choices=['1deg', '5deg', '10deg'], 
                        default='5deg', help="Angle measurement granularity (default: 5deg)")
    
    parser.add_argument("--calibrate", action="store_true", help="Calibrate camera")
    parser.add_argument("--calibrate_source", type=str, help="Source for calibration (video/images)", required=False)
    parser.add_argument("--grid_size", type=int, nargs=2, default=[8, 11], help="Grid size (rows, cols) for calibration.")
    parser.add_argument("--out", type=str, default="assets/calibration.npy", help="Output path for calibration.")

    args = parser.parse_args()

    if args.calibrate:
        calibrator(args)
        return
    else:
        cap, template_img, obj_model, camera_matrix = parser_tasks(args)
    
    print(f"Processing started. Press 'q' to quit, SPACE to pause, 'r' to resume")
    paused = False
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, tag_data_list = process_frame(
                frame,
                template_img=template_img,
                obj_model=obj_model,
                camera_matrix=camera_matrix,
                angle_granularity=args.angle_granularity
            )

            cv2.imshow("Frame", processed_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                paused = True
                cv2.waitKey(0)
            if key == ord('r'):
                paused = False
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()