import cv2
import argparse
import numpy as np
from core.utils import *
import calibration

from core.evaluate import Evaluator

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    parser.add_argument("--calibration", type=str, help="Path to camera calibration matrix (.npy file).", default=None)
    parser.add_argument("--angle-granularity", type=str, choices=['1deg', '5deg', '10deg'], 
                        default='5deg', help="Angle measurement granularity (default: 5deg)")
    
    # Calibration arguments
    parser.add_argument("--calibrate", action="store_true", help="Run in calibration mode.")
    parser.add_argument("--calibrate_source", type=str, help="Source for calibration (video/images).", default=None)
    parser.add_argument("--grid_size", type=int, nargs=2, default=[8, 11], help="Grid size (rows, cols) for calibration.")
    parser.add_argument("--out", type=str, default="assets/calibration.npy", help="Output path for calibration.")
    parser.add_argument("--no_view", action="store_true", help="Disable visualization.")

    args = parser.parse_args()
    
    if args.calibrate:
        if not args.calibrate_source:
            print("Error: --calibrate_source is required for calibration mode.")
            return
        print(f"Starting calibration with source: {args.calibrate_source}")
        view_scale = 0 if args.no_view else 0.5
        calibration.calibrate(args.calibrate_source, tuple(args.grid_size), args.out, view_scale)
        return
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    evaluator = Evaluator()
    paused = False
    
    # Load template image
    template_img = cv2.imread(args.template) if args.template else None
    
    # Load 3D model
    obj_model = None
    if args.model:
        try:
            obj_model = OBJ(args.model, swapyz=True)
            print(f"✓ Loaded 3D model: {args.model}")
        except Exception as e:
            print(f"⚠ Warning: Could not load 3D model: {e}")
    
    # Load camera calibration
    camera_matrix = None
    if args.calibration:
        try:
            if args.calibration.endswith('.npy'):
                camera_matrix = np.load(args.calibration)
            else:
                camera_matrix = np.load(args.calibration)['camera_matrix']
            print(f"✓ Loaded camera calibration: {args.calibration}")
        except Exception as e:
            print(f"⚠ Warning: Could not load calibration: {e}")
    elif args.model:
        # Auto-load professor's calibration for 3D rendering
        import os
        calib_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "calibration_prof.npy"),
        ]
        for path in calib_paths:
            if os.path.exists(path):
                camera_matrix = np.load(path)
                print(f"✓ Auto-loaded calibration: {path}")
                break

    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
    
    print(f"Processing started. Press 'q' to quit, SPACE to pause, 'r' to resume")
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        # Process frame with all features
        processed_frame, tag_data_list = process_frame(
            frame, 
            template_img=template_img,
            obj_model=obj_model,
            camera_matrix=camera_matrix,
            angle_granularity=args.angle_granularity
        )
        
        evaluator.update(tag_data_list)

        display_frame = evaluator.draw_stats(processed_frame)
        cv2.imshow("Frame", display_frame)

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

    evaluator.print_final_report()
    if len(evaluator.jitter_history) > 0:
        print(f"Average Jitter:   {np.mean(evaluator.jitter_history):.3f} px")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()