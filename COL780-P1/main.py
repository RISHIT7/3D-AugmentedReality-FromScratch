import cv2 # OpenCV - Only for video capture and display
import argparse
from core.utils import * # Define custom CV functions in utils.py

from core.evaluate import Evaluator

def main():
    parser = argparse.ArgumentParser(description="AR Tag Detection and Overlay")
    parser.add_argument("--video", type=str, help="Path to video file. If not provided, webcam (0) is used.", default=None)
    parser.add_argument("--template", type=str, help="Path to template image for overlay.", default=None)
    parser.add_argument("--model", type=str, help="Path to .obj model for 3D projection.", default=None)
    
    args = parser.parse_args()
    
    video_source = args.video if args.video else 0
    cap = cv2.VideoCapture(video_source)
    evaluator = Evaluator()
    paused = False

    if not cap.isOpened():
        print(f"Error opening source: {video_source}")
        return
        
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
        processed_frame, tag_data_list = process_frame(frame)
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
    print(f"Average Jitter:   {np.mean(evaluator.jitter_history):.3f} px")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()
