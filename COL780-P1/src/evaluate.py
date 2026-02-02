import cv2
import numpy as np
import time

class Evaluator:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        self.jitter_history = []
        self.prev_corners = None
        
    def update(self, tags):
        self.frame_count += 1
        if len(tags) > 0:
            self.detection_count += 1
            cur = tags[0]['corners']
            if self.prev_corners is not None:
                self.jitter_history.append(np.mean(np.linalg.norm(cur - self.prev_corners, axis=1)))
            self.prev_corners = cur.copy()
        else:
            self.prev_corners = None

    def draw_stats(self, frame):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
        
        cv2.rectangle(frame, (5, 5), (200, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 25), 0, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"DET: {rate:.1f}%", (15, 50), 0, 0.6, (0,255,0), 1)
        return frame

    def print_final_report(self):
        print(f"\nReport: {self.detection_count}/{self.frame_count} frames detected.")
        if self.jitter_history: print(f"Avg Jitter: {np.mean(self.jitter_history):.4f} px")