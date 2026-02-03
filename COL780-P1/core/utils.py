import cv2
from core.cv2_functions import CustomCV2
import numpy as np
import math
from time import time

class TemporalFilter:
    def __init__(self, persistence=4):
        self.persistence = persistence
        self.history = {}

    def process(self, current_tags):
        verified_tags = []
        current_ids = [t["id"] for t in current_tags]

        for tag in current_tags:
            tid = tag["id"]
            self.history[tid] = min(self.history.get(tid, 0) + 1, self.persistence + 5)
            
            if self.history[tid] >= self.persistence:
                verified_tags.append(tag)

        for tid in list(self.history.keys()):
            if tid not in current_ids:
                self.history[tid] -= 1
                if self.history[tid] <= 0:
                    del self.history[tid]
        
        return verified_tags

SIDE = 400
WARP_DST = np.array([[0, 0], [SIDE-1, 0], [SIDE-1, SIDE-1], [0, SIDE-1]], dtype="float32")

TAG_GRID_SIZE = 8
TAG_BORDER_WIDTH = 1
CENTER_PROXIMITY_THRESH_SQUARED = 400
CORE_START_CELL = 2
CORE_END_CELL = 6
FRAME_TRACKER = TemporalFilter(persistence=3)


class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords))


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


def render(img, obj, projection, model, color=False):
    DEFAULT_COLOR = (0, 0, 0)
    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = CustomCV2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]
            cv2.fillConvexPoly(img, imgpts, color)

    return img


def order_points(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]      
    rect[2] = pts[np.argmax(s)]      
    rect[1] = pts[np.argmin(diff)]   
    rect[3] = pts[np.argmax(diff)]   
    return rect


def refine_corners(gray, corners):
    criteria = (CustomCV2.TERM_CRITERIA_EPS + CustomCV2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), criteria)


def decode_tag(warped_tag: np.ndarray):
    """
    Decode AR tag from warped image.
    
    Args:
        warped_tag: Warped and aligned tag image (should be square)
    
    Returns:
        (tag_id, orientation) or (None, None) if invalid
    """
    # Threshold the image
    _, thresh = CustomCV2.threshold(warped_tag, 155, 255, CustomCV2.THRESH_BINARY)
    side = thresh.shape[0]
    
    # Validate minimum size
    if side < 64:
        return None, None
    
    cell = side / TAG_GRID_SIZE
    margin = max(1, int(cell / 2))
    margin = min(margin, side // 4)  # Safety clamp
    
    # Vectorized border sampling
    indices = np.clip(((np.arange(8) + 0.5) * cell).astype(int), 0, side - 1)
    
    border_samples = np.concatenate([
        thresh[margin, indices],
        thresh[indices, side - margin - 1],
        thresh[side - margin - 1, indices][::-1],
        thresh[indices, margin][::-1]
    ])
    
    # Check border (should be black)
    if np.any(border_samples > 150):
        return None, None

    # Extract core region
    start = int(CORE_START_CELL * cell)
    end = int(CORE_END_CELL * cell)
    core_size = end - start
    core_cell = core_size / 4.0
    
    def get_core_val(r: int, c: int) -> float:
        """Sample center 40% of a core cell"""
        y_start = max(0, int(start + (r + 0.3) * core_cell))
        y_end = min(side, int(start + (r + 0.7) * core_cell))
        x_start = max(0, int(start + (c + 0.3) * core_cell))
        x_end = min(side, int(start + (c + 0.7) * core_cell))
        
        if y_end <= y_start or x_end <= x_start:
            return 0
        
        return float(np.mean(thresh[y_start:y_end, x_start:x_end]))

    # Find orientation using anchor corners
    anchors = [(3, 3), (3, 0), (0, 0), (0, 3)]
    intensities = [get_core_val(r, c) for r, c in anchors]
    status = int(np.argmax(intensities))
    
    # Validate orientation marker
    if intensities[status] < 127:
        return None, None

    # Decode data bits based on orientation
    bit_map = {
        0: [(1, 1), (1, 2), (2, 2), (2, 1)],
        1: [(1, 2), (2, 2), (2, 1), (1, 1)],
        2: [(2, 2), (2, 1), (1, 1), (1, 2)],
        3: [(2, 1), (1, 1), (1, 2), (2, 2)]
    }

    bits = [1 if get_core_val(r, c) > 127 else 0 for r, c in bit_map[status]]
    tag_id = (bits[3] << 3 | bits[2] << 2 | bits[1] << 1 | bits[0])
    
    # Validate tag ID range
    if tag_id > 15:  # 4-bit ID should be 0-15
        return None, None
    
    return tag_id, status


def process_frame(frame):
    MIN_TAG_AREA = frame.shape[0] * frame.shape[1] * 0.0003 # 3% of frame


    pre_gray = time()
    gray = CustomCV2.cvtColor(frame, CustomCV2.COLOR_BGR2GRAY)
    post_gray = time()
    print(f"Grayscale Conversion Time: {post_gray - pre_gray:.4f} seconds")


    # blurred = CustomCV2.GaussianBlur(gray, (3, 3), 50)
    blurred = CustomCV2.BoxFilter(gray, (3, 3))
    post_blur = time()
    print(f"Gaussian Blur Time: {post_blur - post_gray:.4f} seconds")
    
    thresh = CustomCV2.adaptiveThreshold(blurred, 255, CustomCV2.ADAPTIVE_THRESH_MEAN_C, 
                                   CustomCV2.THRESH_BINARY_INV, 11, 7)

    # thresh = CustomCV2.Sobel(blurred, 155)

    # thresh = CustomCV2.threshold(blurred, 155, 255, CustomCV2.THRESH_BINARY + CustomCV2.THRESH_OTSU)[1]

    post_thresh = time()
    cv2.imshow("Thresholded", thresh)
    print(f"Adaptive Thresholding Time: {post_thresh - post_blur:.4f} seconds")

    contours, _ = CustomCV2.findContours(thresh, CustomCV2.RETR_TREE, CustomCV2.CHAIN_APPROX_SIMPLE)

    raw_candidates = []
    processed_centers = []

    for cnt in contours:
        if CustomCV2.contourArea(cnt) < MIN_TAG_AREA:
            continue
        
        pre_peri = time()
        peri = CustomCV2.arcLength(cnt, True)
        post_peri = time()
        print(f"Contour Perimeter Time: {post_peri - pre_peri:.4f} seconds")
        # quad = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        quad = CustomCV2.approxPolyDP(cnt, 0.02 * peri, True)
        post_approx = time()
        print(f"Contour Approximation Time: {post_approx - post_peri:.4f} seconds")
        # print("-------")
        # print(quad_cv2)
        # print("=======")
        # print(quad)
        # print("-------")
        # print(quad.shape, type(quad))
        # print(quad_cv2.shape, type(quad_cv2))
        # print(len(quad), len(quad_cv2))
        # print(cv2.isContourConvex(quad_cv2), cv2.isContourConvex(quad))
        # print("-------")
        # cv2.drawContours(frame, [quad], -1, (255, 0, 0), 2)

        # quad = CustomCV2.findCornersQuadrilateral(cnt)
        
        if len(quad) == 4 and CustomCV2.isContourConvex(quad):
            M = CustomCV2.moments(quad)
            if M["m00"] == 0:
                continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            if (len(processed_centers) > 0):
                centers_array = np.array(processed_centers)
                dists = np.sum((centers_array - np.array([cX, cY]))**2, axis=1)
                if np.any(dists < CENTER_PROXIMITY_THRESH_SQUARED):
                    continue

            rect = order_points(quad)
            # rect = refine_corners(gray, rect)
            pre_perspective = time()
            H = CustomCV2.getPerspectiveTransform(rect, WARP_DST)
            post_perspective = time()
            print(f"Perspective Transform Time: {post_perspective - pre_perspective:.4f} seconds")
            warped = CustomCV2.warpPerspective(gray, H, (SIDE, SIDE))

            result = decode_tag(warped)
            
            if result[0] is not None:
                tag_id, orient_idx = result
                rect = np.roll(rect, -orient_idx, axis=0)
                
                processed_centers.append((cX, cY))
                raw_candidates.append({"id": tag_id, "corners": rect, "rotation": orient_idx * 90})

    # stable_tags = FRAME_TRACKER.process(raw_candidates)
    stable_tags = raw_candidates
    
    for tag in stable_tags:
        rect = tag["corners"]
        cv2.polylines(frame, [np.int32(rect)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {tag['id']}", tuple(np.int32(rect[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, f"Rot: {tag['rotation']} deg", tuple(np.int32(rect[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame, stable_tags