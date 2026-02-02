import cv2
from cv2_functions import CustomCV2
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
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
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
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    s = pts.sum(axis=1)
    return np.roll(pts, -np.argmin(s), axis=0)


def refine_corners(gray, corners):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), criteria)


def decode_tag(warped_tag):
    # _, thresh = CustomCV2.threshold(warped_tag, 0, 255, CustomCV2.THRESH_BINARY + CustomCV2.THRESH_TEMPORAL_APPROX_OTSU)
    _, thresh = CustomCV2.threshold(warped_tag, 155, 255, CustomCV2.THRESH_BINARY)
    side = thresh.shape[0]
    cell = side / 8.0

    margin = int(cell / 2)
    border_samples = []
    for i in range(8):
        pos = int((i + 0.5) * cell)
        border_samples.extend([
            thresh[margin, pos],
            thresh[side - margin, pos],
            thresh[pos, margin],
            thresh[pos, side - margin]
        ])
    
    if np.any(np.array(border_samples) > 150):
        return None, None

    start = int(2 * cell)
    end = int(6 * cell)
    core = thresh[start:end, start:end]
    core = cv2.resize(core, (200, 200), interpolation=cv2.INTER_NEAREST)
    c_cell = 50.0

    def get_core_val(r, c):
        y, x = int((r + 0.5) * c_cell), int((c + 0.5) * c_cell)
        return np.mean(core[y - 10:y + 10, x - 10:x + 10])

    anchors = [(3, 3), (3, 0), (0, 0), (0, 3)]
    intensities = [get_core_val(r, c) for r, c in anchors]
    status = np.argmax(intensities)
    
    if intensities[status] < 127:
        return None, None

    bit_map = {
        0: [(1, 1), (1, 2), (2, 2), (2, 1)],
        1: [(1, 2), (2, 2), (2, 1), (1, 1)],
        2: [(2, 2), (2, 1), (1, 1), (1, 2)],
        3: [(2, 1), (1, 1), (1, 2), (2, 2)]
    }

    bits = [1 if get_core_val(r, c) > 127 else 0 for r, c in bit_map[status]]
    tag_id = (bits[3] << 3 | bits[2] << 2 | bits[1] << 1 | bits[0])
    
    return tag_id, status


def process_frame(frame):
    pre_gray = time()
    gray = CustomCV2.cvtColor(frame, CustomCV2.COLOR_BGR2GRAY)
    post_gray = time()
    # print(f"Grayscale Conversion Time: {post_gray - pre_gray:.4f} seconds")


    # blurred = CustomCV2.GaussianBlur(gray, (3, 3), 50)
    blurred = CustomCV2.BoxFilter(gray, (3, 3))
    post_blur = time()
    # print(f"Gaussian Blur Time: {post_blur - post_gray:.4f} seconds")
    
    thresh = CustomCV2.adaptiveThreshold(blurred, 255, CustomCV2.ADAPTIVE_THRESH_MEAN_C, 
                                   CustomCV2.THRESH_BINARY_INV, 11, 7)

    # thresh = CustomCV2.Sobel(blurred, 155)

    # thresh = CustomCV2.threshold(blurred, 155, 255, CustomCV2.THRESH_BINARY + CustomCV2.THRESH_OTSU)[1]

    post_thresh = time()
    cv2.imshow("Thresholded", thresh)
    # print(f"Adaptive Thresholding Time: {post_thresh - post_blur:.4f} seconds")

    contours, _ = CustomCV2.findContours(thresh, CustomCV2.RETR_TREE, CustomCV2.CHAIN_APPROX_SIMPLE)
    post_contours = time()
    # print(f"Contour Detection Time: {post_contours - post_thresh:.4f} seconds")

    raw_candidates = []
    processed_centers = []

    for cnt in contours:
        if CustomCV2.contourArea(cnt) < 1000:
            continue
        
        peri = CustomCV2.arcLength(cnt, True)
        # quad = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        quad = CustomCV2.approxPolyDP(cnt, 0.04 * peri, True)
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

            if any(math.hypot(cX - pc[0], cY - pc[1]) < 20 for pc in processed_centers):
                continue

            rect = order_points(quad)
            rect = refine_corners(gray, rect)
            
            side = 160
            dst_pts = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
            H = cv2.getPerspectiveTransform(rect, dst_pts)
            warped = cv2.warpPerspective(gray, H, (side, side))

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

    return frame, stable_tags