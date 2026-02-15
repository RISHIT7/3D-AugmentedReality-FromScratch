import cv2
import numpy as np
import warnings
from core.cv2_functions import CustomCV2
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import custom_cv2_cpp
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False

SIDE = 400
WARP_DST = np.array([[0, 0], [SIDE-1, 0], [SIDE-1, SIDE-1], [0, SIDE-1]], dtype="float32")

TAG_GRID_SIZE = 8
TAG_BORDER_WIDTH = 1
CENTER_PROXIMITY_THRESH_SQUARED = 400
CORE_START_CELL = 2
CORE_END_CELL = 6

class OBJ:
    def __init__(self, filename, swapyz=False, swapxy=False, swapxz=False):
        self.vertices = []
        self.faces = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    values = line.split()
                    if not values:
                        continue
                    if values[0] == 'v':
                        v = [float(values[1]), float(values[2]), float(values[3])]
                        if swapyz:
                            v = [v[0], v[2], v[1]]
                        if swapxy:
                            v = [v[1], v[0], v[2]]
                        if swapxz:
                            v = [v[2], v[1], v[0]]
                        self.vertices.append(v)
                    elif values[0] == 'f':
                        face = []
                        for vertex in values[1:]:
                            idx = int(vertex.split('/')[0])
                            face.append(idx - 1)
                        if len(face) >= 3:
                            self.faces.append(face)
            self.vertices = np.array(self.vertices, dtype=np.float32)
            print(f"✓ Loaded OBJ: {len(self.vertices)} vertices, {len(self.faces)} faces")
        except Exception as e:
            print(f"Error loading OBJ: {e}")

def _normalize_vertices(obj, scale, tag_size):
    """Normalize OBJ vertices: center, scale to [-1,1], then apply scale and tag_size."""
    vertices_3d = obj.vertices.copy()
    min_v = np.min(vertices_3d, axis=0)
    max_v = np.max(vertices_3d, axis=0)
    center = (min_v + max_v) / 2
    size_range = np.max(max_v - min_v)
    if size_range == 0:
        size_range = 1.0
    vertices_3d = (vertices_3d - center) / size_range
    vertices_3d *= scale * (tag_size / 2.0)
    return vertices_3d


def _project_vertices(vertices_3d, projection):
    """Project 3D vertices using the projection matrix. Returns (points_2d, w, valid_mask) or None."""
    n = len(vertices_3d)
    vertices_homo = np.column_stack([vertices_3d, np.ones(n)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projected = projection @ vertices_homo.T

    if not np.all(np.isfinite(projected)):
        return None

    w = projected[2, :]
    valid_mask = w > 0.001
    if not np.any(valid_mask):
        return None

    with np.errstate(divide='ignore', invalid='ignore'):
        x_2d = projected[0, :] / w
        y_2d = projected[1, :] / w

    points_2d = np.column_stack([x_2d, y_2d]).astype(np.int32)
    return points_2d, w, valid_mask


def _collect_visible_faces(obj, points_2d, w, valid_mask, img_shape, margin=2000):
    """Collect faces that pass validity and bounds checks. Returns list of (avg_depth, face_points)."""
    h, w_img = img_shape[:2]
    face_data = []
    for face_indices in obj.faces:
        if not all(valid_mask[i] for i in face_indices if i < len(valid_mask)):
            continue
        try:
            face_points = points_2d[face_indices]
            if (np.all(face_points[:, 0] < -margin) or np.all(face_points[:, 0] > w_img + margin) or
                np.all(face_points[:, 1] < -margin) or np.all(face_points[:, 1] > h + margin)):
                continue
            avg_depth = np.mean(w[face_indices])
            face_data.append((avg_depth, face_points))
        except Exception:
            continue
    face_data.sort(reverse=True, key=lambda x: x[0])
    return face_data


def render(img, obj, projection, tag_corners=None, tag_size=2.0, color=(100, 100, 100), scale=3.0):
    if obj is None or len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img

    vertices_3d = _normalize_vertices(obj, scale, tag_size)
    result = _project_vertices(vertices_3d, projection)
    if result is None:
        return img
    points_2d, w, valid_mask = result

    face_data = _collect_visible_faces(obj, points_2d, w, valid_mask, img.shape)

    for _depth, face_points in face_data:
        CustomCV2.fillConvexPoly(img, face_points, color)

    return img

def render_with_lighting(img, obj, projection, tag_corners=None, tag_size=2.0,
                          color=(120, 120, 120), scale=3.0):
    if obj is None or len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img

    vertices_3d = _normalize_vertices(obj, scale, tag_size)
    result = _project_vertices(vertices_3d, projection)
    if result is None:
        return img
    points_2d, w, valid_mask = result

    face_data = _collect_visible_faces(obj, points_2d, w, valid_mask, img.shape)

    if not face_data:
        return img

    all_depths = [d for d, _ in face_data]
    depth_min, depth_max = min(all_depths), max(all_depths)
    depth_range = depth_max - depth_min
    if depth_range < 1e-6:
        depth_range = 1.0

    for depth, face_points in face_data:
        t = (depth - depth_min) / depth_range
        shade_factor = max(0.5, min(1.0, 1.0 - 0.5 * t))
        face_color = tuple(int(c * shade_factor) for c in color)
        CustomCV2.fillConvexPoly(img, face_points, face_color)

    return img

def compute_fine_grained_orientation(corners, granularity='1deg'):
    top_edge = corners[1] - corners[0]
    angle_deg = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
    if angle_deg < 0:
        angle_deg += 360

    if granularity == '5deg':
        return round(angle_deg / 5) * 5
    if granularity == '10deg':
        return round(angle_deg / 10) * 10
    return round(angle_deg)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def order_points(pts):
    pts = pts.reshape(4, 2)

    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    if abs(x_sorted[1, 0] - x_sorted[2, 0]) < 1.0:
        y_sorted = pts[np.argsort(pts[:, 1]), :]
        tl = y_sorted[0]
        br = y_sorted[3]
        
        mid_pts = y_sorted[1:3]
        mid_pts_x_sorted = mid_pts[np.argsort(mid_pts[:, 0])]
        bl = mid_pts_x_sorted[0]
        tr = mid_pts_x_sorted[1]
    else:
        left_most = left_most[np.argsort(left_most[:, 1]), :]
        (tl, bl) = left_most

        right_most = right_most[np.argsort(right_most[:, 1]), :]
        (tr, br) = right_most

    return np.array([tl, tr, br, bl], dtype="float32")

def refine_corners(gray, corners):
    criteria = (CustomCV2.TERM_CRITERIA_EPS + CustomCV2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return CustomCV2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), criteria)

def is_valid_quadrilateral(pts, min_side=5.0, max_side_ratio=4.0, min_angle=40.0, max_angle=140.0):
    """Check if a 4-point polygon is geometrically consistent with a (perspective-warped) square.
    Rejects elongated parallelograms, extreme trapezoids, and noise artifacts."""
    pts = pts.reshape(4, 2).astype(np.float64)

    sides = np.array([np.sqrt(np.sum((pts[(i+1) % 4] - pts[i])**2)) for i in range(4)])
    if np.min(sides) < min_side:
        return False
    if np.max(sides) / (np.min(sides) + 1e-8) > max_side_ratio:
        return False

    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        dot = np.dot(v1, v2)
        norms = np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)) + 1e-8
        cos_angle = np.clip(dot / norms, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        if angle_deg < min_angle or angle_deg > max_angle:
            return False

    return True

def check_border(processed, cell, margin):
    side = processed.shape[0]
    indices = np.clip(((np.arange(TAG_BORDER_WIDTH) + 0.5) * cell).astype(int), 0, side - 1)

    for idx in indices:
        if processed[margin, idx] > 127:
            return False
        if processed[side - margin - 1, idx] > 127:
            return False

    for idx in indices:
        if processed[idx, margin] > 127:
            return False
        if processed[idx, side - margin - 1] > 127:
            return False

    return True

def overlay_image_python(frame, overlay, H):
    h_frame, w_frame = frame.shape[:2]
    h_overlay, w_overlay = overlay.shape[:2]

    y_coords, x_coords = np.mgrid[0:h_frame, 0:w_frame]
    coords = np.stack([x_coords.flatten(), y_coords.flatten(), np.ones(h_frame * w_frame)])

    H_inv = np.linalg.inv(H)
    overlay_coords = H_inv @ coords
    overlay_coords /= overlay_coords[2]

    overlay_x = overlay_coords[0].reshape(h_frame, w_frame)
    overlay_y = overlay_coords[1].reshape(h_frame, w_frame)

    valid_mask = (
        (overlay_x >= 0) & (overlay_x < w_overlay - 1) &
        (overlay_y >= 0) & (overlay_y < h_overlay - 1)
    )

    x0 = np.floor(overlay_x[valid_mask]).astype(int)
    y0 = np.floor(overlay_y[valid_mask]).astype(int)
    x1 = np.clip(x0 + 1, 0, w_overlay - 1)
    y1 = np.clip(y0 + 1, 0, h_overlay - 1)

    dx = overlay_x[valid_mask] - x0
    dy = overlay_y[valid_mask] - y0

    y_valid, x_valid = np.where(valid_mask)
    for c in range(3):
        vals = (
            overlay[y0, x0, c] * (1 - dx) * (1 - dy) +
            overlay[y0, x1, c] * dx * (1 - dy) +
            overlay[y1, x0, c] * (1 - dx) * dy +
            overlay[y1, x1, c] * dx * dy
        )
        frame[y_valid, x_valid, c] = vals.astype(np.uint8)

    return frame

ERROR_CODE = {
    "INVALID_BORDER": -1,
    "INVALID_SIZE": -2,
    "INVALID_CHECKSUM": -3,
    "INVALID_ORIENTATION": -4,
    "SUCCESS": 0,
}

def decode_tag(warped_tag: np.ndarray):
    sharpened = CustomCV2.sharpenAndNormalize(warped_tag)
    threshold = CustomCV2.threshold(sharpened, 0, 255, CustomCV2.THRESH_BINARY + CustomCV2.THRESH_OTSU)[1]
    side = threshold.shape[0]

    if side < 64:
        return None, None, ERROR_CODE["INVALID_SIZE"]

    cell = side / TAG_GRID_SIZE
    margin = max(1, int(cell / 2))
    margin = min(margin, side // 4)

    if not check_border(threshold, cell, margin):
        return None, None, ERROR_CODE["INVALID_BORDER"]

    core_indices = [2, 3, 4, 5]

    def get_center_coord(idx):
        return int((idx + 0.5) * cell)

    grid_bits = np.zeros((4, 4), dtype=np.uint8)
    grid_intensities = np.zeros((4, 4), dtype=np.float32)

    for r_idx, grid_row in enumerate(core_indices):
        y = get_center_coord(grid_row)
        row_signal = threshold[y, :]
        row_signal = CustomCV2.GaussianBlur(
            row_signal.reshape(-1, 1).astype(np.float64), (5, 1), 16
        ).flatten()

        col_coords = [get_center_coord(c_idx) for c_idx in core_indices]
        row_values = [row_signal[c] for c in col_coords]

        segment = row_signal[int(2*cell):int(6*cell)].astype(np.float64)
        local_min, local_max = float(np.min(segment)), float(np.max(segment))
        row_thresh = (local_min + local_max) / 2.0

        if (local_max - local_min) < 30:
            row_thresh = 155.0

        for c_idx, val in enumerate(row_values):
            grid_intensities[r_idx, c_idx] = val
            grid_bits[r_idx, c_idx] = 1 if val > row_thresh else 0

    anchor_positions = [
        (get_center_coord(5), get_center_coord(5)),
        (get_center_coord(5), get_center_coord(2)),
        (get_center_coord(2), get_center_coord(2)),
        (get_center_coord(2), get_center_coord(5)),
    ]

    erode_k = max(3, int(cell * 0.15)) | 1
    erode_kernel = np.ones((erode_k, erode_k), dtype=np.uint8)
    threshold_eroded = CustomCV2.erode(threshold, erode_kernel, iterations=2)

    patch_half = max(2, int(cell * 0.2))

    def sample_patch_median(img, cy, cx):
        h, w = img.shape[:2]
        y0 = max(0, cy - patch_half)
        y1 = min(h, cy + patch_half + 1)
        x0 = max(0, cx - patch_half)
        x1 = min(w, cx + patch_half + 1)
        return float(np.median(img[y0:y1, x0:x1].astype(np.float64)))

    eroded_vals = [sample_patch_median(threshold_eroded, *pos) for pos in anchor_positions]

    white_corners = [i for i, v in enumerate(eroded_vals) if v > 127]

    if len(white_corners) == 1:
        orientation = white_corners[0]
    else:
        gs_vals = [sample_patch_median(sharpened, *pos) for pos in anchor_positions]
        sorted_indices = np.argsort(gs_vals)
        max_idx = int(sorted_indices[-1])
        max_val = gs_vals[max_idx]
        all_range = max_val - gs_vals[int(sorted_indices[0])]

        if max_val < 100.0 and all_range < 15.0:
            return None, None, ERROR_CODE["INVALID_ORIENTATION"]

        if len(white_corners) >= 2:
            orientation = max(white_corners, key=lambda i: gs_vals[i])
        else:
            orientation = max_idx

    bit_map = {
        0: [(1, 1), (1, 2), (2, 2), (2, 1)],
        1: [(1, 2), (2, 2), (2, 1), (1, 1)],
        2: [(2, 2), (2, 1), (1, 1), (1, 2)],
        3: [(2, 1), (1, 1), (1, 2), (2, 2)]
    }

    try:
        data_bits = [grid_bits[r, c] for r, c in bit_map[orientation]]
    except KeyError:
        return None, None, ERROR_CODE["INVALID_ORIENTATION"]

    tag_id = (data_bits[0] << 0) | (data_bits[1] << 1) | (data_bits[2] << 2) | (data_bits[3] << 3)

    return tag_id, orientation, ERROR_CODE["SUCCESS"]

def process_contours(gray, thresh, MIN_TAG_AREA, MAX_TAG_AREA):
    contours, _ = CustomCV2.findContours(thresh, CustomCV2.RETR_TREE, CustomCV2.CHAIN_APPROX_SIMPLE)
    candidates = []
    processed_centers = []

    for cnt in contours:
        area = CustomCV2.contourArea(cnt)
        if area < MIN_TAG_AREA or area > MAX_TAG_AREA:
            continue
        
        peri = CustomCV2.arcLength(cnt, True)
        quad = CustomCV2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(quad) == 4 and CustomCV2.isContourConvex(quad) and is_valid_quadrilateral(quad):
            M = CustomCV2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if len(processed_centers) > 0:
                dists = np.sum((np.array(processed_centers) - np.array([cx, cy]))**2, axis=1)
                if np.any(dists < CENTER_PROXIMITY_THRESH_SQUARED):
                    continue

            rect = order_points(quad)
            rect = refine_corners(gray, rect)
            H = CustomCV2.getPerspectiveTransform(rect, WARP_DST)
            warped = CustomCV2.warpPerspective(gray, H, (SIDE, SIDE))
            
            tag_id, orientation, error_code = decode_tag(warped)

            if error_code == ERROR_CODE["INVALID_BORDER"]:
                inner_thresh = CustomCV2.adaptiveThreshold(
                    warped, 255, CustomCV2.ADAPTIVE_THRESH_MEAN_C,
                    CustomCV2.THRESH_BINARY_INV, 11, 7
                )
                inner_contours, _ = CustomCV2.findContours(
                    inner_thresh, CustomCV2.RETR_TREE, CustomCV2.CHAIN_APPROX_SIMPLE
                )
                inner_min_area = SIDE * SIDE * 0.05
                inner_max_area = SIDE * SIDE * 0.85
                for ic in inner_contours:
                    ia = CustomCV2.contourArea(ic)
                    if ia < inner_min_area or ia > inner_max_area:
                        continue
                    ip = CustomCV2.arcLength(ic, True)
                    iq = CustomCV2.approxPolyDP(ic, 0.02 * ip, True)
                    if len(iq) == 4 and CustomCV2.isContourConvex(iq):
                        inner_rect = order_points(iq)
                        pts_h = np.hstack([inner_rect, np.ones((4, 1), dtype=np.float64)])
                        try:
                            H_inv = np.linalg.inv(H)
                        except np.linalg.LinAlgError:
                            continue
                        mapped = (H_inv @ pts_h.T).T
                        mapped_pts = (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)

                        mapped_pts = refine_corners(gray, mapped_pts)
                        H2 = CustomCV2.getPerspectiveTransform(mapped_pts, WARP_DST)
                        warped2 = CustomCV2.warpPerspective(gray, H2, (SIDE, SIDE))

                        tag_id, orientation, error_code = decode_tag(warped2)
                        if tag_id is not None:
                            rect = np.roll(mapped_pts, -orientation, axis=0)
                            processed_centers.append((cx, cy))
                            candidates.append({"id": tag_id, "corners": rect, "orientation": orientation * 90})
                            break
                continue

            if tag_id is not None:
                rect = np.roll(rect, -orientation, axis=0)
                processed_centers.append((cx, cy))
                candidates.append({"id": tag_id, "corners": rect, "orientation": orientation * 90})

    return candidates

def process_frame(frame, template_img=None, obj_model=None, camera_matrix=None, angle_granularity='5deg'):
    MIN_TAG_AREA = frame.shape[0] * frame.shape[1] * 0.0003 
    MAX_TAG_AREA = frame.shape[0] * frame.shape[1] * 0.9

    gray = CustomCV2.cvtColor(frame, CustomCV2.COLOR_BGR2GRAY)

    scales = [
        (8, 5),
        (21, 11)
    ]

    detected_tags = []
    detected_centers = []

    for block_size, blur_size in scales:
        blurred = CustomCV2.bilateralFilter(gray, blur_size, 75, 75)
        thresh = CustomCV2.adaptiveThreshold(
            blurred, 255, CustomCV2.ADAPTIVE_THRESH_MEAN_C,
            CustomCV2.THRESH_BINARY_INV, block_size, 7
        )

        scale_tags = process_contours(gray, thresh, MIN_TAG_AREA, MAX_TAG_AREA)

        for tag in scale_tags:
            tag_center = np.mean(tag["corners"], axis=0)
            too_close = False
            for existing_center in detected_centers:
                if np.sum((tag_center - existing_center)**2) < CENTER_PROXIMITY_THRESH_SQUARED:
                    too_close = True
                    break

            if not too_close:
                detected_tags.append(tag)
                detected_centers.append(tag_center)

    if camera_matrix is None:
        h, w = frame.shape[:2]
        camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)

    for tag in detected_tags:
        tag_id = tag["id"]
        corners = tag["corners"]
        tag['angle'] = compute_fine_grained_orientation(corners, angle_granularity)

        if template_img is not None:
            h_temp, w_temp = template_img.shape[:2]
            src_pts = np.array([[0, 0], [w_temp-1, 0], [w_temp-1, h_temp-1], [0, h_temp-1]], dtype="float32")
            H = CustomCV2.getPerspectiveTransform(src_pts, tag["corners"])
            try:
                H_inv = np.linalg.inv(H)
                if CPP_AVAILABLE:
                    frame = custom_cv2_cpp.overlay_image_cpp(frame, template_img, H_inv)
                else:
                    frame = overlay_image_python(frame, template_img, H)
            except: pass

        if obj_model is not None:
            half_size = 1.0
            obj_pts_3d = np.array([
                [-half_size, -half_size, 0],
                [ half_size, -half_size, 0],
                [ half_size,  half_size, 0],
                [-half_size,  half_size, 0] 
            ], dtype=np.float64)
            
            img_pts = tag["corners"].astype(np.float64)
            success, rvec, tvec = CustomCV2.solvePnP(obj_pts_3d, img_pts, camera_matrix.astype(np.float64), None)
            
            if success:
                R, _ = CustomCV2.Rodrigues(rvec)
                Rt = np.column_stack((R, tvec))
                
                Rx = np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0, -1, 0],
                    [0,  0,  0, 1]
                ], dtype=np.float64)
                
                Extrinsics = np.eye(4, dtype=np.float64)
                Extrinsics[:3, :4] = Rt
                
                Rt_new = Extrinsics @ Rx
                Rt_new = Rt_new[:3, :4]
                
                projection = np.dot(camera_matrix.astype(np.float64), Rt_new)
                
                if not np.all(np.isfinite(projection)):
                    print(f"Projected matrix contains inf/nan: {projection}")
                elif np.max(np.abs(projection)) > 1e10:
                    print(f"Projected matrix values too large: {projection}")
                elif tvec[2] < 0:
                    print(f"Warning: tvec Z is negative ({tvec[2]}), tag behind camera?")
                elif tvec[2] < 1.0:
                     print(f"Warning: tvec Z is very small ({tvec[2]})")
                
                if np.all(np.isfinite(projection)):
                    frame = render_with_lighting(frame, obj_model, projection, tag["corners"],
                                    tag_size=half_size * 2, color=(250, 0, 0), scale=2.5)

        if obj_model is None and template_img is None:  
            cv2.polylines(frame, [np.int32(tag["corners"])], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", tuple(np.int32(tag["corners"][0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {tag['angle']:.1f}deg", 
                    tuple(np.int32(tag["corners"][1]) + np.array([0, 30])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

    return frame, detected_tags