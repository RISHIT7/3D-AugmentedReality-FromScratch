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
    print("C++ acceleration not available, using Python fallback")


SIDE = 400
WARP_DST = np.array([[0, 0], [SIDE-1, 0], [SIDE-1, SIDE-1], [0, SIDE-1]], dtype="float32")

TAG_GRID_SIZE = 8
TAG_BORDER_WIDTH = 1
CENTER_PROXIMITY_THRESH_SQUARED = 400
CORE_START_CELL = 2
CORE_END_CELL = 6

class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.faces = []
        try:
            with open(filename, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    values = line.split()
                    if not values:
                        continue
                    if values[0] == 'v':
                        v = [float(values[1]), float(values[2]), float(values[3])]
                        if swapyz: v = [v[0], v[2], v[1]]
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

def render(img, obj, projection, tag_corners, tag_size=2.0, color=(100, 100, 100), scale=3.0):
    if obj is None or len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img

    vertices_3d = obj.vertices.copy()

    min_v = np.min(vertices_3d, axis=0)
    max_v = np.max(vertices_3d, axis=0)
    center = (min_v + max_v) / 2
    size_range = np.max(max_v - min_v)

    if size_range == 0: size_range = 1.0

    vertices_3d = (vertices_3d - center) / size_range

    vertices_3d *= scale * (tag_size / 2.0)

    n = len(vertices_3d)
    vertices_homo = np.column_stack([vertices_3d, np.ones(n)])

    # Perform projection with warning suppression (spurious warnings observed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        projected = projection @ vertices_homo.T
    
    if not np.all(np.isfinite(projected)):
        return img

    w = projected[2, :]

    valid_mask = w > 0.001 
    if not np.any(valid_mask):
        return img 

    with np.errstate(divide='ignore', invalid='ignore'):
        x_2d = projected[0, :] / w
        y_2d = projected[1, :] / w

    points_2d = np.column_stack([x_2d, y_2d]).astype(np.int32)
    h, w_img = img.shape[:2]

    for face_indices in obj.faces:
        if not all(valid_mask[i] for i in face_indices if i < len(valid_mask)):
            continue

        try:
            face_points = points_2d[face_indices]

            if (np.all(face_points[:, 0] < -2000) or np.all(face_points[:, 0] > w_img + 2000) or
                np.all(face_points[:, 1] < -2000) or np.all(face_points[:, 1] > h + 2000)):
                continue

            CustomCV2.fillConvexPoly(img, face_points, color)
        except Exception:
            continue

    return img

def render_with_lighting(img, obj, projection, camera_matrix, tag_size=2.0, 
                          base_color=(120, 120, 120), scale=3.0):
    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img

    vertices_3d = obj.vertices.copy() * scale

    z_min = vertices_3d[:, 2].min()
    vertices_3d[:, 2] -= z_min
    x_center = (vertices_3d[:, 0].max() + vertices_3d[:, 0].min()) / 2
    y_center = (vertices_3d[:, 1].max() + vertices_3d[:, 1].min()) / 2
    vertices_3d[:, 0] -= x_center
    vertices_3d[:, 1] -= y_center

    n = len(vertices_3d)
    vertices_homo = np.column_stack([vertices_3d, np.ones(n)])

    projected = projection @ vertices_homo.T
    w = projected[2, :]

    valid_mask = w > 0.1
    if not np.any(valid_mask):
        return img

    x_2d = projected[0, :] / w
    y_2d = projected[1, :] / w
    points_2d = np.column_stack([x_2d, y_2d]).astype(np.int32)

    h, w_img = img.shape[:2]

    face_data = []

    for face_indices in obj.faces:
        if not all(valid_mask[i] for i in face_indices if i < len(valid_mask)):
            continue

        try:
            face_points = points_2d[face_indices]

            if (np.all(face_points[:, 0] < -500) or np.all(face_points[:, 0] > w_img + 500) or
                np.all(face_points[:, 1] < -500) or np.all(face_points[:, 1] > h + 500)):
                continue

            face_depths = w[face_indices]
            avg_depth = np.mean(face_depths)

            face_data.append((avg_depth, face_points))
        except:
            continue

    face_data.sort(reverse=True, key=lambda x: x[0])

    for depth, face_points in face_data:
        shade_factor = max(0.5, min(1.0, 1.0 - (depth - 200) / 1000))
        face_color = tuple(int(c * shade_factor) for c in base_color)

        CustomCV2.fillConvexPoly(img, face_points, face_color)

    return img

def compute_fine_grained_orientation(corners, granularity='1deg'):
    top_edge = corners[1] - corners[0]
    angle_deg = np.degrees(np.arctan2(top_edge[1], top_edge[0]))
    if angle_deg < 0: angle_deg += 360

    if granularity == '5deg': return round(angle_deg / 5) * 5
    if granularity == '10deg': return round(angle_deg / 10) * 10
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
        if area < MIN_TAG_AREA or area > MAX_TAG_AREA: continue

        peri = CustomCV2.arcLength(cnt, True)
        quad = CustomCV2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(quad) == 4 and CustomCV2.isContourConvex(quad):
            M = CustomCV2.moments(quad)
            if M["m00"] == 0: continue
            cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            if len(processed_centers) > 0:
                dists = np.sum((np.array(processed_centers) - np.array([cX, cY]))**2, axis=1)
                if np.any(dists < CENTER_PROXIMITY_THRESH_SQUARED): continue

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
                            processed_centers.append((cX, cY))
                            candidates.append({"id": tag_id, "corners": rect, "orientation": orientation * 90})
                            break
                continue

            if tag_id is not None:
                rect = np.roll(rect, -orientation, axis=0)
                processed_centers.append((cX, cY))
                candidates.append({"id": tag_id, "corners": rect, "orientation": orientation * 90})

    return candidates

def process_frame(frame, template_img=None, obj_model=None, camera_matrix=None, angle_granularity='5deg'):
    MIN_TAG_AREA = frame.shape[0] * frame.shape[1] * 0.0003 
    MAX_TAG_AREA = frame.shape[0] * frame.shape[1] * 0.9

    gray = CustomCV2.cvtColor(frame, CustomCV2.COLOR_BGR2GRAY)
    blurred = CustomCV2.bilateralFilter(gray, 9, 75, 75)

    block_sizes = [8, 11, 21]
    thresh = np.zeros_like(gray)
    for bs in block_sizes:
        t = CustomCV2.adaptiveThreshold(blurred, 255, CustomCV2.ADAPTIVE_THRESH_MEAN_C, CustomCV2.THRESH_BINARY_INV, bs, 7)
        thresh = CustomCV2.bitwise_or(thresh, t)

    detected_tags = process_contours(gray, thresh, MIN_TAG_AREA, MAX_TAG_AREA)

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
            # 3D points for the tag corners (normalized).
            # standard: TL, TR, BR, BL
            # We assume the tag is on the XY plane (z=0).
            # Half-size 1.0 means the tag is 2x2 in object space.
            half_size = 1.0
            obj_pts_3d = np.array([
                [-half_size, -half_size, 0], # TL
                [ half_size, -half_size, 0], # TR
                [ half_size,  half_size, 0], # BR
                [-half_size,  half_size, 0]  # BL
            ], dtype=np.float64)
            
            # Image points are already in TL, TR, BR, BL order from order_points
            img_pts = tag["corners"].astype(np.float64)

            # Solve PnP
            # We assume zero distortion for simplicity if not provided.
            # Use ITERATIVE for robustness or IPPE_SQUARE. Let's try ITERATIVE.
            success, rvec, tvec = cv2.solvePnP(obj_pts_3d, img_pts, camera_matrix.astype(np.float64), None, flags=cv2.SOLVEPNP_ITERATIVE)
            
            if success:
                # Convert rvec to Rotation Matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Construct Projection Matrix [R | t]
                Rt = np.column_stack((R, tvec))
                
                # To fix the "upside down" issue:
                # The tag coordinate system has Z pointing INTO the table (standard vision frame X-right, Y-down, Z-forward).
                # The object likely expects +Z or +Y to be "up" (out of the table).
                # We need to rotate the object coordinate system relative to the tag.
                # Only Rotate 180 deg around X-axis: Y -> -Y, Z -> -Z.
                # This makes +Z point OUT of the table (towards camera).
                
                Rx = np.array([
                    [1,  0,  0, 0],
                    [0, -1,  0, 0],
                    [0,  0, -1, 0],
                    [0,  0,  0, 1]
                ], dtype=np.float64)
                
                # We effectively multiply the ModelView matrix by this rotation
                # P = K * [R|t]
                # P_new = K * [R|t] * Rx ? No.
                # Points are P_model. We want P_tag = Rx * P_model.
                # So we project: K * [R|t] * Rx * P_model_homo
                
                # Construct 4x4 Extrinsics matrix to apply rotation easily
                Extrinsics = np.eye(4, dtype=np.float64)
                Extrinsics[:3, :4] = Rt
                
                # Apply rotation to the object frame
                Rt_new = Extrinsics @ Rx
                Rt_new = Rt_new[:3, :4] # Back to 3x4
                
                # Full Projection transformation: K * Rt_new
                projection = np.dot(camera_matrix.astype(np.float64), Rt_new)
                
                # Debug: Check for extreme values
                if not np.all(np.isfinite(projection)):
                    print(f"Projected matrix contains inf/nan: {projection}")
                elif np.max(np.abs(projection)) > 1e10:
                    print(f"Projected matrix values too large: {projection}")
                elif tvec[2] < 0:
                    print(f"Warning: tvec Z is negative ({tvec[2]}), tag behind camera?")
                elif tvec[2] < 1.0: # Very close to camera?
                     print(f"Warning: tvec Z is very small ({tvec[2]})")
                
                # Check for validity
                if np.all(np.isfinite(projection)):
                    frame = render(frame, obj_model, projection, tag["corners"], 
                                    tag_size=half_size * 2, color=(80, 80, 80), scale=2.5)

        if obj_model is None and template_img is None:  
            cv2.polylines(frame, [np.int32(tag["corners"])], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", tuple(np.int32(tag["corners"][0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {tag['angle']:.1f}deg", 
                    tuple(np.int32(tag["corners"][1]) + np.array([0, 30])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

    return frame, detected_tags

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