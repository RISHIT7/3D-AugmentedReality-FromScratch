import cv2
from core.cv2_functions import CustomCV2
import numpy as np
import math
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import custom_cv2_cpp # type: ignore
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    print("C++ acceleration not available, using Python fallback")

tag_trackers = {}

class AdaptiveKalmanFilter:
    """
    Adaptive Kalman Filter that adjusts noise parameters based on motion dynamics.
    Handles fast movements by increasing process noise and measurement trust.
    """
    def __init__(self, process_noise=1e-4, measurement_noise=1e-1, error_estimate=1.0):
        # State vector: [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # Measurement matrix (we only measure position)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        
        # Covariance matrices
        self.P = np.eye(4, dtype=np.float32) * error_estimate
        self.base_Q = np.eye(4, dtype=np.float32) * process_noise
        self.base_R = np.eye(2, dtype=np.float32) * measurement_noise
        self.Q = self.base_Q.copy()
        self.R = self.base_R.copy()
        
        # Motion tracking for adaptation
        self.velocity_history = deque(maxlen=5)
        self.innovation_history = deque(maxlen=3)
        self.last_position = None
        
    def predict(self):
        """Predict next state"""
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.state[:2]
    
    def adapt_noise(self, innovation):
        """
        Adapt process and measurement noise based on motion characteristics.
        
        Args:
            innovation: Difference between measurement and prediction
        """
        # Track innovation (measurement - prediction difference)
        innovation_magnitude = np.linalg.norm(innovation)
        self.innovation_history.append(innovation_magnitude)
        
        # Calculate velocity magnitude
        velocity = self.state[2:4]
        speed = np.linalg.norm(velocity)
        self.velocity_history.append(speed)
        
        # Adaptive strategy:
        # 1. High innovation + high speed → fast motion → increase process noise, trust measurements more
        # 2. Low innovation + low speed → stable → decrease process noise, trust predictions more
        
        avg_innovation = np.mean(self.innovation_history) if len(self.innovation_history) > 0 else 0
        avg_speed = np.mean(self.velocity_history) if len(self.velocity_history) > 0 else 0
        
        # Process noise adaptation (higher for fast motion)
        if avg_speed > 5.0 or avg_innovation > 20.0:
            # Fast motion detected
            process_multiplier = 1.0 + min(avg_speed * 0.05, 5.0)
            measurement_multiplier = 0.5  # Trust measurements more
        elif avg_speed < 1.0 and avg_innovation < 5.0:
            # Stable motion
            process_multiplier = 0.5
            measurement_multiplier = 1.5  # Trust predictions more
        else:
            # Normal motion
            process_multiplier = 1.0
            measurement_multiplier = 1.0
        
        # Update noise matrices
        self.Q = self.base_Q * process_multiplier
        self.R = self.base_R * measurement_multiplier
    
    def update(self, meas_x, meas_y):
        """
        Update state with new measurement.
        
        Args:
            meas_x: Measured x position
            meas_y: Measured y position
            
        Returns:
            Updated position [x, y]
        """
        measurement = np.array([meas_x, meas_y], dtype=np.float32)
        
        # Calculate innovation
        predicted_measurement = np.dot(self.H, self.state)
        innovation = measurement - predicted_measurement
        
        # Adapt noise parameters based on motion
        self.adapt_noise(innovation)
        
        # Kalman update equations
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.state = self.state + np.dot(K, innovation)
        
        I = np.eye(4, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        # Update velocity based on position change
        if self.last_position is not None:
            measured_velocity = measurement - self.last_position
            # Blend measured velocity with predicted velocity (smoothing)
            self.state[2:4] = 0.7 * self.state[2:4] + 0.3 * measured_velocity
        
        self.last_position = measurement.copy()
        
        return self.state[:2]

class ExponentialMovingAverage:
    """
    Simple EMA filter for smoothing.
    Useful as a complementary filter when Kalman becomes unstable.
    """
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # Higher alpha = more responsive, lower = more smooth
        self.value = None
        
    def update(self, measurement):
        if self.value is None:
            self.value = measurement
        else:
            self.value = self.alpha * measurement + (1 - self.alpha) * self.value
        return self.value

class HybridFilter:
    """
    Hybrid filter combining Adaptive Kalman and EMA.
    Automatically switches based on motion characteristics.
    """
    def __init__(self):
        self.kalman = AdaptiveKalmanFilter(
            process_noise=2e-4,  # Slightly higher for better fast motion handling
            measurement_noise=5e-2,  # Lower to trust measurements more
            error_estimate=1.0
        )
        self.ema = ExponentialMovingAverage(alpha=0.6)
        self.is_initialized = False
        self.outlier_threshold = 50.0  # pixels
        self.last_valid_position = None
        
    def is_outlier(self, measurement):
        """Detect if measurement is an outlier"""
        if self.last_valid_position is None:
            return False
        
        distance = np.linalg.norm(measurement - self.last_valid_position)
        return distance > self.outlier_threshold
    
    def update(self, measurement):
        """
        Update filter with new measurement.
        
        Args:
            measurement: 2D position [x, y]
            
        Returns:
            Filtered position [x, y]
        """
        if not self.is_initialized:
            self.kalman.state[:2] = measurement
            self.ema.value = measurement
            self.last_valid_position = measurement
            self.is_initialized = True
            return measurement
        
        # Outlier detection
        if self.is_outlier(measurement):
            # Use prediction only
            prediction = self.kalman.predict()
            return prediction
        
        # Normal update
        kalman_prediction = self.kalman.predict()
        kalman_estimate = self.kalman.update(measurement[0], measurement[1])
        ema_estimate = self.ema.update(measurement)
        
        # Adaptive blending based on motion speed
        speed = np.linalg.norm(self.kalman.state[2:4])
        
        if speed > 10.0:
            # Fast motion - trust Kalman more (better motion model)
            blend_factor = 0.8
        elif speed < 2.0:
            # Slow motion - use EMA more (smoother)
            blend_factor = 0.4
        else:
            # Moderate motion - balanced blend
            blend_factor = 0.6
        
        result = blend_factor * kalman_estimate + (1 - blend_factor) * ema_estimate
        self.last_valid_position = result.copy()
        
        return result

class TagTracker:
    """
    Multi-corner tracker with hybrid filtering for each corner.
    """
    def __init__(self):
        self.filters = [HybridFilter() for _ in range(4)]
        self.is_initialized = False
        
    def update(self, tag_corners):
        """
        Update all corner positions.
        
        Args:
            tag_corners: Array of 4 corner positions [[x,y], [x,y], [x,y], [x,y]]
            
        Returns:
            Smoothed corner positions
        """
        if not self.is_initialized:
            for i in range(4):
                self.filters[i].kalman.state[:2] = tag_corners[i]
                self.filters[i].ema.value = tag_corners[i]
            self.is_initialized = True
            return tag_corners
        
        smoothed_corners = []
        for i in range(4):
            smoothed = self.filters[i].update(tag_corners[i])
            smoothed_corners.append(smoothed)
        
        return np.array(smoothed_corners, dtype=np.float32)

# Constants
SIDE = 400
WARP_DST = np.array([[0, 0], [SIDE-1, 0], [SIDE-1, SIDE-1], [0, SIDE-1]], dtype="float32")

TAG_GRID_SIZE = 8
TAG_BORDER_WIDTH = 1
CENTER_PROXIMITY_THRESH_SQUARED = 400
CORE_START_CELL = 2
CORE_END_CELL = 6

class OBJ:
    """Simple OBJ loader for 3D models"""
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
    """
    Render 3D object on AR tag with auto-normalization.
    """
    if obj is None or len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img
    
    # --- AUTO-NORMALIZATION ---
    vertices_3d = obj.vertices.copy()
    
    # 1. Center the model at (0,0,0)
    min_v = np.min(vertices_3d, axis=0)
    max_v = np.max(vertices_3d, axis=0)
    center = (min_v + max_v) / 2
    size_range = np.max(max_v - min_v)
    
    if size_range == 0: size_range = 1.0
    
    # 2. Scale to Unit Box (-0.5 to 0.5)
    vertices_3d = (vertices_3d - center) / size_range
    
    # 3. Apply the user requested scale 
    # (scale=3.0 means 3x the tag_size)
    vertices_3d *= scale * (tag_size / 2.0)
    
    # Convert to homogeneous coordinates
    n = len(vertices_3d)
    vertices_homo = np.column_stack([vertices_3d, np.ones(n)])  # N x 4
    
    # Project: (3x4) @ (4xN).T = 3xN
    try:
        projected = projection @ vertices_homo.T
    except Exception:
        return img
    
    # Normalize by w coordinate
    w = projected[2, :]
    
    # Filter points behind camera
    valid_mask = w > 0.001 
    if not np.any(valid_mask):
        return img 
    
    # Safe division
    with np.errstate(divide='ignore', invalid='ignore'):
        x_2d = projected[0, :] / w
        y_2d = projected[1, :] / w
    
    points_2d = np.column_stack([x_2d, y_2d]).astype(np.int32)
    h, w_img = img.shape[:2]
    
    # Render faces
    for face_indices in obj.faces:
        if not all(valid_mask[i] for i in face_indices if i < len(valid_mask)):
            continue
        
        try:
            face_points = points_2d[face_indices]
            
            # Frustum culling (skip huge faces)
            if (np.all(face_points[:, 0] < -2000) or np.all(face_points[:, 0] > w_img + 2000) or
                np.all(face_points[:, 1] < -2000) or np.all(face_points[:, 1] > h + 2000)):
                continue
            
            CustomCV2.fillConvexPoly(img, face_points, color)
        except Exception:
            continue
    
    return img

def render_with_lighting(img, obj, projection, camera_matrix, tag_size=2.0, 
                          base_color=(120, 120, 120), scale=3.0):
    """
    Enhanced rendering with simple depth-based shading.
    """
    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        return img
    
    # Prepare vertices
    vertices_3d = obj.vertices.copy() * scale
    
    # Center model
    z_min = vertices_3d[:, 2].min()
    vertices_3d[:, 2] -= z_min
    x_center = (vertices_3d[:, 0].max() + vertices_3d[:, 0].min()) / 2
    y_center = (vertices_3d[:, 1].max() + vertices_3d[:, 1].min()) / 2
    vertices_3d[:, 0] -= x_center
    vertices_3d[:, 1] -= y_center
    
    # To homogeneous
    n = len(vertices_3d)
    vertices_homo = np.column_stack([vertices_3d, np.ones(n)])
    
    # Project
    projected = projection @ vertices_homo.T
    w = projected[2, :]
    
    valid_mask = w > 0.1
    if not np.any(valid_mask):
        return img
    
    x_2d = projected[0, :] / w
    y_2d = projected[1, :] / w
    points_2d = np.column_stack([x_2d, y_2d]).astype(np.int32)
    
    h, w_img = img.shape[:2]
    
    # Collect faces with depth for sorting
    face_data = []
    
    for face_indices in obj.faces:
        if not all(valid_mask[i] for i in face_indices if i < len(valid_mask)):
            continue
        
        try:
            face_points = points_2d[face_indices]
            
            # Skip if outside
            if (np.all(face_points[:, 0] < -500) or np.all(face_points[:, 0] > w_img + 500) or
                np.all(face_points[:, 1] < -500) or np.all(face_points[:, 1] > h + 500)):
                continue
            
            # Calculate average depth
            face_depths = w[face_indices]
            avg_depth = np.mean(face_depths)
            
            face_data.append((avg_depth, face_points))
        except:
            continue
    
    # Sort by depth (back to front - painter's algorithm)
    face_data.sort(reverse=True, key=lambda x: x[0])
    
    # Render sorted faces with depth shading
    for depth, face_points in face_data:
        # Simple depth-based shading
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

def compute_3d_orientation(camera_matrix, homography):
    """
    Compute full 3D orientation (roll, pitch, yaw) from homography.
    
    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        homography: 3x3 homography matrix
        
    Returns:
        roll, pitch, yaw in degrees
    """
    # Get rotation matrix from homography
    K_inv = np.linalg.inv(camera_matrix)
    RT = np.dot(K_inv, homography)
    
    # Normalize
    norm_1 = np.linalg.norm(RT[:, 0])
    norm_2 = np.linalg.norm(RT[:, 1])
    norm = (norm_1 + norm_2) / 2.0
    
    if norm < 1e-6:
        return 0, 0, 0
    
    RT = RT / norm
    
    # Extract rotation vectors
    r1 = RT[:, 0]
    r2 = RT[:, 1]
    r3 = np.cross(r1, r2)
    
    # Construct rotation matrix
    R = np.column_stack((r1, r2, r3))
    
    # Extract Euler angles (ZYX convention)
    # Roll (rotation around X-axis)
    roll = np.arctan2(R[2, 1], R[2, 2])
    
    # Pitch (rotation around Y-axis)
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    
    # Yaw (rotation around Z-axis) - this is the in-plane rotation
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # Normalize to [0, 360)
    yaw_deg = yaw_deg % 360
    
    return roll_deg, pitch_deg, yaw_deg


def get_projection_matrix(camera_matrix, homography):
    """
    Compute projection matrix from homography with proper SVD orthogonalization.
    """
    try:
        K_inv = np.linalg.inv(camera_matrix)
    except np.linalg.LinAlgError:
        return None
        
    RT = np.dot(K_inv, homography)
    
    h1 = RT[:, 0]
    h2 = RT[:, 1]
    h3 = RT[:, 2]
    
    norm1 = np.linalg.norm(h1)
    norm2 = np.linalg.norm(h2)
    lambda_scale = (norm1 + norm2) / 2.0
    
    if lambda_scale < 1e-6:
        return None
        
    r1 = h1 / lambda_scale
    r2 = h2 / lambda_scale
    t = h3 / lambda_scale
    
    r3 = np.cross(r1, r2)
    
    Q = np.column_stack((r1, r2, r3))
    U, _, Vt = np.linalg.svd(Q)
    R = np.dot(U, Vt)
    
    # Fix determinant sign (reflection)
    if np.linalg.det(R) < 0:
        R = -R
        t = -t  # FIX: Flip translation if rotation is flipped
        
    # Enforce positive depth (Object must be in front of camera)
    if t[2] < 0:
        r1 = -r1
        r2 = -r2
        t = -t
        r3 = np.cross(r1, r2) # Recalculate r3
        Q = np.column_stack((r1, r2, r3))
        U, _, Vt = np.linalg.svd(Q)
        R = np.dot(U, Vt)

    return np.column_stack((R, t))

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def order_points(pts):
    """
    Order points in consistent order: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        pts: 4x2 array of corner points
        
    Returns:
        Ordered 4x2 array
    """
    pts = pts.reshape(4, 2)
    
    # Sort by x-coordinate
    x_sorted = pts[np.argsort(pts[:, 0])]
    
    # Split into left and right
    left = x_sorted[:2]
    right = x_sorted[2:]
    
    # Sort each by y-coordinate
    left = left[np.argsort(left[:, 1])]
    (tl, bl) = left
    
    right = right[np.argsort(right[:, 1])]
    (tr, br) = right
    
    return np.array([tl, tr, br, bl], dtype="float32")


def refine_corners(gray, corners):
    """
    Refine corner locations to sub-pixel accuracy.
    
    Args:
        gray: Grayscale image
        corners: Initial corner positions
        
    Returns:
        Refined corners
    """
    criteria = (CustomCV2.TERM_CRITERIA_EPS + CustomCV2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return CustomCV2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), criteria)


def check_border(processed, cell, margin):
    """
    Verify that tag has proper black border.
    
    Args:
        processed: Thresholded tag image
        cell: Size of one grid cell
        margin: Margin for sampling
        
    Returns:
        True if border is valid, False otherwise
    """
    side = processed.shape[0]
    indices = np.clip(((np.arange(TAG_BORDER_WIDTH) + 0.5) * cell).astype(int), 0, side - 1)
    
    # Check top and bottom borders
    for idx in indices:
        if processed[margin, idx] > 127:
            return False
        if processed[side - margin - 1, idx] > 127:
            return False
    
    # Check left and right borders
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
    """
    Decode AR tag from warped image.
    
    Args:
        warped_tag: Warped and aligned tag image (should be square)
    
    Returns:
        (tag_id, orientation) or (None, None) if invalid
    """
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
    
    # Orientation detection via erosion + grayscale fallback
    anchor_positions = [
        (get_center_coord(5), get_center_coord(5)),  # BR
        (get_center_coord(5), get_center_coord(2)),  # BL
        (get_center_coord(2), get_center_coord(2)),  # TL
        (get_center_coord(2), get_center_coord(5)),  # TR
    ]
    
    erode_k = max(3, int(cell * 0.15)) | 1  # ensure odd
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
        # Fallback: use grayscale brightness to pick brightest anchor
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
    
    # Decode data bits based on orientation
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
                # 2-layer decoding: re-threshold warped region to find inner tag
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
                        # Map inner corners back to original image space
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
                            break  # found inner tag, stop searching
                continue  # skip the normal path for this contour
            
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
    
    # Multi-scale adaptive thresholding — captures tags at different distances
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
        smoothed_corners = tag["corners"]
        tag['angle'] = compute_fine_grained_orientation(smoothed_corners, angle_granularity)
        
        # TASK 2: Overlay
        if template_img is not None:
            h_temp, w_temp = template_img.shape[:2]
            src_pts = np.array([[0, 0], [w_temp-1, 0], [w_temp-1, h_temp-1], [0, h_temp-1]], dtype="float32")
            H = CustomCV2.getPerspectiveTransform(src_pts, smoothed_corners)
            try:
                H_inv = np.linalg.inv(H)
                if CPP_AVAILABLE:
                    frame = custom_cv2_cpp.overlay_image_cpp(frame, template_img, H_inv)
                else:
                    frame = overlay_image_python(frame, template_img, H)
            except: pass

        # TASK 3: Render 3D object
        if obj_model is not None:
            # Center of coordinate system is (0,0)
            half_size = 1.0 
            obj_pts_2d = np.array([
                [-half_size, -half_size], 
                [ half_size, -half_size], 
                [ half_size,  half_size], 
                [-half_size,  half_size]
            ], dtype=np.float32)
            
            # Compute Homography for 3D
            H_3d = CustomCV2.getPerspectiveTransform(obj_pts_2d, smoothed_corners)
            
            # Check H_3d validity
            if H_3d is not None and np.all(np.isfinite(H_3d)) and np.linalg.det(H_3d) != 0:
                # get_projection_matrix returns [R|t], need K @ [R|t] for full projection
                Rt = get_projection_matrix(camera_matrix.astype(np.float64), H_3d)
                
                if Rt is not None and np.all(np.isfinite(Rt)):
                    projection = np.dot(camera_matrix.astype(np.float64), Rt)
                    
                    frame = render(frame, obj_model, projection, smoothed_corners, 
                                  tag_size=half_size * 2, color=(80, 80, 80), scale=2.5)
        
        if obj_model is None and template_img is None:  
            cv2.polylines(frame, [np.int32(smoothed_corners)], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {tag_id}", tuple(np.int32(smoothed_corners[0])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Angle: {tag['angle']:.1f}deg", 
                    tuple(np.int32(smoothed_corners[1]) + np.array([0, 30])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    
    return frame, detected_tags


def overlay_image_python(frame, overlay, H):
    """Python fallback for image overlay (slower than C++ version)."""
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