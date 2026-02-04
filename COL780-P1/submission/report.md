1) https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf

2) https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

3) https://en.wikipedia.org/wiki/Otsu%27s_method

4) pybind


-----
- warp prespective
- long long for 4k currently 1080

-----
Todo
- decode_tag
- temporal filters
- corner refinements
- Boxfilter vs gaussian blur
- blocksize of adaptive threshold
- SIMD
- > Optimize adaptiveThreshold Further ★★★★☆
> Current: C++ with integral image
> Proposal: SIMD vectorization for integral image computation
> cpp#include <immintrin.h>  // AVX2
> // Process 8 pixels at once with _mm256_add_epi32
-  > Early Rejection in Contour Processing ★★★★☆
> python# BEFORE contour approximation
> if CustomCV2.contourArea(cnt) < MIN_TAG_AREA or \
>    CustomCV2.contourArea(cnt) > MAX_TAG_AREA:
>     continue
> 
> # Add aspect ratio check
> bbox = cv2.boundingRect(cnt)
> aspect = max(w,h) / min(w,h)
> if aspect > 1.5:  # Tags are square
>     continue
-  > Memory Pooling for Intermediate Buffers ★★★☆☆
> cpp// Reuse warped tag buffers instead of allocating per tag
> thread_local std::vector<uint8_t> warp_buffer(SIDE * SIDE);
- getPerspectiveTransform
- ```
# CURRENT (DISABLED):
# rect = refine_corners(gray, rect)

# FIX - Implement cornerSubPix:
@staticmethod
def cornerSubPix(image, corners, winSize=(5,5), zeroZone=(-1,-1), criteria=None):
    """
    Sub-pixel corner refinement using gradient descent
    """
    refined = corners.copy().astype(np.float32)
    half_win = winSize[0]
    
    for i, corner in enumerate(refined):
        x, y = corner[0]
        x_int, y_int = int(round(x)), int(round(y))
        
        # Extract window
        x1 = max(0, x_int - half_win)
        y1 = max(0, y_int - half_win)
        x2 = min(image.shape[1], x_int + half_win + 1)
        y2 = min(image.shape[0], y_int + half_win + 1)
        
        window = image[y1:y2, x1:x2].astype(np.float32)
        
        # Compute gradients
        grad_x = np.gradient(window, axis=1)
        grad_y = np.gradient(window, axis=0)
        
        # Structure tensor
        Ixx = np.sum(grad_x * grad_x)
        Iyy = np.sum(grad_y * grad_y)
        Ixy = np.sum(grad_x * grad_y)
        
        # Compute shift
        det = Ixx * Iyy - Ixy * Ixy
        if abs(det) > 1e-6:
            # Iterative refinement (simplified)
            # Full implementation uses eigenvalues
            pass
        
        refined[i] = [x, y]  # Updated coordinates
    
    return refined
```


- ```
def adaptive_threshold_selection(gray):
    """
    Dynamically select threshold method based on image statistics
    """
    # Compute histogram
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    
    # Check bimodality (good for global threshold)
    hist_smooth = np.convolve(hist, np.ones(5)/5, mode='same')
    peaks = (hist_smooth[1:-1] > hist_smooth[:-2]) & \
            (hist_smooth[1:-1] > hist_smooth[2:])
    num_peaks = np.sum(peaks)
    
    # Measure lighting uniformity
    mean_vals = []
    for i in range(4):
        for j in range(4):
            h, w = gray.shape
            patch = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
            mean_vals.append(np.mean(patch))
    lighting_variance = np.std(mean_vals)
    
    # Decision tree
    if num_peaks >= 2 and lighting_variance < 30:
        # Good global threshold scenario
        _, thresh = CustomCV2.threshold(
            gray, 0, 255, 
            CustomCV2.THRESH_BINARY + CustomCV2.THRESH_OTSU
        )
    else:
        # Non-uniform lighting, use adaptive
        # Larger block size for gradual lighting changes
        blocksize = 21 if lighting_variance > 50 else 11
        thresh = CustomCV2.adaptiveThreshold(
            gray, 255, 
            CustomCV2.ADAPTIVE_THRESH_MEAN_C,
            CustomCV2.THRESH_BINARY_INV,
            blocksize, 7
        )
    
    return thresh
```

- ```
def decode_tag(warped_tag: np.ndarray):
    # ... existing code ...
    
    # CURRENT: Only checks if anchor is white
    if intensities[status] < 127:
        return None, None
    
    # ENHANCED: Verify anchor is uniquely brightest
    intensities_sorted = sorted(intensities, reverse=True)
    if intensities_sorted[0] - intensities_sorted[1] < 50:
        # Ambiguous orientation
        return None, None
    
    # Verify other corners are sufficiently dark
    other_corners = [intensities[i] for i in range(4) if i != status]
    if any(c > 100 for c in other_corners):
        # False orientation marker
        return None, None
    
    # ... rest of decoding ...
```

- ```
def get_core_val(r: int, c: int) -> float:
    """Sample center 40% of a core cell"""
    # CURRENT: Single sample region
    y_start = max(0, int(start + (r + 0.3) * core_cell))
    y_end = min(side, int(start + (r + 0.7) * core_cell))
    
    # ENHANCED: Multiple samples with voting
    samples = []
    for offset_r in [-0.1, 0, 0.1]:
        for offset_c in [-0.1, 0, 0.1]:
            y_s = int(start + (r + 0.3 + offset_r) * core_cell)
            y_e = int(start + (r + 0.7 + offset_r) * core_cell)
            x_s = int(start + (c + 0.3 + offset_c) * core_cell)
            x_e = int(start + (c + 0.7 + offset_c) * core_cell)
            
            if y_e > y_s and x_e > x_s:
                samples.append(
                    np.mean(thresh[y_s:y_e, x_s:x_e])
                )
    
    # Return median for noise robustness
    return np.median(samples) if samples else 0
```

- ```
def validate_tag_geometry(corners):
    """
    Verify tag has square-like geometry
    """
    # Compute side lengths
    sides = [
        np.linalg.norm(corners[i] - corners[(i+1)%4])
        for i in range(4)
    ]
    
    # Check side length consistency
    side_mean = np.mean(sides)
    side_std = np.std(sides)
    if side_std / side_mean > 0.2:  # 20% variation
        return False
    
    # Check angles (should be ~90 degrees)
    angles = []
    for i in range(4):
        v1 = corners[i] - corners[(i-1)%4]
        v2 = corners[(i+1)%4] - corners[i]
        cos_angle = np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2)
        )
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        angles.append(np.degrees(angle))
    
    # All angles should be 70-110 degrees
    if not all(70 < a < 110 for a in angles):
        return False
    
    return True

# Usage in process_frame:
if len(quad) == 4 and CustomCV2.isContourConvex(quad):
    if not validate_tag_geometry(quad.reshape(4, 2)):
        continue
```

- ```
def match_tags_between_frames(prev_tags, curr_tags, max_dist=50):
    """
    Match tags by position to maintain consistent IDs
    """
    if not prev_tags:
        return curr_tags
    
    matched = []
    unmatched_curr = list(curr_tags)
    
    for p_tag in prev_tags:
        p_center = np.mean(p_tag['corners'], axis=0)
        
        best_match = None
        best_dist = max_dist
        
        for c_tag in unmatched_curr:
            c_center = np.mean(c_tag['corners'], axis=0)
            dist = np.linalg.norm(c_center - p_center)
            
            if dist < best_dist and c_tag['id'] == p_tag['id']:
                best_match = c_tag
                best_dist = dist
        
        if best_match:
            matched.append(best_match)
            unmatched_curr.remove(best_match)
    
    # Add new detections
    matched.extend(unmatched_curr)
    return matched
```

- kalman filtering

- ```from filterpy.kalman import KalmanFilter

class TagTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=8)
        # State: [x1, y1, x2, y2, x3, y3, x4, y4]
        # No velocity tracking (static scene assumption)
        self.kf.F = np.eye(8)  # State transition
        self.kf.H = np.eye(8)  # Measurement
        self.kf.R *= 2  # Measurement noise
        self.kf.P *= 10  # Initial uncertainty
    
    def update(self, corners):
        """
        corners: (4, 2) array
        """
        measurement = corners.flatten()
        self.kf.update(measurement)
        self.kf.predict()
        return self.kf.x.reshape(4, 2)
        ```

- ```
std::pair<int, int> decode_tag_cpp(
    const uint8_t* warped, int side
) {
    // 1. Fast threshold (single pass)
    const int THRESH = 155;
    std::vector<uint8_t> thresh_data(side * side);
    
    #pragma omp parallel for
    for (int i = 0; i < side * side; i++) {
        thresh_data[i] = (warped[i] > THRESH) ? 255 : 0;
    }
    
    // 2. Border validation (vectorized)
    const float cell = side / 8.0f;
    const int margin = std::max(1, (int)(cell / 2));
    
    // Sample 32 border points
    int border_sum = 0;
    for (int i = 0; i < 8; i++) {
        int idx = (int)((i + 0.5) * cell);
        idx = std::min(idx, side - 1);
        
        border_sum += thresh_data[margin * side + idx];
        border_sum += thresh_data[idx * side + (side - margin - 1)];
        border_sum += thresh_data[(side - margin - 1) * side + idx];
        border_sum += thresh_data[idx * side + margin];
    }
    
    if (border_sum > 32 * 150) {
        return {-1, -1};  // Invalid
    }
    
    // 3. Core decoding (unrolled loops)
    const int start = (int)(2 * cell);
    const int core_size = (int)(4 * cell);
    const float core_cell = core_size / 4.0f;
    
    auto sample_core = [&](int r, int c) -> int {
        int y_start = start + (int)((r + 0.3f) * core_cell);
        int y_end = start + (int)((r + 0.7f) * core_cell);
        int x_start = start + (int)((c + 0.3f) * core_cell);
        int x_end = start + (int)((c + 0.7f) * core_cell);
        
        int sum = 0, count = 0;
        for (int y = y_start; y < y_end; y++) {
            for (int x = x_start; x < x_end; x++) {
                sum += thresh_data[y * side + x];
                count++;
            }
        }
        return sum / count;
    };
    
    // Find orientation
    int anchors[4] = {
        sample_core(3, 3),
        sample_core(3, 0),
        sample_core(0, 0),
        sample_core(0, 3)
    };
    
    int status = 0, max_val = anchors[0];
    for (int i = 1; i < 4; i++) {
        if (anchors[i] > max_val) {
            max_val = anchors[i];
            status = i;
        }
    }
    
    if (max_val < 127) return {-1, -1};
    
    // Decode bits
    const int bit_map[4][4][2] = {
        {{1,1}, {1,2}, {2,2}, {2,1}},
        {{1,2}, {2,2}, {2,1}, {1,1}},
        {{2,2}, {2,1}, {1,1}, {1,2}},
        {{2,1}, {1,1}, {1,2}, {2,2}}
    };
    
    int tag_id = 0;
    for (int i = 0; i < 4; i++) {
        int val = sample_core(
            bit_map[status][i][0],
            bit_map[status][i][1]
        );
        if (val > 127) {
            tag_id |= (1 << i);
        }
    }
    
    return {tag_id, status * 90};
}
```

- ```
def order_points_fast(pts):
    """
    Vectorized corner ordering
    """
    pts = pts.reshape(4, 2)
    
    # Compute center
    center = np.mean(pts, axis=0)
    
    # Compute angles from center
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    
    # Ensure top-left is first
    sorted_pts = pts[sorted_indices]
    top_left_idx = np.argmin(np.sum(sorted_pts, axis=1))
    
    # Roll to make top-left first
    return np.roll(sorted_pts, -top_left_idx, axis=0).astype(np.float32)
```

- ```Cache-Friendly Memory Layout ★★☆☆☆
cpp// Current: Row-major access in vertical pass
// Proposal: Tiled processing to improve cache locality

const int TILE_SIZE = 64;
for (int y_tile = 0; y_tile < h; y_tile += TILE_SIZE) {
    for (int x_tile = 0; x_tile < w; x_tile += TILE_SIZE) {
        // Process tile
    }
}```