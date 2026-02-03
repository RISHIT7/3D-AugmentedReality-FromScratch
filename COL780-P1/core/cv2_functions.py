import numpy as np
from typing import Tuple, List, Optional, Any
import math

GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)

class CustomCV2:
    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_OTSU = 8
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
    ADAPTIVE_THRESH_MEAN_C = 0
    RETR_TREE = 3
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 2
    CHAIN_APPROX_NONE = 1
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_MAX_ITER = 1
    COLOR_BGR2GRAY = 6
    FONT_HERSHEY_SIMPLEX = 0
    NORM_MINMAX = 32
    INTER_NEAREST = 0
    INTER_LINEAR = 1
    
    # ---- self implemented temporal OTSU --------
    THRESH_OTSU = 8
    THRESH_TEMPORAL_APPROX_OTSU = 16
    _EMA_THRESH = 155
    _MOMENTUM = 0.0
    _BETA1 = 0.9
    _ALPHA = 0.2
    _FRAME_COUNT = 0
    _UPDATE_FREQ = 10

    @staticmethod
    def cvtColor(src: np.ndarray, code: int) -> np.ndarray:
        if code == CustomCV2.COLOR_BGR2GRAY:
            if src.ndim == 2:
                return src.copy()
            elif src.ndim == 3 and src.shape[2] == 3:
                gray = np.dot(src[..., :3], GRAY_WEIGHTS)
                return gray.astype(np.uint8)
            else:
                raise ValueError("Input image must have 2 or 3 channels for BGR2GRAY conversion")
        else:
            raise NotImplementedError("cvtColor code not implemented")

    @staticmethod
    def BoxFilter(src: np.ndarray, ksize: Tuple[int, int]) -> np.ndarray:
        """
        Highly optimized Box Filter using moving sums (O(1) per pixel relative to kernel size).
        """
        kx, ky = ksize
        # Ensure source is float for precision during accumulation
        res = src.astype(np.float32)
        
        # 1. Horizontal Box Sum
        # cumsum + slice-subtraction is significantly faster than np.convolve
        res = np.cumsum(res, axis=1)
        res[:, kx:] = res[:, kx:] - res[:, :-kx]
        res = res[:, kx-1:] / kx
        
        # 2. Vertical Box Sum
        res = np.cumsum(res, axis=0)
        res[ky:, :] = res[ky:, :] - res[:-ky, :]
        res = res[ky-1:, :] / ky

        # Padding to maintain 'same' mode to match your current implementation
        pad_h = ky // 2
        pad_w = kx // 2
        return np.pad(res, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(src.dtype)

    @staticmethod
    def _make_gaussian_kernel(ksize: int, sigmaX: float) -> np.ndarray:
        ax = np.linspace(-(ksize // 2), ksize // 2, ksize)
        kernel = np.exp(-0.5 * (ax / sigmaX) **2)
        return kernel / kernel.sum()

    @staticmethod
    def _convolve1d(src: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode='same'), axis, src
        )

    @staticmethod
    def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float) -> np.ndarray:
        kx, ky = ksize
        kernel_x = CustomCV2._make_gaussian_kernel(kx, sigmaX)
        kernel_y = CustomCV2._make_gaussian_kernel(ky, sigmaX)

        results = src.astype(np.float32)
        if src.ndim == 2:
            results = CustomCV2._convolve1d(CustomCV2._convolve1d(results, kernel_x, axis=1), kernel_y, axis=0)
        else:
            for c in range(src.shape[2]):
                results[:, :, c] = CustomCV2._convolve1d(
                    CustomCV2._convolve1d(results[:, :, c], kernel_x, axis=1), kernel_y, axis=0
                )
        return np.clip(results, 0, 255).astype(src.dtype)

    @staticmethod
    def findCornersQuadrilateral(contour: np.ndarray) -> np.ndarray:
        """
        Finds the 4 corners of a quadrilateral using Radial Distance Peak Detection.
        More stable than approxPolyDP for AR tags.
        """
        # 1. Standardize Input
        if contour.ndim == 3:
            pts = contour.reshape(-1, 2)
        else:
            pts = contour

        if len(pts) < 4:
            return pts.reshape(-1, 1, 2)

        # 2. Centroid Calculation (Fast Mean)
        # We use the geometric center of the contour points
        centroid = np.mean(pts, axis=0)

        # 3. Distance Signal
        # Compute squared Euclidean distance from center to every boundary point
        diff = pts - centroid
        dists_sq = np.sum(diff**2, axis=1)

        # 4. Signal Smoothing
        # Apply a small convolution to remove pixel noise (jagged edges)
        # which creates fake local peaks.
        kernel_size = 5
        # Wrap padding handles the closed loop structure
        dists_padded = np.pad(dists_sq, (kernel_size//2, kernel_size//2), mode='wrap')
        dists_smooth = np.convolve(dists_padded, np.ones(kernel_size)/kernel_size, mode='valid')

        # 5. Peak Detection (Vectorized)
        # A point is a peak if it is further than its neighbors
        prev_pts = np.roll(dists_smooth, 1)
        next_pts = np.roll(dists_smooth, -1)
        
        # Local Maxima mask
        peaks_mask = (dists_smooth > prev_pts) & (dists_smooth > next_pts)
        peak_indices = np.where(peaks_mask)[0]

        # 6. Filter & Fallback
        if len(peak_indices) > 4:
            # If noise created extra peaks, pick the 4 farthest from center
            peak_dists = dists_smooth[peak_indices]
            # argsort is ascending, so we take the last 4
            top_indices = np.argsort(peak_dists)[-4:]
            final_indices = peak_indices[top_indices]
        elif len(peak_indices) < 4:
            # Fallback to simple bounding box if shape is too distorted
            # (Rare for valid AR tags)
            return CustomCV2.approxPolyDP(contour, 0.02 * CustomCV2.arcLength(contour, True), True)
        else:
            final_indices = peak_indices

        corners = pts[final_indices]
        
        # Return in (4, 1, 2) format to match OpenCV standards
        return corners.reshape(-1, 1, 2).astype(np.int32)

    @staticmethod
    def threshold(src: np.ndarray, thresh: float, maxval: float, type: int) -> Tuple[float, np.ndarray]:
        if type & CustomCV2.THRESH_OTSU:
            hist, _ = np.histogram(src, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            
            bin_centers = np.arange(256)
            
            w_bg = np.cumsum(hist)           
            sum_bg = np.cumsum(hist * bin_centers) 
            
            total_pixels = src.size
            total_sum = sum_bg[-1]
            
            w_fg = total_pixels - w_bg
            sum_fg = total_sum - sum_bg
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_bg = sum_bg / w_bg
                mean_fg = sum_fg / w_fg
                
                inter_class_variance = w_bg * w_fg * (mean_bg - mean_fg) ** 2
            
            inter_class_variance[np.isnan(inter_class_variance)] = 0
            thresh = float(np.argmax(inter_class_variance))
            
            type &= ~CustomCV2.THRESH_OTSU

        if type == CustomCV2.THRESH_TEMPORAL_APPROX_OTSU:
            if CustomCV2._FRAME_COUNT % CustomCV2._UPDATE_FREQ == 0:
                hist, _ = np.histogram(src, bins=256, range=(0, 256))
                hist = hist.astype(np.float64)
                bin_centers = np.arange(256)
                
                w_bg = np.cumsum(hist)
                total_px = src.size
                w_fg = total_px - w_bg
                
                sum_bg = np.cumsum(hist * bin_centers)
                total_sum = sum_bg[-1]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    variance = w_bg * w_fg * (sum_bg/w_bg - (total_sum - sum_bg)/w_fg) ** 2
                    variance[np.isnan(variance)] = 0
                    ideal_thresh = float(np.argmax(variance))                

                # diff = float(ideal_thresh) - CustomCV2._EMA_THRESH
                # CustomCV2._MOMENTUM = (CustomCV2._BETA1 * CustomCV2._MOMENTUM + 
                #                     (1 - CustomCV2._BETA1) * diff)
                # CustomCV2._EMA_THRESH += CustomCV2._ALPHA * CustomCV2._MOMENTUM
                # CustomCV2._EMA_THRESH = (CustomCV2._BETA1 * CustomCV2._EMA_THRESH +
                #                         (1 - CustomCV2._BETA1) * ideal_thresh)
                CustomCV2._EMA_THRESH = ideal_thresh
                # print(f"Updated EMA Threshold to: {CustomCV2._EMA_THRESH:.2f}")
            
            thresh = CustomCV2._EMA_THRESH
            CustomCV2._FRAME_COUNT += 1
            
            type &= ~CustomCV2.THRESH_TEMPORAL_APPROX_OTSU

        if type == CustomCV2.THRESH_BINARY:
            result = np.where(src > thresh, maxval, 0)
        elif type == CustomCV2.THRESH_BINARY_INV:
            result = np.where(src > thresh, 0, maxval)
        else:
            result = src.copy()
            
        return thresh, result.astype(np.uint8)

    @staticmethod
    def adaptiveThreshold(src: np.ndarray, maxValue: float, adaptiveMethod: int,
                         thresholdType: int, blockSize: int, C: float) -> np.ndarray:
        """
        Applies adaptive thresholding.
        Supports both ADAPTIVE_THRESH_MEAN_C (via Integral Image) and 
        ADAPTIVE_THRESH_GAUSSIAN_C (via Convolution).
        """
        if blockSize % 2 == 0:
            blockSize += 1  # Block size must be odd
            
        if adaptiveMethod == CustomCV2.ADAPTIVE_THRESH_GAUSSIAN_C:
            # OpenCV formula to derive sigma from ksize if not provided
            sigma = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8
            mean = CustomCV2.GaussianBlur(src, (blockSize, blockSize), sigma)
            # Depending on precision, we might want mean to be float
            mean = mean.astype(np.float32)
            
        else:
            # ADAPTIVE_THRESH_MEAN_C
            # Use the fast Integral Image (Box Filter) approach
            h, w = src.shape
            half = blockSize // 2
            
            # Pad and compute integral image
            integral = np.pad(src.astype(np.float64).cumsum(axis=0).cumsum(axis=1), 
                             ((1, 1), (1, 1)), mode='edge') 
            
            y = np.arange(h)
            x = np.arange(w)
            
            y1 = np.maximum(0, y - half)
            y2 = np.minimum(h, y + half + 1)
            x1 = np.maximum(0, x - half)
            x2 = np.minimum(w, x + half + 1)

            block_sum = (integral[y2[:, None], x2] - integral[y1[:, None], x2] - 
                        integral[y2[:, None], x1] + integral[y1[:, None], x1])

            counts = (y2[:, None] - y1[:, None]) * (x2 - x1)
            mean = (block_sum / counts)

        # Apply the threshold calculation: T(x,y) = mean(x,y) - C
        diff = src.astype(np.float32) - (mean - C)

        if thresholdType == CustomCV2.THRESH_BINARY_INV:
            output = np.where(diff <= 0, maxValue, 0)
        else:
            output = np.where(diff > 0, maxValue, 0)
            
        return output.astype(np.uint8)

    @staticmethod
    def findContours(image: np.ndarray, mode: int = 0, method: int = 1) -> List[np.ndarray]:
        binary = (image > 0).astype(np.uint8)
        binary = np.pad(binary, 1, mode='constant', constant_values=0)
        h, w = binary.shape
        
        visited = np.zeros_like(binary, dtype=bool)
        contours = []
        
        OFFSETS_Y = [-1, -1, 0, 1, 1, 1, 0, -1]
        OFFSETS_X = [0, 1, 1, 1, 0, -1, -1, -1]
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if binary[y, x] == 1 and not visited[y, x]:
                    is_border = False
                    if binary[y-1, x] == 0 or binary[y+1, x] == 0 or \
                       binary[y, x-1] == 0 or binary[y, x+1] == 0:
                        is_border = True
                        
                    if is_border:
                        contour_pts = []
                        cy, cx = y, x
                        backtrack = 0
                        
                        while True:
                            contour_pts.append([(cx - 1), (cy - 1)])
                            visited[cy, cx] = True
                            
                            found = False
                            for i in range(8):
                                idx = (backtrack + i) % 8
                                dy = OFFSETS_Y[idx]
                                dx = OFFSETS_X[idx]
                                ny, nx = cy + dy, cx + dx
                                
                                if binary[ny, nx] == 1:
                                    cy, cx = ny, nx
                                    backtrack = (idx + 5) % 8
                                    found = True
                                    break
                            
                            if not found or (cy == y and cx == x):
                                break
                        
                        if len(contour_pts) > 2:
                            contours.append(np.array(contour_pts, dtype=np.int32).reshape(-1, 1, 2))

        return contours, None    

    @staticmethod
    def Sobel(src: np.ndarray, threshold: int = 0) -> np.ndarray:
        """
        Highly optimized Sobel Edge Detection (3x3).
        Uses vectorized slicing and separable kernels for maximum FPS.
        Returns: Gradient Magnitude (L1 Norm approximation for speed).
        """
        img = src.astype(np.int16)
        p = np.pad(img, 1, mode='reflect')
        smooth_y = p[:-2, :] + (2 * p[1:-1, :]) + p[2:, :]
        
        gx = smooth_y[:, 2:] - smooth_y[:, :-2]
        smooth_x = p[:, :-2] + (2 * p[:, 1:-1]) + p[:, 2:]
        
        gy = smooth_x[2:, :] - smooth_x[:-2, :]
        
        magnitude = np.abs(gx) + np.abs(gy)
        
        # binarize edges
        magnitude = np.where(magnitude > threshold, 255, 0)

        return magnitude.astype(np.uint8)

    @staticmethod
    def contourArea(contour: np.ndarray, oriented: bool = False) -> float:
        if contour.ndim == 3:
            pts = contour.reshape(-1, 2)
        else:
            pts = contour

        if len(pts) < 3:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        area = 0.5 * (np.dot(x, y_next) - np.dot(x_next, y))

        if oriented:
            return float(area)
        else:
            return abs(float(area))
    
    @staticmethod
    def arcLength(curve: np.ndarray, closed: bool) -> float:
        if curve.ndim == 3:
            pts = curve.reshape(-1, 2)
        else:
            pts = curve

        if len(pts) < 2:
            return 0.0
        
        if closed:
            diffs = pts - np.roll(pts, -1, axis=0)
        else:
            diffs = pts[:-1] - pts[1:]

        distances = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
        
        return float(np.sum(distances))
    
    @staticmethod
    def approxPolyDP(curve: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
        if curve.ndim == 3:
            pts = curve.reshape(-1, 2).copy()
        else:
            pts = curve.copy()

        if len(pts) < 3:
            return curve.copy()
        
        def perpendicular_distance(pt, line_start, line_end):
            line_vec = line_end - line_start
            line_len_sq = np.dot(line_vec, line_vec)
            if line_len_sq < 1e-10:
                return np.linalg.norm(pt - line_start)
            
            point_vec = pt - line_start
            t = np.dot(point_vec, line_vec) / line_len_sq
            t = np.clip(t, 0.0, 1.0)

            closest = line_start + t * line_vec
            return np.linalg.norm(pt - closest)
        
        def recurse(start, end, keep_mask):
            if end - start <= 1:
                return
            
            max_dist = 0.0
            max_idx = start

            for i in range(start + 1, end):
                dist = perpendicular_distance(pts[i], pts[start], pts[end])
                if dist > max_dist:
                    max_dist = dist
                    max_idx = i
            
            if max_dist > epsilon:
                keep_mask[max_idx] = True
                recurse(start, max_idx, keep_mask)
                recurse(max_idx, end, keep_mask)

        n = len(pts)
        keep_mask = np.zeros(n, dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True

        recurse(0, n - 1, keep_mask)
        
        approx = pts[keep_mask]

        if len(approx) > 4 and np.allclose(approx[0], approx[-1], atol=1.0) and closed:
            approx = approx[:-1]

        return approx.reshape(-1, 1, 2).astype(np.int32)
    
    @staticmethod
    def isContourConvex(contour: np.ndarray) -> bool:
        if contour.ndim == 3:
            pts = contour.reshape(-1, 2)
        else:
            pts = contour

        n = len(pts)
        if n < 4:
            return True
        
        sign = 0
        for i in range(n):
            d1 = pts[(i + 1) % n] - pts[i]
            d2 = pts[(i + 2) % n] - pts[(i + 1) % n]
            cross = d1[0] * d2[1] - d1[1] * d2[0]

            if abs(cross) < 1e-10:
                continue

            current_sign = np.sign(cross)
            if current_sign != 0:
                if sign == 0:
                    sign = current_sign
                elif sign != current_sign:
                    return False
        return True

    @staticmethod
    def moments(contour: np.ndarray, binaryImage: bool = False) -> dict:
        if contour.ndim == 3:
            pts = contour.reshape(-1, 2).astype(np.float64)
        else:
            pts = contour.astype(np.float64)
        
        n = len(pts)
        
        # Initialize dictionary with zeros
        mo = {k: 0.0 for k in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03',
                               'mu20', 'mu11', 'mu02', 'mu30', 'mu21', 'mu12', 'mu03',
                               'nu20', 'nu11', 'nu02', 'nu30', 'nu21', 'nu12', 'nu03']}
        
        if n < 3:
            return mo
        
        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        
        # Standard Shoelace component
        a = x * y_next - x_next * y
        signed_area = 0.5 * np.sum(a)
        
        if abs(signed_area) < 1e-9:
            return mo

        # Calculate spatial moments
        # We take the absolute of the final sum for m00 to match OpenCV's non-oriented behavior
        mo["m00"] = abs(signed_area)
        
        # m10 and m01 must be adjusted by the sign of the area to remain consistent
        # regardless of clockwise or counter-clockwise point ordering.
        area_sign = 1.0 if signed_area > 0 else -1.0
        
        mo["m10"] = (1/6) * np.sum((x + x_next) * a) * area_sign
        mo["m01"] = (1/6) * np.sum((y + y_next) * a) * area_sign
        
        # Note: If you need m20, m02, etc., they also require the area_sign adjustment.
        
        return mo
    
    @staticmethod
    def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Calculates the 3x3 perspective transform matrix (Homography) 
        that maps points 'src' to 'dst'.
        Solves the system: dst_i = M * src_i
        """
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Source and Destination must contain exactly 4 points")

        # We need to solve for 8 coefficients (h22 is fixed to 1)
        # The system is Ah = b
        A = np.zeros((8, 8), dtype=np.float64)
        b = np.zeros((8), dtype=np.float64)

        for i in range(4):
            x, y = src[i]
            u, v = dst[i]
            
            # Equation 1 for x-coordinate (u)
            # h00*x + h01*y + h02 - h20*x*u - h21*y*u = u
            A[2*i] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
            b[2*i] = u
            
            # Equation 2 for y-coordinate (v)
            # h10*x + h11*y + h12 - h20*x*v - h21*y*v = v
            A[2*i+1] = [0, 0, 0, x, y, 1, -x*v, -y*v]
            b[2*i+1] = v

        # Solve linear system
        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.eye(3)

        # Reshape to 3x3 matrix (append h22 = 1)
        M = np.append(h, 1.0).reshape(3, 3)
        return M
    
    @staticmethod
    def warpPerspective(src: np.ndarray, M: np.ndarray, dsize: Tuple[int, int],
                       flags: int = INTER_LINEAR) -> np.ndarray:
        """
        Applies perspective transformation using Backward Mapping (Inverse warping).
        Vectorized for performance (avoids slow Python loops).
        """
        width, height = dsize
        
        # 1. Invert the matrix because we need to map Destination(x,y) -> Source(u,v)
        # to interpolate pixel values.
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.zeros((height, width), dtype=src.dtype)

        # 2. Create a grid of coordinates for the Destination image
        # x_coords: [[0, 1, 2...], [0, 1, 2...]]
        # y_coords: [[0, 0, 0...], [1, 1, 1...]]
        x_idxs, y_idxs = np.meshgrid(np.arange(width), np.arange(height))
        
        # 3. Apply Perspective Equation to the grid: 
        # src_x = (m00*x + m01*y + m02) / (m20*x + m21*y + m22)
        # src_y = (m10*x + m11*y + m12) / (m20*x + m21*y + m22)
        
        # Pre-extract matrix values for speed
        h00, h01, h02 = M_inv[0]
        h10, h11, h12 = M_inv[1]
        h20, h21, h22 = M_inv[2]

        # Calculate denominator (w')
        denom = h20 * x_idxs + h21 * y_idxs + h22
        # Avoid division by zero
        denom[denom == 0] = 1e-9
        
        # Calculate source coordinates
        src_x = (h00 * x_idxs + h01 * y_idxs + h02) / denom
        src_y = (h10 * x_idxs + h11 * y_idxs + h12) / denom
        
        # 4. Bilinear Interpolation (Vectorized)
        h_src, w_src = src.shape[:2]
        src_x = np.clip(src_x, 0, w_src - 1)
        src_y = np.clip(src_y, 0, h_src - 1)
        
        # Get integer and fractional parts
        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Calculate weights
        dx = src_x - x0
        dy = src_y - y0
        
        # Clamp coordinates to image bounds
        x0 = np.clip(x0, 0, w_src - 1)
        x1 = np.clip(x1, 0, w_src - 1)
        y0 = np.clip(y0, 0, h_src - 1)
        y1 = np.clip(y1, 0, h_src - 1)
        
        # Sample pixels
        # If image is grayscale (2D) vs Color (3D)
        if src.ndim == 2:
            Ia = src[y0, x0]
            Ib = src[y1, x0]
            Ic = src[y0, x1]
            Id = src[y1, x1]
            
            # Bilinear Formula: 
            # val = Ia*(1-dx)(1-dy) + Ib*(1-dx)dy + Ic*dx(1-dy) + Id*dx*dy
            wa = (1 - dx) * (1 - dy)
            wb = (1 - dx) * dy
            wc = dx * (1 - dy)
            wd = dx * dy
            
            warped = (Ia * wa + Ib * wb + Ic * wc + Id * wd)
            
        else:
            # For 3 Channel images, operations broadcast automatically
            Ia = src[y0, x0]
            Ib = src[y1, x0]
            Ic = src[y0, x1]
            Id = src[y1, x1]
            
            wa = (1 - dx) * (1 - dy)
            wb = (1 - dx) * dy
            wc = dx * (1 - dy)
            wd = dx * dy
            
            # Reshape weights for broadcasting: (H, W) -> (H, W, 1)
            warped = (Ia * wa[..., None] + Ib * wb[..., None] + 
                      Ic * wc[..., None] + Id * wd[..., None])

        return warped.astype(np.uint8)
    
    @staticmethod
    def cornerSubPix(image: np.ndarray, corners: np.ndarray, winSize: Tuple[int, int],
                     zeroZone: Tuple[int, int], criteria: Tuple[int, int, float]) -> np.ndarray:
        """
        Refine corner locations to sub-pixel accuracy.
        
        Args:
            image: Input grayscale image
            corners: Initial corner coordinates
            winSize: Half of search window side length
            zeroZone: Dead region in the middle of search zone
            criteria: Termination criteria
            
        Returns:
            Refined corners
        """
        raise NotImplementedError("cornerSubPix needs implementation")
    
    @staticmethod
    def resize(src: np.ndarray, dsize: Tuple[int, int], interpolation: int = INTER_LINEAR) -> np.ndarray:
        dst_w, dst_h = dsize
        src_h, src_w = src.shape[:2]

        if interpolation == CustomCV2.INTER_NEAREST:
            row_ratio = src_h / dst_h
            col_ratio = src_w / dst_w

            dst_y, dst_x = np.indices((dst_h, dst_w))
            src_x = np.clip((dst_x * col_ratio).astype(np.int32),
                            0, src_w - 1)
            src_y = np.clip((dst_y * row_ratio).astype(np.int32),
                            0, src_h - 1)
            
            return src[src_y, src_x]
        elif interpolation == CustomCV2.INTER_LINEAR:
            x = np.linspace(0, src_w-1, dst_w)
            y = np.linspace(0, src_h-1, dst_h)

            x_idx, y_idx = np.meshgrid(x, y)
            x_floor, y_floor = np.floor(x_idx).astype(np.int32), np.floor(y_idx).astype(np.int32)
            x_ceil = np.ceil(x_idx).astype(np.int32)
            y_ceil = np.ceil(y_idx).astype(np.int32)

            dx = x_idx - x_floor
            dy = y_idx - y_floor

            if src.ndim == 3:
                dx = dx[..., np.newaxis]
                dy = dy[..., np.newaxis]

            top_left = src[np.clip(y_floor, 0, src_h-1), np.clip(x_floor, 0, src_w-1)]
            top_right = src[np.clip(y_floor, 0, src_h-1), np.clip(x_ceil, 0, src_w-1)]
            bottom_left = src[np.clip(y_ceil, 0, src_h-1), np.clip(x_floor, 0, src_w-1)]
            bottom_right = src[np.clip(y_ceil, 0, src_w-1), np.clip(x_ceil, 0, src_w-1)]

            top = top_left * (1 - dx) + top_right * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            resized = top * (1 - dy) + bottom * dy
            return resized.astype(src.dtype)
    
    @staticmethod
    def perspectiveTransform(src: np.ndarray, m: np.ndarray) -> np.ndarray:
        if src.ndim == 3:
            pts = src.reshape(-1, 2)
        else:
            pts = src

        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homogenous_pts = np.hstack([pts, ones])

        transformed_pts = np.dot(homogenous_pts, m.T)

        w = transformed_pts[:, 2:3]
        w[w == 0] = 1e-9

        normalized_pts = transformed_pts[:, :2] / w

        if src.ndim == 3:
            return normalized_pts.reshape(-1, 1, 2).astype(np.float32)
        else:
            return normalized_pts.astype(np.float32)

    @staticmethod
    def normalize(src: np.ndarray, dst: Optional[np.ndarray], alpha: float, 
                  beta: float, norm_type: int) -> np.ndarray:
        if norm_type == CustomCV2.NORM_MINMAX:
            src_min = src.min()
            src_max = src.max()

            if src_max - src_min < 1e-9:
                return np.full_like(src, alpha)
            
            scale = (beta - alpha) / (src_max - src_min)
            shift = alpha - src_min * scale

            res = src * scale + shift

            if src.dtype == np.uint8:
                res = np.clip(res, 0, 255).astype(np.uint8)
            else:
                res = res.astype(src.dtype)
            
            return res
        else:
            raise NotImplementedError("Only NORM_MINMAX is implemented in normalize")