import numpy as np
from typing import Tuple, List, Optional

import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import custom_cv2_cpp
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False

class CustomCV2:
    COLOR_BGR2GRAY = 6
    GRAY_WEIGHTS = np.array([0.3333, 0.3333, 0.3333], dtype=np.float32)
    MIN_CONTOUR_POINTS = 10
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
    ADAPTIVE_THRESH_MEAN_C = 0
    INTER_LINEAR = 1
    NORM_MINMAX = 32
    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_OTSU = 8
    THRESH_TEMPORAL_APPROX_OTSU = 16
    RETR_TREE = 3
    CHAIN_APPROX_SIMPLE = 2
    INTER_NEAREST = 0
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_MAX_ITER = 1
    TERM_CRITERIA_COUNT = 1

    update_freq: int = 10
    min_pixels: int = 256

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
    def _compute_otsu_threshold(src: np.ndarray) -> float:
        if src.size < CustomCV2.min_pixels:
            return float(np.median(src))

        hist, _ = np.histogram(src.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)

        bin_centers = np.arange(256)
        weight_bg = np.cumsum(hist)
        weight_fg = src.size - weight_bg

        sum_bg = np.cumsum(hist * bin_centers)
        sum_total = sum_bg[-1]

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        var_between[~np.isfinite(var_between)] = 0
        return float(np.argmax(var_between))

    @staticmethod
    def _compute_temporal_threshold(src: np.ndarray) -> float:
        thresh, count = CustomCV2._threshold_state.get_and_increment()
        if count % CustomCV2.update_freq == 0:
            new_thresh = CustomCV2._compute_otsu_threshold(src)
            CustomCV2._threshold_state.update(new_thresh)
            return new_thresh
        else:
            return thresh

    @staticmethod
    def _apply_threshold(src: np.ndarray, thresh: float, maxval: float, type: int) -> np.ndarray:
        if type == CustomCV2.THRESH_BINARY:
            return np.where(src > thresh, maxval, 0).astype(np.uint8)
        elif type == CustomCV2.THRESH_BINARY_INV:
            return np.where(src > thresh, 0, maxval).astype(np.uint8)
        else:
            return src.copy()

    @staticmethod
    def _bilinear_sample(src: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h, w = src.shape[:2]

        src_x = np.clip(x, 0, w - 1)
        src_y = np.clip(y, 0, h - 1)

        x0 = np.floor(src_x).astype(int)
        y0 = np.floor(src_y).astype(int)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dx = src_x - x0
        dy = src_y - y0

        wa = (1 - dx) * (1 - dy)
        wb = (1 - dx) * dy
        wc = dx * (1 - dy)
        wd = dx * dy

        if src.ndim == 2:
            return (src[y0, x0] * wa + src[y0, x1] * wb + src[y1, x0] * wc + src[y1, x1] * wd).astype(src.dtype)
        else:
            wa, wb, wc, wd = [w[..., None] for w in (wa, wb, wc, wd)]
            return (src[y0, x0] * wa + src[y1, x0] * wb + src[y0, x1] * wc + src[y1, x1] * wd).astype(src.dtype)

    @staticmethod
    def _manual_gradient_2d(img):
        img = img.astype(np.float64)
        grad_axis_0 = np.zeros_like(img)
        grad_axis_1 = np.zeros_like(img)

        grad_axis_0[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        grad_axis_1[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        grad_axis_0[0, :] = img[1, :] - img[0, :]
        grad_axis_1[:, 0] = img[:, 1] - img[:, 0]

        grad_axis_0[-1, :] = img[-1, :] - img[-2, :]
        grad_axis_1[:, -1] = img[:, -1] - img[:, -2]

        return grad_axis_0, grad_axis_1

    @staticmethod
    def cvtColor(src: np.ndarray, code: int) -> np.ndarray:
        if CPP_AVAILABLE:
            return custom_cv2_cpp.cvtColor_cpp(src)
        if code == CustomCV2.COLOR_BGR2GRAY:
            if src.ndim == 2:
                return src.copy()
            elif src.ndim == 3 and src.shape[2] == 3:
                gray = np.dot(src[..., :3], CustomCV2.GRAY_WEIGHTS)
                return gray.astype(np.uint8)
            else:
                raise ValueError("Input image must have 2 or 3 channels for BGR2GRAY conversion")
        else:
            raise NotImplementedError("cvtColor code not implemented")

    @staticmethod
    def BoxFilter(src: np.ndarray, ksize: Tuple[int, int]) -> np.ndarray:
        if CPP_AVAILABLE:
            return custom_cv2_cpp.boxFilter_cpp(src, ksize[0])
        kx, ky = ksize
        res = src.astype(np.float32)

        res = np.cumsum(res, axis=1)
        res[:, kx:] = res[:, kx:] - res[:, :-kx]
        res = res[:, kx-1:] / kx

        res = np.cumsum(res, axis=0)
        res[ky:, :] = res[ky:, :] - res[:-ky, :]
        res = res[ky-1:, :] / ky

        pad_h = ky // 2
        pad_w = kx // 2
        return np.pad(res, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge').astype(src.dtype)

    @staticmethod
    def GaussianBlur(src: np.ndarray, ksize: Tuple[int, int], sigmaX: float) -> np.ndarray:
        if CPP_AVAILABLE:
            return custom_cv2_cpp.gaussianBlur_cpp(src, ksize[0], sigmaX)

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
    def sharpenAndNormalize(src: np.ndarray) -> np.ndarray:
        if CPP_AVAILABLE:
            return custom_cv2_cpp.sharpenAndNormalize_cpp(src)

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

        padded = np.pad(src, 1, mode='edge').astype(np.float32)
        res = padded[1:-1, 1:-1] * kernel[1, 1]
        res -= (padded[0:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, 0:-2] + padded[1:-1, 2:])

        min_val, max_val = res.min(), res.max()

        if (max_val - min_val) < 1e-5:
            return np.zeros_like(src, dtype=np.uint8)
        normalized = (res - min_val) * (255.0 / (max_val - min_val))
        return np.clip(normalized, 0, 255).astype(np.uint8)


    @staticmethod
    def findCornersQuadrilateral(contour: np.ndarray) -> np.ndarray:
        if contour.ndim == 3:
            pts = contour.reshape(-1, 2)
        else:
            pts = contour

        if len(pts) < 4:
            return pts.reshape(-1, 1, 2)

        centroid = np.mean(pts, axis=0)

        diff = pts - centroid
        dists_sq = np.sum(diff**2, axis=1)

        kernel_size = 5
        dists_padded = np.pad(dists_sq, (kernel_size//2, kernel_size//2), mode='wrap')
        dists_smooth = np.convolve(dists_padded, np.ones(kernel_size)/kernel_size, mode='valid')

        prev_pts = np.roll(dists_smooth, 1)
        next_pts = np.roll(dists_smooth, -1)

        peaks_mask = (dists_smooth > prev_pts) & (dists_smooth > next_pts)
        peak_indices = np.where(peaks_mask)[0]

        if len(peak_indices) > 4:
            peak_dists = dists_smooth[peak_indices]
            top_indices = np.argsort(peak_dists)[-4:]
            final_indices = peak_indices[top_indices]
        elif len(peak_indices) < 4:
            return CustomCV2.approxPolyDP(contour, 0.02 * CustomCV2.arcLength(contour, True), True)
        else:
            final_indices = peak_indices

        corners = pts[final_indices]

        return corners.reshape(-1, 1, 2).astype(np.int32)

    @staticmethod
    def threshold(src: np.ndarray, thresh: float, maxval: float, type: int) -> Tuple[float, np.ndarray]:
        if src.size == 0:
            return thresh, np.zeros_like(src, dtype=np.uint8)

        use_otsu = bool(type & CustomCV2.THRESH_OTSU)
        use_temporal = bool(type & CustomCV2.THRESH_TEMPORAL_APPROX_OTSU)
        base_type = type & ~(CustomCV2.THRESH_OTSU | CustomCV2.THRESH_TEMPORAL_APPROX_OTSU)
        if use_otsu:
            thresh = CustomCV2._compute_otsu_threshold(src)
        elif use_temporal:
            thresh = CustomCV2._compute_temporal_threshold(src)

        return thresh, CustomCV2._apply_threshold(src, thresh, maxval, base_type)

    @staticmethod
    def adaptiveThreshold(src: np.ndarray, maxValue: float, adaptiveMethod: int,
                         thresholdType: int, blockSize: int, C: float) -> np.ndarray:
        if blockSize % 2 == 0:
            blockSize += 1

        if CPP_AVAILABLE:
            res = custom_cv2_cpp.adaptiveThreshold_cpp(src, maxValue, blockSize, C)

            if thresholdType == CustomCV2.THRESH_BINARY_INV:
                res = maxValue - res

            return res

        if adaptiveMethod == CustomCV2.ADAPTIVE_THRESH_GAUSSIAN_C:
            sigma = 0.3 * ((blockSize - 1) * 0.5 - 1) + 0.8
            mean = CustomCV2.GaussianBlur(src, (blockSize, blockSize), sigma)
            mean = mean.astype(np.float32)

        else:
            h, w = src.shape
            half = blockSize // 2

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

        if CPP_AVAILABLE:
            return custom_cv2_cpp.findContours_cpp(binary, CustomCV2.MIN_CONTOUR_POINTS), None

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
        img = src.astype(np.int16)
        p = np.pad(img, 1, mode='reflect')
        smooth_y = p[:-2, :] + (2 * p[1:-1, :]) + p[2:, :]

        gx = smooth_y[:, 2:] - smooth_y[:, :-2]
        smooth_x = p[:, :-2] + (2 * p[:, 1:-1]) + p[:, 2:]

        gy = smooth_x[2:, :] - smooth_x[:-2, :]

        magnitude = np.abs(gx) + np.abs(gy)

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

        x = pts[:, 0].astype(np.float64)
        y = pts[:, 1].astype(np.float64)
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
        if CPP_AVAILABLE:
            return custom_cv2_cpp.approxPolyDP_cpp(curve, epsilon, closed)
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

        mo = {k: 0.0 for k in ['m00', 'm10', 'm01', 'm20', 'm11', 'm02']}

        if len(pts) < 3:
            return mo

        x = pts[:, 0]
        y = pts[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        cross = x * y_next - x_next * y
        signed_area = 0.5 * np.sum(cross)

        mo["m00"] = abs(signed_area)

        if abs(signed_area) > 1e-10:
            sign_correction = np.sign(signed_area)
            mo["m10"] = (1/6) * np.sum((x + x_next) * cross) * sign_correction
            mo["m01"] = (1/6) * np.sum((y + y_next) * cross) * sign_correction
        return mo

    @staticmethod
    def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        if src.shape != (4, 2) or dst.shape != (4, 2):
            raise ValueError("Source and Destination must contain exactly 4 points")

        A = np.zeros((8, 8), dtype=np.float64)
        b = np.zeros((8), dtype=np.float64)

        for i in range(4):
            x, y = src[i]
            u, v = dst[i]

            A[2*i] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
            b[2*i] = u

            A[2*i+1] = [0, 0, 0, x, y, 1, -x*v, -y*v]
            b[2*i+1] = v

        try:
            h = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.eye(3)

        M = np.append(h, 1.0).reshape(3, 3)
        return M

    @staticmethod
    def warpPerspective(src: np.ndarray, M: np.ndarray, dsize: Tuple[int, int],
                       tile_height: int = 64) -> np.ndarray:
        width, height = dsize
        if CPP_AVAILABLE:
            return custom_cv2_cpp.warpPerspective_cpp(src, M, width, height)

        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            return np.zeros((height, width), dtype=src.dtype)

        h00, h01, h02 = M_inv[0]
        h10, h11, h12 = M_inv[1]
        h20, h21, h22 = M_inv[2]

        if src.ndim == 2:
            output = np.zeros((height, width), dtype=src.dtype)
        else:
            output = np.zeros((height, width, src.shape[2]), dtype=src.dtype)

        for y_start in range(0, height, tile_height):
            y_end = min(y_start + tile_height, height)
            tile_h = y_end - y_start

            x_idxs, y_idxs = np.meshgrid(np.arange(width), np.arange(y_start, y_end))

            denom = h20 * x_idxs + h21 * y_idxs + h22
            valid_mask = np.abs(denom) > 1e-6
            denom[~valid_mask] = 1.0

            src_x = (h00 * x_idxs + h01 * y_idxs + h02) / denom
            src_y = (h10 * x_idxs + h11 * y_idxs + h12) / denom

            warped_tile = CustomCV2._bilinear_sample(
                src, src_x, src_y
            )
            output[y_start:y_end, :] = warped_tile

        return output

    @staticmethod
    def cornerSubPix(image: np.ndarray, corners: np.ndarray, winSize: Tuple[int, int],
                     zeroZone: Tuple[int, int], criteria: Tuple[int, int, float]) -> np.ndarray:

        crit_type, max_iter, epsilon = criteria
        if not (crit_type & CustomCV2.TERM_CRITERIA_MAX_ITER):
            max_iter = 100
        if not (crit_type & CustomCV2.TERM_CRITERIA_EPS):
            epsilon = 1e-6

        if CPP_AVAILABLE:
            orig_shape = corners.shape
            c3d = corners.reshape(-1, 1, 2).astype(np.float32)
            result = custom_cv2_cpp.cornerSubPix_cpp(image, c3d, winSize[0], winSize[1], max_iter, epsilon)
            return result.reshape(orig_shape)

        img = image.astype(np.float64)
        h, w = img.shape[:2]
        win_w, win_h = winSize
        zero_w, zero_h = zeroZone

        padded = np.pad(img, 1, mode='reflect')
        gx = (padded[1:-1, 2:] - padded[1:-1, :-2]) * 0.5
        gy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) * 0.5

        refined = corners.copy().astype(np.float32)

        for i in range(len(refined)):
            cx, cy = float(refined[i, 0]), float(refined[i, 1])

            for _ in range(max_iter):
                ix, iy = int(round(cx)), int(round(cy))

                x0 = max(0, ix - win_w)
                x1 = min(w - 1, ix + win_w)
                y0 = max(0, iy - win_h)
                y1 = min(h - 1, iy + win_h)

                if x1 <= x0 or y1 <= y0:
                    break


                A = np.zeros((2, 2), dtype=np.float64)
                b_vec = np.zeros(2, dtype=np.float64)

                for py in range(y0, y1 + 1):
                    for px in range(x0, x1 + 1):
                        if zero_w >= 0 and zero_h >= 0:
                            if abs(px - ix) <= zero_w and abs(py - iy) <= zero_h:
                                continue

                        dx = gx[py, px]
                        dy = gy[py, px]

                        A[0, 0] += dx * dx
                        A[0, 1] += dx * dy
                        A[1, 0] += dx * dy
                        A[1, 1] += dy * dy

                        b_vec[0] += dx * dx * px + dx * dy * py
                        b_vec[1] += dx * dy * px + dy * dy * py

                det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                if abs(det) < 1e-10:
                    break

                new_cx = (A[1, 1] * b_vec[0] - A[0, 1] * b_vec[1]) / det
                new_cy = (A[0, 0] * b_vec[1] - A[1, 0] * b_vec[0]) / det

                shift = np.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
                cx, cy = new_cx, new_cy

                if shift < epsilon:
                    break

            refined[i, 0] = cx
            refined[i, 1] = cy

        return refined

    @staticmethod
    def resize(src: np.ndarray, dsize: Tuple[int, int], interpolation: int) -> np.ndarray:
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
    def bilateralFilter(src: np.ndarray, d: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
        if CPP_AVAILABLE:
            return custom_cv2_cpp.bilateralFilter_cpp(src, d, sigmaColor, sigmaSpace)
        radius = d // 2
        kernel_size = 2 * radius + 1

        padded = np.pad(src, ((radius, radius), (radius, radius)), mode='reflect').astype(np.float32)
        src_f = src.astype(np.float32)

        results_num = np.zeros_like(src_f)
        results_den = np.zeros_like(src_f)

        y_idx, x_idx = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
        spatial_kernel = np.exp(-(x_idx**2 + y_idx**2) / (2 * sigmaSpace**2))

        for ky in range(kernel_size):
            for kx in range(kernel_size):
                neighbor = padded[ky : ky + src.shape[0], kx : kx + src.shape[1]]

                diff_sq = (neighbor - src_f) ** 2
                range_weight = np.exp(-diff_sq / (2 * sigmaColor**2))

                total_weight = spatial_kernel[ky, kx] * range_weight

                results_num += neighbor * total_weight
                results_den += total_weight

        output = results_num / (results_den + 1e-10)
        return np.clip(output, 0, 255).astype(src.dtype)

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

    @staticmethod
    def erode(src: np.ndarray, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
        if CPP_AVAILABLE:
            try:
                return custom_cv2_cpp.erode_cpp(src.astype(np.uint8), kernel.astype(np.uint8), iterations)
            except Exception:
                pass

        result = src.copy()
        kh, kw = kernel.shape[:2]
        pad_h, pad_w = kh // 2, kw // 2
        ky, kx = np.where(kernel > 0)

        for _ in range(iterations):
            padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
            h, w = result.shape[:2]
            out = np.full_like(result, 255)
            for dy, dx in zip(ky, kx):
                shifted = padded[dy:dy + h, dx:dx + w]
                out = np.minimum(out, shifted)
            result = out

        return result

    @staticmethod
    def bitwise_or(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
        if CPP_AVAILABLE:
            try:
                return custom_cv2_cpp.bitwise_or_cpp(src1.astype(np.uint8), src2.astype(np.uint8))
            except Exception:
                pass
        return np.bitwise_or(src1, src2)

    @staticmethod
    def fillConvexPoly(img: np.ndarray, points: np.ndarray, color) -> np.ndarray:
        if CPP_AVAILABLE:
            try:
                if isinstance(color, (list, tuple)):
                    color = list(color)
                else:
                    color = [int(color)] * 3
                return custom_cv2_cpp.fillConvexPoly_cpp(img, points.astype(np.int32), color)
            except Exception:
                pass

        pts = points.reshape(-1, 2).astype(np.int32)
        h, w = img.shape[:2]
        n = len(pts)
        if n < 3:
            return img

        y_min = max(0, int(pts[:, 1].min()))
        y_max = min(h - 1, int(pts[:, 1].max()))

        for y in range(y_min, y_max + 1):
            x_intersections = []
            for i in range(n):
                j = (i + 1) % n
                y0, y1 = pts[i, 1], pts[j, 1]
                if y0 == y1:
                    continue
                if min(y0, y1) <= y < max(y0, y1):
                    x = pts[i, 0] + (y - y0) * (pts[j, 0] - pts[i, 0]) / (y1 - y0)
                    x_intersections.append(x)
            if len(x_intersections) >= 2:
                x_intersections.sort()
                x_start = max(0, int(np.ceil(x_intersections[0])))
                x_end = min(w - 1, int(np.floor(x_intersections[-1])))
                if x_start <= x_end:
                    if img.ndim == 3:
                        img[y, x_start:x_end + 1] = color
                    else:
                        img[y, x_start:x_end + 1] = color[0] if isinstance(color, (list, tuple)) else color
        return img