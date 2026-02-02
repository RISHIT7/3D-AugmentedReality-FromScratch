import numpy as np
from typing import Tuple, List, Optional, Any
import math
from concurrent.futures import ThreadPoolExecutor

GRAY_WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)

class CustomCV2:
    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_OTSU = 8
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
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
            
        print(thresh)
        return thresh, result.astype(np.uint8)

    @staticmethod
    def adaptiveThreshold(src: np.ndarray, maxValue: float, adaptiveMethod: int,
                         thresholdType: int, blockSize: int, C: float) -> np.ndarray:
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
        local_thresh = (block_sum / counts) - C
        
        if thresholdType == CustomCV2.THRESH_BINARY_INV:
            output = np.where(src > local_thresh, 0, maxValue)
        else:
            output = np.where(src > local_thresh, maxValue, 0)
            
        return output.astype(np.uint8)

    @staticmethod
    def findContours(image: np.ndarray, mode: int = 0, method: int = 1) -> List[np.ndarray]:
        """
        Fast Proxy-Contours. 
        Downscales image to trace faster, then upscales coordinates.
        Speedup: ~16x for SCALE=4.
        """
        SCALE = 4
        
        small_img = image[::SCALE, ::SCALE]
        
        binary = (small_img > 0).astype(np.uint8)
        binary = np.pad(binary, 1, mode='constant', constant_values=0)
        h, w = binary.shape
        
        visited = np.zeros_like(binary, dtype=bool)
        contours = []
        
        # Pre-calculate offsets to avoid list creation in loop
        # N, NE, E, SE, S, SW, W, NW
        OFFSETS_Y = [-1, -1, 0, 1, 1, 1, 0, -1]
        OFFSETS_X = [0, 1, 1, 1, 0, -1, -1, -1]
        
        # 3. Scan for start points
        # Optimization: Only scan pixels that are potential borders? 
        # For now, standard scan on small image is fast enough.
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if binary[y, x] == 1 and not visited[y, x]:
                    # Check boundary condition (at least one 0 neighbor)
                    # Inline check for speed
                    is_border = False
                    if binary[y-1, x] == 0 or binary[y+1, x] == 0 or \
                       binary[y, x-1] == 0 or binary[y, x+1] == 0:
                        is_border = True
                        
                    if is_border:
                        contour_pts = []
                        cy, cx = y, x
                        backtrack = 0
                        
                        while True:
                            # Add point (Remove padding + Rescale to original size)
                            contour_pts.append([(cx - 1) * SCALE, (cy - 1) * SCALE])
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
    def contourArea(contour: np.ndarray, oriented: bool = False) -> float:
        """
        Calculate contour area.
        
        Args:
            contour: Input contour
            oriented: If True, returns signed area
            
        Returns:
            Area value
        """
        raise NotImplementedError("contourArea needs implementation")
    
    @staticmethod
    def arcLength(curve: np.ndarray, closed: bool) -> float:
        """
        Calculate contour perimeter.
        
        Args:
            curve: Input contour
            closed: Whether contour is closed
            
        Returns:
            Perimeter length
        """
        raise NotImplementedError("arcLength needs implementation")
    
    @staticmethod
    def approxPolyDP(curve: np.ndarray, epsilon: float, closed: bool) -> np.ndarray:
        """
        Approximate polygonal curve with specified precision.
        Uses Douglas-Peucker algorithm.
        
        Args:
            curve: Input contour
            epsilon: Approximation accuracy
            closed: Whether curve is closed
            
        Returns:
            Approximated polygon
        """
        raise NotImplementedError("approxPolyDP needs implementation")
    
    @staticmethod
    def isContourConvex(contour: np.ndarray) -> bool:
        """
        Test if contour is convex.
        
        Args:
            contour: Input contour
            
        Returns:
            True if convex, False otherwise
        """
        raise NotImplementedError("isContourConvex needs implementation")
    
    @staticmethod
    def moments(array: np.ndarray, binaryImage: bool = False) -> dict:
        """
        Calculate all moments up to third order.
        
        Args:
            array: Raster image or contour
            binaryImage: If True, all non-zero pixels are treated as 1
            
        Returns:
            Dictionary of moments
        """
        raise NotImplementedError("moments needs implementation")
    
    @staticmethod
    def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Calculate perspective transformation matrix.
        
        Args:
            src: Coordinates of 4 source points
            dst: Coordinates of 4 destination points
            
        Returns:
            3x3 perspective transformation matrix
        """
        raise NotImplementedError("getPerspectiveTransform needs implementation")
    
    @staticmethod
    def warpPerspective(src: np.ndarray, M: np.ndarray, dsize: Tuple[int, int],
                       flags: int = INTER_LINEAR) -> np.ndarray:
        """
        Apply perspective transformation to image.
        
        Args:
            src: Input image
            M: 3x3 transformation matrix
            dsize: Size of output image
            flags: Interpolation method
            
        Returns:
            Warped image
        """
        raise NotImplementedError("warpPerspective needs implementation")
    
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
        """
        Resize image.
        
        Args:
            src: Input image
            dsize: Output image size
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        raise NotImplementedError("resize needs implementation")
    
    @staticmethod
    def perspectiveTransform(src: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Perform perspective transformation on points.
        
        Args:
            src: Input points
            m: 3x3 transformation matrix
            
        Returns:
            Transformed points
        """
        raise NotImplementedError("perspectiveTransform needs implementation")
    
    @staticmethod
    def polylines(img: np.ndarray, pts: List[np.ndarray], isClosed: bool, 
                  color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
        """
        Draw polylines on image.
        
        Args:
            img: Input/output image
            pts: Array of polygonal curves
            isClosed: Whether polylines are closed
            color: Polyline color
            thickness: Line thickness
            
        Returns:
            Image with drawn polylines
        """
        raise NotImplementedError("polylines needs implementation")
    
    @staticmethod
    def putText(img: np.ndarray, text: str, org: Tuple[int, int], fontFace: int,
                fontScale: float, color: Tuple[int, int, int], thickness: int = 1) -> np.ndarray:
        """
        Draw text on image.
        
        Args:
            img: Input/output image
            text: Text string to draw
            org: Bottom-left corner of text
            fontFace: Font type
            fontScale: Font scale factor
            color: Text color
            thickness: Line thickness
            
        Returns:
            Image with drawn text
        """
        raise NotImplementedError("putText needs implementation")
    
    @staticmethod
    def fillConvexPoly(img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Fill convex polygon.
        
        Args:
            img: Input/output image
            points: Polygon vertices
            color: Fill color
            
        Returns:
            Image with filled polygon
        """
        raise NotImplementedError("fillConvexPoly needs implementation")
    
    @staticmethod
    def normalize(src: np.ndarray, dst: Optional[np.ndarray], alpha: float, 
                  beta: float, norm_type: int) -> np.ndarray:
        """
        Normalize array to range.
        
        Args:
            src: Input array
            dst: Output array (can be None)
            alpha: Lower range boundary
            beta: Upper range boundary
            norm_type: Normalization type
            
        Returns:
            Normalized array
        """
        raise NotImplementedError("normalize needs implementation")
