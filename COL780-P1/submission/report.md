# COL780 Project 1: AR Tag Detection & Rendering

## 1. Introduction
This project implements a robust Augmented Reality (AR) tag detection system capable of identifying fiducial markers (AR tags) in video streams, estimating their 3D pose, and rendering virtual objects with dynamic lighting. The pipeline is optimized for performance using C++ bindings (PyBind11) with OpenMP parallelization and SIMD instructions, ensuring real-time performance even with complex image processing operations.

## 2. Pipeline Architecture
The image processing pipeline transitions from raw input frames to the final AR overlay through the following stages:

### 2.1. Preprocessing & Multi-Scale Thresholding
To handle varying lighting conditions and motion blur, we employ a **Multi-Scale Adaptive Thresholding** strategy. Instead of relying on a single threshold parameters, we process the image at multiple scales:
1.  **Fine Scale (Block Size 8, Blur 5)**: optimal for sharp, high-contrast tags and detecting precise corners.
2.  **Coarse Scale (Block Size 21, Blur 11)**: robust against motion blur and noise, filling in gaps that appear at finer scales.

The input frame is converted to grayscale (`CustomCV2.cvtColor`) and smoothed using a **Bilateral Filter** (implemented in C++) to preserve edges while reducing noise.

### 2.2. Contour Extraction & Geometric Gating
We extract contours from the thresholded binary images using `CustomCV2.findContours`. To filter out irrelevant shapes early (optimization), we implemented a rigorous **Geometric Gating** mechanism (`is_valid_quadrilateral`):
-   **Convexity Check**: Ensures the shape is closed and convex.
-   **Quadrilateral Approximation**: Uses `approxPolyDP` to verify the shape has 4 vertices.
-   **Side Ratio Constraint**: Rejects elongated shapes (max/min side ratio > 4.0).
-   **Angle Constraint**: Ensures interior angles are within [40°, 140°] to allow for perspective distortion while rejecting extreme artifacts.

### 2.3. Homography & Perspective Warping
For each valid candidate, we compute the homography matrix $H$ mapping the detected corners to a canonical square tag (400x400 pixels).
$$ H = \text{getPerspectiveTransform}(pts_{src}, pts_{dst}) $$
The tag region is then warped to this canonical frame using `CustomCV2.warpPerspective` (SIMD-optimized C++ implementation).

### 2.4. Tag Decoding & "Double Decode" Strategy
The warped tag is grid-sampled (8x8 grid) to read the binary ID. We verify the tag by:
1.  **Border Validation**: Checking if the outer border is strictly white.
2.  **Orientation Detection**: Identifying the unique white anchor in the inner 4x4 grid.

**Advancement: Double Decoding**
If the border check fails (often due to thresholding artifacts leaking into the border), we trigger a recovery mechanism:
1.  The warped canonical image is re-thresholded locally.
2.  We search for contours *within* the warped image.
3.  If a valid inner quad is found, we compute a secondary homography $H_{inv}$ to map the inner coordinates back to the original frame, refining the corner positions. This significantly improves recall for tags with poor lighting or occlusion.

### 2.5. Pose Estimation (PnP)
For detected tags, we check the alignment with the camera using Perspective-n-Point (PnP).
We solve for the rotation ($R$) and translation ($t$) vectors that minimize the reprojection error of the 3D tag corners onto the 2D image plane:
$$ s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} R & t \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} $$
where $K$ is the camera intrinsic matrix.

### 2.6. Rendering with Lighting
We render 3D OBJ models onto the tags. The rendering pipeline includes:
-   **Projection**: 3D vertices are projected to 2D using the computed pose and camera matrix.
-   **Painter's Algorithm**: Faces are sorted by average depth ($w$) to handle occlusion correctly.
-   **Dynamic Lighting**: We implement a depth-based shading model (`render_with_lighting`). Faces further from the camera are shaded darker, providing a sense of depth and 3D structure. The rasterization of these faces is accelerated using `CustomCV2.fillConvexPoly` (C++).

## 3. Key Advancements & Optimizations

### 3.1. Robustness Improvements
-   **Multi-Scale Independent Detection**: By processing scales independently and merging results based on spatial proximity (deduplication), we detect both small, sharp tags and large, blurry tags simultaneously.
-   **Sub-Pixel Refinement**: We use `cornerSubPix` (implemented in C++) to refine corner locations with gradient descent, improving jitter stability.

### 3.2. C++ Performance Optimizations
We identified and optimized critical bottlenecks in the C++ extension (`custom_cv2.cpp`):

1.  **Branchless Bilateral Filter**:
    -   **Problem**: The naive implementation had 4 conditional bounds checks per pixel in the inner loop ($O(N \cdot d^2)$).
    -   **Solution**: We pad the source image by the kernel radius. This allows the inner loop to execute distinct load operations without any branching logic.
    -   **Gain**: Eliminates ~81 branch instructions per pixel (for d=9), yielding massive speedups.

2.  **Cache-Friendly Gaussian Blur**:
    -   **Problem**: The vertical pass iterated column-by-column ($y$ varying in inner loop), causing cache trashing.
    -   **Solution**: Restructured to iterate row-by-row ($x$ varying in inner loop), utilizing spatial locality of reference.

3.  **SIMD Auto-Vectorization**:
    -   We enabled `-march=native` in the build configuration. This grants the compiler permission to generate AVX2/NEON instructions, automatically vectorizing loops in `warpPerspective`, `threshold`, and `integral_image` computation.

## 4. Assumptions & Compliance

### 4.1. Assumptions
-   **Planar Geometry**: We assume AR tags lie on a planar surface, allowing the use of Homography decomposition for pose estimation.
-   **Lighting Conditions**: While our adaptive thresholding handles varying light, we assume sufficient contrast exists between the tag border (black) and the background.
-   **Camera Calibration**: We assume the camera intrinsic matrix $K$ is provided or can be approximated. For 3D rendering, accurate $K$ is critical for correct perspective.

### 4.2. Compliance with Constraints
-   **Custom Implementations**: As per assignment requirements, we strictly implemented our own versions of:
    -   **Homography**: `CustomCV2.getPerspectiveTransform` (Linear system solver).
    -   **Inverse Homography**: `CustomCV2.warpPerspective` (Inverse mapping with bilinear interpolation).
    -   **Pose Estimation**: `CustomCV2.solvePnP` (Homography decomposition method), avoiding `cv2.solvePnP` entirely.
    -   **Rendering**: Custom logic for 3D projection and rasterization.
-   **Allowed Libraries**: `cv2` is used only for basic I/O (reading/writing images, video capture, drawing visualizations). All core logic is custom C++/Python.

## 4. References

1.  **Algorithm References**:
    -   Rothe, R., et al. "Efficient Marker Detection for Augmented Reality." [Link](https://people.scs.carleton.ca/~roth/iit-publications-iti/docs/gerh-50002.pdf)
    -   OpenCV Documentation - Adaptive Thresholding: [Link](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
    -   Otsu's Method: [Link](https://en.wikipedia.org/wiki/Otsu%27s_method)

2.  **Implementation**:
    -   **PyBind11**: Used for seamless C++ to Python interoperability.
    -   **OpenMP**: Used for multi-threaded parallelization of image filters.
    -   **Render / Render with Lighting**: Custom implementation in `core/utils.py` leveraging standard computer graphics projection and shading techniques, accelerated by C++ rasterization primitives.