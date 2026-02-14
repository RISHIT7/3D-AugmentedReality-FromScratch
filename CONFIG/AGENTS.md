# Agent Instructions: Custom AR Tag Pipeline

You operate within a 3-layer architecture designed to optimize high-performance Computer Vision tasks (C++/Python hybrid).

## The 3-Layer Architecture

**Layer 1: Directive (The Assignment Logic)**
- [cite_start]**Goal**: Detect custom AR tags, solve pose, and project 2D/3D content[cite: 28, 46].
- **Constraints**: No built-in `cv2.findHomography` or `cv2.warpPerspective`. [cite_start]Must use custom implementations[cite: 88, 90].
- **SOPs**:
  - [cite_start]`directives/calibrate.md`: Procedure for capturing checkerboard frames and generating `camera_params.npz`[cite: 78].
  - [cite_start]`directives/detect_and_decode.md`: Pipeline for thresholding, contour finding, and bit decoding[cite: 46, 50].
  - [cite_start]`directives/render_3d.md`: Math for recovering pose ($R, t$) from Homography and projecting `.obj` vertices[cite: 69, 73].

**Layer 2: Orchestration (Python Glue)**
- **Role**: High-level logic in `main.py` or `utils.py`.
- **Responsibilities**:
  - Capture video frames.
  - Route intensive tasks to the C++ backend.
  - [cite_start]Manage state (Kalman Filters for smoothing)[cite: 85].
  - Handle user input (switching between 2D overlay and 3D projection).

**Layer 3: Execution (C++ Backend)**
- **Deterministic Core**: `custom_cv2.cpp` compiled as a Python extension.
- **Tasks**:
  - [cite_start]Adaptive Thresholding & Contours (OpenMP optimized)[cite: 31].
  - [cite_start]Inverse Homography Warping (Pixel-wise operations)[cite: 87].
  - Matrix Math (Adjugate matrices, projection calculations).
- **Principle**: Python handles *logic*, C++ handles *pixels*.

## Operating Principles

**1. Optimization First**
Always check if a function exists in `custom_cv2_cpp` before writing it in Python. If a loop iterates over pixels, move it to C++.

**2. Math over Libraries**
Do not use forbidden OpenCV functions (`findHomography`, `warpPerspective`). [cite_start]Instead, implement the Linear Algebra manually (DLT, SVD, Inverse) in Layer 3[cite: 88].

**3. Self-Annealing (Calibration)**
If 3D projection looks wrong (sliding/floating), re-run the calibration directive. Bad $K$ matrix = Bad AR.

## File Organization

- `core/` - C++ source files (`custom_cv2.cpp`) and CMake/Setup scripts.
- [cite_start]`utils.py` - Python wrappers, decoding logic, and rendering helpers[cite: 31].
- [cite_start]`assets/` - `.obj` models (e.g., Wolf), template images, and calibration data (`camera_params.npz`)[cite: 69].
- `.env` - Debug flags (`DEBUG_WARP=True`, `USE_CPP=True`).