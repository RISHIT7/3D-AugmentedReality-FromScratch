# Project Skills & Capabilities

## Core Competencies
- **Hybrid Development**: Ability to write Python scripts that interface with compiled C++ extensions using `pybind11`.
- [cite_start]**Linear Algebra**: Manual implementation of Homography estimation (DLT algorithm), Matrix Inversion (Adjugate method), and Change of Basis[cite: 87].
- **Computer Vision**:
  - [cite_start]**Fiducial Marker Detection**: Contour approximation, bit-grid decoding, and orientation resolution[cite: 36, 40].
  - [cite_start]**Camera Calibration**: Intrinsic parameter estimation ($f_x, f_y, c_x, c_y$) using checkerboards[cite: 78].
  - [cite_start]**Pose Estimation**: Recovering Rotation ($R$) and Translation ($t$) from planar Homography[cite: 73].

## Technical Stack
- **Languages**: Python 3.x, C++17 (OpenMP support).
- **Libraries**: OpenCV (Basic I/O only), NumPy (Vector math), PyBind11 (Bindings).
- **Math Concepts**:
  - Homogeneous Coordinates.
  - Euler Angles (Yaw/Pitch/Roll) decomposition.
  - [cite_start]Projection Matrix ($P = K[R|t]$)[cite: 74].

## Specialized Tasks
- [cite_start]**"Anti-Gravity" Rendering**: Projecting 3D vertices onto a 2D plane to create the illusion of objects standing or floating on a tag[cite: 69].
- [cite_start]**Stabilization**: Implementing Kalman Filters to smooth jittery corner detections[cite: 85].