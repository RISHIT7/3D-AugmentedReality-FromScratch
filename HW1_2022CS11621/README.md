# AR Tag Detection & Rendering Project

This project implements a custom Augmented Reality (AR) pipeline for detecting fiducial markers, estimating their 3D pose, and rendering virtual objects with dynamic lighting. Key components are optimized in C++ for performance.

## 1. Prerequisites
-   **Python 3.8+**
-   **C++ Compiler**: Clang (macOS) or GCC (Linux) with OpenMP support.
-   **Dependencies**:
    -   `numpy`
    -   `opencv-python` (for I/O only)
    -   `pybind11` (for C++ extensions)
    -   `setuptools`

## 2. Compilation & Setup
Before running the Python scripts, you must compile the C++ extension module `custom_cv2_cpp`.

### MacOS (with Homebrew)
Ensure `libomp` is installed for OpenMP support:
```bash
brew install libomp
```

### Build Command
Run the following from the project root:
```bash
python setup.py build_ext --inplace
```
This generates a shared object file (e.g., `custom_cv2_cpp.cpython-39-darwin.so`) in the root directory.

## 3. Usage
The main entry point is `main.py`.

### Basic Usage (Webcam)
```bash
python main.py
```

### Video File Input
```bash
python main.py --video_source path/to/video.mp4
```

### Custom AR Content
To overlay a specific image template and 3D model:
```bash
python main.py --template_path data/logo.png --obj_model_path data/wolf.obj
```

### Camera Calibration
To provide a custom camera matrix (YAML or XML from OpenCV calibration), ensuring accurate 3D projection:
```bash
python main.py --camera_matrix_path data/camera_calib.yaml
```

## 4. Project Structure
-   `main.py`: Main application loop and argument parsing.
-   `core/`: Python modules for image processing logic.
    -   `cv2_functions.py`: Wrappers redirecting to C++ implementation if available.
    -   `utils.py`: High-level AR pipeline (detection, decoding, PnP, rendering).
-   `cpp/`: C++ source code.
    -   `custom_cv2.cpp`: Optimized implementations of standard CV algorithms (Bilateral Filter, Gaussian Blur, Warp Perspective, etc.).
-   `setup.py`: Build configuration script.

## 5. Troubleshooting
-   **"ImportError: No module named custom_cv2_cpp"**: Ensure you ran the build command successfully. The `.so` file must be in the same directory where you run python.
-   **OpenMP Errors on macOS**: You might need to set environment variables if the compiler can't find `libomp`:
    ```bash
    export CFLAGS="-I/opt/homebrew/opt/libomp/include"
    export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
    python setup.py build_ext --inplace
    ```
