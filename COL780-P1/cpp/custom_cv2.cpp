#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>

namespace py = pybind11;

inline uint8_t clip_u8(float val) {
    return (val < 0.0f) ? 0 : (val > 255.0f) ? 255 : static_cast<uint8_t>(val);
}

inline double get_dist_sq(int x1, int y1, int x2, int y2) {
    double dx = static_cast<double>(x1) - static_cast<double>(x2);
    double dy = static_cast<double>(y1) - static_cast<double>(y2);
    return dx * dx + dy * dy;
}

void invert_3x3(const double* mat, double* invMat) {
    double det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) -
                 mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
                 mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);

    if (std::abs(det) < 1e-9) {
        std::cerr << "Warning: Singular matrix, cannot invert. Returning identity matrix." << std::endl;
        for (int i = 0; i < 9; i++) {
            invMat[i] = (i % 4 == 0) ? 1.0 : 0.0;
        }
        return;
    }

    double invDet = 1.0 / det;
    invMat[0] = (mat[4] * mat[8] - mat[5] * mat[7]) * invDet;
    invMat[1] = (mat[2] * mat[7] - mat[1] * mat[8]) * invDet;
    invMat[2] = (mat[1] * mat[5] - mat[2] * mat[4]) * invDet;
    invMat[3] = (mat[5] * mat[6] - mat[3] * mat[8]) * invDet;
    invMat[4] = (mat[0] * mat[8] - mat[2] * mat[6]) * invDet;
    invMat[5] = (mat[2] * mat[3] - mat[0] * mat[5]) * invDet;
    invMat[6] = (mat[3] * mat[7] - mat[4] * mat[6]) * invDet;
    invMat[7] = (mat[1] * mat[6] - mat[0] * mat[7]) * invDet;
    invMat[8] = (mat[0] * mat[4] - mat[1] * mat[3]) * invDet;
}

double perpendicular_distance(int x, int y, int x1, int y1, int x2, int y2) {
    double l2 = get_dist_sq(x1, y1, x2, y2);
    if (l2 < 1e-9) {
        return std::sqrt(get_dist_sq(x, y, x1, y1));
    }

    double t = (static_cast<double>(x - x1) * (x2 - x1) + 
                static_cast<double>(y - y1) * (y2 - y1)) / l2;
    t = std::max(0.0, std::min(1.0, t));

    double proj_x = x1 + t * (x2 - x1);
    double proj_y = y1 + t * (y2 - y1);

    double dx = x - proj_x;
    double dy = y - proj_y;

    return std::sqrt(dx * dx + dy * dy);
}

void dp_recursive(const std::vector<std::array<int, 2>>& points, int start, int end, double epsilon, std::vector<bool>& keep) {
    if (end <= start + 1) {
        return;
    }

    double max_dist = 0.0;
    int index = start;

    for (int i = start + 1; i < end; i++) {
        double dist = perpendicular_distance(points[i][0], points[i][1],
                                             points[start][0], points[start][1],
                                             points[end][0], points[end][1]);
        if (dist > max_dist) {
            index = i;
            max_dist = dist;
        }
    }

    if (max_dist > epsilon) {
        keep[index] = true;
        dp_recursive(points, start, index, epsilon, keep);
        dp_recursive(points, index, end, epsilon, keep);
    }
}

py::array_t<uint8_t> warpPerspective_cpp(const py::array_t<uint8_t> src, const py::array_t<double> M, int d_w, int d_h) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    auto result = py::array_t<uint8_t>({d_h, d_w});

    uint8_t* raw_src = (uint8_t*)buf_src.ptr;
    uint8_t* raw_dst = (uint8_t*)result.request().ptr;

    double mat[9], invM[9];
    auto r_M = M.unchecked<2>();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i * 3 + j] = r_M(i, j);
        }
    }
    invert_3x3(mat, invM);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < d_h; y++) {
        for (int x = 0; x < d_w; x++) {
            double denom = invM[6] * x + invM[7] * y + invM[8];
            
            if (std::abs(denom) < 1e-9) {
                raw_dst[y * d_w + x] = 0;
                continue;
            }

            double src_x = (invM[0] * x + invM[1] * y + invM[2]) / denom;
            double src_y = (invM[3] * x + invM[4] * y + invM[5]) / denom;

            src_x = std::max(0.0, std::min(src_x, static_cast<double>(s_w - 1)));
            src_y = std::max(0.0, std::min(src_y, static_cast<double>(s_h - 1)));

            int x0 = static_cast<int>(std::floor(src_x));
            int y0 = static_cast<int>(std::floor(src_y));

            int x1 = std::min(x0 + 1, s_w - 1);
            int y1 = std::min(y0 + 1, s_h - 1);

            float dx = static_cast<float>(src_x - x0);
            float dy = static_cast<float>(src_y - y0);

            x0 = std::max(0, std::min(x0, s_w - 1));
            x1 = std::max(0, std::min(x1, s_w - 1));
            y0 = std::max(0, std::min(y0, s_h - 1));
            y1 = std::max(0, std::min(y1, s_h - 1));

            float val = raw_src[y0 * s_w + x0] * (1.0f - dx) * (1.0f - dy) +
                        raw_src[y0 * s_w + x1] * dx * (1.0f - dy) +
                        raw_src[y1 * s_w + x0] * (1.0f - dx) * dy +
                        raw_src[y1 * s_w + x1] * dx * dy;
            raw_dst[y * d_w + x] = clip_u8(val);
        }
    }

    return result;
}

py::list findContours_cpp(const py::array_t<uint8_t> src, int min_points) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    std::vector<uint8_t> visited(s_h * s_w, 0);
    py::list contours;

    int dir_x[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    int dir_y[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    for (int y = 1; y < s_h - 1; y++) {
        uint8_t* row_ptr = raw_src + y * s_w;
        uint8_t* visited_row_ptr = visited.data() + y * s_w;

        for (int x = 1; x < s_w - 1; x++) {
            if (row_ptr[x] != 0 && visited_row_ptr[x] == 0) {
                bool is_border = (row_ptr[x - 1] == 0 || row_ptr[x + 1] == 0 ||
                                 raw_src[(y - 1) * s_w + x] == 0 || raw_src[(y + 1) * s_w + x] == 0);
                if (is_border) {
                    std::vector<std::array<int, 2>> contour;
                    contour.reserve(256);
                    
                    int cy = y, cx = x;
                    int start_x = x, start_y = y;
                    int backtrack = 0;

                    while (true) {
                        contour.push_back({cx - 1, cy - 1});
                        visited[cy * s_w + cx] = 1;

                        bool found_next = false;
                        for (int d = 0; d < 8; d++) {
                            int nd = (backtrack + d) % 8;
                            int nx = cx + dir_x[nd];
                            int ny = cy + dir_y[nd];
                            
                            if (nx >= 1 && nx < s_w - 1 && ny >= 1 && ny < s_h - 1 &&
                                raw_src[ny * s_w + nx] != 0) {
                                cx = nx;
                                cy = ny;
                                backtrack = (nd + 5) % 8;
                                found_next = true;
                                break;
                            }
                        }
                        
                        if (!found_next || (cx == start_x && cy == start_y)) {
                            break;
                        }
                    }

                    if (contour.size() >= static_cast<size_t>(std::max(min_points, 3))) {
                        py::array_t<int> contour_array({static_cast<int>(contour.size()), 1, 2});
                        auto r_contour = contour_array.mutable_unchecked<3>();
                        for (size_t i = 0; i < contour.size(); i++) {
                            r_contour(i, 0, 0) = contour[i][0];
                            r_contour(i, 0, 1) = contour[i][1];
                        }
                    contours.append(contour_array);
                    }
                }
            }
        }
    }
    return contours;
}

py::array_t<int> approxPolyDP_cpp(const py::array_t<int>& curve, double epsilon, bool closed = false) {
    auto buf_curve = curve.request();
    if (buf_curve.ndim != 3 || buf_curve.shape[1] != 1 || buf_curve.shape[2] != 2) {
        throw std::runtime_error("Input curve must have shape (N, 1, 2)");
    }

    int n_points = buf_curve.shape[0];
    if (n_points < 3) {
        return curve;
    }
    auto r_curve = curve.unchecked<3>();

    std::vector<std::array<int, 2>> points(n_points);
    for (int i = 0; i < n_points; i++) {
        points[i][0] = r_curve(i, 0, 0);
        points[i][1] = r_curve(i, 0, 1);
    }

    std::vector<bool> keep(n_points, false);
    keep[0] = true;
    keep[n_points - 1] = true;

    dp_recursive(points, 0, n_points - 1, epsilon, keep);

    std::vector<std::array<int, 2>> result_points;
    result_points.reserve(n_points);
    for (int i = 0; i < n_points; i++) {
        if (keep[i]) {
            result_points.push_back(points[i]);
        }
    }

    if (closed && result_points.size() > 1) {
        auto& first = result_points[0];
        auto& last = result_points[result_points.size() - 1];

        double dx = std::abs(static_cast<double>(first[0]) - static_cast<double>(last[0]));
        double dy = std::abs(static_cast<double>(first[1]) - static_cast<double>(last[1]));
        if (dx <= 1.0 && dy <= 1.0) {
            result_points.pop_back();
        }
    }

    if (result_points.size() < 3) {
        result_points.clear();
        for (int i = 0; i < n_points; i++) {
            if (keep[i]) {
                result_points.push_back(points[i]);
            }
        }
    }

    py::array_t<int> result_array({static_cast<int>(result_points.size()), 1, 2});
    auto r_result = result_array.mutable_unchecked<3>();
    for (size_t i = 0; i < result_points.size(); i++) {
        r_result(i, 0, 0) = result_points[i][0];
        r_result(i, 0, 1) = result_points[i][1];
    }

    return result_array;
}

py::array_t<uint8_t> adaptiveThreshold_cpp(const py::array_t<uint8_t> src, double maxValue, int blockSize, double C) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    auto result = py::array_t<uint8_t>({s_h, s_w});
    auto raw_dst = result.mutable_unchecked<2>();

    std::vector<int> integral((s_h + 1) * (s_w + 1), 0);
    int stride = s_w + 1;

    for (int y = 1; y <= s_h; y++) {
        int row_sum = 0;
        for (int x = 1; x <= s_w; x++) {
            row_sum += raw_src[(y - 1) * s_w + (x - 1)];
            integral[y * stride + x] = integral[(y - 1) * stride + x] + row_sum;
        }
    }

    int half_block = blockSize / 2;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < s_h; y++) {
        for (int x = 0; x < s_w; x++) {
            int x1 = std::max(0, x - half_block);
            int y1 = std::max(0, y - half_block);
            int x2 = std::min(s_w - 1, x + half_block);
            int y2 = std::min(s_h - 1, y + half_block);

            int count = (x2 - x1 + 1) * (y2 - y1 + 1);

            int sum = integral[(y2 + 1) * stride + (x2 + 1)]
                      - integral[(y1) * stride + (x2 + 1)]
                      - integral[(y2 + 1) * stride + (x1)]
                      + integral[(y1) * stride + (x1)];

            if (static_cast<int>(raw_src[y * s_w + x]) * count < sum - C * count) {
                raw_dst(y, x) = static_cast<uint8_t>(maxValue);
            } else {
                raw_dst(y, x) = 0;
            }
        }
    }

    return result;
}

PYBIND11_MODULE(custom_cv2_cpp, m) {
    m.doc() = "Custom OpenCV-like functions implemented in C++ with OpenMP";

    m.def("warpPerspective_cpp", &warpPerspective_cpp, 
          py::arg("src"), py::arg("M"), py::arg("d_w"), py::arg("d_h"),
          "Applies a perspective warp to the input image using the given transformation matrix M.");

    m.def("findContours_cpp", &findContours_cpp, 
          py::arg("src"), py::arg("min_points") = 10,
          "Finds contours in a binary image and returns those with number of points >= min_points.");

    m.def("approxPolyDP_cpp", &approxPolyDP_cpp, 
          py::arg("curve"), py::arg("epsilon"), py::arg("closed") = false,
          "Approximates a polygonal curve with the specified precision epsilon using the Douglas-Peucker algorithm.");

    m.def("adaptiveThreshold_cpp", &adaptiveThreshold_cpp, 
          py::arg("src"), py::arg("maxValue"), py::arg("blockSize"), py::arg("C"),
          "Applies adaptive thresholding to the input grayscale image.");
}