#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <iostream>

namespace py = pybind11;

inline int get_center_coord(int idx, float cell_size) {
    return static_cast<int>((idx + 0.5f) * cell_size);
}

inline int sample_cell(const uint8_t* img,  const int start, const float cell_size, const int r, const int c, const int side) {
    int y_s = static_cast<int>(start + (r + 0.4f) * cell_size);
    int x_s = static_cast<int>(start + (c + 0.4f) * cell_size);
    int y_e = static_cast<int>(start + (r + 0.5f) * cell_size);
    int x_e = static_cast<int>(start + (c + 0.5f) * cell_size);

    y_s = std::max(y_s, 0);
    x_s = std::max(x_s, 0);
    y_e = std::min(y_e, side);
    x_e = std::min(x_e, side);

    if (y_s >= y_e || x_s >= x_e) {
        return 0;
    }

    int sum = 0;
    int count = 0;
    for (int y = y_s; y < y_e; y++) {
        for (int x = x_s; x < x_e; x++) {
            sum += img[y * side + x];
            count++;
        }
    }
    return (count == 0) ? 0 : (sum / count);
}

inline uint8_t clip_u8(float val) {
    return (val < 0.0f) ? 0 : (val > 255.0f) ? 255 : static_cast<uint8_t>(val);
}

inline double get_dist_sq(int x1, int y1, int x2, int y2) {
    double dx = static_cast<double>(x1) - static_cast<double>(x2);
    double dy = static_cast<double>(y1) - static_cast<double>(y2);
    return dx * dx + dy * dy;
}

inline void gaussian_blur_1d(const std::vector<float>& input, std::vector<float>& output, int size) {
    const float kernel[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
    output.resize(size);

    for (int i = 0; i < size; i++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        for (int k = -2; k <= 2; k++) {
            int idx = i + k;
            if (idx >= 0 && idx < size) {
                sum += input[idx] * kernel[k + 2];
                weight_sum += kernel[k + 2];
            }
        }
        output[i] = sum / weight_sum;
    }
}

bool normalize_invert_3x3(const double* mat, double* invMat) {
    double max = 0.0;
    for (int i = 0; i < 9; i++) {
        max = std::max(max, std::abs(mat[i]));
    }
    if (max < 1e-9) return false;

    double normMat[9];
    double scale = 1.0 / max;
    for (int i = 0; i < 9; i++) {
        normMat[i] = mat[i] * scale;
    }

    double det = mat[0] * (mat[4] * mat[8] - mat[5] * mat[7]) -
                 mat[1] * (mat[3] * mat[8] - mat[5] * mat[6]) +
                 mat[2] * (mat[3] * mat[7] - mat[4] * mat[6]);
    if (std::abs(det) < 1e-9) {
        return false;
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

    for (int i = 0; i < 9; i++) {
        invMat[i] *= scale;
    }

    return true;
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
    uint8_t* raw_dst = (uint8_t*)result.request().ptr;
    uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    double mat[9], invM[9];
    auto r_M = M.unchecked<2>();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i * 3 + j] = r_M(i, j);
        }
    }

    if (!normalize_invert_3x3(mat, invM)) {
        std::fill(raw_dst, raw_dst + d_w * d_h, 0);
        return result;
    }

    #pragma omp parallel for
    for (int y = 0; y < d_h; y++) {

        double num_x = invM[1] * y + invM[2];
        double num_y = invM[4] * y + invM[5];
        double denom = invM[7] * y + invM[8];

        const double row_num_x = invM[0];
        const double row_num_y = invM[3];
        const double row_den = invM[6];

        uint8_t* row_ptr = raw_dst + y * d_w;

        for (int x = 0; x < d_w; x++) {
            if (std::abs(denom) < 1e-6) {
                row_ptr[x] = 0;
                num_x += row_num_x;
                num_y += row_num_y;
                denom += row_den;
                continue;
            }

            double w_inv = 1.0 / denom;
            double src_x = num_x * w_inv;
            double src_y = num_y * w_inv;

            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);

            if (x0 >= 0 && x0 < s_w - 1 && y0 >= 0 && y0 < s_h - 1) {
                float dx = static_cast<float>(src_x - x0);
                float dy = static_cast<float>(src_y - y0);

                int offset = y0 * s_w + x0;
                uint8_t p00 = raw_src[offset];
                uint8_t p01 = raw_src[offset + 1];
                uint8_t p10 = raw_src[offset + s_w];
                uint8_t p11 = raw_src[offset + s_w + 1];

                float val = p00 * (1.0f - dx) * (1.0f - dy) +
                            p01 * dx * (1.0f - dy) +
                            p10 * (1.0f - dx) * dy +
                            p11 * dx * dy;
                row_ptr[x] = clip_u8(val);
            } else {
                row_ptr[x] = 0;
            }
            num_x += row_num_x;
            num_y += row_num_y;
            denom += row_den;
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

py::array_t<uint8_t> boxFilter_cpp(const py::array_t<uint8_t> src, const int ksize) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    const uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    std::vector<int> temp(s_h * s_w);
    auto result = py::array_t<uint8_t>({s_h, s_w});
    auto raw_dst = result.mutable_unchecked<2>();

    int half_ksize = ksize / 2;
    float norm = 1.0f / (ksize * ksize);

    #pragma omp parallel for
    for (int y = 0; y < s_h; y++) {
        const uint8_t* row_ptr = raw_src + y * s_w;
        int* temp_row_ptr = temp.data() + y * s_w;

        int sum = 0;
        for (int i = 0; i < ksize; i++) {
            sum += row_ptr[std::min(i, s_w - 1)];
        }

        for (int x = 0; x < s_w; x++) {
            temp_row_ptr[x] = sum;
            
            int idx_remove = x - half_ksize;
            int idx_add = x + half_ksize + 1;

            int idx_remove_clipped = std::max(0, idx_remove);
            int idx_add_clipped = std::min(s_w - 1, idx_add);

            sum -= row_ptr[idx_remove_clipped];
            sum += row_ptr[idx_add_clipped];
        }
    }

    #pragma omp parallel for
    for (int x = 0; x < s_w; x++) {
        int sum = 0;
        for (int i = 0; i < ksize; i++) {
            sum += temp[std::min(i, s_h - 1) * s_w + x];
        }
        for (int y = 0; y < s_h; y++) {
            raw_dst(y, x) = static_cast<uint8_t>(sum * norm);

            int idx_remove = y - half_ksize;
            int idx_add = y + half_ksize + 1;

            int idx_remove_clipped = std::max(0, idx_remove);
            int idx_add_clipped = std::min(s_h - 1, idx_add);

            sum -= temp[idx_remove_clipped * s_w + x];
            sum += temp[idx_add_clipped * s_w + x];
        }
    }
    return result;
}

py::array_t<uint8_t> gaussianBlur_cpp(const py::array_t<uint8_t> src, const int ksize, const float sigma) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    const uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    int half_ksize = ksize / 2;
    for (int i = 0; i < ksize; i++) {
        int x = i - half_ksize;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; i++) {
        kernel[i] /= sum;
    }

    std::vector<float> temp(s_h * s_w);
    auto result = py::array_t<uint8_t>({s_h, s_w});
    auto raw_dst = result.mutable_unchecked<2>();

    #pragma omp parallel for
    for (int y = 0; y < s_h; y++) {
        const uint8_t* row_ptr = raw_src + y * s_w;
        float* temp_row_ptr = temp.data() + y * s_w;

        for (int x = 0; x < s_w; x++) {
            float val = 0.0f;
            for (int k = -half_ksize; k <= half_ksize; k++) {
                int idx = std::min(std::max(x + k, 0), s_w - 1);
                val += row_ptr[idx] * kernel[k + half_ksize];
            }
            temp_row_ptr[x] = val;
        }
    }

    #pragma omp parallel for
    for (int x = 0; x < s_w; x++) {
        for (int y = 0; y < s_h; y++) {
            float val = 0.0f;
            for (int k = -half_ksize; k <= half_ksize; k++) {
                int idx = std::min(std::max(y + k, 0), s_h - 1);
                val += temp[idx * s_w + x] * kernel[k + half_ksize];
            }
            raw_dst(y, x) = clip_u8(val);
        }
    }
    return result;
}

py::array_t<uint8_t> cvtColor_cpp(const py::array_t<uint8_t> src) {
    auto buf_src = src.request();
    if (buf_src.ndim != 3 || buf_src.shape[2] != 3) {
        throw std::runtime_error("Input image must have 3 channels (H, W, 3)");
    }
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    const uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    auto result = py::array_t<uint8_t>({s_h, s_w});
    auto buf_res = result.request();
    uint8_t* raw_dst = (uint8_t*)buf_res.ptr;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < s_h; y++) {
        const uint8_t* row_ptr = raw_src + y * s_w * 3;
        uint8_t* dst_row_ptr = raw_dst + y * s_w;

        for (int x = 0; x < s_w; x++) {
            float gray = 0.333f * row_ptr[x * 3 + 2] + 0.333f * row_ptr[x * 3 + 1] + 0.333f * row_ptr[x * 3 + 0];
            dst_row_ptr[x] = clip_u8(gray);
        }
    }
    return result;
}

py::tuple decode_tag_cpp(const py::array_t<uint8_t> warped) {
    auto buf_warped = warped.request();

    if (buf_warped.ndim != 2)
    {
        return py::make_tuple(py::none(), py::none());
    }

    int SIDE = static_cast<int>(buf_warped.shape[0]);
    int width = static_cast<int>(buf_warped.shape[1]);
    if (SIDE != width || SIDE < 64) {
        return py::make_tuple(py::none(), py::none());
    }

    const uint8_t* raw_warped = static_cast<const uint8_t*>(buf_warped.ptr);

    float cell_size = static_cast<float>(SIDE) / 8.0f;
    int margin  = std::max(1, static_cast<int>(cell_size *0.5f));
    margin = std::min(margin, SIDE / 4);

    const int border_threshold = 150;

    for (int i = 0; i < 8; i++) {
        int idx = (int) (( i + 0.5f) * cell_size);
        if (idx < 0 || idx >= SIDE) continue;
        if (raw_warped[margin * SIDE + idx] > border_threshold) return py::make_tuple(py::none(), py::none());
        if (raw_warped[(SIDE - 1 - margin) * SIDE + idx] > border_threshold) return py::make_tuple(py::none(), py::none());
    }   

    for (int i = 0; i < 8; i++) {
        int idx = (int) (( i + 0.5f) * cell_size);
        if (idx < 0 || idx >= SIDE) continue;
        if (raw_warped[idx * SIDE + margin] > border_threshold) return py::make_tuple(py::none(), py::none());
        if (raw_warped[idx * SIDE + (SIDE - 1 - margin)] > border_threshold) return py::make_tuple(py::none(), py::none());
    }

    const int CORE_INDICES[4] = {2, 3, 4, 5};
    const float ORIENTATION_THRESHOLD = 155.0f;
    const float ADAPTIVE_THRESH_MIN_RANGE = 30.0f;
    const int MAX_TAG_ID = 15;

    uint8_t grid_bits[4][4] = {{0}};
    float grid_intensities[4][4] = {{0.0f}};

    for (int r_idx = 0; r_idx < 4; ++r_idx) {
        int grid_row = CORE_INDICES[r_idx];
        int y = get_center_coord(grid_row, cell_size);

        if (y < 0 || y >= SIDE) {
            return py::make_tuple(py::none(), py::none());
        }

        std::vector<float> row_signal(SIDE);
        for (int x = 0; x < SIDE; ++x) {
            row_signal[x] = static_cast<float>(raw_warped[y * SIDE + x]);
        }

        std::vector<float> blurred_row;
        gaussian_blur_1d(row_signal, blurred_row, SIDE);

        float row_value[4];
        for (int c_idx = 0; c_idx < 4; ++c_idx) {
            int x = get_center_coord(CORE_INDICES[c_idx], cell_size);
            if (x < 0 || x >= SIDE) {
                return py::make_tuple(py::none(), py::none());
            }

            row_value[c_idx] = blurred_row[x];
        }

        int data_start_x = static_cast<int>(2 * cell_size);
        int data_end_x = static_cast<int>(6 * cell_size);

        data_start_x = std::max(0, std::min(data_start_x, SIDE - 1));
        data_end_x = std::max(data_start_x + 1, std::min(data_end_x, SIDE));

        float local_min = blurred_row[data_start_x];
        float local_max = blurred_row[data_start_x];

        for (int x = data_start_x; x < data_end_x; x++) {
            local_min = std::min(local_min, blurred_row[x]);
            local_max = std::max(local_max, blurred_row[x]);
        }

        float row_threshold = (local_min + local_max) * 0.5f;

        if ((local_max - local_min) < ADAPTIVE_THRESH_MIN_RANGE) {
            row_threshold = 155.0f;
        }

        for (int c_idx = 0; c_idx < 4; ++c_idx) {
            grid_intensities[r_idx][c_idx] = row_value[c_idx];
            grid_bits[r_idx][c_idx] = (row_value[c_idx] > row_threshold) ? 1 : 0;
        }
    }

    float anchor_values[4] = {
        grid_intensities[3][3],
        grid_intensities[3][0],
        grid_intensities[0][0],
        grid_intensities[0][3]
    };

    int status = 0;
    float max_val = anchor_values[0];
    for (int i = 1; i < 4; i++) {
        if (anchor_values[i] > max_val) {
            max_val = anchor_values[i];
            status = i;
        }
    }

    if (max_val < ORIENTATION_THRESHOLD) {
        return py::make_tuple(py::none(), py::none());
    }

    const int bit_map[4][4][2] = {
        {{1, 1}, {1, 2}, {2, 2}, {2, 1}},
        {{1, 2}, {2, 2}, {2, 1}, {1, 1}},
        {{2, 2}, {2, 1}, {1, 1}, {1, 2}},
        {{2, 1}, {1, 1}, {1, 2}, {2, 2}}
    };

    int tag_id = 0;
    for (int r = 0; r < 4; ++r) {
        int br = bit_map[status][r][0];
        int bc = bit_map[status][r][1];
        uint8_t bit = grid_bits[br][bc];
        tag_id |= (bit << r);
    }

    if (tag_id == 12)
    {
        for (int x = 0; x < 4; ++x) {
            for (int y = 0; y < 4; ++y) {
                std::cout << (int)grid_intensities[x][y] << " ";
            }
            std::cout << std::endl;
        }
    }

    if (tag_id < 0 || tag_id > MAX_TAG_ID) {
        return py::make_tuple(py::none(), py::none());
    }
    return py::make_tuple(tag_id, status);
}

py::array_t<uint8_t> bilateralFilter_cpp(const py::array_t<uint8_t> src, int d, double sigmaColor, double sigmaSpace) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];

    py::array_t<uint8_t> result = py::array_t<uint8_t>({s_h, s_w});
    const uint8_t* raw_src = (uint8_t*)buf_src.ptr;
    uint8_t* raw_dst = (uint8_t*)result.request().ptr;

    int radius = d / 2;
    int kernel_size = 2 * radius + 1;

    std::vector<float> space_weights(kernel_size * kernel_size);
    std::vector<float> space_offs_x(kernel_size * kernel_size);
    std::vector<float> space_offs_y(kernel_size * kernel_size);
    float space_coeff = -0.5f / (sigmaSpace * sigmaSpace);

    int k_idx = 0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float r2 = static_cast<float>(x * x + y * y);
            space_weights[k_idx] = std::exp(r2 * space_coeff);
            space_offs_x[k_idx] = x;
            space_offs_y[k_idx] = y;
            k_idx++;
        }
    }

    float range_coeff = -0.5f / (sigmaColor * sigmaColor);
    std::vector<float> range_weights(256);
    for (int i = 0; i < 256; i++) {
        range_weights[i] = std::exp(i * i * range_coeff);
    }

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < s_h; y++) {
        const uint8_t* row_ptr = raw_src + y * s_w;
        uint8_t* dst_row_ptr = raw_dst + y * s_w;

        for (int x = 0; x < s_w; x++) {
            float sum = 0.0f;
            float w_sum = 0.0f;
            int center_val = row_ptr[x];

            for (int k = 0; k < kernel_size * kernel_size; k++) {
                int nx = x + static_cast<int>(space_offs_x[k]);
                int ny = y + static_cast<int>(space_offs_y[k]);

                if (nx >= 0 && nx < s_w && ny >= 0 && ny < s_h) {
                    int neighbor_val = raw_src[ny * s_w + nx];
                    float weight = space_weights[k] * range_weights[std::abs(neighbor_val - center_val)];
                    sum += neighbor_val * weight;
                    w_sum += weight;
                }
            }
            dst_row_ptr[x] = clip_u8(sum / w_sum);
        }
    }
    return result;
}

py::array_t<uint8_t> sharpenAndNormalize_cpp(const py::array_t<uint8_t> src) {
    auto buf_src = src.request();
    int s_h = buf_src.shape[0];
    int s_w = buf_src.shape[1];
    const uint8_t* raw_src = (uint8_t*)buf_src.ptr;

    auto result = py::array_t<uint8_t>({s_h, s_w});
    auto raw_dst = (uint8_t*)result.request().ptr;

    std::vector<float> temp(s_h * s_w);
    float min_val = 1e9f;
    float max_val = -1e9f;

    #pragma omp parallel for collapse(2) reduction(min:min_val) reduction(max:max_val)
    for (int y = 0; y < s_h; y++) {
        for (int x = 0; x < s_w; x++) {
            int c = raw_src[y * s_w + x];
            int u = (y > 0) ? raw_src[(y - 1) * s_w + x] : c;
            int d = (y < s_h - 1) ? raw_src[(y + 1) * s_w + x] : c;
            int l = (x > 0) ? raw_src[y * s_w + (x - 1)] : c;
            int r = (x < s_w - 1) ? raw_src[y * s_w + (x + 1)] : c;

            float val = 5.0f * c - u - d - l - r;

            temp[y * s_w + x] = val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    float range = max_val - min_val;
    if (range < 1e-6f) {
        range = 1.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < s_h * s_w; i++) {
        raw_dst[i] = clip_u8((temp[i] - min_val) * 255.0f / range);
    }

    return result;
}

py::array_t<float> cornerSubPix_cpp(const py::array_t<uint8_t> image, py::array_t<float> corners, int win_w, int win_h, int max_iter, double epsilon) {
    auto buf_image = image.request();
    int s_h = buf_image.shape[0];
    int s_w = buf_image.shape[1];
    const uint8_t* raw_image = (uint8_t*)buf_image.ptr;

    auto buf_corners = corners.request();
    int n_corners = buf_corners.shape[0];

    py::array_t<float> result = py::array_t<float>({n_corners, 1, 2});
    auto r_corners = corners.unchecked<3>();
    auto r_result = result.mutable_unchecked<3>();

    for (int i = 0; i < n_corners; i++) {
        r_result(i, 0, 0) = r_corners(i, 0, 0);
        r_result(i, 0, 1) = r_corners(i, 0, 1);
    }

    for (int i = 0; i < n_corners; i++) {
        float cx = r_result(i, 0, 0);
        float cy = r_result(i, 0, 1);

        for (int iter = 0; iter < max_iter; iter++) {
            float gx = 0, gy = 0;
            double Gxx = 0, Gxy = 0, Gyy = 0;
            double bx = 0, by = 0;

            int Ix = static_cast<int>(cx);
            int Iy = static_cast<int>(cy);

            int left = std::max(1, Ix - win_w);
            int right = std::min(s_w - 2, Ix + win_w);
            int top = std::max(1, Iy - win_h);
            int bottom = std::min(s_h - 2, Iy + win_h);

            for (int y = top; y <= bottom; y++) {
                const uint8_t* row_ptr = raw_image + y * s_w;
                for (int x = left; x <= right; x++) {
                    float dx = 0.5f * (row_ptr[x + 1] - row_ptr[x - 1]);
                    float dy = 0.5f * (raw_image[(y + 1) * s_w + x] - raw_image[(y - 1) * s_w + x]);

                    gx += dx;
                    gy += dy;

                    Gxx += dx * dx;
                    Gxy += dx * dy;
                    Gyy += dy * dy;

                    bx += dx * dx * x + dx * dy * y;
                    by += dx * dy * x + dy * dy * y;
                }
            }

            double det = Gxx * Gyy - Gxy * Gxy;
            if (std::abs(det) < 1e-9) {
                break;
            }
            double inv_det = 1.0 / det;
            double new_cx = (Gyy * bx - Gxy * by) * inv_det;
            double new_cy = (Gxx * by - Gxy * bx) * inv_det;

            double diff_x = new_cx - cx;
            double diff_y = new_cy - cy;

            cx = static_cast<float>(new_cx);
            cy = static_cast<float>(new_cy);

            if (diff_x * diff_x + diff_y * diff_y < epsilon * epsilon) {
                break;
            }
        }
        r_result(i, 0, 0) = cx;
        r_result(i, 0, 1) = cy;
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
    
    m.def("boxFilter_cpp", &boxFilter_cpp, 
          py::arg("src"), py::arg("ksize"),
          "Applies a box filter to the input image with the specified kernel size.");
          
    m.def("gaussianBlur_cpp", &gaussianBlur_cpp,
          py::arg("src"), py::arg("ksize"), py::arg("sigma"),
          "Applies Gaussian blur to the input image with the specified kernel size and sigma.");

    m.def("cvtColor_cpp", &cvtColor_cpp,
          py::arg("src"),
          "Converts a BGR image to grayscale.");
    
    m.def("decode_tag_cpp", &decode_tag_cpp,
          py::arg("warped"),
          "Decodes a tag from a warped grayscale image and returns the tag ID and orientation status.");

    m.def("bilateralFilter_cpp", &bilateralFilter_cpp,
          py::arg("src"), py::arg("d"), py::arg("sigmaColor"), py::arg("sigmaSpace"),
          "Applies a bilateral filter to the input image with the specified diameter, sigmaColor, and sigmaSpace.");

    m.def("sharpenAndNormalize_cpp", &sharpenAndNormalize_cpp,
          py::arg("src"),
          "Sharpens the input image using a simple kernel and normalizes the result to the full 0-255 range.");

    m.def("cornerSubPix_cpp", &cornerSubPix_cpp,
          py::arg("image"), py::arg("corners"), py::arg("win_w"), py::arg("win_h"), py::arg("max_iter"), py::arg("epsilon"),
          "Refines corner locations to sub-pixel accuracy using the specified window size, maximum iterations, and convergence epsilon.");
}