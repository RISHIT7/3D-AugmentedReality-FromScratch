#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

inline int get_center_coord(int idx, float cell_size) {
  return static_cast<int>((idx + 0.5f) * cell_size);
}

inline int sample_cell(const uint8_t *img, const int start,
                       const float cell_size, const int r, const int c,
                       const int side) {
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

inline void gaussian_blur_1d(const std::vector<float> &input,
                             std::vector<float> &output, int size) {
  const float kernel[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f,
                           4.0f / 16.0f, 1.0f / 16.0f};
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

bool normalize_invert_3x3(const double *mat, double *invMat) {
  double maxVal = 0.0;
  for (int i = 0; i < 9; i++) {
    maxVal = std::max(maxVal, std::abs(mat[i]));
  }
  if (maxVal < 1e-15)
    return false;

  double normMat[9];
  double scale = 1.0 / maxVal;
  for (int i = 0; i < 9; i++) {
    normMat[i] = mat[i] * scale;
  }

  double det =
      normMat[0] * (normMat[4] * normMat[8] - normMat[5] * normMat[7]) -
      normMat[1] * (normMat[3] * normMat[8] - normMat[5] * normMat[6]) +
      normMat[2] * (normMat[3] * normMat[7] - normMat[4] * normMat[6]);
  if (std::abs(det) < 1e-15) {
    return false;
  }

  double invDet = 1.0 / det;
  invMat[0] = (normMat[4] * normMat[8] - normMat[5] * normMat[7]) * invDet;
  invMat[1] = (normMat[2] * normMat[7] - normMat[1] * normMat[8]) * invDet;
  invMat[2] = (normMat[1] * normMat[5] - normMat[2] * normMat[4]) * invDet;
  invMat[3] = (normMat[5] * normMat[6] - normMat[3] * normMat[8]) * invDet;
  invMat[4] = (normMat[0] * normMat[8] - normMat[2] * normMat[6]) * invDet;
  invMat[5] = (normMat[2] * normMat[3] - normMat[0] * normMat[5]) * invDet;
  invMat[6] = (normMat[3] * normMat[7] - normMat[4] * normMat[6]) * invDet;
  invMat[7] = (normMat[1] * normMat[6] - normMat[0] * normMat[7]) * invDet;
  invMat[8] = (normMat[0] * normMat[4] - normMat[1] * normMat[3]) * invDet;

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
              static_cast<double>(y - y1) * (y2 - y1)) /
             l2;
  t = std::max(0.0, std::min(1.0, t));

  double proj_x = x1 + t * (x2 - x1);
  double proj_y = y1 + t * (y2 - y1);

  double dx = x - proj_x;
  double dy = y - proj_y;

  return std::sqrt(dx * dx + dy * dy);
}

void dp_recursive(const std::vector<std::array<int, 2>> &points, int start,
                  int end, double epsilon, std::vector<bool> &keep) {
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

py::array_t<uint8_t> warpPerspective_cpp(const py::array_t<uint8_t> src,
                                         const py::array_t<double> M, int d_w,
                                         int d_h) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];

  auto result = py::array_t<uint8_t>({d_h, d_w});
  uint8_t *raw_dst = (uint8_t *)result.request().ptr;
  uint8_t *raw_src = (uint8_t *)buf_src.ptr;

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

    uint8_t *row_ptr = raw_dst + y * d_w;

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

        float val = p00 * (1.0f - dx) * (1.0f - dy) + p01 * dx * (1.0f - dy) +
                    p10 * (1.0f - dx) * dy + p11 * dx * dy;
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
  uint8_t *raw_src = (uint8_t *)buf_src.ptr;

  std::vector<uint8_t> visited(s_h * s_w, 0);
  py::list contours;

  int dir_x[8] = {0, 1, 1, 1, 0, -1, -1, -1};
  int dir_y[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

  for (int y = 1; y < s_h - 1; y++) {
    uint8_t *row_ptr = raw_src + y * s_w;
    uint8_t *visited_row_ptr = visited.data() + y * s_w;

    for (int x = 1; x < s_w - 1; x++) {
      if (row_ptr[x] != 0 && visited_row_ptr[x] == 0) {
        bool is_border = (row_ptr[x - 1] == 0 || row_ptr[x + 1] == 0 ||
                          raw_src[(y - 1) * s_w + x] == 0 ||
                          raw_src[(y + 1) * s_w + x] == 0);
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
            py::array_t<int> contour_array(
                {static_cast<int>(contour.size()), 1, 2});
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

py::array_t<int> approxPolyDP_cpp(const py::array_t<int> &curve, double epsilon,
                                  bool closed = false) {
  auto buf_curve = curve.request();
  if (buf_curve.ndim != 3 || buf_curve.shape[1] != 1 ||
      buf_curve.shape[2] != 2) {
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
    auto &first = result_points[0];
    auto &last = result_points[result_points.size() - 1];

    double dx =
        std::abs(static_cast<double>(first[0]) - static_cast<double>(last[0]));
    double dy =
        std::abs(static_cast<double>(first[1]) - static_cast<double>(last[1]));
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

py::array_t<uint8_t> adaptiveThreshold_cpp(const py::array_t<uint8_t> src,
                                           double maxValue, int blockSize,
                                           double C) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];
  uint8_t *raw_src = (uint8_t *)buf_src.ptr;

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

      int sum = integral[(y2 + 1) * stride + (x2 + 1)] -
                integral[(y1)*stride + (x2 + 1)] -
                integral[(y2 + 1) * stride + (x1)] +
                integral[(y1)*stride + (x1)];

      if (static_cast<int>(raw_src[y * s_w + x]) * count < sum - C * count) {
        raw_dst(y, x) = static_cast<uint8_t>(maxValue);
      } else {
        raw_dst(y, x) = 0;
      }
    }
  }

  return result;
}

py::array_t<uint8_t> boxFilter_cpp(const py::array_t<uint8_t> src,
                                   const int ksize) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];
  const uint8_t *raw_src = (uint8_t *)buf_src.ptr;

  std::vector<int> temp(s_h * s_w);
  auto result = py::array_t<uint8_t>({s_h, s_w});
  auto raw_dst = result.mutable_unchecked<2>();

  int half_ksize = ksize / 2;
  float norm = 1.0f / (ksize * ksize);

#pragma omp parallel for
  for (int y = 0; y < s_h; y++) {
    const uint8_t *row_ptr = raw_src + y * s_w;
    int *temp_row_ptr = temp.data() + y * s_w;

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

py::array_t<uint8_t> gaussianBlur_cpp(const py::array_t<uint8_t> src,
                                      const int ksize, const float sigma) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];
  const uint8_t *raw_src = (uint8_t *)buf_src.ptr;

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
  uint8_t *raw_dst = (uint8_t *)result.request().ptr;

#pragma omp parallel for
  for (int y = 0; y < s_h; y++) {
    const uint8_t *row_ptr = raw_src + y * s_w;
    float *temp_row_ptr = temp.data() + y * s_w;

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
  for (int y = 0; y < s_h; y++) {
    for (int x = 0; x < s_w; x++) {
      float val = 0.0f;
      for (int k = -half_ksize; k <= half_ksize; k++) {
        int iy = std::min(std::max(y + k, 0), s_h - 1);
        val += temp[iy * s_w + x] * kernel[k + half_ksize];
      }
      raw_dst[y * s_w + x] = clip_u8(val);
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
  const uint8_t *raw_src = (uint8_t *)buf_src.ptr;

  auto result = py::array_t<uint8_t>({s_h, s_w});
  auto buf_res = result.request();
  uint8_t *raw_dst = (uint8_t *)buf_res.ptr;

#pragma omp parallel for collapse(2)
  for (int y = 0; y < s_h; y++) {
    const uint8_t *row_ptr = raw_src + y * s_w * 3;
    uint8_t *dst_row_ptr = raw_dst + y * s_w;

    for (int x = 0; x < s_w; x++) {
      float gray = 0.333f * row_ptr[x * 3 + 2] + 0.333f * row_ptr[x * 3 + 1] +
                   0.333f * row_ptr[x * 3 + 0];
      dst_row_ptr[x] = clip_u8(gray);
    }
  }
  return result;
}

static uint8_t otsu_threshold_value(const uint8_t *data, int n) {
  int hist[256] = {0};
  for (int i = 0; i < n; i++)
    hist[data[i]]++;
  float sum = 0;
  for (int i = 0; i < 256; i++)
    sum += i * hist[i];
  float sum_b = 0;
  int w_b = 0;
  float max_var = 0;
  int best_t = 0;
  for (int t = 0; t < 256; t++) {
    w_b += hist[t];
    if (w_b == 0)
      continue;
    int w_f = n - w_b;
    if (w_f == 0)
      break;
    sum_b += t * hist[t];
    float mu_b = sum_b / w_b;
    float mu_f = (sum - sum_b) / w_f;
    float diff = mu_b - mu_f;
    float var = (float)w_b * w_f * diff * diff;
    if (var > max_var) {
      max_var = var;
      best_t = t;
    }
  }
  return (uint8_t)best_t;
}

static void sharpen_normalize_inplace(const uint8_t *src, uint8_t *sharpened,
                                      uint8_t *thresholded, int side) {
  std::vector<float> temp(side * side);
  float mn = 1e9f, mx = -1e9f;
  for (int y = 0; y < side; y++) {
    for (int x = 0; x < side; x++) {
      int c = src[y * side + x];
      int u = (y > 0) ? src[(y - 1) * side + x] : c;
      int d = (y < side - 1) ? src[(y + 1) * side + x] : c;
      int l = (x > 0) ? src[y * side + x - 1] : c;
      int r = (x < side - 1) ? src[y * side + x + 1] : c;
      float val = 5.0f * c - u - d - l - r;
      temp[y * side + x] = val;
      mn = std::min(mn, val);
      mx = std::max(mx, val);
    }
  }
  float range = mx - mn;
  if (range < 1e-6f)
    range = 1.0f;
  for (int i = 0; i < side * side; i++)
    sharpened[i] = clip_u8((temp[i] - mn) * 255.0f / range);
  uint8_t otsu_t = otsu_threshold_value(sharpened, side * side);
  for (int i = 0; i < side * side; i++)
    thresholded[i] = (sharpened[i] > otsu_t) ? 255 : 0;
}

static void erode_inplace(const uint8_t *src, uint8_t *dst, int h, int w,
                          int ksize, int iterations) {
  int pad = ksize / 2;
  std::vector<uint8_t> buf_a(src, src + h * w), buf_b(h * w);
  uint8_t *cur = buf_a.data(), *nxt = buf_b.data();
  for (int iter = 0; iter < iterations; iter++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        uint8_t mn = 255;
        for (int ky = -pad; ky <= pad; ky++) {
          for (int kx = -pad; kx <= pad; kx++) {
            int ny = std::max(0, std::min(h - 1, y + ky));
            int nx = std::max(0, std::min(w - 1, x + kx));
            mn = std::min(mn, cur[ny * w + nx]);
          }
        }
        nxt[y * w + x] = mn;
      }
    }
    std::swap(cur, nxt);
  }
  std::memcpy(dst, cur, h * w);
}

static float patch_median(const uint8_t *img, int h, int w, int cy, int cx,
                          int half) {
  int y0 = std::max(0, cy - half);
  int y1 = std::min(h, cy + half + 1);
  int x0 = std::max(0, cx - half);
  int x1 = std::min(w, cx + half + 1);
  std::vector<uint8_t> vals;
  for (int y = y0; y < y1; y++)
    for (int x = x0; x < x1; x++)
      vals.push_back(img[y * w + x]);
  if (vals.empty())
    return 0;
  std::sort(vals.begin(), vals.end());
  return (float)vals[vals.size() / 2];
}

py::tuple decode_tag_cpp(const py::array_t<uint8_t> warped) {
  auto buf_warped = warped.request();
  if (buf_warped.ndim != 2)
    return py::make_tuple(py::none(), py::none(), -2);

  int SIDE = (int)buf_warped.shape[0];
  int width = (int)buf_warped.shape[1];
  if (SIDE != width || SIDE < 64)
    return py::make_tuple(py::none(), py::none(), -2);

  const uint8_t *raw = (const uint8_t *)buf_warped.ptr;

  std::vector<uint8_t> sharpened(SIDE * SIDE), threshold_img(SIDE * SIDE);
  sharpen_normalize_inplace(raw, sharpened.data(), threshold_img.data(), SIDE);

  float cell = (float)SIDE / 8.0f;
  int margin = std::max(1, (int)(cell / 2.0f));
  margin = std::min(margin, SIDE / 4);

  for (int i = 0; i < 1; i++) {
    int idx = (int)((i + 0.5f) * cell);
    idx = std::max(0, std::min(idx, SIDE - 1));
    if (threshold_img[margin * SIDE + idx] > 127)
      return py::make_tuple(py::none(), py::none(), -1);
    if (threshold_img[(SIDE - 1 - margin) * SIDE + idx] > 127)
      return py::make_tuple(py::none(), py::none(), -1);
    if (threshold_img[idx * SIDE + margin] > 127)
      return py::make_tuple(py::none(), py::none(), -1);
    if (threshold_img[idx * SIDE + (SIDE - 1 - margin)] > 127)
      return py::make_tuple(py::none(), py::none(), -1);
  }

  const int CI[4] = {2, 3, 4, 5};
  uint8_t grid_bits[4][4] = {{0}};

  for (int r_idx = 0; r_idx < 4; r_idx++) {
    int y = get_center_coord(CI[r_idx], cell);
    if (y < 0 || y >= SIDE)
      return py::make_tuple(py::none(), py::none(), -2);

    std::vector<float> row_sig(SIDE);
    for (int x = 0; x < SIDE; x++)
      row_sig[x] = (float)threshold_img[y * SIDE + x];

    std::vector<float> blurred;
    gaussian_blur_1d(row_sig, blurred, SIDE);

    int ds = std::max(0, std::min((int)(2 * cell), SIDE - 1));
    int de = std::max(ds + 1, std::min((int)(6 * cell), SIDE));
    float lmin = blurred[ds], lmax = blurred[ds];
    for (int x = ds; x < de; x++) {
      lmin = std::min(lmin, blurred[x]);
      lmax = std::max(lmax, blurred[x]);
    }
    float row_thresh = (lmin + lmax) * 0.5f;
    if ((lmax - lmin) < 30.0f)
      row_thresh = 155.0f;

    for (int c_idx = 0; c_idx < 4; c_idx++) {
      int x = get_center_coord(CI[c_idx], cell);
      if (x < 0 || x >= SIDE)
        return py::make_tuple(py::none(), py::none(), -2);
      grid_bits[r_idx][c_idx] = (blurred[x] > row_thresh) ? 1 : 0;
    }
  }

  int anchor_y[4] = {get_center_coord(5, cell), get_center_coord(5, cell),
                     get_center_coord(2, cell), get_center_coord(2, cell)};
  int anchor_x[4] = {get_center_coord(5, cell), get_center_coord(2, cell),
                     get_center_coord(2, cell), get_center_coord(5, cell)};

  int erode_k = std::max(3, (int)(cell * 0.15f)) | 1;
  std::vector<uint8_t> eroded(SIDE * SIDE);
  erode_inplace(threshold_img.data(), eroded.data(), SIDE, SIDE, erode_k, 2);

  int ph = std::max(2, (int)(cell * 0.2f));

  float eroded_vals[4];
  for (int i = 0; i < 4; i++)
    eroded_vals[i] =
        patch_median(eroded.data(), SIDE, SIDE, anchor_y[i], anchor_x[i], ph);

  std::vector<int> white_corners;
  for (int i = 0; i < 4; i++)
    if (eroded_vals[i] > 127)
      white_corners.push_back(i);

  int orientation;
  if (white_corners.size() == 1) {
    orientation = white_corners[0];
  } else {
    float gs_vals[4];
    for (int i = 0; i < 4; i++)
      gs_vals[i] = patch_median(sharpened.data(), SIDE, SIDE, anchor_y[i],
                                anchor_x[i], ph);

    int max_idx = 0;
    float max_v = gs_vals[0];
    float min_v = gs_vals[0];
    for (int i = 1; i < 4; i++) {
      if (gs_vals[i] > max_v) {
        max_v = gs_vals[i];
        max_idx = i;
      }
      if (gs_vals[i] < min_v)
        min_v = gs_vals[i];
    }

    if (max_v < 100.0f && (max_v - min_v) < 15.0f)
      return py::make_tuple(py::none(), py::none(), -4);

    if (white_corners.size() >= 2) {
      orientation = white_corners[0];
      float best = gs_vals[white_corners[0]];
      for (size_t i = 1; i < white_corners.size(); i++) {
        if (gs_vals[white_corners[i]] > best) {
          best = gs_vals[white_corners[i]];
          orientation = white_corners[i];
        }
      }
    } else {
      orientation = max_idx;
    }
  }

  const int bit_map[4][4][2] = {{{1, 1}, {1, 2}, {2, 2}, {2, 1}},
                                {{1, 2}, {2, 2}, {2, 1}, {1, 1}},
                                {{2, 2}, {2, 1}, {1, 1}, {1, 2}},
                                {{2, 1}, {1, 1}, {1, 2}, {2, 2}}};

  int tag_id = 0;
  for (int r = 0; r < 4; r++) {
    int br = bit_map[orientation][r][0];
    int bc = bit_map[orientation][r][1];
    tag_id |= (grid_bits[br][bc] << r);
  }

  if (tag_id < 0 || tag_id > 15)
    return py::make_tuple(py::none(), py::none(), -3);

  return py::make_tuple(tag_id, orientation, 0);
}

py::array_t<uint8_t> bilateralFilter_cpp(const py::array_t<uint8_t> src, int d,
                                         double sigmaColor, double sigmaSpace) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];

  py::array_t<uint8_t> result = py::array_t<uint8_t>({s_h, s_w});
  const uint8_t *raw_src = (uint8_t *)buf_src.ptr;
  uint8_t *raw_dst = (uint8_t *)result.request().ptr;

  int radius = d / 2;
  int kernel_size = 2 * radius + 1;
  int k_count = kernel_size * kernel_size;

  struct KernelEntry {
    int dy, dx;
    float space_w;
  };
  std::vector<KernelEntry> kernel_entries(k_count);
  float space_coeff = -0.5f / (float)(sigmaSpace * sigmaSpace);

  int k_idx = 0;
  for (int ky = -radius; ky <= radius; ky++) {
    for (int kx = -radius; kx <= radius; kx++) {
      float r2 = (float)(kx * kx + ky * ky);
      kernel_entries[k_idx] = {ky, kx, std::exp(r2 * space_coeff)};
      k_idx++;
    }
  }

  float range_coeff = -0.5f / (float)(sigmaColor * sigmaColor);
  float range_weights[256];
  for (int i = 0; i < 256; i++) {
    range_weights[i] = std::exp((float)(i * i) * range_coeff);
  }

  int pad_w = s_w + 2 * radius;
  int pad_h = s_h + 2 * radius;
  std::vector<uint8_t> padded(pad_h * pad_w);

  for (int y = 0; y < pad_h; y++) {
    int sy = std::max(0, std::min(s_h - 1, y - radius));
    const uint8_t *src_row = raw_src + sy * s_w;
    uint8_t *pad_row = padded.data() + y * pad_w;
    for (int x = 0; x < pad_w; x++) {
      int sx = std::max(0, std::min(s_w - 1, x - radius));
      pad_row[x] = src_row[sx];
    }
  }

  const uint8_t *pad_ptr = padded.data();

  std::vector<int> k_offsets(k_count);
  for (int k = 0; k < k_count; k++) {
    k_offsets[k] = kernel_entries[k].dy * pad_w + kernel_entries[k].dx;
  }

#pragma omp parallel for
  for (int y = 0; y < s_h; y++) {
    uint8_t *dst_row = raw_dst + y * s_w;
    const uint8_t *center_row = pad_ptr + (y + radius) * pad_w + radius;

    for (int x = 0; x < s_w; x++) {
      const uint8_t *center_ptr = center_row + x;
      int center_val = *center_ptr;
      float sum = 0.0f;
      float w_sum = 0.0f;

      for (int k = 0; k < k_count; k++) {
        int neighbor_val = center_ptr[k_offsets[k]];
        float weight = kernel_entries[k].space_w *
                       range_weights[std::abs(neighbor_val - center_val)];
        sum += neighbor_val * weight;
        w_sum += weight;
      }
      dst_row[x] = clip_u8(sum / w_sum);
    }
  }
  return result;
}

py::array_t<uint8_t> sharpenAndNormalize_cpp(const py::array_t<uint8_t> src) {
  auto buf_src = src.request();
  int s_h = buf_src.shape[0];
  int s_w = buf_src.shape[1];
  const uint8_t *raw_src = (uint8_t *)buf_src.ptr;

  auto result = py::array_t<uint8_t>({s_h, s_w});
  auto raw_dst = (uint8_t *)result.request().ptr;

  std::vector<float> temp(s_h * s_w);
  float min_val = 1e9f;
  float max_val = -1e9f;

#pragma omp parallel for collapse(2) reduction(min : min_val)                  \
    reduction(max : max_val)
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

py::array_t<float> cornerSubPix_cpp(const py::array_t<uint8_t> image,
                                    py::array_t<float> corners, int win_w,
                                    int win_h, int max_iter, double epsilon) {
  auto buf_image = image.request();
  int s_h = buf_image.shape[0];
  int s_w = buf_image.shape[1];
  const uint8_t *raw_image = (uint8_t *)buf_image.ptr;

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
      double Gxx = 0, Gxy = 0, Gyy = 0;
      double bx = 0, by = 0;

      int Ix = static_cast<int>(cx);
      int Iy = static_cast<int>(cy);

      int left = std::max(1, Ix - win_w);
      int right = std::min(s_w - 2, Ix + win_w);
      int top = std::max(1, Iy - win_h);
      int bottom = std::min(s_h - 2, Iy + win_h);

      for (int y = top; y <= bottom; y++) {
        const uint8_t *row_ptr = raw_image + y * s_w;
        for (int x = left; x <= right; x++) {
          float dx = 0.5f * (row_ptr[x + 1] - row_ptr[x - 1]);
          float dy = 0.5f * (raw_image[(y + 1) * s_w + x] -
                             raw_image[(y - 1) * s_w + x]);

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

py::array_t<uint8_t> overlay_image_cpp(py::array_t<uint8_t> frame,
                                       py::array_t<uint8_t> overlay,
                                       py::array_t<double> H_inv) {
  auto buf_frame = frame.request();
  auto buf_overlay = overlay.request();

  int f_h = buf_frame.shape[0];
  int f_w = buf_frame.shape[1];
  int f_c = buf_frame.shape[2];

  int o_h = buf_overlay.shape[0];
  int o_w = buf_overlay.shape[1];
  int o_c = buf_overlay.shape[2];

  uint8_t *raw_frame = (uint8_t *)buf_frame.ptr;
  const uint8_t *raw_overlay = (uint8_t *)buf_overlay.ptr;

  auto r_H = H_inv.unchecked<2>();
  double h00 = r_H(0, 0), h01 = r_H(0, 1), h02 = r_H(0, 2);
  double h10 = r_H(1, 0), h11 = r_H(1, 1), h12 = r_H(1, 2);
  double h20 = r_H(2, 0), h21 = r_H(2, 1), h22 = r_H(2, 2);

#pragma omp parallel for collapse(2)
  for (int y = 0; y < f_h; y++) {
    for (int x = 0; x < f_w; x++) {
      double w = h20 * x + h21 * y + h22;
      if (std::abs(w) < 1e-5)
        continue;

      double inv_w = 1.0 / w;
      double u = (h00 * x + h01 * y + h02) * inv_w;
      double v = (h10 * x + h11 * y + h12) * inv_w;

      if (u >= 0 && u < o_w - 1 && v >= 0 && v < o_h - 1) {
        int overlay_x = static_cast<int>(u);
        int overlay_y = static_cast<int>(v);
        float dx = u - overlay_x;
        float dy = v - overlay_y;

        int f_idx = (y * f_w + x) * f_c;

        for (int k = 0; k < 3; k++) {
          int o_idx00 = (overlay_y * o_w + overlay_x) * o_c + k;
          int o_idx01 = (overlay_y * o_w + overlay_x + 1) * o_c + k;
          int o_idx10 = ((overlay_y + 1) * o_w + overlay_x) * o_c + k;
          int o_idx11 = ((overlay_y + 1) * o_w + overlay_x + 1) * o_c + k;

          float val = (1 - dx) * (1 - dy) * raw_overlay[o_idx00] +
                      dx * (1 - dy) * raw_overlay[o_idx01] +
                      (1 - dx) * dy * raw_overlay[o_idx10] +
                      dx * dy * raw_overlay[o_idx11];

          if (val > 10) {
            raw_frame[f_idx + k] = clip_u8(val);
          }
        }
      }
    }
  }
  return frame;
}

py::array_t<uint8_t> erode_cpp(const py::array_t<uint8_t> src,
                               const py::array_t<uint8_t> kernel,
                               int iterations) {
  auto s = src.unchecked<2>();
  auto k = kernel.unchecked<2>();
  int h = s.shape(0), w = s.shape(1);
  int kh = k.shape(0), kw = k.shape(1);
  int pad_h = kh / 2, pad_w = kw / 2;

  std::vector<std::pair<int, int>> offsets;
  for (int ky = 0; ky < kh; ky++)
    for (int kx = 0; kx < kw; kx++)
      if (k(ky, kx) > 0)
        offsets.push_back({ky - pad_h, kx - pad_w});

  std::vector<uint8_t> buf_a(h * w), buf_b(h * w);
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
      buf_a[i * w + j] = s(i, j);

  uint8_t *cur = buf_a.data(), *nxt = buf_b.data();

  for (int iter = 0; iter < iterations; iter++) {
#pragma omp parallel for schedule(static)
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        uint8_t mn = 255;
        for (auto &off : offsets) {
          int ny = std::max(0, std::min(h - 1, y + off.first));
          int nx = std::max(0, std::min(w - 1, x + off.second));
          mn = std::min(mn, cur[ny * w + nx]);
        }
        nxt[y * w + x] = mn;
      }
    }
    std::swap(cur, nxt);
  }

  py::array_t<uint8_t> result({h, w});
  auto r = result.mutable_unchecked<2>();
  for (int i = 0; i < h; i++)
    for (int j = 0; j < w; j++)
      r(i, j) = cur[i * w + j];
  return result;
}

py::array_t<uint8_t> bitwise_or_cpp(const py::array_t<uint8_t> src1,
                                    const py::array_t<uint8_t> src2) {
  auto s1 = src1.unchecked<2>();
  auto s2 = src2.unchecked<2>();
  int h = s1.shape(0), w = s1.shape(1);

  py::array_t<uint8_t> result({h, w});
  auto r = result.mutable_unchecked<2>();

#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      r(y, x) = s1(y, x) | s2(y, x);

  return result;
}

py::array_t<uint8_t> bitwise_and_cpp(const py::array_t<uint8_t> src1,
                                     const py::array_t<uint8_t> src2) {
  auto s1 = src1.unchecked<2>();
  auto s2 = src2.unchecked<2>();
  int h = s1.shape(0), w = s1.shape(1);

  py::array_t<uint8_t> result({h, w});
  auto r = result.mutable_unchecked<2>();

#pragma omp parallel for schedule(static)
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      r(y, x) = s1(y, x) & s2(y, x);

  return result;
}

void fillConvexPoly_cpp(py::array_t<uint8_t> img, const py::array_t<int> points,
                        std::vector<int> color) {
  auto im = img.mutable_unchecked<3>();
  auto pts = points.unchecked<2>();
  int h = im.shape(0), w = im.shape(1);
  int n = pts.shape(0);
  if (n < 3)
    return;

  int y_min = h, y_max = 0;
  for (int i = 0; i < n; i++) {
    y_min = std::min(y_min, pts(i, 1));
    y_max = std::max(y_max, pts(i, 1));
  }
  y_min = std::max(0, y_min);
  y_max = std::min(h - 1, y_max);

  uint8_t c0 = (uint8_t)color[0], c1 = (uint8_t)color[1],
          c2 = (uint8_t)color[2];

#pragma omp parallel for schedule(static)
  for (int y = y_min; y <= y_max; y++) {
    double x_min_d = 1e9, x_max_d = -1e9;
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      int y0 = pts(i, 1), y1 = pts(j, 1);
      if (y0 == y1)
        continue;
      if ((y0 <= y && y < y1) || (y1 <= y && y < y0)) {
        double x =
            pts(i, 0) + (double)(y - y0) * (pts(j, 0) - pts(i, 0)) / (y1 - y0);
        x_min_d = std::min(x_min_d, x);
        x_max_d = std::max(x_max_d, x);
      }
    }
    if (x_min_d > x_max_d)
      continue;
    int xs = std::max(0, (int)std::ceil(x_min_d));
    int xe = std::min(w - 1, (int)std::floor(x_max_d));
    for (int x = xs; x <= xe; x++) {
      im(y, x, 0) = c0;
      im(y, x, 1) = c1;
      im(y, x, 2) = c2;
    }
  }
}

py::array_t<double> rodrigues_cpp(py::array_t<double> src) {
  auto buf = src.request();
  double *p = (double *)buf.ptr;

  bool is_mat = (buf.ndim == 2 && buf.shape[0] == 3 && buf.shape[1] == 3);

  if (is_mat) {
    double R[9];
    for (int i = 0; i < 9; i++)
      R[i] = p[i];
    double cos_t =
        std::max(-1.0, std::min(1.0, (R[0] + R[4] + R[8] - 1.0) / 2.0));
    double theta = std::acos(cos_t);
    auto result = py::array_t<double>({3, 1});
    double *out = (double *)result.request().ptr;
    if (theta < 1e-6) {
      out[0] = out[1] = out[2] = 0;
      return result;
    }
    double v[3] = {R[7] - R[5], R[2] - R[6], R[3] - R[1]};
    double s = 2.0 * std::sin(theta);
    if (std::abs(s) < 1e-10) {
      double S[3] = {(R[0] + 1) / 2.0, (R[4] + 1) / 2.0, (R[8] + 1) / 2.0};
      for (int i = 0; i < 3; i++)
        S[i] = std::max(0.0, S[i]);
      double k[3] = {std::sqrt(S[0]), std::sqrt(S[1]), std::sqrt(S[2])};
      if (k[0] > 1e-6) {
        if (R[1] < 0)
          k[1] = -k[1];
        if (R[2] < 0)
          k[2] = -k[2];
      } else if (k[1] > 1e-6) {
        if (R[5] < 0)
          k[2] = -k[2];
      }
      for (int i = 0; i < 3; i++)
        out[i] = k[i] * theta;
      return result;
    }
    for (int i = 0; i < 3; i++)
      out[i] = (theta / s) * v[i];
    return result;
  } else {
    int n = 1;
    for (int i = 0; i < buf.ndim; i++)
      n *= buf.shape[i];
    double r[3] = {p[0], p[1], (n >= 3) ? p[2] : 0.0};
    double theta = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    auto result = py::array_t<double>({3, 3});
    double *out = (double *)result.request().ptr;
    if (theta < 1e-6) {
      for (int i = 0; i < 9; i++)
        out[i] = (i % 4 == 0) ? 1.0 : 0.0;
      return result;
    }
    double k[3] = {r[0] / theta, r[1] / theta, r[2] / theta};
    double c = std::cos(theta), s = std::sin(theta);
    out[0] = c + (1 - c) * k[0] * k[0];
    out[1] = (1 - c) * k[0] * k[1] - s * k[2];
    out[2] = (1 - c) * k[0] * k[2] + s * k[1];
    out[3] = (1 - c) * k[1] * k[0] + s * k[2];
    out[4] = c + (1 - c) * k[1] * k[1];
    out[5] = (1 - c) * k[1] * k[2] - s * k[0];
    out[6] = (1 - c) * k[2] * k[0] - s * k[1];
    out[7] = (1 - c) * k[2] * k[1] + s * k[0];
    out[8] = c + (1 - c) * k[2] * k[2];
    return result;
  }
}

py::tuple solvePnP_cpp(py::array_t<double> objectPoints,
                       py::array_t<double> imagePoints,
                       py::array_t<double> cameraMatrix) {
  auto obj_buf = objectPoints.request();
  auto img_buf = imagePoints.request();
  auto cam_buf = cameraMatrix.request();
  double *obj = (double *)obj_buf.ptr;
  double *img_p = (double *)img_buf.ptr;
  double *K = (double *)cam_buf.ptr;

  double obj_2d[4][2], img_2d[4][2];
  for (int i = 0; i < 4; i++) {
    obj_2d[i][0] = obj[i * 3 + 0];
    obj_2d[i][1] = obj[i * 3 + 1];
    img_2d[i][0] = img_p[i * 2 + 0];
    img_2d[i][1] = img_p[i * 2 + 1];
  }

  double A[8][9];
  std::memset(A, 0, sizeof(A));
  for (int i = 0; i < 4; i++) {
    double X = obj_2d[i][0], Y = obj_2d[i][1];
    double u = img_2d[i][0], v = img_2d[i][1];
    A[2 * i][0] = X;
    A[2 * i][1] = Y;
    A[2 * i][2] = 1;
    A[2 * i][6] = -u * X;
    A[2 * i][7] = -u * Y;
    A[2 * i][8] = -u;
    A[2 * i + 1][3] = X;
    A[2 * i + 1][4] = Y;
    A[2 * i + 1][5] = 1;
    A[2 * i + 1][6] = -v * X;
    A[2 * i + 1][7] = -v * Y;
    A[2 * i + 1][8] = -v;
  }

  for (int col = 0; col < 8; col++) {
    int pivot = -1;
    double max_v = 0;
    for (int row = col; row < 8; row++) {
      if (std::abs(A[row][col]) > max_v) {
        max_v = std::abs(A[row][col]);
        pivot = row;
      }
    }
    if (pivot < 0 || max_v < 1e-12)
      return py::make_tuple(false, py::none(), py::none());
    if (pivot != col)
      for (int j = 0; j < 9; j++)
        std::swap(A[col][j], A[pivot][j]);
    double scale = A[col][col];
    for (int j = col; j < 9; j++)
      A[col][j] /= scale;
    for (int row = 0; row < 8; row++) {
      if (row == col)
        continue;
      double f = A[row][col];
      for (int j = col; j < 9; j++)
        A[row][j] -= f * A[col][j];
    }
  }

  double H[9];
  for (int i = 0; i < 8; i++)
    H[i] = -A[i][8];
  H[8] = 1.0;

  double K_inv[9];
  double det_K = K[0] * (K[4] * K[8] - K[5] * K[7]) -
                 K[1] * (K[3] * K[8] - K[5] * K[6]) +
                 K[2] * (K[3] * K[7] - K[4] * K[6]);
  if (std::abs(det_K) < 1e-12)
    return py::make_tuple(false, py::none(), py::none());
  double inv_det = 1.0 / det_K;
  K_inv[0] = (K[4] * K[8] - K[5] * K[7]) * inv_det;
  K_inv[1] = (K[2] * K[7] - K[1] * K[8]) * inv_det;
  K_inv[2] = (K[1] * K[5] - K[2] * K[4]) * inv_det;
  K_inv[3] = (K[5] * K[6] - K[3] * K[8]) * inv_det;
  K_inv[4] = (K[0] * K[8] - K[2] * K[6]) * inv_det;
  K_inv[5] = (K[2] * K[3] - K[0] * K[5]) * inv_det;
  K_inv[6] = (K[3] * K[7] - K[4] * K[6]) * inv_det;
  K_inv[7] = (K[1] * K[6] - K[0] * K[7]) * inv_det;
  K_inv[8] = (K[0] * K[4] - K[1] * K[3]) * inv_det;

  double M[9];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      M[i * 3 + j] = 0;
      for (int k = 0; k < 3; k++)
        M[i * 3 + j] += K_inv[i * 3 + k] * H[k * 3 + j];
    }

  double h1[3] = {M[0], M[3], M[6]};
  double h2[3] = {M[1], M[4], M[7]};
  double h3[3] = {M[2], M[5], M[8]};

  double norm_h1 = std::sqrt(h1[0] * h1[0] + h1[1] * h1[1] + h1[2] * h1[2]);
  if (norm_h1 < 1e-12)
    return py::make_tuple(false, py::none(), py::none());
  double lam = 1.0 / norm_h1;
  if (h3[2] * lam < 0)
    lam = -lam;

  double r1[3], r2[3], t[3];
  for (int i = 0; i < 3; i++) {
    r1[i] = lam * h1[i];
    r2[i] = lam * h2[i];
    t[i] = lam * h3[i];
  }

  double r3[3] = {r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2],
                  r1[0] * r2[1] - r1[1] * r2[0]};

  double r2n[3] = {r3[1] * r1[2] - r3[2] * r1[1], r3[2] * r1[0] - r3[0] * r1[2],
                   r3[0] * r1[1] - r3[1] * r1[0]};
  for (int i = 0; i < 3; i++)
    r2[i] = r2n[i];

  auto norm3 = [](double *v) {
    double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 1e-12)
      for (int i = 0; i < 3; i++)
        v[i] /= n;
  };
  norm3(r1);
  norm3(r2);
  norm3(r3);

  double R[9] = {r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]};
  double det_R = R[0] * (R[4] * R[8] - R[5] * R[7]) -
                 R[1] * (R[3] * R[8] - R[5] * R[6]) +
                 R[2] * (R[3] * R[7] - R[4] * R[6]);
  if (det_R < 0) {
    for (int i = 0; i < 9; i++)
      R[i] = -R[i];
    for (int i = 0; i < 3; i++)
      t[i] = -t[i];
  }

  auto R_arr = py::array_t<double>({3, 3});
  std::memcpy(R_arr.request().ptr, R, 9 * sizeof(double));
  auto rvec = rodrigues_cpp(R_arr);

  auto tvec = py::array_t<double>({3, 1});
  double *tv = (double *)tvec.request().ptr;
  tv[0] = t[0];
  tv[1] = t[1];
  tv[2] = t[2];

  return py::make_tuple(true, rvec, tvec);
}

PYBIND11_MODULE(custom_cv2_cpp, m) {
  m.doc() = "Custom OpenCV-like functions implemented in C++ with OpenMP";

  m.def("warpPerspective_cpp", &warpPerspective_cpp, py::arg("src"),
        py::arg("M"), py::arg("d_w"), py::arg("d_h"),
        "Applies a perspective warp to the input image using the given "
        "transformation matrix M.");

  m.def("findContours_cpp", &findContours_cpp, py::arg("src"),
        py::arg("min_points") = 10,
        "Finds contours in a binary image and returns those with number of "
        "points >= min_points.");

  m.def("approxPolyDP_cpp", &approxPolyDP_cpp, py::arg("curve"),
        py::arg("epsilon"), py::arg("closed") = false,
        "Approximates a polygonal curve with the specified precision epsilon "
        "using the Douglas-Peucker algorithm.");

  m.def("adaptiveThreshold_cpp", &adaptiveThreshold_cpp, py::arg("src"),
        py::arg("maxValue"), py::arg("blockSize"), py::arg("C"),
        "Applies adaptive thresholding to the input grayscale image.");

  m.def("boxFilter_cpp", &boxFilter_cpp, py::arg("src"), py::arg("ksize"),
        "Applies a box filter to the input image with the specified kernel "
        "size.");

  m.def("gaussianBlur_cpp", &gaussianBlur_cpp, py::arg("src"), py::arg("ksize"),
        py::arg("sigma"),
        "Applies Gaussian blur to the input image with the specified kernel "
        "size and sigma.");

  m.def("cvtColor_cpp", &cvtColor_cpp, py::arg("src"),
        "Converts a BGR image to grayscale.");

  m.def("decode_tag_cpp", &decode_tag_cpp, py::arg("warped"),
        "Decodes a tag from a warped grayscale image.");

  m.def("rodrigues_cpp", &rodrigues_cpp, py::arg("src"),
        "Rodrigues rotation conversion.");

  m.def("solvePnP_cpp", &solvePnP_cpp, py::arg("objectPoints"),
        py::arg("imagePoints"), py::arg("cameraMatrix"),
        "Solve PnP for planar objects.");

  m.def("bilateralFilter_cpp", &bilateralFilter_cpp, py::arg("src"),
        py::arg("d"), py::arg("sigmaColor"), py::arg("sigmaSpace"),
        "Applies a bilateral filter to the input image with the specified "
        "diameter, sigmaColor, and sigmaSpace.");

  m.def("sharpenAndNormalize_cpp", &sharpenAndNormalize_cpp, py::arg("src"),
        "Sharpens the input image using a simple kernel and normalizes the "
        "result to the full 0-255 range.");

  m.def("cornerSubPix_cpp", &cornerSubPix_cpp, py::arg("image"),
        py::arg("corners"), py::arg("win_w"), py::arg("win_h"),
        py::arg("max_iter"), py::arg("epsilon"),
        "Refines corner locations to sub-pixel accuracy using the specified "
        "window size, maximum iterations, and convergence epsilon.");

  m.def("overlay_image_cpp", &overlay_image_cpp, py::arg("frame"),
        py::arg("overlay"), py::arg("H_inv"),
        "Overlays the given image onto the frame using the inverse homography "
        "matrix H_inv for perspective transformation.");

  m.def("erode_cpp", &erode_cpp, py::arg("src"), py::arg("kernel"),
        py::arg("iterations"),
        "Erodes a grayscale image using the given kernel for the specified "
        "number of iterations.");

  m.def("bitwise_or_cpp", &bitwise_or_cpp, py::arg("src1"), py::arg("src2"),
        "Element-wise bitwise OR of two uint8 images.");

  m.def("bitwise_and_cpp", &bitwise_and_cpp, py::arg("src1"), py::arg("src2"),
        "Element-wise bitwise AND of two uint8 images.");

  m.def("fillConvexPoly_cpp", &fillConvexPoly_cpp, py::arg("img"),
        py::arg("points"), py::arg("color"),
        "Fills a convex polygon on the image using scanline rasterization.");
}