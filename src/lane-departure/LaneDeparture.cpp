#include "LaneDeparture.h"
#include <opencv2/imgproc.hpp>

cv::Mat LaneDeparture::process(cv::Mat &frame) {
  cv::Mat hls, combined_mask, masked_frame;
  cv::cvtColor(frame, hls, cv::COLOR_BGR2HLS);
  const cv::Mat yellow_mask = apply_yellow_lane_mask(hls);
  const cv::Mat white_mask = apply_white_lane_mask(hls);
  cv::bitwise_or(yellow_mask, white_mask, combined_mask);
  cv::bitwise_and(frame, frame, masked_frame, combined_mask);


  const cv::Mat canny_frame = apply_canny_edge_detection(masked_frame);
  const cv::Mat roi_frame = apply_region_of_interest(canny_frame);
  const std::vector<cv::Vec4i> segments = get_hough_probabilistic_lines(roi_frame);
  const LaneState state = analyze_lane(segments, frame.cols, frame.rows);
  draw_overlay(state, frame);

  return frame;
}

std::string LaneDeparture::get_feature_name() { return this->name; }

cv::Mat LaneDeparture::apply_canny_edge_detection(const cv::Mat &frame) {
  constexpr double lower_thresh = 10;
  constexpr double upper_thresh = 50;
  cv::Mat gray, blur, canny_frame;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0.0);
  cv::Canny(blur, canny_frame, lower_thresh, upper_thresh);
  return canny_frame;
}
cv::Mat LaneDeparture::apply_scharr_edge_detection(const cv::Mat &frame) {
  cv::Mat edge_x, img_hls;
  cv::cvtColor(frame, img_hls, cv::COLOR_BGR2HLS);
  img_hls.convertTo(img_hls, CV_32F);
  std::vector<cv::Mat> hls_channels;
  cv::split(img_hls, hls_channels);

  cv::Scharr(img_hls, edge_x, CV_32F, 1, 0);
  // np.absolute(edge_x)
  cv::Mat abs_edge_x = cv::abs(edge_x);

  // np.uint8(255 * edge_x / np.max(edge_x))
  double max_val;
  cv::minMaxLoc(abs_edge_x, nullptr, &max_val);

  cv::Mat scaled;
  if (max_val > 0.0) {
    abs_edge_x.convertTo(scaled, CV_8U, 255.0 / max_val);
  } else {
    scaled = cv::Mat::zeros(abs_edge_x.size(), CV_8U);
  }

  return scaled;
}
void LaneDeparture::calculate_image_points(const cv::Mat &frame) {
  if (horizon_y_start_pixel <= 0 || dashboard_y_ratio <= 0 || car_x_center_pixel <= 0) {
    horizon_y_start_pixel = static_cast<int>(frame.rows * horizon_y_ratio);
    dashboard_y_end_pixel = static_cast<int>(frame.rows * dashboard_y_ratio);
    car_x_center_pixel = static_cast<int>(frame.cols * car_x_center_percentage);
  }
}
std::vector<cv::Point> LaneDeparture::get_trapezoid_roi(const int height, const int width) const {
  const int trapezoid_bottom = static_cast<int>(height * dashboard_y_ratio);
  const int trapezoid_top = static_cast<int>(height * horizon_y_ratio);
  return std::vector<cv::Point>{
      // Bottom-Left Point
      {static_cast<int>(width * 0.05), trapezoid_bottom},
      // Top-Left Point
      {static_cast<int>(width * 0.45), trapezoid_top},
      // Top-Right Point
      {static_cast<int>(width * 0.55), trapezoid_top},
      // Bottom-Right Point
      {static_cast<int>(width * 0.95), trapezoid_bottom},
  };
}
std::vector<cv::Point> LaneDeparture::get_perspective_roi(const int height, const int width) {
  const int roi_bottom = static_cast<int>(height * dashboard_y_ratio);
  const int roi_top = static_cast<int>(height * horizon_y_ratio);
  return std::vector<cv::Point>{
      // Bottom-Left Point
      {static_cast<int>(width * 0.05), roi_bottom},
      // Top-Left Point
      {static_cast<int>(width * 0.05), roi_top},
      // Top-Right Point
      {static_cast<int>(width * 0.95), roi_top},
      // Bottom-Right Point
      {static_cast<int>(width * 0.95), roi_bottom},
  };
}
cv::Mat LaneDeparture::get_wrapped_roi(const cv::Mat &frame) {
  const auto width = static_cast<float>(frame.cols);
  const auto height = static_cast<float>(frame.rows);

  const float dashboard_end = height * 0.85f;
  const float roi_top = height * 0.55f;

  // Trapezoid in original camera image
  const std::vector<cv::Point2f> src = {
      {width * 0.05f, dashboard_end}, // 0 bottom-left
      {width * 0.30f, roi_top},       // 1 top-left
      {width * 0.60f, roi_top},       // 2 top-right
      {width * 0.95f, dashboard_end}  // 3 bottom-right
  };

  // Rectangle target
  const std::vector<cv::Point2f> dst = {
      {width * 0.05f, dashboard_end}, // 0 bottom-left
      {width * 0.05f, roi_top},       // 1 top-left
      {width * 0.95f, roi_top},       // 2 top-right
      {width * 0.95f, dashboard_end}  // 3 bottom-right
  };

  cv::Mat debug = frame.clone();

  auto drawShape = [](cv::Mat &image, const std::vector<cv::Point2f> &points, const cv::Scalar &color,
                      const std::string &name) {
    std::vector<cv::Point> poly;
    for (const auto &p : points) {
      poly.emplace_back(cvRound(p.x), cvRound(p.y));
    }

    cv::polylines(image, poly, true, color, 3);

    for (size_t i = 0; i < points.size(); ++i) {
      cv::circle(image, points[i], 6, color, -1);

      cv::putText(image, name + std::to_string(i), points[i] + cv::Point2f(8.f, -8.f), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  color, 2);
    }
  };

  drawShape(debug, src, cv::Scalar(0, 255, 0), "S"); // green trapezoid
  drawShape(debug, dst, cv::Scalar(0, 0, 255), "D"); // red rectangle

  const cv::Mat M = cv::getPerspectiveTransform(src, dst);
  const cv::Mat I = cv::getPerspectiveTransform(dst, src);

  cv::Mat warped, inv;
  cv::warpPerspective(frame, warped, M, frame.size(), cv::INTER_LANCZOS4);
  cv::warpPerspective(frame, inv, I, frame.size(), cv::INTER_LANCZOS4);
  return warped;
}

cv::Mat LaneDeparture::apply_region_of_interest(const cv::Mat &canny_frame) {
  // NAIVE WAY TO FILTER OUT ROAD
  // Using a trapezoid instead of a triangle
  const int height = canny_frame.rows;
  const int width = canny_frame.cols;

  // Using percentages instead of raw pixel values
  const std::vector<cv::Point> trapezoid_points = get_trapezoid_roi(height, width);
  cv::Mat mask = cv::Mat::zeros(canny_frame.size(), canny_frame.type());
  cv::fillPoly(mask, trapezoid_points, cv::Scalar(255));

  cv::Mat masked_frame;
  // Only show the image in the ROI by CannyEdgeFrame & Trapezoid(Intersection),
  // thus everything outside the Trapezoid gets ignored
  cv::bitwise_and(canny_frame, mask, masked_frame);

  return masked_frame;
}
cv::Mat LaneDeparture::get_region_of_interest(const cv::Mat &canny_frame) {}

std::vector<cv::Vec4i> LaneDeparture::get_hough_probabilistic_lines(const cv::Mat &roi_frame) {
  constexpr double theta = CV_PI / 180;
  constexpr double threshold = 60;
  constexpr double min_line_length = 15;
  constexpr double max_line_gap = 20;

  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(roi_frame, segments, 1, theta, threshold, min_line_length, max_line_gap);
  return segments;
}
cv::Mat LaneDeparture::apply_lines(const cv::Mat &frame, const std::vector<cv::Vec4i> &lines) {
  const auto no_warning_lane_color = cv::Scalar(235, 149, 70);
  cv::Mat frame_lines = cv::Mat::zeros(frame.size(), frame.type());
  for (const auto &line : lines) {
    cv::line(frame_lines, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), no_warning_lane_color, 2,
             cv::LINE_AA);
  }
  return frame_lines;
}
void LaneDeparture::draw_lines(const std::vector<cv::Vec4i> &lines, cv::Mat &original_frame) {
  const auto no_warning_lane_color = cv::Scalar(235, 149, 70);
  for (const auto &s : lines) {
    cv::line(original_frame, cv::Point(s[0], s[1]), cv::Point(s[2], s[3]), no_warning_lane_color, 2, cv::LINE_AA);
  }
}
/**
 * Generated by GPT *  Prompt with LaneDeparture.cpp, LaneDeparture.h, and
LaneState.h upload: " Using the following upload code that uses C++ opencv for a
lane departure warning system create a function that will perform the following:
  - Mark only the lanes in the current lane the car is
  - Detect the middle of the lane
  - Detect if a lane crossing has been changed
"
Comments were provided by Amiel for clear understanding

 * @param segments
 * @param width
 * @param height
 * @return
 */
LaneState LaneDeparture::analyze_lane(const std::vector<cv::Vec4i> &segments, int width, int height) {
  constexpr double vertical_line_threshold = 0.3;

  LaneState state;
  state.x_car_center = width / 2;

  // Convert percentage ratios to actual Y-pixel coordinates
  const int y_top = static_cast<int>(height * horizon_y_ratio);
  const int y_bottom = static_cast<int>(height * dashboard_y_ratio);

  // 1. Separate raw segments into left and right candidates
  auto [left_segments, right_segments] = separate_lanes(segments, state.x_car_center, vertical_line_threshold);

  // 2. Filter out outer lanes/bike lanes by clustering around closest x-intercept
  state.left_lane = calculate_closest_lane(left_segments, y_bottom, y_top, state.x_car_center, width);
  state.right_lane = calculate_closest_lane(right_segments, y_bottom, y_top, state.x_car_center, width);

  // 3. Calculate lane center
  if (state.left_lane && state.right_lane) {
    int left_bottom_x = state.left_lane.value()[0];
    int right_bottom_x = state.right_lane.value()[0];
    state.x_lane_center = (left_bottom_x + right_bottom_x) / 2;
  } else {
    state.x_lane_center = state.x_car_center;
  }

  // 4. Evaluate departure status based on percentage of frame width
  evaluate_departure_status(state, width);

  return state;
}
void LaneDeparture::draw_overlay(const LaneState &state, cv::Mat &frame) {
  const cv::Scalar center_lane_color(246, 191, 3);
  const cv::Scalar warning_lane_color(41, 139, 252);
  const cv::Scalar departure_lane_color(26, 43, 251);
  const cv::Scalar car_center_color(219, 48, 134);

  // Helper lambda to map status to color
  auto get_color = [&](const DepartureStatus status) {
    if (status == DepartureStatus::ALERT)
      return departure_lane_color;
    if (status == DepartureStatus::WARNING)
      return warning_lane_color;
    return center_lane_color;
  };

  // Draw Left Lane
  if (state.left_lane.has_value()) {
    auto l = state.left_lane.value();
    cv::line(frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), get_color(state.left_status), 5, cv::LINE_AA);
  }

  // Draw Right Lane
  if (state.right_lane.has_value()) {
    auto l = state.right_lane.value();
    cv::line(frame, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), get_color(state.right_status), 5, cv::LINE_AA);
  }

  // Draw Circle to reference where the car is at.
  cv::circle(frame, cv::Point(state.x_lane_center, frame.rows - 50), 8, car_center_color, -1);

  // Draw Car Center (Camera Center) for reference
  cv::circle(frame, cv::Point(state.x_car_center, frame.rows - 50), 5, cv::Scalar(255, 255, 255), -1);
}

cv::Vec4i LaneDeparture::get_line(const double slope, const double intercept) const {
  // y = mx + b -> x = (y-b)/m , b-intercept, m-slope
  const int y_bottom = dashboard_y_end_pixel;
  const int y_top = horizon_y_start_pixel;

  const int x_bottom = static_cast<int>((y_bottom - intercept) * slope);
  const int x_top = static_cast<int>((y_top - intercept) * slope);

  return {x_bottom, y_bottom, x_top, y_top};
}

cv::Vec2d LaneDeparture::get_avg_fitting_line(const std::vector<cv::Vec2d> &fits) {
  cv::Vec2d avg(0.0, 0.0);
  for (const auto &fit : fits)
    avg += fit;

  if (!fits.empty())
    avg *= 1.0 / static_cast<double>(fits.size());
  return avg;
}

std::pair<cv::Vec4i, cv::Vec4i> LaneDeparture::get_avg_left_and_right_line(const std::vector<cv::Vec4i> &hough_lines) {
  constexpr double vertical_line_threshold = 0.3;
  // Find the line of best fit for all hough lines
  // slope and intercept vectors
  std::vector<cv::Vec2d> left_line_fit;
  std::vector<cv::Vec2d> right_line_fit;

  // Compute line fitting
  for (const auto &line : hough_lines) {
    const double x1 = line[0];
    const double y1 = line[1];
    const double x2 = line[2];
    const double y2 = line[3];

    // Ignore vertical lines to avoid zero division
    if (std::abs(x2 - x1) < 1e-6)
      continue;

    const double slope = (y2 - y1) / (x2 - x1);

    // Filter out non-vertical lines
    if (std::abs(slope) < vertical_line_threshold)
      continue;

    const double intercept = y1 - slope * x1;

    // OpenCV Graph is flipped at the x-axis compared to Cartesian graph:
    // (0,0) --------------------> x
    //   |        /     \
    //   |      /        \
    //   y
    if (slope < 0)
      /*            / flipped               */
      left_line_fit.emplace_back(slope, intercept);
    else
      /*            \ flipped               */
      right_line_fit.emplace_back(slope, intercept);
  }

  cv::Vec2d left_average = get_avg_fitting_line(left_line_fit);
  cv::Vec2d right_average = get_avg_fitting_line(right_line_fit);

  cv::Vec4i left_line = get_line(left_average[0], left_average[1]);
  cv::Vec4i right_line = get_line(right_average[0], right_average[1]);

  return {left_line, right_line};
}
LaneState LaneDeparture::get_lane_state(const std::pair<cv::Vec4i, cv::Vec4i>& left_and_right_line, int width, int height) {
  LaneState state;
  state.x_car_center = car_x_center_pixel;



  int offset = state.x_car_center - state.x_lane_center;

  // state.left_lane = left_and_right_line.first;
  // state.right_lane = left_and_right_line.second;




}
std::vector<LineData> LaneDeparture::extract_line_data(const std::vector<cv::Vec4i> &segments, int width, int height) {}
std::optional<cv::Vec4i> LaneDeparture::calculate_lane_boundary(std::vector<LineData> &lines, int width, int height,
                                                                bool is_left) {
  std::vector<LineData> side_lines;

  // Filter for left or right side based on slope and position
  for (const auto& l : lines) {
    if (is_left && l.slope < 0 && l.bottom_x < width / 2) {
      side_lines.push_back(l);
    } else if (!is_left && l.slope > 0 && l.bottom_x > width / 2) {
      side_lines.push_back(l);
    }
  }

  if (side_lines.empty()) return std::nullopt;

  // SORTING: This is the secret to ignoring bike lanes.
  // We sort by bottom_x to find the line closest to the car center.
  if (is_left) {
    std::sort(side_lines.begin(), side_lines.end(), [](const LineData& a, const LineData& b) { return a.bottom_x > b.bottom_x; });
  } else {
    std::sort(side_lines.begin(), side_lines.end(), [](const LineData& a, const LineData& b) { return a.bottom_x < b.bottom_x; });
  }
  // Identify the innermost line's bottom X coordinate
  const int innermost_x = side_lines.front().bottom_x;
  constexpr int cluster_tolerance = 150; // Pixels. Only average lines within this distance of the innermost line.

  double slope_sum = 0, intercept_sum = 0, weight_sum = 0;

  for (const auto& l : side_lines) {
    // Only include lines that belong to the primary lane marker cluster
    if (std::abs(l.bottom_x - innermost_x) <= cluster_tolerance) {
      slope_sum += l.slope * l.length;
      intercept_sum += l.intercept * l.length;
      weight_sum += l.length;
    }
  }

  if (weight_sum == 0) return std::nullopt;

  const double m = slope_sum / weight_sum;
  const double b = intercept_sum / weight_sum;

  const int y_bottom = height;
  const int y_top = static_cast<int>(height * 0.6);
  const int bottom_x = static_cast<int>((y_bottom - b) / m);
  const int top_x = static_cast<int>((y_top - b) / m);

  return cv::Vec4i(bottom_x, y_bottom, top_x, y_top);
}
std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>> LaneDeparture::separate_lanes(
    const std::vector<cv::Vec4i>& segments, int car_center_x, double vertical_line_threshold) {
  std::vector<cv::Vec4i> left_segments, right_segments;

  for (const auto &s : segments) {
    const double x1 = s[0], y1 = s[1], x2 = s[2], y2 = s[3];
    if (x1 == x2) continue;

    const double slope = (y2 - y1) / (x2 - x1);
    if (std::abs(slope) < vertical_line_threshold) continue;

    if (slope < 0 && x1 < car_center_x && x2 < car_center_x) {
      left_segments.push_back(s);
    } else if (slope > 0 && x1 > car_center_x && x2 > car_center_x) {
      right_segments.push_back(s);
    }
  }
  return {left_segments, right_segments};
}

void LaneDeparture::evaluate_departure_status(LaneState &state, const int width) const {
  int offset = state.x_car_center - state.x_lane_center;
  // Calculate dynamic pixel thresholds based on frame width percentage
  const double warning_pixels = width * warning_threshold_pct;
  const double alert_pixels = width * alert_threshold_pct;

  if (offset > alert_pixels) {
    state.left_status = DepartureStatus::ALERT;
  } else if (offset > warning_pixels) {
    state.left_status = DepartureStatus::WARNING;
  }

  if (offset < -alert_pixels) {
    state.right_status = DepartureStatus::ALERT;
  } else if (offset < -warning_pixels) {
    state.right_status = DepartureStatus::WARNING;
  }
}
std::optional<cv::Vec4i> LaneDeparture::calculate_closest_lane(
    const std::vector<cv::Vec4i>& segments, int y_bottom, int y_top, int car_center_x, int width) {
  if (segments.empty()) return std::nullopt;

  struct SegmentData {
    double slope;
    double intercept;
    double length;
    double bottom_x;
  };

  std::vector<SegmentData> processed_segments;
  double closest_distance = std::numeric_limits<double>::max();
  double target_bottom_x = 0;

  for (const auto& s : segments) {
    double slope = static_cast<double>(s[3] - s[1]) / (s[2] - s[0]);
    double intercept = s[1] - slope * s[0];
    double length = std::hypot(s[3] - s[1], s[2] - s[0]);
    double bottom_x = (y_bottom - intercept) / slope;

    processed_segments.push_back({slope, intercept, length, bottom_x});

    double distance_to_center = std::abs(car_center_x - bottom_x);
    if (distance_to_center < closest_distance) {
      closest_distance = distance_to_center;
      target_bottom_x = bottom_x;
    }
  }
  // Convert cluster threshold percentage to actual pixel margin for this resolution
  const double cluster_pixel_threshold = width * cluster_threshold_pct;
  double slope_sum = 0, intercept_sum = 0, weight = 0;

  for (const auto& ps : processed_segments) {
    if (std::abs(ps.bottom_x - target_bottom_x) <= cluster_pixel_threshold) {
      slope_sum += ps.slope * ps.length;
      intercept_sum += ps.intercept * ps.length;
      weight += ps.length;
    }
  }

  if (weight == 0) return std::nullopt;

  const double m = slope_sum / weight;
  const double b = intercept_sum / weight;

  const int bottom_x = static_cast<int>((y_bottom - b) / m);
  const int top_x = static_cast<int>((y_top - b) / m);

  return cv::Vec4i(bottom_x, y_bottom, top_x, y_top);
}
cv::Mat LaneDeparture::apply_yellow_lane_mask(const cv::Mat &hls_frame) {
  cv::Mat yellow_mask, orange_mask;

  const auto lower_yellow_thresh = cv::Scalar(0, 35, 40);
  const auto upper_yellow_thresh = cv::Scalar(35, 255, 255);
  cv::inRange(hls_frame, lower_yellow_thresh, upper_yellow_thresh, yellow_mask);

  return yellow_mask;
}
cv::Mat LaneDeparture::apply_white_lane_mask(const cv::Mat &hls_frame) {
  cv::Mat white_mask;

  const auto lower_white_thresh = cv::Scalar(0, 100, 0);
  const auto upper_white_thresh = cv::Scalar(180, 255, 255);
  cv::inRange(hls_frame, lower_white_thresh, upper_white_thresh, white_mask);

  return white_mask;
}
