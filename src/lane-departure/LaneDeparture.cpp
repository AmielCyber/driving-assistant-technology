#include "LaneDeparture.h"
#include <opencv2/imgproc.hpp>

cv::Mat LaneDeparture::process(cv::Mat &frame) {
  cv::Mat hls, combined_mask, masked_frame;
  // Extract yellow colors and white  for lane detection and noise reduction
  cv::cvtColor(frame, hls, cv::COLOR_BGR2HLS);                // HLS Conversion
  const cv::Mat yellow_mask = apply_yellow_lane_mask(hls);    // Extract Yellow
  const cv::Mat white_mask = apply_white_lane_mask(hls);      // Extract White
  cv::bitwise_or(yellow_mask, white_mask, combined_mask);     // Combine yellow & white
  cv::bitwise_and(frame, frame, masked_frame, combined_mask); // Combine yellow_white w/ original

  const cv::Mat canny_frame = apply_canny_edge_detection(masked_frame); // Canny Edge
  const cv::Mat roi_frame = apply_region_of_interest(canny_frame);      // ROI
  const std::vector<cv::Vec4i> hough_lines = get_probabilistic_hough_lines(roi_frame);
  const LaneState state = analyze_lane(hough_lines, frame.cols, frame.rows);
  draw_overlay(state, frame);

  return frame;
}

std::string LaneDeparture::get_feature_name() { return this->name; }

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

cv::Mat LaneDeparture::apply_canny_edge_detection(const cv::Mat &frame) {
  constexpr double lower_thresh = 10;
  constexpr double upper_thresh = 50;
  cv::Mat gray, blur, canny_frame;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0.0);
  cv::Canny(blur, canny_frame, lower_thresh, upper_thresh);
  return canny_frame;
}

cv::Mat LaneDeparture::apply_region_of_interest(const cv::Mat &canny_frame) const {
  // Using a trapezoid to approximate Road View
  const int height = canny_frame.rows;
  const int width = canny_frame.cols;

  const std::vector<cv::Point> trapezoid_points = get_trapezoid_roi(height, width);
  cv::Mat mask = cv::Mat::zeros(canny_frame.size(), canny_frame.type());
  cv::fillPoly(mask, trapezoid_points, cv::Scalar(255));

  cv::Mat masked_frame;
  cv::bitwise_and(canny_frame, mask, masked_frame);

  return masked_frame;
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

std::vector<cv::Vec4i> LaneDeparture::get_probabilistic_hough_lines(const cv::Mat &roi_frame) {
  constexpr double theta = CV_PI / 180;
  constexpr double threshold = 60;
  constexpr double min_line_length = 15;
  constexpr double max_line_gap = 20;

  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(roi_frame, segments, 1, theta, threshold, min_line_length, max_line_gap);
  return segments;
}

LaneState LaneDeparture::analyze_lane(const std::vector<cv::Vec4i> &lines, const int width, const int height) const {
  LaneState state;
  state.x_car_center = static_cast<int>(car_x_center_ratio * width);

  // Convert percentage ratios to actual Y-pixel coordinates
  const int y_top = static_cast<int>(height * horizon_y_ratio);
  const int y_bottom = static_cast<int>(height * dashboard_y_ratio);

  // Separate lines into left and right candidates
  auto [left_segments, right_segments] = separate_lanes(lines, state.x_car_center);

  // 2. Filter out outer lanes/bike lanes by clustering around closest x-intercept
  state.left_lane = calculate_closest_lane(left_segments, y_bottom, y_top, state.x_car_center, width);
  state.right_lane = calculate_closest_lane(right_segments, y_bottom, y_top, state.x_car_center, width);

  // 3. Calculate lane center
  if (state.left_lane && state.right_lane) {
    const int left_bottom_x = state.left_lane.value()[0];
    const int right_bottom_x = state.right_lane.value()[0];
    state.x_lane_center = (left_bottom_x + right_bottom_x) / 2;
  } else {
    state.x_lane_center = state.x_car_center;
  }

  // 4. Evaluate departure status based on percentage of frame width
  evaluate_departure_status(state, width);

  return state;
}

std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
LaneDeparture::separate_lanes(const std::vector<cv::Vec4i> &segments, const int car_center_x) {
  std::vector<cv::Vec4i> left_segments, right_segments;
  constexpr double vertical_line_threshold = 0.3;

  for (const auto &s : segments) {
    const double x1 = s[0], y1 = s[1], x2 = s[2], y2 = s[3];
    if (x1 == x2)
      continue;

    const double slope = (y2 - y1) / (x2 - x1);
    if (std::abs(slope) < vertical_line_threshold)
      continue;

    if (slope < 0 && x1 < car_center_x && x2 < car_center_x) {
      left_segments.push_back(s);
    } else if (slope > 0 && x1 > car_center_x && x2 > car_center_x) {
      right_segments.push_back(s);
    }
  }
  return {left_segments, right_segments};
}

std::optional<cv::Vec4i> LaneDeparture::calculate_closest_lane(const std::vector<cv::Vec4i> &lines, const int y_bottom,
                                                               const int y_top, const int car_center_x,
                                                               const int width) const {
  if (lines.empty())
    return std::nullopt;

  struct LineData {
    double slope;
    double intercept;
    double length;
    double bottom_x;
  };

  std::vector<LineData> processed_lines;
  double closest_distance = std::numeric_limits<double>::max();
  double target_bottom_x = 0;

  for (const auto &line : lines) {
    const double slope = static_cast<double>(line[3] - line[1]) / (line[2] - line[0]);
    const double intercept = line[1] - slope * line[0];
    const double length = std::hypot(line[3] - line[1], line[2] - line[0]);
    const double bottom_x = (y_bottom - intercept) / slope;

    processed_lines.push_back({slope, intercept, length, bottom_x});

    double distance_to_center = std::abs(car_center_x - bottom_x);
    if (distance_to_center < closest_distance) {
      closest_distance = distance_to_center;
      target_bottom_x = bottom_x;
    }
  }
  // Convert cluster threshold percentage to actual pixel margin for this resolution
  const double cluster_pixel_threshold = width * cluster_threshold_pct;
  double slope_sum = 0, intercept_sum = 0, weight = 0;

  for (const auto &line : processed_lines) {
    if (std::abs(line.bottom_x - target_bottom_x) <= cluster_pixel_threshold) {
      slope_sum += line.slope * line.length;
      intercept_sum += line.intercept * line.length;
      weight += line.length;
    }
  }

  if (weight == 0)
    return std::nullopt;

  const double m = slope_sum / weight;
  const double b = intercept_sum / weight;

  const int bottom_x = static_cast<int>((y_bottom - b) / m);
  const int top_x = static_cast<int>((y_top - b) / m);

  return cv::Vec4i(bottom_x, y_bottom, top_x, y_top);
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

void LaneDeparture::evaluate_departure_status(LaneState &state, const int width) const {
  const int offset = state.x_car_center - state.x_lane_center;
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
