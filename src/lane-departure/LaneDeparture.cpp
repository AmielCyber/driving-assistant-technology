#include "LaneDeparture.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat LaneDeparture::process(cv::Mat &frame) {
  cv::Mat hls, combined_mask, masked_frame;
  // Extract yellow colors and white  for lane detection and noise reduction
  cv::cvtColor(frame, hls, cv::COLOR_BGR2HLS);                    // HLS Conversion
  const cv::Mat yellow_mask = apply_yellow_orange_lane_mask(hls); // Extract Yellow-Orange
  const cv::Mat white_mask = apply_white_lane_mask(hls);          // Extract Dark White

  cv::bitwise_or(yellow_mask, white_mask, combined_mask);
  // Applying binary smoothing to smooth out the lanes.
  // https://docs.opencv.org/5.0/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 10));
  cv::morphologyEx(combined_mask, combined_mask, cv::MORPH_CLOSE, kernel);

  cv::bitwise_and(frame, frame, masked_frame, combined_mask);

  const cv::Mat canny_frame = apply_canny_edge_detection(masked_frame); // Perform Canny Edge
  const cv::Mat roi_frame = apply_region_of_interest(canny_frame);      // Perform Region of Interest (ROI)
  const std::vector<cv::Vec4i> hough_lines = get_probabilistic_hough_lines(roi_frame); // Perform HoughP
  const LaneState state = analyze_lane(hough_lines, frame.cols, frame.rows);           // Get Lane State
  // lane_state_logger.log_lane_departure_status(state);
  draw_overlay(state, frame); // Draw Annotations based on the lane state

  return frame;
}

std::string LaneDeparture::get_feature_name() { return this->name; }

cv::Mat LaneDeparture::apply_yellow_orange_lane_mask(const cv::Mat &hls_frame) {
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
  // Optimal Parameters from Assignment 4 Problem 2 for finding lanes
  const auto kernel_size = cv::Size(5, 5);
  cv::Mat gray, blur, canny_frame;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, kernel_size, 0);

  cv::Mat tmp;
  const double upper_thresh = cv::threshold(blur, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  const double lower_thresh = upper_thresh * 0.5;
  cv::Canny(blur, canny_frame, lower_thresh, upper_thresh);
  return canny_frame;
}

cv::Mat LaneDeparture::apply_region_of_interest(const cv::Mat &canny_frame) const {
  // Using a trapezoid to approximate Road View Extraction
  const int height = canny_frame.rows;
  const int width = canny_frame.cols;

  const std::vector<cv::Point> trapezoid_points = get_trapezoid_roi(height, width);
  cv::Mat mask = cv::Mat::zeros(canny_frame.size(), canny_frame.type());
  // Create white trapezoid and place it in the blacked-out frame with the same size of the canny frame.
  cv::fillPoly(mask, trapezoid_points, cv::Scalar(255));

  cv::Mat masked_frame;
  // Perform and operations on canny frame and white trapezoid, leaving only canny frame pixels within the trapezoid
  cv::bitwise_and(canny_frame, mask, masked_frame);

  return masked_frame;
}

std::vector<cv::Point> LaneDeparture::get_trapezoid_roi(const int height, const int width) const {
  const int trapezoid_top = static_cast<int>(height * horizon_y_ratio);
  const int trapezoid_bottom = static_cast<int>(height * dashboard_y_ratio);
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
  // Optimal Parameters from Assignment 4 Problem 2 for finding lanes
  constexpr double theta = CV_PI / 180;
  constexpr double threshold = 30;
  constexpr double min_line_length = 15;
  constexpr double max_line_gap = 300;

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

  // Separate lines into left and right line candidates
  auto [left_lines, right_lines] = separate_lanes(lines, state.x_car_center);

  // Filter out outer lanes/bike lanes by clustering around closest x-intercept
  state.left_lane = calculate_closest_lane(left_lines, y_bottom, y_top, state.x_car_center, width);
  state.right_lane = calculate_closest_lane(right_lines, y_bottom, y_top, state.x_car_center, width);

  // Calculate lane center
  if (state.left_lane && state.right_lane) {
    const int left_bottom_x = state.left_lane.value()[0];
    const int right_bottom_x = state.right_lane.value()[0];
    state.x_lane_center = (left_bottom_x + right_bottom_x) / 2;
  } else {
    state.x_lane_center = state.x_car_center;
  }

  // Evaluate departure status based on percentage of frame width
  evaluate_departure_status(state);

  return state;
}

std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
LaneDeparture::separate_lanes(const std::vector<cv::Vec4i> &lines, const int car_center_x) {
  std::vector<cv::Vec4i> left_lines, right_lines;
  constexpr double min_slope_threshold = 0.40;

  for (const auto &l : lines) {
    const double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    double dx = x2 - x1;
    if (x1 == x2)
      // Avoid division by zero
        // May change if i get too many false positives
      dx = 0.0001;
    const double slope = (y2 - y1) / dx;
    if (std::abs(slope) < min_slope_threshold)
      continue;

    if (slope < 0 && x1 < car_center_x && x2 < car_center_x) {
      //  /  x_center
      // Estimated Left lane contender
      left_lines.push_back(l);
    } else if (slope > 0 && x1 > car_center_x && x2 > car_center_x) {
      //  x_center  \
      // Estimated right lane contender
      right_lines.push_back(l);
    }
  }
  return {left_lines, right_lines};
}

// https://learning.oreilly.com/library/view/applied-deep-learning/9781838646301/e5a39ce5-e62a-44a1-827e-6da9ab20ff70.xhtml#uuid-efab72e9-73f7-43ef-9d14-46225fd65ed9
// Optimizing the detected road markings
std::optional<cv::Vec4i> LaneDeparture::calculate_closest_lane(const std::vector<cv::Vec4i> &lines, const int y_bottom_limit,
                                                               const int y_top_limit, const int car_center_x,
                                                               const int width) const {
  if (lines.empty())
    return std::nullopt;

  std::vector<LineData> processed_lines;
  // Target distance line  & bottom_x to achieve the closest lane
  // in case there are other lanes such as sidewalk, bike lane and other lanes not in the current one
  double closest_distance = std::numeric_limits<double>::max();
  int target_bottom_x = 0;

  // Convert each line segment into slope intercept form
  for (const auto &line : lines) {
    // m = (y2-y1)/(x2-x1)
    const double slope = static_cast<double>(line[3] - line[1]) / (line[2] - line[0]);
    // b = y1 - m * x1
    const double intercept = line[1] - slope * line[0];
    // d = sqrt((x^2-x^1)+(y^2-y^1))
    // Replaced sqrt(x2 - y2) for better performance
    const double length = std::hypot(line[3] - line[1], line[2] - line[0]);
    // Compute where line starts from ROI aka dashboard
    const int bottom_x = static_cast<int>((y_bottom_limit - intercept) / slope);

    processed_lines.emplace_back(slope, intercept, length, bottom_x);

    if (const double distance_to_center = std::abs(car_center_x - bottom_x); distance_to_center < closest_distance) {
      // Found a closer line towards the center
      closest_distance = distance_to_center;
      target_bottom_x = bottom_x;
    }
  }
  // Convert cluster threshold percentage to actual pixel margin for this resolution
  // Cluster lines to determine which lines belong to the other same lines (merge lines together)
  const double cluster_pixel_threshold = width * line_cluster_threshold_ratio;
  double slope_sum = 0, intercept_sum = 0, weight = 0;

  // Get weighted average of all lines to get a smooth line to approximate actual lane
  // average_slop_intercept
  //    ////// ->    /
  for (const auto &line : processed_lines) {
    if (std::abs(line.bottom_x - target_bottom_x) <= cluster_pixel_threshold) {
      slope_sum += line.slope * line.length;
      intercept_sum += line.intercept * line.length;
      weight += line.length;  // To get average slope and intercept
    }
  }

  if (weight == 0)
    return std::nullopt;

  // Get the best fitting lane from the list of lines found
  const double m = slope_sum / weight;
  const double b = intercept_sum / weight;

  // y = mx + b -> x = (y-b)/m
  const int x_bottom = static_cast<int>((y_bottom_limit - b) / m);
  const int x_top = static_cast<int>((y_top_limit - b) / m);

  //    (x_t, y_t)
  //        |
  //        |
  //    (x_b, y_b)
  return cv::Vec4i(x_bottom, y_bottom_limit, x_top, y_top_limit);
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

  const int left_wheel_x = state.x_car_center - state.x_car_center * alert_threshold_from_center_ratio;
  const int right_wheel_x = state.x_car_center + state.x_car_center * alert_threshold_from_center_ratio;
  const int warning_left_line = state.x_car_center - state.x_car_center * warning_threshold_from_center_ratio;
  const int warning_right_line = state.x_car_center + state.x_car_center * warning_threshold_from_center_ratio;

  cv::line(frame, cv::Point(left_wheel_x, frame.rows - 50), cv::Point(left_wheel_x, frame.rows - 25),
           departure_lane_color, 2, cv::LINE_AA);
  cv::line(frame, cv::Point(right_wheel_x, frame.rows - 50), cv::Point(right_wheel_x, frame.rows - 25),
           departure_lane_color, 2, cv::LINE_AA);

  cv::line(frame, cv::Point(warning_left_line, frame.rows - 50), cv::Point(warning_left_line, frame.rows - 25),
           warning_lane_color, 2, cv::LINE_AA);
  cv::line(frame, cv::Point(warning_right_line, frame.rows - 50), cv::Point(warning_right_line, frame.rows - 25),
           warning_lane_color, 2, cv::LINE_AA);

  if (state.right_status == DepartureStatus::ALERT || state.left_status == DepartureStatus::ALERT) {
    draw_alert_triangle(frame);
  }
}

void LaneDeparture::evaluate_departure_status(LaneState &state) const {
  // Calculate dynamic pixel thresholds based on frame width percentage
  const double warning_pixels_offset = state.x_car_center * warning_threshold_from_center_ratio;
  const double alert_pixels_offset = state.x_car_center * alert_threshold_from_center_ratio;

  if (state.left_lane.has_value()) {
    if (state.left_lane.value()[0] >= state.x_car_center - alert_pixels_offset) {
      state.left_status = DepartureStatus::ALERT;
    } else if (state.left_lane.value()[0] >= state.x_car_center - warning_pixels_offset) {
      state.left_status = DepartureStatus::WARNING;
    }
  }

  if (state.right_lane.has_value()) {
    if (state.right_lane.value()[0] <= state.x_car_center + alert_pixels_offset) {
      state.right_status = DepartureStatus::ALERT;
    } else if (state.right_lane.value()[0] <= state.x_car_center + warning_pixels_offset) {
      state.right_status = DepartureStatus::WARNING;
    }
  }
}

// GPT GENERATED CODE: in C++ opencv 4 how I can draw a small triangle with the character '!' in red in the middle of
// frame
void LaneDeparture::draw_alert_triangle(cv::Mat &frame) {
  const cv::Point center(frame.cols / 2, frame.rows / 2); // Calculate center point
  const cv::Scalar color(0, 0, 255);
  constexpr int size = 50, thickness = 2;
  constexpr int hw = static_cast<int>(size * 0.9); // Horizontal width

  // Draw Triangle
  const std::vector<cv::Point> pts = {
      {center.x, center.y - size}, {center.x - hw, center.y + size / 2}, {center.x + hw, center.y + size / 2}};
  cv::polylines(frame, std::vector<std::vector<cv::Point>>{pts}, true, color, thickness, cv::LINE_AA);

  // Draw Exclamation Mark
  constexpr double fontScale = std::max(0.5, size / 33.0);
  int baseline = 0;
  cv::Size sz = cv::getTextSize("!", cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
  cv::putText(frame, "!", {center.x - sz.width / 2, center.y + sz.height / 2}, cv::FONT_HERSHEY_DUPLEX, fontScale,
              color, thickness, cv::LINE_AA);
}