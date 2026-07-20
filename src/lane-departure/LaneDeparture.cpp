#include "LaneDeparture.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <limits>

cv::Mat LaneDeparture::process(cv::Mat &frame) {
  cv::Mat hls, combined_mask, masked_frame;
  // Extract yellow colors and white  for lane detection and noise reduction
  cv::cvtColor(frame, hls, cv::COLOR_BGR2HLS);                    // HLS Conversion
  const cv::Mat yellow_mask = apply_yellow_orange_lane_mask(hls); // Extract Yellow-Orange
  const cv::Mat white_mask = apply_white_lane_mask(hls);          // Extract Dark White
  cv::bitwise_or(yellow_mask, white_mask, combined_mask);         // Combine yellow-orange & white imagee
  cv::bitwise_and(frame, frame, masked_frame, combined_mask);     // Combine yellow_white w/ original

  const cv::Mat canny_frame = apply_canny_edge_detection(masked_frame); // Perform Canny Edge
  const cv::Mat roi_frame = apply_region_of_interest(canny_frame);      // Perform Region of Interest (ROI)
  const std::vector<cv::Vec4i> hough_lines = get_probabilistic_hough_lines(roi_frame); // Perform HoughP
  const LaneState state = analyze_lane(hough_lines, frame.cols, frame.rows);           // Get Lane State
  // lane_state_logger.log_lane_departure_status(state);
  draw_overlay(state, frame); // Draw Annotations based on the lane state

  return frame;
}

std::string LaneDeparture::get_feature_name() { return this->name; }

cv::Mat LaneDeparture::apply_yellow_orange_lane_mask(
    const cv::Mat &hls_frame
) {
  cv::Mat yellow_orange_mask;

  /*
   * OpenCV HLS order:
   * H = hue
   * L = lightness
   * S = saturation
   *
   * This range includes common yellow and orange road markings while
   * excluding much of the low-saturation gray pavement.
   */
  const cv::Scalar lower_yellow_orange(8, 30, 70);
  const cv::Scalar upper_yellow_orange(40, 255, 255);

  cv::inRange(
      hls_frame,
      lower_yellow_orange,
      upper_yellow_orange,
      yellow_orange_mask
  );

  return clean_lane_mask(yellow_orange_mask);
}

cv::Mat LaneDeparture::apply_white_lane_mask(
    const cv::Mat &hls_frame
) {
  cv::Mat white_mask;

  /*
   * Permit moderately dark/faded white markings by starting lightness
   * at 115, but require relatively low saturation so bright colored
   * objects are less likely to pass the mask.
   */
  const cv::Scalar lower_white(0, 115, 0);
  const cv::Scalar upper_white(180, 255, 110);

  cv::inRange(
      hls_frame,
      lower_white,
      upper_white,
      white_mask
  );

  return clean_lane_mask(white_mask);
}

cv::Mat LaneDeparture::clean_lane_mask(const cv::Mat &mask) {
  cv::Mat cleaned_mask;

  const cv::Mat close_kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(5, 5)
  );

  const cv::Mat open_kernel = cv::getStructuringElement(
      cv::MORPH_RECT,
      cv::Size(3, 3)
  );

  // Close small gaps in worn or dashed lane markings.
  cv::morphologyEx(
      mask,
      cleaned_mask,
      cv::MORPH_CLOSE,
      close_kernel
  );

  // Remove isolated single-pixel and small-region noise.
  cv::morphologyEx(
      cleaned_mask,
      cleaned_mask,
      cv::MORPH_OPEN,
      open_kernel
  );

  return cleaned_mask;
}

cv::Mat LaneDeparture::apply_canny_edge_detection(const cv::Mat &frame) {
  // Optimal Parameters from Assignment 4 Problem 2 for finding lanes
  constexpr double lower_thresh = 10;
  constexpr double upper_thresh = 50;
  const auto kernel_size = cv::Size(5, 5);
  cv::Mat gray, blur, canny_frame;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, kernel_size, 0);
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
  constexpr double threshold = 60;
  constexpr double min_line_length = 15;
  constexpr double max_line_gap = 20;

  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(roi_frame, segments, 1, theta, threshold, min_line_length, max_line_gap);
  return segments;
}

LaneState LaneDeparture::analyze_lane(
    const std::vector<cv::Vec4i> &lines,
    const int width,
    const int height
) {
  LaneState state;
  state.x_car_center = static_cast<int>(
      car_x_center_ratio * width
  );

  const int y_top = static_cast<int>(
      height * horizon_y_ratio
  );

  const int y_bottom = static_cast<int>(
      height * dashboard_y_ratio
  );

  auto [left_lines, right_lines] =
      separate_lanes(lines, state.x_car_center);

  state.left_lane = calculate_closest_lane(
      left_lines,
      y_bottom,
      y_top,
      state.x_car_center,
      width
  );

  state.right_lane = calculate_closest_lane(
      right_lines,
      y_bottom,
      y_top,
      state.x_car_center,
      width
  );

  /*
   * Learn lane width only from frames where both sides were directly
   * detected. Do this before attempting an inferred lane.
   */
  if (state.left_lane.has_value() &&
      state.right_lane.has_value()) {
    update_lane_width_estimate(state, width);
  }

  /*
   * If the left lane is visible and the right fog line is absent,
   * infer the right lane from the previously learned lane width.
   */
  if (state.left_lane.has_value() &&
      !state.right_lane.has_value()) {
    estimate_missing_right_lane(state, width);
  }

  if (state.left_lane.has_value() &&
      state.right_lane.has_value()) {
    const int left_bottom_x =
        state.left_lane.value()[0];

    const int right_bottom_x =
        state.right_lane.value()[0];

    state.x_lane_center =
        (left_bottom_x + right_bottom_x) / 2;
  } else {
    state.x_lane_center = state.x_car_center;
  }

  evaluate_departure_status(state);

  return state;
}

std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
LaneDeparture::separate_lanes(const std::vector<cv::Vec4i> &lines, const int car_center_x) {
  std::vector<cv::Vec4i> left_lines, right_lines;
  constexpr double min_slope_threshold = 0.4;

  for (const auto &l : lines) {
    const double x1 = l[0], y1 = l[1], x2 = l[2], y2 = l[3];
    if (x1 == x2)
      continue;

    const double slope = (y2 - y1) / (x2 - x1);
    if (std::abs(slope) < min_slope_threshold)
      continue;

    if (slope < 0 && x1 < car_center_x && x2 < car_center_x) {
      // Estimated Left lane contender
      left_lines.push_back(l);
    } else if (slope > 0 && x1 > car_center_x && x2 > car_center_x) {
      // Estimated right lane contender
      right_lines.push_back(l);
    }
  }
  return {left_lines, right_lines};
}

std::optional<cv::Vec4i> LaneDeparture::calculate_closest_lane(const std::vector<cv::Vec4i> &lines, const int y_bottom,
                                                               const int y_top, const int car_center_x,
                                                               const int width) const {
  if (lines.empty())
    return std::nullopt;

  std::vector<LineData> processed_lines;
  // Target distance line  & bottom_x to achieve the closest lane
  // in case there are other lanes such as sidewalk, bike lane and others
  double closest_distance = std::numeric_limits<double>::max();
  int target_bottom_x = 0;

  // Convert each line segment into slope intercept form
  for (const auto &line : lines) {
    const double slope = static_cast<double>(line[3] - line[1]) / (line[2] - line[0]);
    const double intercept = line[1] - slope * line[0];
    // Replaced sqrt(x2 - y2) for better performance
    const double length = std::hypot(line[3] - line[1], line[2] - line[0]);
    // Compute where line starts from ROI aka dashboard
    const int bottom_x = static_cast<int>((y_bottom - intercept) / slope);

    processed_lines.emplace_back(slope, intercept, length, bottom_x);

    double distance_to_center = std::abs(car_center_x - bottom_x);
    if (distance_to_center < closest_distance) {
      // Found a closer line towards the center
      closest_distance = distance_to_center;
      target_bottom_x = bottom_x;
    }
  }
  // Convert cluster threshold percentage to actual pixel margin for this resolution
  // Cluster lines to determine which lines belong to the other same lines
  const double cluster_pixel_threshold = width * line_cluster_threshold_ratio;
  double slope_sum = 0, intercept_sum = 0, weight = 0;

  // Get weighted average of all lines to get a smooth line
  for (const auto &line : processed_lines) {
    if (std::abs(line.bottom_x - target_bottom_x) <= cluster_pixel_threshold) {
      slope_sum += line.slope * line.length;
      intercept_sum += line.intercept * line.length;
      weight += line.length;
    }
  }

  if (weight == 0)
    return std::nullopt;

  // Get the best fitting lane from the list of lines found
  const double m = slope_sum / weight;
  const double b = intercept_sum / weight;

  const int x_bottom = static_cast<int>((y_bottom - b) / m);
  const int x_top = static_cast<int>((y_top - b) / m);

  return cv::Vec4i(x_bottom, y_bottom, x_top, y_top);
}

void LaneDeparture::update_lane_width_estimate(
    const LaneState &state,
    const int frame_width
) {
  if (!state.left_lane.has_value() ||
      !state.right_lane.has_value()) {
    return;
  }

  const cv::Vec4i &left = state.left_lane.value();
  const cv::Vec4i &right = state.right_lane.value();

  const double bottom_width =
      static_cast<double>(right[0] - left[0]);

  const double top_width =
      static_cast<double>(right[2] - left[2]);

  /*
   * Reject geometrically implausible measurements. These limits are
   * intentionally broad and should be calibrated against the videos.
   */
  const double minimum_bottom_width =
      frame_width * 0.25;

  const double maximum_bottom_width =
      frame_width * 0.90;

  const double minimum_top_width =
      frame_width * 0.03;

  const double maximum_top_width =
      frame_width * 0.40;

  const bool bottom_width_is_valid =
      bottom_width >= minimum_bottom_width &&
      bottom_width <= maximum_bottom_width;

  const bool top_width_is_valid =
      top_width >= minimum_top_width &&
      top_width <= maximum_top_width;

  if (!bottom_width_is_valid || !top_width_is_valid) {
    return;
  }

  if (!smoothed_lane_width_bottom.has_value()) {
    smoothed_lane_width_bottom = bottom_width;
  } else {
    smoothed_lane_width_bottom =
        lane_width_smoothing_alpha * bottom_width +
        (1.0 - lane_width_smoothing_alpha) *
            smoothed_lane_width_bottom.value();
  }

  if (!smoothed_lane_width_top.has_value()) {
    smoothed_lane_width_top = top_width;
  } else {
    smoothed_lane_width_top =
        lane_width_smoothing_alpha * top_width +
        (1.0 - lane_width_smoothing_alpha) *
            smoothed_lane_width_top.value();
  }
}

void LaneDeparture::estimate_missing_right_lane(
    LaneState &state,
    const int frame_width
) const {
  if (!state.left_lane.has_value()) {
    return;
  }

  if (state.right_lane.has_value()) {
    return;
  }

  if (!smoothed_lane_width_bottom.has_value() ||
      !smoothed_lane_width_top.has_value()) {
    return;
  }

  const cv::Vec4i &left = state.left_lane.value();

  const int estimated_bottom_x = static_cast<int>(
      std::lround(
          left[0] + smoothed_lane_width_bottom.value()
      )
  );

  const int estimated_top_x = static_cast<int>(
      std::lround(
          left[2] + smoothed_lane_width_top.value()
      )
  );

  /*
   * Reject the estimate if the projected right boundary is outside
   * the frame or does not remain on the right side of the vehicle.
   */
  if (estimated_bottom_x <= state.x_car_center ||
      estimated_bottom_x >= frame_width ||
      estimated_top_x < 0 ||
      estimated_top_x >= frame_width) {
    return;
  }

  state.right_lane = cv::Vec4i(
      estimated_bottom_x,
      left[1],
      estimated_top_x,
      left[3]
  );

  state.right_lane_estimated = true;
}

void LaneDeparture::shade_lane_region(
    const LaneState &state,
    cv::Mat &frame
) {
  if (!state.left_lane.has_value() ||
      !state.right_lane.has_value()) {
    return;
  }

  const bool has_alert =
      state.left_status == DepartureStatus::ALERT ||
      state.right_status == DepartureStatus::ALERT;

  if (has_alert) {
    return;
  }

  const bool has_warning =
      state.left_status == DepartureStatus::WARNING ||
      state.right_status == DepartureStatus::WARNING;

  // OpenCV uses BGR order.
  const cv::Scalar safe_fill_color(0, 180, 0);
  const cv::Scalar warning_fill_color(0, 165, 255);

  const cv::Scalar fill_color =
      has_warning
          ? warning_fill_color
          : safe_fill_color;

  const cv::Vec4i &left = state.left_lane.value();
  const cv::Vec4i &right = state.right_lane.value();

  std::vector<cv::Point> lane_polygon{
      cv::Point(left[0], left[1]),
      cv::Point(left[2], left[3]),
      cv::Point(right[2], right[3]),
      cv::Point(right[0], right[1])
  };

  cv::Mat shading_layer = frame.clone();

  cv::fillConvexPoly(
      shading_layer,
      lane_polygon,
      fill_color,
      cv::LINE_AA
  );

  constexpr double shading_opacity = 0.20;

  cv::addWeighted(
      shading_layer,
      shading_opacity,
      frame,
      1.0 - shading_opacity,
      0.0,
      frame
  );
}

void LaneDeparture::draw_overlay(const LaneState &state, cv::Mat &frame) {
  // Shade first so lane lines and warning symbols remain visible.
  shade_lane_region(state, frame);

  const cv::Scalar center_lane_color(246, 191, 3);
  const cv::Scalar warning_lane_color(41, 139, 252);
  const cv::Scalar departure_lane_color(26, 43, 251);
  const cv::Scalar estimated_lane_color(180, 180, 180);
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
  const cv::Vec4i &lane = state.right_lane.value();

  const cv::Scalar right_lane_color =
      state.right_lane_estimated
          ? estimated_lane_color
          : get_color(state.right_status);

  const int thickness =
      state.right_lane_estimated ? 3 : 5;

  cv::line(
      frame,
      cv::Point(lane[0], lane[1]),
      cv::Point(lane[2], lane[3]),
      right_lane_color,
      thickness,
      cv::LINE_AA
  );

  if (state.right_lane_estimated) {
    cv::putText(
        frame,
        "Estimated right lane",
        cv::Point(
            std::max(10, lane[2] - 80),
            std::max(25, lane[3] - 10)
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        estimated_lane_color,
        1,
        cv::LINE_AA
    );
  }
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
  const cv::Point center(frame.cols / 2, frame.rows / 2);
  const cv::Scalar color(0, 0, 255);
  const int size = 50, thickness = 2;
  const int hw = static_cast<int>(size * 0.9);

  // 1. Draw Triangle
  std::vector<cv::Point> pts = {
      {center.x, center.y - size}, {center.x - hw, center.y + size / 2}, {center.x + hw, center.y + size / 2}};
  cv::polylines(frame, std::vector<std::vector<cv::Point>>{pts}, true, color, thickness, cv::LINE_AA);

  // 2. Draw Exclamation Mark
  double fontScale = std::max(0.5, size / 33.0);
  int baseline = 0;
  cv::Size sz = cv::getTextSize("!", cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, &baseline);
  cv::putText(frame, "!", {center.x - sz.width / 2, center.y + sz.height / 2}, cv::FONT_HERSHEY_DUPLEX, fontScale,
              color, thickness, cv::LINE_AA);
}