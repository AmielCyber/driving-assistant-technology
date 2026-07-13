#include "LaneDeparture.h"
#include <opencv2/imgproc.hpp>

cv::Mat LaneDeparture::process(cv::Mat frame) {
  const cv::Mat canny_frame = apply_canny_edge_detection(frame);
  const cv::Mat roi_frame = apply_region_of_interest(canny_frame);
  const std::vector<cv::Vec4i> segments = get_hough_probabilistic_lines(roi_frame);
  const cv::Mat lines = apply_lines(frame, segments);

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
std::vector<cv::Point> LaneDeparture::get_trapezoid_roi(const int height, const int width) {
  const int trapezoid_bottom = static_cast<int>(height * 0.85);
  const int trapezoid_top = static_cast<int>(height * 0.55);
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
  const int roi_bottom = static_cast<int>(height * 0.85);
  const int roi_top = static_cast<int>(height * 0.55);
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
  constexpr int warning_threshold = 40; // Pixels off-center for orange warning
  constexpr int alert_threshold = 100;  // Pixels off-center for red alert
  LaneState state;
  // Assume camera is in the center, may change this with an algorithm
  state.x_car_center = width / 2;

  // Get average of left and right lanes
  double left_slope_sum = 0, left_intercept_sum = 0, left_weight = 0;
  double right_slope_sum = 0, right_intercept_sum = 0, right_weight = 0;

  for (const auto &s : segments) {
    const double x1 = s[0], y1 = s[1], x2 = s[2], y2 = s[3];

    // Avoid division by zero
    if (x1 == x2)
      continue;

    const double slope = (y2 - y1) / (x2 - x1);
    const double intercept = y1 - slope * x1;
    // Intensive operation, may need to rewrite for optimization
    // Using Euclidean distance equation
    const double length = std::sqrt(std::pow(y2 - y1, 2) + std::pow(x2 - x1, 2));

    // Filter out non-vertical lines
    if (std::abs(slope) < vertical_line_threshold)
      continue;

    if (slope < 0 && x1 < static_cast<double>(width) / 2 && x2 < static_cast<double>(width) / 2) {
      // Left lane detection
      left_slope_sum += slope * length;
      left_intercept_sum += intercept * length;
      left_weight += length;
    } else if (slope > 0 && x1 > static_cast<double>(width) / 2 && x2 > static_cast<double>(width) / 2) {
      // Right lane detection
      right_slope_sum += slope * length;
      right_intercept_sum += intercept * length;
      right_weight += length;
    }
  }

  // Calculate vertical line boundaries from bottom and top of ROI
  const int y_bottom = height;
  const int y_top = static_cast<int>(height * 0.6);

  int left_bottom_x = 0;
  int right_bottom_x = width;

  if (left_weight > 0) {
    // Set left lane
    /* / */
    const double m = left_slope_sum / left_weight;
    const double b = left_intercept_sum / left_weight;
    left_bottom_x = static_cast<int>((y_bottom - b) / m);
    const int top_x = static_cast<int>((y_top - b) / m);
    state.left_lane = cv::Vec4i(left_bottom_x, y_bottom, top_x, y_top);
  }

  if (right_weight > 0) {
    // Set right lane
    /* \ */
    const double m = right_slope_sum / right_weight;
    const double b = right_intercept_sum / right_weight;
    right_bottom_x = static_cast<int>((y_bottom - b) / m);
    const int top_x = static_cast<int>((y_top - b) / m);
    state.right_lane = cv::Vec4i(right_bottom_x, y_bottom, top_x, y_top);
  }

  // Detect middle of the lane
  if (state.left_lane && state.right_lane) {
    state.x_lane_center = (left_bottom_x + right_bottom_x) / 2;
  } else {
    // Fallback to the middle of the car center
    state.x_lane_center = state.x_car_center;
  }

  // Detect lane crossing
  int offset = state.x_car_center - state.x_lane_center;

  // Check left lane proximity
  if (offset > alert_threshold) {
    // leaving lane
    state.left_status = DepartureStatus::ALERT;
  } else if (offset > warning_threshold) {
    // Approaching lane
    state.left_status = DepartureStatus::WARNING;
  }

  // Check right lane proximity
  if (offset < -alert_threshold) {
    // leaving lane
    state.right_status = DepartureStatus::ALERT;
  } else if (offset < -warning_threshold) {
    // Approaching lane
    state.right_status = DepartureStatus::WARNING;
  }
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
cv::Vec4i LaneDeparture::get_line(const int y_bottom, const int y_top, double slope, double intercept) {
  // y = mx + b -> x = (y-b)/m , b-intercept, m-slope
  const int x_bottom = static_cast<int>((y_bottom - intercept) * slope);
  const int x_top = static_cast<int>((y_top - intercept) * slope);

  return {x_bottom, y_bottom, x_top, y_top};
}
