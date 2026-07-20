#ifndef LANE_LINE_H
#define LANE_LINE_H
#include <optional>
#include <opencv2/core/matx.hpp>

enum class DepartureStatus {
  SAFE,
  WARNING,
  ALERT,
};

class LaneState {
  public:
  // {x_bottom, y_bottom, x_top, y_top}
  std::optional<cv::Vec4i> left_lane;
  std::optional<cv::Vec4i> right_lane;
  // True only when a lane was inferred rather than directly detected.
  bool left_lane_estimated{false};
  bool right_lane_estimated{false};

  int x_lane_center{0};
  int x_car_center{0};

  DepartureStatus left_status{DepartureStatus::SAFE};
  DepartureStatus right_status{DepartureStatus::SAFE};
};

  // Used to simplify iteration of lines into slop intercept form
struct LineData {
  double slope{0.0};
  double intercept{0.0};
  double length{0};
  int bottom_x{0};

  LineData() = default;
  LineData(const double slope, const double intercept, const double length, const int bottom_x): slope{slope}, intercept{intercept}, length{length}, bottom_x {bottom_x} {};
};

#endif // LANE_LINE_H
