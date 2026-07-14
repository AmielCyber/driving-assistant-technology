#ifndef LANE_LINE_H
#define LANE_LINE_H
#include <optional>
#include <opencv2/core/matx.hpp>

enum class DepartureStatus {
  SAFE,
  WARNING,
  ALERT,
};

struct LaneState {
  std::optional<cv::Vec4i> left_lane;
  std::optional<cv::Vec4i> right_lane;

  int x_lane_center{0};
  int x_car_center{0};

  DepartureStatus left_status{DepartureStatus::SAFE};
  DepartureStatus right_status{DepartureStatus::SAFE};
};

struct LineData {
  double slope;
  double intercept;
  double length;
  int bottom_x;
};

#endif // LANE_LINE_H
