#ifndef LANE_LINE_H
#define LANE_LINE_H
#include <opencv2/core/matx.hpp>
#include <optional>

enum class DepartureStatus {
  SAFE,
  WARNING,
  ALERT,
};

class LaneState {
public:
  std::deque<cv::Vec4i> previous_left_lanes;
  std::deque<cv::Vec4i> previous_right_lanes;
  // {x_bottom, y_bottom, x_top, y_top}
  std::optional<cv::Vec4i> left_lane;
  std::optional<cv::Vec4i> right_lane;
  static constexpr size_t max_lane_history_frames = 10;
  int x_lane_center{0};
  int x_car_center{0};
  bool left_lane_estimate{false};
  bool right_lane_estimate{false};

  DepartureStatus left_status{DepartureStatus::SAFE};
  DepartureStatus right_status{DepartureStatus::SAFE};

  void reset() {
    left_lane.reset();
    right_lane.reset();
    left_lane_estimate = false;
    right_lane_estimate = false;
    left_status = DepartureStatus::SAFE;
    right_status = DepartureStatus::SAFE;
  }

  static void remember_lane(std::deque<cv::Vec4i> &history, const cv::Vec4i &lane) {
    history.push_back(lane);
    if (history.size() > max_lane_history_frames) {
      history.pop_front();
    }
  }

  static std::optional<cv::Vec4i> estimate_lane(const std::deque<cv::Vec4i> &history) {
    if (history.empty())
      return std::nullopt;

    long x_bottom_sum{0}, y_bottom_sum{0}, x_top_sum{0}, y_top_sum{0};
    for (const auto &lane : history) {
      x_bottom_sum += lane[0];
      y_bottom_sum += lane[1];
      x_top_sum += lane[2];
      y_top_sum += lane[3];
    }
    const auto count = static_cast<long>(history.size());
    return cv::Vec4i(static_cast<int>(x_bottom_sum / count), static_cast<int>(y_bottom_sum / count),
                     static_cast<int>(x_top_sum / count), static_cast<int>(y_top_sum / count));
  }
};

// Used to simplify iteration of lines into slop intercept form
struct LineData {
  double slope{0.0};
  double intercept{0.0};
  double length{0};
  int bottom_x{0};

  LineData() = default;
  LineData(const double slope, const double intercept, const double length, const int bottom_x)
      : slope{slope}, intercept{intercept}, length{length}, bottom_x{bottom_x} {};
};

#endif // LANE_LINE_H
