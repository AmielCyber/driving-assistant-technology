#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "../ADASFeature.h"
#include "LaneState.h"
#include "LaneStateLogger.h"

#include "../ADASFeature.h"
#include "LaneState.h"
#include "LaneStateLogger.h"

#include <optional>

class LaneDeparture : public ADASFeature {
  std::string name{"Lane Departure Warning System"};
  // Hard Coded Image Location Ratios
  const double horizon_y_ratio{0.60};
  const double dashboard_y_ratio{0.78};
  const double car_x_center_ratio{0.5};

  // Threshold for lanes
  const double warning_threshold_from_center_ratio{0.20};
  const double alert_threshold_from_center_ratio{0.15};
  // Threshold to select a line out of a cluster of lines from houghP
  double line_cluster_threshold_ratio{0.04};
  LaneStateLogger lane_state_logger{};

  // Learned apparent lane width in pixels.
  // Separate measurements are needed because perspective makes the lane
  // narrower near the horizon than at the bottom of the ROI.
  std::optional<double> smoothed_lane_width_bottom;
  std::optional<double> smoothed_lane_width_top;

  // Exponential moving-average weight.
  // Smaller values provide more temporal stability.
  const double lane_width_smoothing_alpha{0.15};

public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(cv::Mat &frame) override;
  std::string get_feature_name() override;

private:
  static cv::Mat apply_yellow_orange_lane_mask(const cv::Mat &hls_frame);
  static cv::Mat apply_white_lane_mask(const cv::Mat &hls_frame);
  static cv::Mat apply_canny_edge_detection(const cv::Mat &frame);
  [[nodiscard]] cv::Mat apply_region_of_interest(const cv::Mat &canny_frame) const;
  [[nodiscard]] std::vector<cv::Point> get_trapezoid_roi(int height, int width) const;
  static std::vector<cv::Vec4i> get_probabilistic_hough_lines(const cv::Mat &roi_frame);
  [[nodiscard]] LaneState analyze_lane(
      const std::vector<cv::Vec4i> &lines,
      int width,
      int height
  );
  static std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
  separate_lanes(const std::vector<cv::Vec4i> &lines, int car_center_x);
  [[nodiscard]] std::optional<cv::Vec4i> calculate_closest_lane(const std::vector<cv::Vec4i> &lines, int y_bottom,
                                                                int y_top, int car_center_x, int width) const;
  void evaluate_departure_status(LaneState &state) const;
  void draw_overlay(const LaneState &state, cv::Mat &frame);
  static void draw_alert_triangle(cv::Mat& frame);

  void update_lane_width_estimate(
      const LaneState &state,
      int frame_width
  );

  void estimate_missing_right_lane(
      LaneState &state,
      int frame_width
  ) const;

  static cv::Mat clean_lane_mask(const cv::Mat &mask);

  static void shade_lane_region(
      const LaneState &state,
      cv::Mat &frame
  );
};

#endif // LANE_DEPARTURE_H
