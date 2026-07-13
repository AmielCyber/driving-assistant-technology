#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "../ADASFeature.h"
#include "LaneState.h"

class LaneDeparture : public ADASFeature {
  std::string name{"Lane Departure Warning System"};
  const double horizon_y_ratio{0.55};
  const double dashboard_y_ratio{0.85};
  const double car_x_center_percentage{0.5};

  const double warning_threshold_pct{0.10};
  const double alert_threshold_pct{0.20};
  double cluster_threshold_pct{0.05};

  int horizon_y_start_pixel = 0;
  int dashboard_y_end_pixel = 0;
  int car_x_center_pixel = 0;

public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(cv::Mat &frame) override;
  std::string get_feature_name() override;

private:
  cv::Mat apply_canny_edge_detection(const cv::Mat &frame);
  cv::Mat apply_scharr_edge_detection(const cv::Mat &frame);
  void calculate_image_points(const cv::Mat &frame);
  std::vector<cv::Point> get_trapezoid_roi(int height, int width) const;
  std::vector<cv::Point> get_perspective_roi(int height, int width);
  cv::Mat get_wrapped_roi(const cv::Mat &frame);
  cv::Mat apply_region_of_interest(const cv::Mat &canny_frame);
  cv::Mat get_region_of_interest(const cv::Mat &canny_frame);
  std::vector<cv::Vec4i> get_hough_probabilistic_lines(const cv::Mat &roi_frame);
  cv::Mat apply_lines(const cv::Mat &frame, const std::vector<cv::Vec4i> &lines);
  void draw_lines(const std::vector<cv::Vec4i> &lines, cv::Mat &original_frame);
  LaneState analyze_lane(const std::vector<cv::Vec4i> &segments, int width, int height);
  void draw_overlay(const LaneState &state, cv::Mat &frame);
  [[nodiscard]] cv::Vec4i get_line(double slope, double intercept) const;
  cv::Vec2d get_avg_fitting_line(const std::vector<cv::Vec2d> &fits);
  std::pair<cv::Vec4i, cv::Vec4i> get_avg_left_and_right_line(const std::vector<cv::Vec4i> &hough_lines);
  LaneState get_lane_state(const std::pair<cv::Vec4i, cv::Vec4i> &left_and_right_line, int width, int height);
  std::vector<LineData> extract_line_data(const std::vector<cv::Vec4i> &segments, int width, int height);
  std::optional<cv::Vec4i> calculate_lane_boundary(std::vector<LineData> &lines, int width, int height, bool is_left);
  void evaluate_departure_status(LaneState &state, int width) const;
  std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>>
  separate_lanes(const std::vector<cv::Vec4i> &segments, int car_center_x,
                                double vertical_line_threshold);
  std::optional<cv::Vec4i> calculate_closest_lane(const std::vector<cv::Vec4i> &segments, int y_bottom,
                                                                 int y_top, int car_center_x, int width);
  static cv::Mat apply_yellow_lane_mask(const cv::Mat &hls_frame);
  static cv::Mat apply_white_lane_mask(const cv::Mat &hls_frame);
};

#endif // LANE_DEPARTURE_H
