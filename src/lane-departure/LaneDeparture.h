#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "../ADASFeature.h"
#include "LaneState.h"

class LaneDeparture : public ADASFeature {
  std::string name{"Lane Departure Warning System"};
  const double horizon_y_start_percentage = 0.55;
  const double dashboard_y_end_percentage = 0.85;
  const double car_x_center_percentage = 0.5;

  int horizon_y_start_pixel = 0;
  int dashboard_y_end_pixel = 0;
public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(const cv::Mat &frame) override;
  std::string get_feature_name() override;

private:
  cv::Mat apply_canny_edge_detection(const cv::Mat &frame);
  cv::Mat apply_scharr_edge_detection(const cv::Mat &frame);
  void calculate_road_y_range(const cv::Mat &frame);
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
};

#endif // LANE_DEPARTURE_H
