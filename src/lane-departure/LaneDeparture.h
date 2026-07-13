#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "../ADASFeature.h"
#include "LaneState.h"

class LaneDeparture : public ADASFeature {
  std::string name{"Lane Departure Warning System"};

public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(cv::Mat frame) override;
  std::string get_feature_name() override;

private:
  static cv::Mat apply_canny_edge_detection(const cv::Mat &frame);
  static cv::Mat apply_scharr_edge_detection(const cv::Mat &frame);
  static std::vector<cv::Point> get_trapezoid_roi(int height, int width);
  static std::vector<cv::Point> get_perspective_roi(int height, int width);
  static cv::Mat get_wrapped_roi(const cv::Mat &frame);
  static cv::Mat apply_region_of_interest(const cv::Mat &canny_frame);
  static cv::Mat get_region_of_interest(const cv::Mat &canny_frame);
  static std::vector<cv::Vec4i> get_hough_probabilistic_lines(const cv::Mat &roi_frame);
  static cv::Mat apply_lines(const cv::Mat &frame, const std::vector<cv::Vec4i> &lines);
  static void draw_lines(const std::vector<cv::Vec4i> &lines, cv::Mat &original_frame);
  static LaneState analyze_lane(const std::vector<cv::Vec4i> &segments, int width, int height);
  static void draw_overlay(const LaneState &state, cv::Mat &frame);
  static cv::Vec4i get_line(int y_bottom, int y_top, double slope, double intercept);
  static std::vector<cv::Vec4i> get_avg_left_and_right_line(const std::vector<cv::Vec4i> &lines);
};

#endif // LANE_DEPARTURE_H
