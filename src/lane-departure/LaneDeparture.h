#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "LaneState.h"
#include "ADASFeature.h"

class LaneDeparture: public ADASFeature {
public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(cv::Mat frame) override;
  std::string get_feature_name() override;
private:
  std::string name{"Lane Departure Warning System"};
  static cv::Mat apply_canny_edge_detection(const cv::Mat& frame);
  static cv::Mat apply_region_of_interest(const cv::Mat& edges);
  static std::vector<cv::Vec4i> get_hough_probabilistic_lines(const cv::Mat& frame);
  static void draw_lines(const std::vector<cv::Vec4i> &lines, cv::Mat& original_frame);
  static LaneState  analyze_lane(const std::vector<cv::Vec4i> &segments, int width, int height);
  static void draw_overlay(const LaneState& state, cv::Mat& frame);
};

#endif // LANE_DEPARTURE_H
