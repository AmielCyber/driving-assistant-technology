#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
#include "../ADASFeature.h"
#include "../AsyncLogger.h"
#include "LaneState.h"
#include "LaneStateLogger.h"

class LaneDeparture : public ADASFeature {
public:
  /**
   * Draws lines on the highway road.
   * @param frame Copy frame to draw lane departure annotations.
   * @return Returns a frame with lane departure annotations.
   */
  cv::Mat process(cv::Mat &frame) override;
  std::string get_feature_name() override;
  explicit LaneDeparture(bool log_data = false);

private:
  std::optional<LaneStateLogger> lane_state_logger;
  LaneState lane_state;
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

  /**
   * Applies yellow-orange lane color threshold to an image and returns the updated image.
   * @param hls_frame An HLS image to extract yellow and orange lane colors
   * @return The extracted yellow and orange lane colors image in binary.
   */
  static cv::Mat apply_yellow_orange_lane_mask(const cv::Mat &hls_frame);
  /**
   * Applies white lane color threshold to an image and returns the updated image.
   * @param hls_frame An HLS image to extract white lane colors
   * @return The extracted white lane colors image in binary.
   */
  static cv::Mat apply_white_lane_mask(const cv::Mat &hls_frame);
  /**
   * Performs grayscale to an image and then blurs, and finally applies Canny Edge Detection.
   * This program expects an image with 3 channels.
   * @param frame The image to perform Canny Edge Detection
   * @return An image with Canny Edge Detection
   */
  static cv::Mat apply_canny_edge_detection(const cv::Mat &frame);
  /**
   * Applies region of interest to an image. The ROI is shaped into a trapezoid to approximate the road in the image.
   *  Based on Optimal Parameters from Assignment 4 Problem 2 for finding lanes.
   * @param canny_frame Canny Edge Frame image to select the Region of Interest
   * @return A binary image with the same size as the original but containing only values inside the ROI and everything
   * else zero out.
   */
  [[nodiscard]] cv::Mat apply_region_of_interest(const cv::Mat &canny_frame) const;
  /**
   * Gets a trapezoid shape to be used for region of interest. The size of the trapezoid will depend on the arguments
   * passed to create a trapezoid based on the ratio of the original frame size.
   * @param frame_height The total height of the frame.
   * @param frame_width The total width of the frame.
   * @return The Trapezoid shape Points.
   */
  [[nodiscard]] std::vector<cv::Point> get_trapezoid_roi(int frame_height, int frame_width) const;
  /**
   * Performs Hough Line Probabilistic on an image.
   * @param roi_frame Perform hough lines probabilistic method on an image
   * @return A list of all lines found from the Hough Line Probabilistic Method
   */
  static std::vector<cv::Vec4i> get_probabilistic_hough_lines(const cv::Mat &roi_frame);
  /**
   * Analyze the lines from Hough Lines to reduce to left and right lanes based on all lane candidates and perform
   * analyzes to determine the lane departure status.
   *  Based on Optimal Parameters from Assignment 4 Problem 2 for finding lanes.
   * @param lines Hough lines list
   * @param width The total width of the image containing the lines
   * @param height The total height of the image containing the lines
   * @return The LaneState containing left and right locations, along with departure status.
   */
  [[nodiscard]] LaneState analyze_lane(const std::vector<cv::Vec4i> &lines, int width, int height);
  /**
   * Separate lines into the left lane candidates and right lane candidates.
   * @param lines Hough Lines to separate left and right lane candidates.
   * @param car_center_x The center point where to seperate the lines.
   * @return A pair of two list of lines containing the left and right lane candidates in that order.
   */
  static std::pair<std::vector<cv::Vec4i>, std::vector<cv::Vec4i>> separate_lanes(const std::vector<cv::Vec4i> &lines,
                                                                                  int car_center_x);
  /**
   * Analyze all lines that are lane candidates and chooses the best fitting line to be the best lane candidate by
   * performing sum of slopes. Bounds the line with bounds passed in the arguments.
   *
   * Using the following python reference to calculate the sum of slopes:
   * Nair, Balu, Summit Ranjan, S. Senthamilarasu, "Finding Road Markings Using OpenCV." Applied Deep Learning and
   Computer Vision for Self-Driving Cars. Packt Publishing, 2020.  O’Reilly Learning Platform,
   https://learning.oreilly.com/library/view/applied-deep-learning/9781838646301/e5a39ce5-e62a-44a1-827e-6da9ab20ff70.xhtml.
   *
   * @param lines The lines that are candidates for a lane.
   * @param y_bottom_limit Bounds the bottom line point to this value.
   * @param y_top_limit Bounds the top line point to this value.
   * @param car_center_x To calculate distance from center and get the closest lane.
   * @param width The maximum width with an applied ratio in this function to draw our lines.
   * @return A point representing the line that will be our lane if found any.
   */
  [[nodiscard]] std::optional<cv::Vec4i> calculate_closest_lane(const std::vector<cv::Vec4i> &lines, int y_bottom_limit,
                                                                int y_top_limit, int car_center_x, int width) const;
  /**
   * Evaluate departure status on a lane state, when lane state contains the left and right lines that are on a lane, if
   * any.
   * @param state The lane state to update the departure status
   */
  void evaluate_departure_status(LaneState &state) const;
  /**
   * Draws all annotations depending on the LaneState values for lane positions and departure status.
   * @param state The lane state containing lane positions and lane departure status
   * @param frame The destination frame to add the annotations to.
   */
  void draw_overlay(const LaneState &state, cv::Mat &frame);
  /**
   * Draws Alert triangle to indicate lane crossing.
   *
   * ChatGPT GENERATED CODE: in C++ opencv 4 how I can draw a small triangle with the character '!' in red in the
   * middle of frame
   *
   * @param frame The destination frame to draw the alert on.
   */
  static void draw_alert_triangle(cv::Mat &frame);
  /**
   * Shades a lane based on the LaneState if it contains both left and right lanes.
   *
   * ChatGPT GENERATED CODE with LineDeparture.h, LaneDeparture.cpp, LaneState.h as uploads, before this commit with
   * prompt: "Implement a shade_lane_region function that will shade the two lines from the lane state if it has two
   * lanes or neither of status are DepartureStatus::Alert. Make the color shading green if both status are
   * DepartureStatus:::SAFE, color orange if either one status is DepartureStatus::WARNING"
   *
   * @param state The lane state to determine what color to shade the road
   * @param frame The destination frame to draw the shaded region.
   */
  void shade_lane_region(const LaneState &state, cv::Mat &frame);
};

#endif // LANE_DEPARTURE_H
