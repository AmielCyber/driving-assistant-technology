#include "LaneDeparture.h"

#include <opencv2/imgproc.hpp>
cv::Mat LaneDeparture::process(cv::Mat frame) {
  const cv::Mat edges = apply_canny_edge_detection(frame);
  const cv::Mat roi_edges = apply_region_of_interest(edges);
  const std::vector<cv::Vec4i> segments =
      get_hough_probabilistic_lines(roi_edges);
  draw_lines(segments, frame);

  return frame;
}
std::string LaneDeparture::get_feature_name() { return this->name; }

cv::Mat LaneDeparture::apply_canny_edge_detection(const cv::Mat &frame) {
  constexpr double lower_thresh = 10;
  constexpr double upper_thresh = 50;
  cv::Mat gray, blur, edges;

  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0.0);
  cv::Canny(blur, edges, lower_thresh, upper_thresh);
  return edges;
}
cv::Mat LaneDeparture::apply_region_of_interest(const cv::Mat &edges) {
  // NAIVE WAY TO FILTER OUT ROAD
  // JUST DRAW A TRIANGLE from corners of the image up to the center of the
  // image
  cv::Mat mask = cv::Mat::zeros(edges.size(), edges.type());

  int width = edges.cols;
  int height = edges.rows;

  const std::vector<cv::Point> triangle = {
      cv::Point(0, height),             // bottom-left image corner
      cv::Point(width / 2, height / 2), // horizontal center of image
      cv::Point(width, height)          // bottom-right image corner
  };

  cv::fillConvexPoly(mask, triangle, cv::Scalar(255));

  cv::Mat masked_edges;
  cv::bitwise_and(edges, mask, masked_edges);

  return masked_edges;
}

std::vector<cv::Vec4i>
LaneDeparture::get_hough_probabilistic_lines(const cv::Mat &frame) {
  constexpr double theta = CV_PI / 180;
  constexpr double threshold = 60;
  constexpr double min_line_length = 15;
  constexpr double max_line_gap = 20;

  std::vector<cv::Vec4i> segments;
  cv::HoughLinesP(frame, segments, 1, theta, threshold, min_line_length,
                  max_line_gap);
  return segments;
}
void LaneDeparture::draw_lines(const std::vector<cv::Vec4i> &lines,
                               cv::Mat &original_frame) {
  const auto no_warning_lane_color = cv::Scalar(235, 149, 70);
  for (const auto &s : lines) {
    cv::line(original_frame, cv::Point(s[0], s[1]), cv::Point(s[2], s[3]),
             no_warning_lane_color, 2, cv::LINE_AA);
  }
}