#include "DetectStopSign.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <stdexcept>

DetectStopSign::DetectStopSign(
    const std::string &detector_path) {

  if (!hog.load(detector_path)) {
    throw std::runtime_error(
        "Could not load HOG stop-sign detector: " + detector_path);
  }

  hog.nlevels = number_of_levels;
}

cv::Mat DetectStopSign::process(cv::Mat& frame) {
  if (frame.empty()) {
    return frame;
  }

  const cv::Size window_stride(
      win_stride_width,
      win_stride_height);

  std::vector<cv::Rect> detections;

  hog.detectMultiScale(
      frame,
      detections,
      hit_threshold,
      window_stride,
      cv::Size(0, 0),
      pyramid_scale,
      group_threshold);

  draw_detections(frame, detections);

  return frame;
}

std::string DetectStopSign::get_feature_name() {
  return name;
}

void DetectStopSign::draw_detections(
    cv::Mat &frame,
    const std::vector<cv::Rect> &detections) {

  const cv::Scalar detection_color(0, 255, 0);

  for (const cv::Rect &detection : detections) {
    cv::rectangle(
        frame,
        detection,
        detection_color,
        3);

    const int label_y =
        std::max(detection.y - 5, 20);

    cv::putText(
        frame,
        "STOP SIGN",
        cv::Point(detection.x, label_y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        detection_color,
        2,
        cv::LINE_AA);
  }
}