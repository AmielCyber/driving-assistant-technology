#include "LaneDeparture.h"

#include <opencv2/imgproc.hpp>
cv::Mat LaneDeparture::process(cv::Mat frame) {
  cv::putText(frame, "Lane Departure Warning System", cv::Point(10, 30),
              cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
  return frame;
}
std::string LaneDeparture::get_feature_name() {
  return this->name;
}