#ifndef DETECT_STOP_SIGN_H
#define DETECT_STOP_SIGN_H

#include "../ADASFeature.h"

#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

/**
 * Detects stop signs using a trained OpenCV HOG detector.
 */
class DetectStopSign : public ADASFeature {
public:
  /**
   * Construct the feature and load the trained HOG detector.
   *
   * @param detector_path Path to the HOG detector YAML file.
   */
  explicit DetectStopSign(const std::string &detector_path);

  /**
   * Detect and annotate stop signs in one frame.
   *
   * @param frame Frame to analyze and annotate.
   * @return Frame containing stop-sign annotations.
   */
  cv::Mat process(cv::Mat frame) override;

  /**
   * Return the display name of this feature.
   */
  std::string get_feature_name() override;

private:
  /**
   * Draw rectangles and labels for detected stop signs.
   */
  static void draw_detections(
      cv::Mat &frame,
      const std::vector<cv::Rect> &detections);

  std::string name{"HOG Stop Sign Detection"};

  cv::HOGDescriptor hog;

  int win_stride_width{16};
  int win_stride_height{16};
  int group_threshold{8};
  int number_of_levels{13};

  double hit_threshold{-0.75};
  double pyramid_scale{1.10};
};

#endif // DETECT_STOP_SIGN_H