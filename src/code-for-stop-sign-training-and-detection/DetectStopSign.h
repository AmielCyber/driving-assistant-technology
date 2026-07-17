#ifndef DETECT_STOP_SIGN_H
#define DETECT_STOP_SIGN_H

#include "../ADASFeature.h"

#include <opencv2/objdetect.hpp>

#include <string>
#include <vector>

#include <opencv2/core.hpp>

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
  cv::Mat process(cv::Mat& frame) override;

  /**
   * Return the display name of this feature.
   */
  std::string get_feature_name() override;

private:

  double roi_height_fraction{0.30};
  double minimum_red_pixel_ratio{0.05};
  double minimum_white_pixel_ratio{0.01};
  bool require_octagon_shape{true};

  struct StopSignDetection {
    cv::Rect rectangle;
    double red_pixel_ratio;
    double white_pixel_ratio;
    int vertex_count;
    bool octagon_like;
    };

  /**
   * Return a full-width ROI covering the bottom portion of the frame.
   */
  static cv::Rect getBottomVerticalRegionOfInterest(
      const cv::Size &frameSize,
      double retainedHeightFraction);

  /**
   * Return a full-width ROI centered vertically within the frame.
   */
  static cv::Rect getMiddleVerticalRegionOfInterest(
      const cv::Size &frameSize,
      double retainedHeightFraction);

  /**
   * Create a binary mask containing pixels classified as red.
   */
  static cv::UMat createRedColorMask(
      const cv::UMat &bgrImage);

  /**
   * Create a binary mask containing pixels classified as white.
   */
  static cv::UMat createWhiteColorMask(
      const cv::UMat &bgrImage);

  /**
   * Calculate the proportion of nonzero pixels inside
   * a detection rectangle.
   */
  static double calculateMaskPixelRatio(
      const cv::UMat &redOrWhiteMask,
      const cv::Rect &detectionRectangle);

  /**
   * Determine whether the largest red region inside a HOG
   * detection resembles a stop-sign shape.
   */
  static bool isOctagonLikeRedRegion(
      const cv::UMat &redMask,
      const cv::Rect &detectionRectangle,
      int &detectedVertexCount);

  /**
   * Draw rectangles and labels for detected stop signs.
   */
  static void draw_detections(
    cv::Mat &frame,
    const std::vector<StopSignDetection> &detections);

  std::string name{"HOG Stop Sign Detection"};

  cv::HOGDescriptor hog;

  int win_stride_width{16};
  int win_stride_height{16};
  int group_threshold{8};
  int number_of_levels{13};

  double hit_threshold{-0.75};
  double pyramid_scale{1.05};
};

#endif // DETECT_STOP_SIGN_H