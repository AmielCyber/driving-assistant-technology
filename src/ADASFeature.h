#ifndef ADAS_FEATURE_H
#define ADAS_FEATURE_H
#include <opencv2/core/mat.hpp>
#include <string>

/**
 * ADAS Feature that each our feature will implement in order
 * to add feature annotations to the passed frame.
 */
class ADASFeature {
public:
  virtual ~ADASFeature() = default;
  /**
   * Process a frame for ADAS features and returns an annotated frame based
   * on the feature.
   * @param frame The frame to analyze the features
   * @return A frame that will display the features of the passed frame
   */
  virtual cv::Mat process(cv::Mat &frame) = 0;
  /**
   * Returns the name for this feature. Can be used to name a window
   * @return Returns the name for this feature
   */
  virtual std::string get_feature_name() = 0;
};
#endif // ADAS_FEATURE_H
