#ifndef LANE_DEPARTURE_H
#define LANE_DEPARTURE_H
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
};

#endif // LANE_DEPARTURE_H
