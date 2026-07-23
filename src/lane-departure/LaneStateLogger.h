#ifndef LANE_STATE_LOGGER_H
#define LANE_STATE_LOGGER_H
#include "../AsyncLogger.h"
#include "LaneState.h"


class LaneStateLogger {
  long frame_number{1};
  AsyncLogger logger;
  public:
  LaneStateLogger(const std::string& filename, std::string_view header);
  void log_lane_departure_status(const LaneState &state);
    static std::string get_string_lane_status(const DepartureStatus &departure_status);
};

#endif // LANE_STATE_LOGGER_H
