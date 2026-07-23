#ifndef LANE_STATE_LOGGER_H
#define LANE_STATE_LOGGER_H
#include "../AsyncLogger.h"
#include "LaneState.h"

class LaneStateLogger {
  long frame_number{1}; // Use to reference for analysis
  AsyncLogger logger;   // Used to simplified non-blocking writes
public:
  /**
   * Creates the setup for logging lane states and performs a header csv write when initiated.
   * @param filename The file to write the CSV logs.
   * @param header The header to write at the beginning of the CSV log representing the columns.
   */
  LaneStateLogger(const std::string &filename, std::string_view header);
  /**
   * Logs the lane status to the stored CSV file with the following format:
   * left-lane-prediction:boolean, right_lane-prediction:boolean, left-lane-status:enum, right-lane-status:enum
   * @param state The lane state to log.
   */
  void log_lane_departure_status(const LaneState &state);
  static std::string get_string_lane_status(const DepartureStatus &departure_status);
};

#endif // LANE_STATE_LOGGER_H
