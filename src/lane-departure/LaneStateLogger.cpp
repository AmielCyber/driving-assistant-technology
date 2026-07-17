#include "LaneStateLogger.h"
#include <iostream>

void LaneStateLogger::log_lane_departure_status(const LaneState &state) {
  const auto detected_left_lane = state.left_lane.has_value();
  const auto detected_right_lane = state.right_lane.has_value();
  const auto left_lane_status = get_string_lane_status(state.left_status);
  const auto right_lane_status = get_string_lane_status(state.right_status);
  std::cout
  << frame_number << ','
  << detected_left_lane << ','
  << detected_right_lane << ','
  << left_lane_status << ','
  << right_lane_status << std::endl;

  ++frame_number;
}
std::string LaneStateLogger::get_string_lane_status(const DepartureStatus &departure_status) {
  switch (departure_status) {
  case DepartureStatus::SAFE:
    return "safe";
  case DepartureStatus::WARNING:
    return "warning";
  case DepartureStatus::ALERT:
    return "alert";
  default:
    return "safe";
  }
}