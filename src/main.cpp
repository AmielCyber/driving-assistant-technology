#include "ADASFeature.h"
#include "LaneDeparture.h"
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>

struct AppConfig {
  // Will change type to the interface
  std::vector<std::shared_ptr<ADASFeature>> features{};
  std::optional<std::string> store_filename;
  std::string video_source{};
  bool show_help{false};
};

std::optional<AppConfig> parse_arguments(int argc, char **argv);

int main(int argc, char **argv) {
  // Parse User Arguments if any.
  auto config = parse_arguments(argc, argv);
  if (!config) {
    return EXIT_FAILURE;
  }
  if (config->show_help) {
    return EXIT_SUCCESS;
  }

  // Set up video capture from video
  cv::VideoCapture cap{};
  cap.open(config->video_source);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video from source\n";
    return EXIT_FAILURE;
  }

  // Set up video writer for storing results onto a video
  cv::VideoWriter writer;
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps < 1.0 || fps > 60.0)
    fps = 30.0; // Cap at 30 fps

  constexpr int ESCAPE_KEY = 27;
  cv::Mat frame, dst;
  for (int input_key = 0; input_key != ESCAPE_KEY && cap.read(frame);
       input_key = cv::waitKey(1)) {
    // Exit if no frame captured
    if (frame.empty())
      break;
    // Display result/s
    for (auto &feature : config->features) {
      cv::Mat feature_frame = feature->process(frame);
      cv::imshow(feature->get_feature_name(), feature_frame);
    }
    // Save output to video filename if set
    if (config->store_filename.has_value()) {
      if (!writer.isOpened()) {
        writer.open(config->store_filename.value(),
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                    frame.size());
        if (!writer.isOpened()) {
          // Check again if writer failed to open
          std::cerr << "Failed to open output video stream\n";
          return EXIT_FAILURE;
        }
      }
      /************************** Merge features HERE *************************/
      // Needs to change to write the merged features.
      writer.write(frame);
    }
  }
  // Manual Clean Up, not really needed with RAII
  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}

std::optional<AppConfig> parse_arguments(const int argc, char **argv) {
  const std::string keys = {
      "{help h | | Print this message.}"
      "{video v | | Video file name}"
      "{show s | | Features to show separated by commas. Example: "
      "--show=stops,lanes,objects}"
      "{store o | | Store the results back in a video file name.}"};

  const cv::CommandLineParser parser{argc, argv, keys};
  if (!parser.check()) {
    parser.printErrors();
    return std::nullopt;
  }

  AppConfig config{};
  // --help
  if (parser.has("help")) {
    parser.printMessage();
    config.show_help = true;
    return config;
  }
  // --video=
  if (const auto video_file = parser.get<std::string>("video");
      !video_file.empty()) {
    config.video_source = video_file;
    std::cout << "Using video file: " << video_file << '\n';
  } else {
    std::cerr << "You must specify a video file.\nSee --help for more info.";
    return std::nullopt;
  }
  // --show=
  if (const auto features = parser.get<std::string>("show");
      !features.empty()) {
    std::stringstream ss(features);
    std::string feature;
    while (std::getline(ss, feature, ',')) {
      if (feature == "lanes") {
        config.features.push_back(std::make_shared<LaneDeparture>());
      } else if (feature == "objects") {
        /************* Object Detection Implementation HERE ******************/
        std::cout << "Object Detection feature not implemented\n";
      } else if (feature == "stops") {
        /************* Stop Sign Detection Implementation HERE ****************/
        std::cout << "Stop Detection feature not implemented\n";
      }
    }
  }
  // --store=
  if (auto out_video = parser.get<std::string>("store"); !out_video.empty()) {
    config.store_filename = out_video;
    std::cout << "Saving results to video filename: " << out_video << '\n';
  }
  return config;
}
