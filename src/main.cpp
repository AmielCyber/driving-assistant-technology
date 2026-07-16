#include "./code-for-stop-sign-training-and-detection/DetectStopSign.h"
#include "./Object-Detection/YOLOVideoDetector.h"
#include "./lane-departure/LaneDeparture.h"
#include "ADASFeature.h"
#include <chrono>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using std::chrono::steady_clock;

struct AppConfig {
  // Will change type to the interface
  std::vector<std::shared_ptr<ADASFeature>> features{};
  std::optional<std::string> store_filename;
  std::string video_source{};
  bool show_help{false};
};

std::optional<AppConfig> parse_arguments(int argc, char **argv);
void put_fps_text(cv::Mat &src, const steady_clock::time_point &start_time,
                  const steady_clock::time_point &end_time);

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
  std::chrono::time_point<steady_clock> start_time;
  std::chrono::time_point<steady_clock> end_time;

  // These are required to resize the output video
  const cv::Size videoSize(640, 480);
  cv::Mat encodedFrame;
  cv::Mat resizedFrame;
  encodedFrame.setTo(cv::Scalar::all(0));

  for (int input_key = 1; input_key != ESCAPE_KEY && cap.read(frame);
       input_key = cv::waitKey(1)) {
    // Exit if no frame captured
    if (frame.empty())
      break;
    // Display result/s

    // This is the output frame to be resized
    cv::Mat output_frame = frame.clone();
    
    for (auto &feature : config->features) {
      // Clock start time
      start_time = steady_clock::now();
      output_frame = feature->process(output_frame);
      // Clock end time
      end_time = steady_clock::now();
      put_fps_text(output_frame, start_time, end_time);
      cv::imshow(feature->get_feature_name(), output_frame);
    }
    // Save output to video filename if set
    if (config->store_filename.has_value()) {

      // Resizing
      const double scale = std::min(
            static_cast<double>(videoSize.width) / output_frame.cols,
            static_cast<double>(videoSize.height) / output_frame.rows);

      const cv::Size resizedSize(
          static_cast<int>(output_frame.cols * scale),
          static_cast<int>(output_frame.rows * scale));

      const cv::Point offset(
          (videoSize.width - resizedSize.width) / 2,
          (videoSize.height - resizedSize.height) / 2);

      encodedFrame = cv::Mat(
          videoSize, output_frame.type(), cv::Scalar::all(0));

      cv::resize(output_frame, resizedFrame, resizedSize,
                 0.0, 0.0, cv::INTER_AREA);

      resizedFrame.copyTo(encodedFrame(cv::Rect(offset, resizedSize)));
      
      if (!writer.isOpened()) {
        writer.open(config->store_filename.value(),
                    cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                    videoSize);
        if (!writer.isOpened()) {
          // Check again if writer failed to open
          std::cerr << "Failed to open output video stream\n";
          return EXIT_FAILURE;
        }
      }
      /************************** Merge features HERE *************************/
      // Needs to change to write the merged features.
      writer.write(encodedFrame);
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
        const std::string model_path =
              "data/Object-Detection/best.onnx";
          const std::string classes_path =
              "data/Object-Detection/classes.names";
          config.features.push_back(
              std::make_shared<YOLOVideoDetector>(model_path, classes_path));
          std::cout << "Object detection enabled\n";
      } else if (feature == "stops") {
        /************* Stop Sign Detection Implementation HERE ****************/
        const std::string detector_path =
            "src/code-for-stop-sign-training-and-detection/training/"
            "stop_sign_hog_detector.yml";

        config.features.push_back(
            std::make_shared<DetectStopSign>(detector_path));

        std::cout << "Stop sign detection enabled\n";
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

// Circular version to show processing speed
void put_fps_text(cv::Mat &src,
                  const steady_clock::time_point &start_time,
                  const steady_clock::time_point &end_time)
{
    const double elapsed =
        std::chrono::duration<double>(end_time - start_time).count();

    // Avoid division by zero and keep the display within 0.0–99.9.
    const double fps = (elapsed > 0.0)
                           ? std::min(99.9, 1.0 / elapsed)
                           : 99.9;

    std::stringstream speedText;
    speedText << std::fixed << std::setprecision(1) << fps;

    const std::string fpsLabel = "FPS";

    const int radius = 62;
    const cv::Point center(radius + 12, radius + 12);

    // Black circle with thin green perimeter.
    cv::circle(src, center, radius, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::circle(src, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    const double speedScale = 1.15;
    const int speedThickness = 2;
    const cv::Size speedSize = cv::getTextSize(
        speedText.str(),
        cv::FONT_HERSHEY_SIMPLEX,
        speedScale,
        speedThickness,
        nullptr);

    const double labelScale = 0.55;
    const int labelThickness = 1;
    const cv::Size labelSize = cv::getTextSize(
        fpsLabel,
        cv::FONT_HERSHEY_SIMPLEX,
        labelScale,
        labelThickness,
        nullptr);

    const int lineGap = 8;
    const int totalHeight = speedSize.height + lineGap + labelSize.height;
    const int speedY = center.y - totalHeight / 2 + speedSize.height;
    const int labelY = speedY + lineGap + labelSize.height;

    // Speed: white, one decimal place.
    cv::putText(
        src,
        speedText.str(),
        cv::Point(center.x - speedSize.width / 2, speedY),
        cv::FONT_HERSHEY_SIMPLEX,
        speedScale,
        cv::Scalar(255, 255, 255),
        speedThickness,
        cv::LINE_AA);

    // Label: smaller white text below.
    cv::putText(
        src,
        fpsLabel,
        cv::Point(center.x - labelSize.width / 2, labelY),
        cv::FONT_HERSHEY_SIMPLEX,
        labelScale,
        cv::Scalar(255, 255, 255),
        labelThickness,
        cv::LINE_AA);
}

