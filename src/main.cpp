#include "./Object-Detection/YOLOVideoDetector.h"
#include "./code-for-stop-sign-training-and-detection/DetectStopSign.h"
#include "./lane-departure/LaneDeparture.h"
#include "ADASFeature.h"
#include "ADASFeatureThreadPool.h"
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
  std::vector<std::shared_ptr<ADASFeature>> features{};
  std::optional<std::string> store_filename;
  std::string video_source{};
  bool show_help{false};
};

std::optional<AppConfig> parse_arguments(int argc, char **argv);
void put_fps_text(cv::Mat &src, const steady_clock::time_point &start_time, const steady_clock::time_point &end_time);
cv::Mat run_features_async(const cv::Mat &frame, const std::vector<std::shared_ptr<ADASFeature>> &adas_features,
                           ADASFeatureThreadPool &pool);

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

  const size_t pool_size = config->features.size();
  ADASFeatureThreadPool features_pool{pool_size};

  constexpr int ESCAPE_KEY = 27;
  cv::Mat frame, dst, annotated_frame;
  std::chrono::time_point<steady_clock> start_time;
  std::chrono::time_point<steady_clock> end_time;
  for (int input_key = 1; input_key != ESCAPE_KEY && cap.read(frame); input_key = cv::waitKey(1)) {
    // Exit if no frame captured
    if (frame.empty())
      break;
    start_time = steady_clock::now();
    annotated_frame = run_features_async(frame, config->features, features_pool);
    end_time = steady_clock::now();
    put_fps_text(annotated_frame, start_time, end_time);
    // Save output to video filename if set
    if (config->store_filename.has_value()) {
      if (!writer.isOpened()) {
        writer.open(config->store_filename.value(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frame.size());
        if (!writer.isOpened()) {
          // Check again if writer failed to open
          std::cerr << "Failed to open output video stream\n";
          return EXIT_FAILURE;
        }
      }
      /************************** Merge features HERE *************************/
      // Needs to change to write the merged features.
      writer.write(annotated_frame);
    }
    cv::imshow("ADAS", annotated_frame);
  }
  // Manual Clean Up, not really needed with RAII
  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}

std::optional<AppConfig> parse_arguments(const int argc, char **argv) {
  const std::string keys = {"{help h | | Print this message.}"
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
  if (const auto video_file = parser.get<std::string>("video"); !video_file.empty()) {
    config.video_source = video_file;
    std::cout << "Using video file: " << video_file << '\n';
  } else {
    std::cerr << "You must specify a video file.\nSee --help for more info.";
    return std::nullopt;
  }
  // --show=
  if (const auto features = parser.get<std::string>("show"); !features.empty()) {
    std::stringstream ss(features);
    std::string feature;
    while (std::getline(ss, feature, ',')) {
      if (feature == "lanes") {
        config.features.push_back(std::make_shared<LaneDeparture>());
      } else if (feature == "objects") {
        /************* Object Detection Implementation HERE ******************/
        const std::string model_path = "data/Object-Detection/best.onnx";
        const std::string classes_path = "data/Object-Detection/classes.names";
        config.features.push_back(std::make_shared<YOLOVideoDetector>(model_path, classes_path));
        std::cout << "Object detection enabled\n";
      } else if (feature == "stops") {
        /************* Stop Sign Detection Implementation HERE ****************/
        const std::string detector_path = "src/code-for-stop-sign-training-and-detection/training/"
                                          "stop_sign_hog_detector.yml";

        config.features.push_back(std::make_shared<DetectStopSign>(detector_path));

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
void put_fps_text(cv::Mat &src, const steady_clock::time_point &start_time, const steady_clock::time_point &end_time) {
  const double elapsed = std::chrono::duration<double>(end_time - start_time).count();

  // Avoid division by zero and keep the display within 0.0–999.9.
  const double fps = (elapsed > 0.0) ? std::min(999.9, 1.0 / elapsed) : 999.9;

  std::stringstream speedText;
  speedText << std::fixed << std::setprecision(1) << fps;

  const std::string fpsLabel = "FPS";

  const int radius = 62;
  const cv::Point center(radius + 12, radius + 12);

  // Black circle with thin green perimeter.
  cv::circle(src, center, radius, cv::Scalar(0, 0, 0), cv::FILLED);
  cv::circle(src, center, radius, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

  constexpr double speedScale = 1.15;
  constexpr int speedThickness = 2;
  const cv::Size speedSize =
      cv::getTextSize(speedText.str(), cv::FONT_HERSHEY_SIMPLEX, speedScale, speedThickness, nullptr);

  constexpr double labelScale = 0.55;
  constexpr int labelThickness = 1;
  const cv::Size labelSize = cv::getTextSize(fpsLabel, cv::FONT_HERSHEY_SIMPLEX, labelScale, labelThickness, nullptr);

  constexpr int lineGap = 8;
  const int totalHeight = speedSize.height + lineGap + labelSize.height;
  const int speedY = center.y - totalHeight / 2 + speedSize.height;
  const int labelY = speedY + lineGap + labelSize.height;

  // Speed: white, one decimal place.
  cv::putText(src, speedText.str(), cv::Point(center.x - speedSize.width / 2, speedY), cv::FONT_HERSHEY_SIMPLEX,
              speedScale, cv::Scalar(255, 255, 255), speedThickness, cv::LINE_AA);

  // Label: smaller white text below.
  cv::putText(src, fpsLabel, cv::Point(center.x - labelSize.width / 2, labelY), cv::FONT_HERSHEY_SIMPLEX, labelScale,
              cv::Scalar(255, 255, 255), labelThickness, cv::LINE_AA);
}

cv::Mat run_features_async(const cv::Mat &frame, const std::vector<std::shared_ptr<ADASFeature>> &adas_features,
                           ADASFeatureThreadPool &pool) {
  std::vector<std::future<cv::Mat>> futures;
  futures.reserve(adas_features.size());

  for (auto &feature : adas_features) {
    futures.push_back(pool.submit([feature, &frame]() -> cv::Mat {
      try {
        cv::Mat feature_annotation = frame.clone();
        return feature->process(feature_annotation);
      } catch (const std::exception &e) {
        std::cerr << feature->get_feature_name() << " failed to process: " << e.what() << '\n';
        return frame.clone();  // Return original to not crash other threads
      }
    }));
  }

  // The following code block was generated by the free version of Claude
  // prompt: "in C++ opencv how can I merged three images that contain the same image but with different
  // annotations"
  cv::Mat merged_annotations = frame.clone();
  for (auto &future : futures) {
    cv::Mat feature_result = future.get();

    cv::Mat diff, mask;
    cv::absdiff(feature_result, frame, diff);
    cv::cvtColor(diff, mask, cv::COLOR_BGR2GRAY);
    cv::threshold(mask, mask, 10, 255, cv::THRESH_BINARY);

    // Stamp feature drawing onto final frame without fading background
    feature_result.copyTo(merged_annotations, mask);
  }

  return merged_annotations;
}
