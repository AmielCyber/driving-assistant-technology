#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <variant>

struct AppConfig {
  std::string method{"houghp"};
  std::variant<int, std::string> video_source;
};

std::optional<AppConfig> parse_arguments(int argc, char **argv);

int main(int argc, char **argv) {
  // Parse User Arguments if any.
  auto config = parse_arguments(argc, argv);
  if (!config) {
    return EXIT_FAILURE;
  }

  // Set up video capture from video or from camera.
  cv::VideoCapture cap{};
  if (std::holds_alternative<int>(config->video_source)) {
    int camera_id = std::get<int>(config->video_source);
    cap.open(camera_id);
  } else if (std::holds_alternative<std::string>(config->video_source)) {
    auto file_path = std::get<std::string>(config->video_source);
    cap.open(file_path);
  }
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video from source\n";
    return EXIT_FAILURE;
  }

  std::string src_win_name = "Source";
  constexpr int ESCAPE_KEY = 27;
  cv::Mat frame, dst;
  for (int input_key = 0; input_key != ESCAPE_KEY && cap.read(frame);
       input_key = cv::waitKey(1)) {
    if (frame.empty()) {
      break;
    }
    cv::imshow("Source", frame);
  }
  // Manual Clean Up, not really needed with RAII
  cap.release();
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}

std::optional<AppConfig> parse_arguments(const int argc, char **argv) {
  const std::string keys = {
      "{help h | | Print this message.}"
      "{video v |  | Video source file name to perform Hough Lines.}"
      "{camera c | 0 | Camera device ID. Default if no args are passed. Set "
      "with 0 as default built-in "
      "camera.}"};

  const cv::CommandLineParser parser{argc, argv, keys};
  if (parser.has("help") || !parser.check()) {
    parser.printMessage();
    if (!parser.check()) {
      parser.printErrors();
    }
    return std::nullopt;
  }

  AppConfig config;
  int camera_id = parser.get<int>("camera");
  if (auto video_file = parser.get<std::string>("video"); !video_file.empty()) {
    config.video_source = video_file;
    std::cout << "Using video file: " << video_file << '\n';
  } else {
    config.video_source = camera_id;
    std::cout << "Using camera with id " << camera_id << '\n';
  }
  return config;
}
