#include "opencv2/imgcodecs.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace std;
using namespace cv;
class App {
public:
  App(CommandLineParser &cmd);
  void run();
  void handleKey(char key);
  void hogWorkBegin();
  void hogWorkEnd();
  string hogWorkFps() const;
  void workBegin();
  void workEnd();
  string workFps() const;

private:
  App operator=(App &);
  // Args args;
  bool running;
  bool make_gray;
  double scale;
  double resize_scale;
  int win_width;
  int win_stride_width, win_stride_height;
  int gr_threshold;
  int nlevels;
  double hit_threshold;
  bool gamma_corr;
  int64 hog_work_begin;
  double hog_work_fps;
  int64 work_begin;
  double work_fps;
  string img_source;
  string vdo_source;
  int camera_id;
  bool show_debug;
  string store_output;
};
int main(int argc, char **argv) {
  const char *keys =
      "{ h help      |                | print help message }"
      "{ i input     |                | specify input image}"
      "{ c camera    | -1             | enable camera capturing }"
      "{ v video     |        | use video as input }"
      "{ g gray      |                | convert image to gray one or not}"
      "{ s scale     | 1.0            | resize the image before detect}"
      "{ store       |                | store annotated output video filename }"
      "{ show        | false          | show imshow debug windows }";
  CommandLineParser cmd(argc, argv, keys);
  if (cmd.has("help")) {
    cmd.printMessage();
    return EXIT_SUCCESS;
  }
  App app(cmd);
  try {
    app.run();
  } catch (const Exception &e) {
    return cout << "error: " << e.what() << endl, 1;
  } catch (const exception &e) {
    return cout << "error: " << e.what() << endl, 1;
  } catch (...) {
    return cout << "unknown exception" << endl, 1;
  }
  return EXIT_SUCCESS;
}
App::App(CommandLineParser &cmd) {
  cout << "\nControls:\n"
       << "\tESC - exit\n"
       << "\tm - change mode GPU <-> CPU\n"
       << "\tg - convert image to gray or not\n"
       << "\t--store filename - save annotated video\n"
       << "\t1/q - increase/decrease HOG scale\n"
       << "\t2/w - increase/decrease levels count\n"
       << "\t3/e - increase/decrease HOG group threshold\n"
       << "\t4/r - increase/decrease hit threshold\n"
       << endl;
  make_gray = cmd.has("gray");
  resize_scale = cmd.get<double>("s");
  vdo_source = cmd.get<string>("v");
  img_source = cmd.get<string>("i");
  camera_id = cmd.get<int>("c");
  win_width = 64;
  win_stride_width = 16;
  win_stride_height = 16;
  gr_threshold = 8;
  nlevels = 13;
  hit_threshold = -0.75;
  scale = 1.10;
  gamma_corr = true;
  cout << "Group threshold: " << gr_threshold << endl;
  cout << "Levels number: " << nlevels << endl;
  cout << "Win width: " << win_width << endl;
  cout << "Win stride: (" << win_stride_width << ", " << win_stride_height
       << ")\n";
  cout << "Hit threshold: " << hit_threshold << endl;
  cout << "Gamma correction: " << gamma_corr << endl;
  cout << endl;
  show_debug = cmd.has("show");
  store_output = cmd.get<string>("store");
}
void App::run() {
  running = true;
  VideoWriter video_writer;
  // remove bc its for the pedestrian detector
  // Size win_size(win_width, win_width * 2);
  Size win_stride(win_stride_width, win_stride_height);
  // Create HOG descriptors and detectors here
  HOGDescriptor hog;

  // Replace the pedestrian detector yaml file with our yaml file here
  if (!hog.load("data/stop-sign/stop_sign_hog_detector.yml")) {
    throw runtime_error("Could not load stop_sign_hog_detector.yml");
  }
  while (running) {
    VideoCapture vc;
    UMat frame;
    if (vdo_source != "") {
      vc.open(vdo_source.c_str());
      if (!vc.isOpened())
        throw runtime_error(string("can't open video file: " + vdo_source));
      vc >> frame;
    } else if (camera_id != -1) {
      vc.open(camera_id);
      if (!vc.isOpened()) {
        stringstream msg;
        msg << "can't open camera: " << camera_id;
        throw runtime_error(msg.str());
      }
      vc >> frame;
    } else {
      imread(img_source).copyTo(frame);
      if (frame.empty())
        throw runtime_error(string("can't open image file: " + img_source));
    }
    UMat img_aux, img, img_to_show;
    // Iterate over all frames
    while (running && !frame.empty()) {
      workBegin();
      // Change format of the image
      if (make_gray)
        cvtColor(frame, img_aux, COLOR_BGR2GRAY);
      else
        frame.copyTo(img_aux);
      img = img_aux;
      img.copyTo(img_to_show);
      // Resize image
      /*
      if (abs(scale-1.0)>0.001)
      {
          Size sz((int)((double)img_aux.cols/resize_scale),
      (int)((double)img_aux.rows/resize_scale)); resize(img_aux, img, sz, 0, 0,
      INTER_LINEAR_EXACT);
      }
      else img = img_aux;
      */
      // img_aux.copyTo(img_to_show);
      hog.nlevels = nlevels;
      vector<Rect> found;
      // Perform HOG classification
      hogWorkBegin();
      hog.detectMultiScale(img, found, hit_threshold, win_stride, Size(0, 0),
                           scale, gr_threshold);

      cout << "Detections found: " << found.size() << endl;
      hogWorkEnd();
      // Draw positive classified windows
      for (size_t i = 0; i < found.size(); i++) {
        rectangle(img_to_show, found[i], Scalar(0, 255, 0), 3);

        putText(img_to_show, "STOP SIGN",
                Point(found[i].x, max(found[i].y - 5, 0)), FONT_HERSHEY_SIMPLEX,
                0.6, Scalar(0, 255, 0), 2);
      }
      putText(img_to_show, ocl::useOpenCL() ? "Mode: OpenCL" : "Mode: CPU",
              Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
      // putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65),
      // FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
      putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105),
              FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
      imshow("opencv_hog", img_to_show);
      // imshow("64x64 HOG Input", img);
      imshow("Original Image", img_to_show);
      if (vdo_source != "" || camera_id != -1)
        vc >> frame;
      workEnd();
      if (store_output != "") {
        if (!video_writer.isOpened()) {
          double input_fps = 24.0;

          if (vdo_source != "") {
            double cap_fps = vc.get(CAP_PROP_FPS);
            if (cap_fps > 0)
              input_fps = cap_fps;
          }

          video_writer.open(store_output,
                            VideoWriter::fourcc('m', 'p', '4', 'v'), input_fps,
                            img_to_show.size(), true);

          if (!video_writer.isOpened())
            throw runtime_error("Could not create output video: " +
                                store_output);
        }

        Mat output_frame;
        img_to_show.copyTo(output_frame);

        if (output_frame.channels() == 1)
          cvtColor(output_frame, output_frame, COLOR_GRAY2BGR);
        else if (output_frame.channels() == 4)
          cvtColor(output_frame, output_frame, COLOR_BGRA2BGR);

        video_writer.write(output_frame);
      }
      handleKey((char)waitKey(3));
    }
  }
}
void App::handleKey(char key) {
  switch (key) {
  case 27:
    running = false;
    break;
  case 'm':
  case 'M':
    ocl::setUseOpenCL(!cv::ocl::useOpenCL());
    cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU")
         << " mode\n";
    break;
  case 'g':
  case 'G':
    make_gray = !make_gray;
    cout << "Convert image to gray: " << (make_gray ? "YES" : "NO") << endl;
    break;
  case '1':
    scale *= 1.05;
    cout << "Scale: " << scale << endl;
    break;
  case 'q':
  case 'Q':
    scale /= 1.05;
    cout << "Scale: " << scale << endl;
    break;
  case '2':
    nlevels++;
    cout << "Levels number: " << nlevels << endl;
    break;
  case 'w':
  case 'W':
    nlevels = max(nlevels - 1, 1);
    cout << "Levels number: " << nlevels << endl;
    break;
  case '3':
    gr_threshold++;
    cout << "Group threshold: " << gr_threshold << endl;
    break;
  case 'e':
  case 'E':
    gr_threshold = max(0, gr_threshold - 1);
    cout << "Group threshold: " << gr_threshold << endl;
    break;
  case '4':
    hit_threshold += 0.25;
    cout << "Hit threshold: " << hit_threshold << endl;
    break;
  case 'r':
  case 'R':
    hit_threshold = max(0.0, hit_threshold - 0.25);
    cout << "Hit threshold: " << hit_threshold << endl;
    break;
  case 'c':
  case 'C':
    gamma_corr = !gamma_corr;
    cout << "Gamma correction: " << gamma_corr << endl;
    break;
  }
}
inline void App::hogWorkBegin() { hog_work_begin = getTickCount(); }
inline void App::hogWorkEnd() {
  int64 delta = getTickCount() - hog_work_begin;
  double freq = getTickFrequency();
  hog_work_fps = freq / delta;
}
inline string App::hogWorkFps() const {
  stringstream ss;
  ss << hog_work_fps;
  return ss.str();
}
inline void App::workBegin() { work_begin = getTickCount(); }
inline void App::workEnd() {
  int64 delta = getTickCount() - work_begin;
  double freq = getTickFrequency();
  work_fps = freq / delta;
}
inline string App::workFps() const {
  stringstream ss;
  ss << work_fps;
  return ss.str();
}
