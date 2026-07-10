// yolo11_video_demo.cpp
// Created by Carlos G Cazares
// Usage:
//     ./yolo11_video_demo_threaded video.mp4
//
// Files expected in current directory:
//     yolo11n.onnx
//     classes.names      (or your own classes file)
//
// Compile:
// g++ yolo11_video_demo_threaded.cpp -o yolo11_demo_threaded `pkg-config
// --cflags --libs opencv4`

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

using namespace cv;
using namespace cv::dnn;
using namespace std;

static const float SCORE_THRESHOLD = 0.25f;
static const float CONFIDENCE_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;

static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;
static const int OUTPUT_WIDTH = 640;
static const int OUTPUT_HEIGHT = 480;

// Structure to get frames and count them
struct FrameItem {
  int frameNumber;
  Mat frame;
};

// Creating a queue so threads can work on frames coordinatingly
template <typename T> class SafeQueue {
public:
  SafeQueue(size_t maxSize = 8) : maxSize(maxSize) {} // Constructor

  void push(T item) { // Function to store the frames
    unique_lock<mutex> lock(mtx);

    notFull.wait(lock, [&] { return q.size() < maxSize || finished; });

    if (finished)
      return;

    q.push(std::move(item));
    notEmpty.notify_one();
  }

  bool pop(T &item) { // Function to take out the frame
    unique_lock<mutex> lock(mtx);

    notEmpty.wait(lock, [&] { return !q.empty() || finished; });

    if (q.empty())
      return false;

    item = std::move(q.front());
    q.pop();

    notFull.notify_one();
    return true;
  }

  void setFinished() { // Stablish where the queue ends.
    unique_lock<mutex> lock(mtx);
    finished = true;
    notEmpty.notify_all();
    notFull.notify_all();
  }

  size_t size() // Provide the current size of the queue
  {
    lock_guard<mutex> lock(mtx);
    return q.size();
  }

private:                       // Define class attributes
  queue<T> q;                  // The queue
  mutex mtx;                   // to ensure only one thread use the resource.
  condition_variable notEmpty; // if the queue is empty
  condition_variable notFull;  // if the queue is not full
  bool finished = false;       // If the queue is completed
  size_t maxSize;              // Maximum size
};

vector<string>
loadClasses(const string &filename) { // Structure to store the model classes
  vector<string>
      classes; // Vector to store the classes and use it for the detection
  ifstream ifs(filename); // File from where the classes are stored

  string line;
  while (getline(ifs, line)) // Read class by line
    classes.push_back(line); // enter the class in the vector

  return classes; // Return a vector with all the classes.
}

// Read the video as a independent thread
void readerThread(VideoCapture &cap, SafeQueue<FrameItem> &rawQueue) {
  Mat frame;
  int frameNumber = 0; // Count the read frames

  while (cap.read(frame)) {
    Mat smallFrame; // To convert the frame to optimal size Yolo frame
    resize(frame, smallFrame,
           Size(640, 480)); // Keep constant size frame optimal for Yolo
    // Read frame is pushed in the threaded queue so it can be available for
    // multi-threads
    rawQueue.push({frameNumber, smallFrame});
    frameNumber++; // Increase the number of read frames
  }

  // Set the queue as complete. To control the queue so it would stop a the
  // right time.
  rawQueue.setFinished();
}

// This si the detection thread.
void yoloThread(
    SafeQueue<FrameItem>
        &rawQueue, // Using the queue to pull frames when they are available
    SafeQueue<FrameItem>
        &processedQueue, // The queue with the frames that have proceesed.
    Net &net,            // The model handler
    const vector<string> &classNames, // The list of the classes for the model
    double videoFPS, // VAriable to calculate the processing rate
    int totalFrames  // Number of total expected frames
) {                  // Prepare the file to record each detected object info.
  ofstream logFile("detections.csv");
  // write the detected objects using these fields. They are separated by comas
  logFile << "frame,time_sec,class,confidence," << "center_x,center_y,"
          << "left,top,width,height\n";

  FrameItem item; // item is the current frame

  int64 prevTick =
      cv::getTickCount(); // Get the time stamp the frame was readed.

  while (rawQueue.pop(item)) { // Keep the loop until there are available frames
    Mat frame = item.frame;    // Frame gets the read frame
    int frameNumber = item.frameNumber; // framenumber is the id in the record

    int img_w = frame.cols; // Frame width
    int img_h = frame.rows; // frame height

    Mat blob;
    blobFromImage( // convert the read frame in the format the Yolo model needs
                   // 1 × 3 × 640 × 640
        frame, blob, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(),
        true, // color channels switch from B G R to R G B
        false // image is just resize it, it is not cropped
    );

    net.setInput(blob); // passes the converted frame to the model

    // taking the raw output of the model and converting it into a 2D matrix for
    // processing.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    Mat output = outputs[0]; // output contains the detection tensor

    // read the dimensions
    if (output.dims == 3) {
      int rows = output.size[1];
      int cols = output.size[2];

      // Convert to a 2D matrix
      output = output.reshape(1, rows);

      // Make sure we have the correct order.
      if (rows < cols)
        transpose(output, output);
    }

    if (frameNumber % 100 == 0) // since the original video is not showed, this
                                // displayed progess info
    {
      // Calculate the processing speed in frames per second
      double percent = 100.0 * frameNumber / totalFrames;

      // Provide the update
      cout << "\rProcessing frame " << frameNumber << " / " << totalFrames
           << " (" << fixed << setprecision(1) << percent << "%)             "
           << flush;
    }
    // Display how the queue has been moving along
    if (frameNumber % 270 == 0) {
      cout << "\rFrame " << frameNumber << "  Raw Queue: " << rawQueue.size()
           << "  Processed Queue: " << processedQueue.size() << flush;
    }
    // Variables to prepare the boxes nad labels
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    // Calculate object's lcoation
    // Dimensions of the frame
    float xFactor = static_cast<float>(img_w) / INPUT_WIDTH;
    float yFactor = static_cast<float>(img_h) / INPUT_HEIGHT;

    // for each of the detected objects
    for (int i = 0; i < output.rows; i++) {
      float *data = output.ptr<float>(i);

      // Calculate the location of the detected object
      float cx = data[0];
      float cy = data[1];
      float w = data[2];
      float h = data[3];

      // Show the label with the detected object class
      Mat scores(1, output.cols - 4, CV_32FC1, data + 4);

      Point classIdPoint;
      double score;

      // Make sure the labels are displayed in the frame
      minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

      // If the score is less than the threshold, the object is skipped
      if (score < SCORE_THRESHOLD)
        continue;

      // Prepare the box of the detected object
      float left = (cx - 0.5f * w) * xFactor;
      float top = (cy - 0.5f * h) * yFactor;
      float width = w * xFactor;
      float height = h * yFactor;

      // Draw the box
      boxes.emplace_back(Rect(static_cast<int>(left), static_cast<int>(top),
                              static_cast<int>(width),
                              static_cast<int>(height)));

      // store the class and the detection confidence
      classIds.push_back(classIdPoint.x);
      confidences.push_back(static_cast<float>(score));
    }

    // To make sure boxes are not overlapping
    vector<int> indices;

    NMSBoxes( // Do not overlap boxes, try to keep only the highest confident
              // box
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    // For detected objects in the frame
    for (int idx : indices) { // prepare the boxes
      Rect box = boxes[idx];

      double timeSec = frameNumber / videoFPS;

      float centerX = box.x + box.width * 0.5f;
      float centerY = box.y + box.height * 0.5f;

      string className; // display the class name instead of the class id

      if (classIds[idx] >= 0 &&
          static_cast<size_t>(classIds[idx]) < classNames.size()) {
        className = classNames[classIds[idx]];
      } else {
        className = to_string(classIds[idx]);
      }

      // Pass the info for the log
      logFile << frameNumber << "," << fixed << setprecision(3) << timeSec
              << "," << className << "," << confidences[idx] << "," << centerX
              << "," << centerY << "," << box.x << "," << box.y << ","
              << box.width << "," << box.height << "\n";

      // draw the box in the frame
      rectangle(frame, box, Scalar(0, 255, 0), 2);
      // Tag the box with the class and confidence rate
      string label = className + cv::format(" %.2f", confidences[idx]);

      int baseline = 0;

      // Set the font type size and other attributes
      Size tsize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

      int y = max(box.y, tsize.height);

      // label background
      rectangle(frame, Point(box.x, y - tsize.height - 4),
                Point(box.x + tsize.width, y + baseline), Scalar(0, 255, 0),
                FILLED);
      // label text infront of the background
      putText(frame, label, Point(box.x, y - 2), FONT_HERSHEY_SIMPLEX, 0.5,
              Scalar(0, 0, 0), 1);
    }

    // Get the time stamp at the end of the process
    int64 currentTick = cv::getTickCount();
    // Calculate the elapsed time
    double elapsed = (currentTick - prevTick) / cv::getTickFrequency();
    prevTick = currentTick;

    // Calculate the processing speed in frames per second
    double fps = 1.0 / elapsed;
    // Display the processing speed
    string fpsText = cv::format("FPS: %.2f", fps);

    int baseline = 0;

    // Prepare the box where the processing speed would be printed
    Size fpsSize =
        getTextSize(fpsText, FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    // Print the processing speed
    Point fpsOrigin(frame.cols - fpsSize.width - 10, 30);

    rectangle(
        frame, Point(fpsOrigin.x - 5, fpsOrigin.y - fpsSize.height - 5),
        Point(fpsOrigin.x + fpsSize.width + 5, fpsOrigin.y + baseline + 5),
        Scalar(0, 0, 0), FILLED);

    putText(frame, fpsText, fpsOrigin, FONT_HERSHEY_SIMPLEX, 0.7,
            Scalar(255, 255, 255), 2);

    // create a new frame that would be used in the output video
    Mat outFrame;
    resize(frame, outFrame, Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

    // Read the next frame from the queue
    processedQueue.push({frameNumber, outFrame});
  }
  // Close the log file and reset the processingqueue
  logFile.close();
  processedQueue.setFinished();
}

// Writer in a different thread to record the output video at the same time
// other frame is prossed
void writerThread(SafeQueue<FrameItem> &processedQueue, VideoWriter &writer) {
  FrameItem item;

  while (processedQueue.pop(item)) { // Get the processed frame
    writer.write(item.frame); // write the processed frame in the output video
  }
}

int main(int argc, char **argv) {
  // Timestamp of the begining of the program
  auto start = chrono::steady_clock::now();
  // If a video was not provided, indicate what needs to be provided
  if (argc != 2) {
    cout << "Usage:\n";
    cout << argv[0] << " video_file\n";
    return -1;
  }
  // Read the file classes.names with the name of the classes
  vector<string> classNames =
      loadClasses("data/Object-Detection/classes.names");
  // Indicate the program is starting
  cout << "Loading YOLO model..." << endl;
  Net net = readNet("data/Object-Detection/best.onnx");
  cout << "Model loaded successfully." << endl;
  // tell YOLO what are the preference for execution
  net.setPreferableBackend(DNN_BACKEND_OPENCV);
  net.setPreferableTarget(DNN_TARGET_CPU);
  // Show the video file was open
  cout << "Opening video: " << argv[1] << endl;

  VideoCapture cap(argv[1]); // Read the video file

  if (!cap.isOpened()) // if the video file cannot be open
  {
    cerr << "Cannot open video." << endl;
    return -1; // Terminate the program
  }

  cout << "Video opened successfully." << endl;

  double videoFPS =
      cap.get(CAP_PROP_FPS); // Video file expected speed in frames
  int totalFrames = static_cast<int>(
      cap.get(CAP_PROP_FRAME_COUNT)); // expected total number of frames in the
                                      // video file
  int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));   // video size x
  int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)); // video size y

  // Provide general info about the video before analysis
  cout << "Resolution : " << width << " x " << height << endl;

  cout << "FPS        : " << videoFPS << endl;

  cout << "Frames     : " << totalFrames << endl;

  // if video speed was not provided, default is 30
  if (videoFPS <= 0.0)
    videoFPS = 30.0;

  // Prepare a the output file
  VideoWriter writer("object_detection_demom.mp4",
                     VideoWriter::fourcc('m', 'p', '4', 'v'), videoFPS,
                     Size(OUTPUT_WIDTH, OUTPUT_HEIGHT), true);

  // If the output file cannot be opened, inform and terminate the program
  if (!writer.isOpened()) {
    cerr << "Cannot open writer.\n";
    return -1;
  }

  // start the queues
  SafeQueue<FrameItem> rawQueue(4);
  SafeQueue<FrameItem> processedQueue(4);
  // Define the first thread "read"
  thread reader(readerThread, ref(cap), ref(rawQueue));
  // Define the second thread "process/analysis"
  thread yolo(yoloThread, ref(rawQueue), ref(processedQueue), ref(net),
              cref(classNames), videoFPS, totalFrames);
  // Define the last thread "write"
  thread writerWorker(writerThread, ref(processedQueue), ref(writer));
  // provide model general info.
  cout << "----------------------------------------" << endl;
  cout << "YOLO11 Video Detection" << endl;
  cout << "----------------------------------------" << endl;
  cout << "Input video : " << argv[1] << endl;
  cout << "Resolution  : " << cap.get(CAP_PROP_FRAME_WIDTH) << " x "
       << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
  cout << "FPS         : " << cap.get(CAP_PROP_FPS) << endl;
  cout << "Frames      : " << totalFrames << endl;
  cout << "Output      : object_detection_demo.mp4" << endl;
  cout << "Log file    : detections.csv" << endl;
  cout << "----------------------------------------" << endl;
  // provide progress updates
  cout << "Starting processing..." << endl;

  cout << "[Reader] Started.\n" << endl;
  reader.join();
  cout << "\n[Reader] Finished." << endl;

  cout << "[YOLO] Started." << endl;
  yolo.join();
  cout << "[YOLO] Finished." << endl;

  cout << "[Writer] Started." << endl;
  writerWorker.join();
  cout << "[Writer] Finished." << endl;
  // When the analysis is done take a time stamp
  auto end = chrono::steady_clock::now();
  // Calculate teh duration of the analysis
  double seconds = chrono::duration<double>(end - start).count();
  // Show the summary of the process
  cout << endl;
  cout << "--------------------------------------" << endl;
  cout << "Processing completed." << endl;
  cout << "Frames processed : " << totalFrames << endl;
  cout << "Elapsed time     : " << fixed << setprecision(2) << seconds
       << " seconds" << endl;

  cout << "Average FPS      : " << totalFrames / seconds << endl;
  // Release resources
  writer.release();
  cap.release();
  // Terminate the program
  return 0;
}
