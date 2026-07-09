// yolo11_video_demo_threaded_ort.cpp
// ONNX Runtime + OpenCV version

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <atomic>
#include <chrono>

using namespace cv;
using namespace std;

static const float SCORE_THRESHOLD = 0.25f;
static const float CONFIDENCE_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;

static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;
static const int OUTPUT_WIDTH = 640;
static const int OUTPUT_HEIGHT = 480;

struct FrameItem {
    int frameNumber;
    Mat frame;
};

template <typename T>
class SafeQueue {
public:
    SafeQueue(size_t maxSize = 8) : maxSize(maxSize) {}

    void push(T item) {
        unique_lock<mutex> lock(mtx);
        notFull.wait(lock, [&] { return q.size() < maxSize || finished; });

        if (finished)
            return;

        q.push(std::move(item));
        notEmpty.notify_one();
    }

    bool pop(T& item) {
        unique_lock<mutex> lock(mtx);
        notEmpty.wait(lock, [&] { return !q.empty() || finished; });

        if (q.empty())
            return false;

        item = std::move(q.front());
        q.pop();

        notFull.notify_one();
        return true;
    }

    void setFinished() {
        unique_lock<mutex> lock(mtx);
        finished = true;
        notEmpty.notify_all();
        notFull.notify_all();
    }

    size_t size() {
        lock_guard<mutex> lock(mtx);
        return q.size();
    }

private:
    queue<T> q;
    mutex mtx;
    condition_variable notEmpty;
    condition_variable notFull;
    bool finished = false;
    size_t maxSize;
};

vector<string> loadClasses(const string& filename) {
    vector<string> classes;
    ifstream ifs(filename);

    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    return classes;
}

vector<float> preprocessImage(const Mat& frame) {
    Mat resized;
    resize(frame, resized, Size(INPUT_WIDTH, INPUT_HEIGHT));

    Mat rgb;
    cvtColor(resized, rgb, COLOR_BGR2RGB);

    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    vector<float> inputTensorValues(1 * 3 * INPUT_HEIGHT * INPUT_WIDTH);

    int channelSize = INPUT_HEIGHT * INPUT_WIDTH;

    for (int y = 0; y < INPUT_HEIGHT; y++) {
        for (int x = 0; x < INPUT_WIDTH; x++) {
            Vec3f pixel = rgb.at<Vec3f>(y, x);

            inputTensorValues[0 * channelSize + y * INPUT_WIDTH + x] = pixel[0];
            inputTensorValues[1 * channelSize + y * INPUT_WIDTH + x] = pixel[1];
            inputTensorValues[2 * channelSize + y * INPUT_WIDTH + x] = pixel[2];
        }
    }

    return inputTensorValues;
}

void readerThread(VideoCapture& cap, SafeQueue<FrameItem>& rawQueue) {
    Mat frame;
    int frameNumber = 0;

    while (cap.read(frame)) {
        Mat smallFrame;
        resize(frame, smallFrame, Size(640, 480));

        rawQueue.push({frameNumber, smallFrame});
        frameNumber++;
    }

    rawQueue.setFinished();
}

void yoloThread(
    SafeQueue<FrameItem>& rawQueue,
    SafeQueue<FrameItem>& processedQueue,
    Ort::Session& session,
    const char* inputName,
    const char* outputName,
    const vector<string>& classNames,
    double videoFPS,
    int totalFrames
) {
    ofstream logFile("detections.csv");

    logFile << "frame,time_sec,class,confidence,"
            << "center_x,center_y,"
            << "left,top,width,height\n";

    Ort::MemoryInfo memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    array<int64_t, 4> inputShape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

    FrameItem item;
    int64 prevTick = cv::getTickCount();

    while (rawQueue.pop(item)) {
        Mat frame = item.frame;
        int frameNumber = item.frameNumber;

        int img_w = frame.cols;
        int img_h = frame.rows;

        vector<float> inputTensorValues = preprocessImage(frame);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size()
        );

        const char* inputNames[] = {inputName};
        const char* outputNames[] = {outputName};

        auto outputTensors = session.Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1
        );

        float* outputData = outputTensors[0].GetTensorMutableData<float>();

        auto outputShape =
            outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t dim1 = outputShape[1];
        int64_t dim2 = outputShape[2];

        int rows;
        int cols;
        bool transposedLayout;

        // YOLO commonly outputs either [1, 84, 8400] or [1, 8400, 84]
        if (dim1 < dim2) {
            cols = static_cast<int>(dim1);
            rows = static_cast<int>(dim2);
            transposedLayout = true;
        } else {
            rows = static_cast<int>(dim1);
            cols = static_cast<int>(dim2);
            transposedLayout = false;
        }

        if (frameNumber % 100 == 0 && totalFrames > 0) {
            double percent = 100.0 * frameNumber / totalFrames;

            cout << "\rProcessing frame "
                 << frameNumber
                 << " / "
                 << totalFrames
                 << " ("
                 << fixed << setprecision(1)
                 << percent
                 << "%)             "
                 << flush;
        }

        if (frameNumber % 270 == 0) {
            cout << "\rFrame "
                 << frameNumber
                 << "  Raw Queue: "
                 << rawQueue.size()
                 << "  Processed Queue: "
                 << processedQueue.size()
                 << flush;
        }

        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        float xFactor = static_cast<float>(img_w) / INPUT_WIDTH;
        float yFactor = static_cast<float>(img_h) / INPUT_HEIGHT;

        auto getValue = [&](int r, int c) -> float {
            if (transposedLayout) {
                return outputData[c * rows + r];
            } else {
                return outputData[r * cols + c];
            }
        };

        for (int i = 0; i < rows; i++) {
            float cx = getValue(i, 0);
            float cy = getValue(i, 1);
            float w  = getValue(i, 2);
            float h  = getValue(i, 3);

            int bestClassId = -1;
            float bestScore = 0.0f;

            for (int c = 4; c < cols; c++) {
                float score = getValue(i, c);

                if (score > bestScore) {
                    bestScore = score;
                    bestClassId = c - 4;
                }
            }

            if (bestScore < SCORE_THRESHOLD)
                continue;

            float left = (cx - 0.5f * w) * xFactor;
            float top = (cy - 0.5f * h) * yFactor;
            float width = w * xFactor;
            float height = h * yFactor;

            boxes.emplace_back(
                Rect(
                    static_cast<int>(left),
                    static_cast<int>(top),
                    static_cast<int>(width),
                    static_cast<int>(height)
                )
            );

            classIds.push_back(bestClassId);
            confidences.push_back(bestScore);
        }

        vector<int> indices;

        cv::dnn::NMSBoxes(
            boxes,
            confidences,
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
            indices
        );

        for (int idx : indices) {
            Rect box = boxes[idx];

            double timeSec = frameNumber / videoFPS;

            float centerX = box.x + box.width * 0.5f;
            float centerY = box.y + box.height * 0.5f;

            string className;

            if (classIds[idx] >= 0 &&
                static_cast<size_t>(classIds[idx]) < classNames.size()) {
                className = classNames[classIds[idx]];
            } else {
                className = to_string(classIds[idx]);
            }

            logFile << frameNumber << ","
                    << fixed << setprecision(3) << timeSec << ","
                    << className << ","
                    << confidences[idx] << ","
                    << centerX << ","
                    << centerY << ","
                    << box.x << ","
                    << box.y << ","
                    << box.width << ","
                    << box.height << "\n";

            rectangle(frame, box, Scalar(0, 255, 0), 2);

            string label = className + cv::format(" %.2f", confidences[idx]);

            int baseline = 0;

            Size tsize = getTextSize(
                label,
                FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
                &baseline
            );

            int y = max(box.y, tsize.height);

            rectangle(
                frame,
                Point(box.x, y - tsize.height - 4),
                Point(box.x + tsize.width, y + baseline),
                Scalar(0, 255, 0),
                FILLED
            );

            putText(
                frame,
                label,
                Point(box.x, y - 2),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(0, 0, 0),
                1
            );
        }

        int64 currentTick = cv::getTickCount();
        double elapsed = (currentTick - prevTick) / cv::getTickFrequency();
        prevTick = currentTick;

        double fps = elapsed > 0.0 ? 1.0 / elapsed : 0.0;

        string fpsText = cv::format("FPS: %.2f", fps);

        int baseline = 0;

        Size fpsSize = getTextSize(
            fpsText,
            FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
            &baseline
        );

        Point fpsOrigin(frame.cols - fpsSize.width - 10, 30);

        rectangle(
            frame,
            Point(fpsOrigin.x - 5, fpsOrigin.y - fpsSize.height - 5),
            Point(fpsOrigin.x + fpsSize.width + 5, fpsOrigin.y + baseline + 5),
            Scalar(0, 0, 0),
            FILLED
        );

        putText(
            frame,
            fpsText,
            fpsOrigin,
            FONT_HERSHEY_SIMPLEX,
            0.7,
            Scalar(255, 255, 255),
            2
        );

        Mat outFrame;
        resize(frame, outFrame, Size(OUTPUT_WIDTH, OUTPUT_HEIGHT));

        processedQueue.push({frameNumber, outFrame});
    }

    logFile.close();
    processedQueue.setFinished();
}

void writerThread(
    SafeQueue<FrameItem>& processedQueue,
    VideoWriter& writer
) {
    FrameItem item;

    while (processedQueue.pop(item)) {
        writer.write(item.frame);
    }
}

int main(int argc, char** argv) {
    auto start = chrono::steady_clock::now();

    if (argc != 2) {
        cout << "Usage:\n";
        cout << argv[0] << " video_file\n";
        return -1;
    }

    vector<string> classNames = loadClasses("classes.names");

    cout << "Loading YOLO model with ONNX Runtime..." << endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO11");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, "best.onnx", sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputNameAllocated = session.GetInputNameAllocated(0, allocator);
    auto outputNameAllocated = session.GetOutputNameAllocated(0, allocator);

    const char* inputName = inputNameAllocated.get();
    const char* outputName = outputNameAllocated.get();

    cout << "Model loaded successfully." << endl;
    cout << "Input name  : " << inputName << endl;
    cout << "Output name : " << outputName << endl;

    cout << "Opening video: " << argv[1] << endl;

    VideoCapture cap(argv[1]);

    if (!cap.isOpened()) {
        cerr << "Cannot open video." << endl;
        return -1;
    }

    cout << "Video opened successfully." << endl;

    double videoFPS = cap.get(CAP_PROP_FPS);
    int totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

    cout << "Resolution : " << width << " x " << height << endl;
    cout << "FPS        : " << videoFPS << endl;
    cout << "Frames     : " << totalFrames << endl;

    if (videoFPS <= 0.0)
        videoFPS = 30.0;

    VideoWriter writer(
        "object_detection_demo.mp4",
        VideoWriter::fourcc('m', 'p', '4', 'v'),
        videoFPS,
        Size(OUTPUT_WIDTH, OUTPUT_HEIGHT),
        true
    );

    if (!writer.isOpened()) {
        cerr << "Cannot open writer.\n";
        return -1;
    }

    SafeQueue<FrameItem> rawQueue(4);
    SafeQueue<FrameItem> processedQueue(4);

    thread reader(readerThread, ref(cap), ref(rawQueue));

    thread yolo(
        yoloThread,
        ref(rawQueue),
        ref(processedQueue),
        ref(session),
        inputName,
        outputName,
        cref(classNames),
        videoFPS,
        totalFrames
    );

    thread writerWorker(
        writerThread,
        ref(processedQueue),
        ref(writer)
    );

    cout << "----------------------------------------" << endl;
    cout << "YOLO11 Video Detection - ONNX Runtime" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Input video : " << argv[1] << endl;
    cout << "Output      : object_detection_demo.mp4" << endl;
    cout << "Log file    : detections.csv" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Starting processing..." << endl;

    reader.join();
    cout << "\n[Reader] Finished." << endl;

    yolo.join();
    cout << "[YOLO] Finished." << endl;

    writerWorker.join();
    cout << "[Writer] Finished." << endl;

    auto end = chrono::steady_clock::now();

    double seconds =
        chrono::duration<double>(end - start).count();

    cout << endl;
    cout << "--------------------------------------" << endl;
    cout << "Processing completed." << endl;
    cout << "Frames processed : " << totalFrames << endl;
    cout << "Elapsed time     : "
         << fixed << setprecision(2)
         << seconds
         << " seconds" << endl;

    cout << "Average FPS      : "
         << totalFrames / seconds
         << endl;

    writer.release();
    cap.release();

    return 0;
}
