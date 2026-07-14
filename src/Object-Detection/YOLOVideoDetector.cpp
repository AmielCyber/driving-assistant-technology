// YOLOVideoDetector.cpp
#include "YOLOVideoDetector.h"

#include <iostream>
#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Constructor using parameters
YOLOVideoDetector::YOLOVideoDetector(const string& modelPath,
                                     const string& classFilePath)
{
    loadClasses(classFilePath);
    loadModel(modelPath);
}

// Helper to load the classes names used in the detection Default is classes.names
void YOLOVideoDetector::loadClasses(const string& filename)
{
    ifstream ifs(filename);

    if (!ifs.is_open())
    {
        cerr << "Cannot open classes file: " << filename << endl;
        return;
    }

    string line;

    while (getline(ifs, line))
    {
        classNames.push_back(line);
    }
}

// Helper to load the YOLO trained Model used in the objects detection
void YOLOVideoDetector::loadModel(const string& modelPath)
{
    net = readNet(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
}

// Main method to detect objects
cv::Mat YOLOVideoDetector::process(cv::Mat& frame)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    detectObjects(frame, classIds, confidences, boxes);

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        indices
    );

    for (int idx : indices)
    {
        cv::Rect box = boxes[idx];
        std::string className = getClassName(classIds[idx]);

        drawDetection(frame, box, className, confidences[idx]);
    }

    return frame;
}

std::string YOLOVideoDetector::get_feature_name()
{
    return "objects";
}

// Mian logic to detect the objects, this is coming from YOLO model
void YOLOVideoDetector::detectObjects(const Mat& frame, // Frame
                                      vector<int>& classIds, // Detected Class id
                                      vector<float>& confidences, // detection confidence
                                      vector<Rect>& boxes) // List of boxes created by YOLO model
{
    // Variables needed for the calculation
    int imgWidth = frame.cols;
    int imgHeight = frame.rows;

    Mat blob;

    // Convert the frame in blob which is the format the neural networks expect
    blobFromImage(
        frame, // Provided frame
        blob, // Image to be processed by the model
        1.0 / 255.0, // Scale factor this is the most common
        Size(INPUT_WIDTH, INPUT_HEIGHT), // Image size
        Scalar(),// mean substractions
        true,// swap green to blue as YOLO use a different order for color channels
        false// Keep the image info intact, do not crop just resize it.
    );

    net.setInput(blob); // pass the blob to the neural network

    vector<Mat> outputs; // use a vector to store outputs
    // the net passed its results
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // Received the results, format them and stored
    Mat output = outputs[0];

    if (output.dims == 3)
    {
        int rows = output.size[1];
        int cols = output.size[2];

        output = output.reshape(1, rows);

        if (rows < cols)
            transpose(output, output);
    }

    // Preparing how the outputs will be received in the main loop
    float xFactor = static_cast<float>(imgWidth) / INPUT_WIDTH;
    float yFactor = static_cast<float>(imgHeight) / INPUT_HEIGHT;

    // Calculate each detected object
    for (int i = 0; i < output.rows; i++)
    {
        float* data = output.ptr<float>(i);

        float cx = data[0];
        float cy = data[1];
        float w = data[2];
        float h = data[3];

        Mat scores(
            1,
            output.cols - 4,
            CV_32FC1,
            data + 4
        );

        Point classIdPoint;
        double score;

        minMaxLoc(scores, nullptr, &score, nullptr, &classIdPoint);

        // Pass only detected objects that meet the threshold
        // We can apply a stronger filter by increasing the threshold
        if (score < SCORE_THRESHOLD)
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

        // Pass the class id and the confidence score for each of the detected objects
        classIds.push_back(classIdPoint.x);
        confidences.push_back(static_cast<float>(score));
    }
}

// Method to draw the detected objects
void YOLOVideoDetector::drawDetection(Mat& frame, // the canvas
                                      const Rect& box, // box attributes
                                      const string& label, // class
                                      float confidence) // Confidence
{
    // Draw the box in the frame
    rectangle(frame, box, Scalar(0, 255, 0), 2);

    // This is the class
    string displayLabel = label;

    // format the confidence score
    char conf[32];
    snprintf(conf, sizeof(conf), " %.2f", confidence);
    // Concatenate the class with the score
    displayLabel += conf;

    int baseline = 0;

    // format the label (class and confidence)
    Size textSize = getTextSize(
        displayLabel,
        FONT_HERSHEY_SIMPLEX,
        0.5,
        1,
        &baseline
    );

    // Calculate the location of the box
    int y = max(box.y, textSize.height);

    // draw the box in the frame
    rectangle(
        frame,
        Point(box.x, y - textSize.height - 4),
        Point(box.x + textSize.width, y + baseline),
        Scalar(0, 255, 0),
        FILLED
    );

    // Write the label at the top left corner of the box
    putText(
        frame,
        displayLabel,
        Point(box.x, y - 2),
        FONT_HERSHEY_SIMPLEX,
        0.5,
        Scalar(0, 0, 0),
        1
    );
}

// Display the processing speed
void YOLOVideoDetector::drawFPS(Mat& frame, // the canvas
                                int64& previousTick) // previous timestamp to calculate speed
{
    int64 currentTick = getTickCount();// current timestamp to calculate processing speed

    // Calculate the processing time for the frame
    double elapsed =
        static_cast<double>(currentTick - previousTick) / getTickFrequency();

    previousTick = currentTick; // Now previoustick is current tick

    double fps = 1.0 / elapsed; // The speed is the elapsed time to process 1 frame

    string fpsText = format("FPS: %.2f", fps);// Format the speed to be displayed

    int baseline = 0;

    // Formating
    Size fpsSize = getTextSize(
        fpsText,
        FONT_HERSHEY_SIMPLEX,
        0.7,
        2,
        &baseline
    );

    // Location for the processing speed is in the yop right corner
    Point fpsOrigin(frame.cols - fpsSize.width - 10, 30);

    // Draw the background for the processing speed
    rectangle(
        frame,
        Point(fpsOrigin.x - 5, fpsOrigin.y - fpsSize.height - 5),
        Point(fpsOrigin.x + fpsSize.width + 5,
              fpsOrigin.y + baseline + 5),
        Scalar(0, 0, 0),
        FILLED
    );

    // Write the processing speed over the background using white color
    putText(
        frame,
        fpsText,
        fpsOrigin,
        FONT_HERSHEY_SIMPLEX,
        0.7,
        Scalar(255, 255, 255),
        2
    );
}

// Helper to derive the class name using the class id provided by the detection
string YOLOVideoDetector::getClassName(int classId) const
{
    if (classId >= 0 &&
        static_cast<size_t>(classId) < classNames.size())
    {
        return classNames[classId];
    }

    return to_string(classId);
}
