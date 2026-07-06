// yolo11_video_demo.cpp
// Created by Carlos G Cazares
// Usage:
//     ./yolo11_video_demo video.mp4
//
// Files expected in current directory:
//     yolo11n.onnx
//     classes.names      (or your own classes file)
//
// Compile:
// g++ yolo11_video_demo.cpp -o yolo11_demo `pkg-config --cflags --libs opencv4`

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#include <iomanip>

using namespace cv;
using namespace cv::dnn;
using namespace std;
// to prevent objects with little confidence in the detection
static const float SCORE_THRESHOLD = 0.25f;
static const float CONFIDENCE_THRESHOLD = 0.25f;
static const float NMS_THRESHOLD = 0.45f;
// To save disk sapce recording the detection
static const int INPUT_WIDTH = 640;
static const int INPUT_HEIGHT = 640;
// Using a vector to have the name of the classes and tag the objects with its detcted class
// The function receive the file name and return a vector with each class
vector<string> loadClasses(const string& filename)
{
    vector<string> classes;
    ifstream ifs(filename);

    string line;
    while (getline(ifs, line))
        classes.push_back(line);

    return classes;
}

// the name of the video to analyze is received as argument 1. Only one argument.
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage:\n";
        cout << argv[0] << " video_file\n";
        return -1;
    }

    // the list of classes need to be defined in classes.name file
    // a text file with the name of each class in order. Ine class name per line only.
    vector<string> classNames = loadClasses("classes.names");
    
    // the YOLO11 model needs to be converted to ONNX format
    // This is to read the model for this program
    Net net = readNet("best.onnx");

    // Trying to give stability to the model by defining OpenCV as the backend
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    // This computer use CPU for detection
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Open the video
    VideoCapture cap(argv[1]);
    // Verify the video file exist and it can be opened.
    if (!cap.isOpened())
    {
        cerr << "Cannot open video.\n";
        return -1;
    }
    
    // This is the file to keep the log of all object detecte and when and where were detected.
    ofstream logFile("detections.csv");

    logFile << "frame,time_sec,class,confidence,"
            << "center_x,center_y,"
            << "left,top,width,height\n";

    Mat frame;

    double fps = 0.0;
    int64 prevTick = cv::getTickCount();
    
    // Prepare the writer to create the video in color. Default
    VideoWriter writer(
                       "object_detection_demo.mp4",
                       VideoWriter::fourcc('m', 'p', '4', 'v'),
                       cap.get(CAP_PROP_FPS),
                       Size(640, 480),
                       true     // false = grayscale
                       );
    
    // Validate the defualt output file name can be openend for writing.
    if (!writer.isOpened())
    {
        cout << "Cannot open writer" << endl;
        return -1;
    };
    
    // We need to count frames so the log can ahve the frame number
    int frameNumber = 0;
    double videoFPS = cap.get(CAP_PROP_FPS);
           
    // Main loop, read each frame from the file
    while (cap.read(frame))
    {
        // since the video can ahve any resolution we store the number of columns and roww for reference
        int img_w = frame.cols;
        int img_h = frame.rows;

        //  preprocesses an image into the format expected by a neural network.
        Mat blob;

        blobFromImage(frame,
                      blob,
                      1.0 / 255.0,
                      Size(INPUT_WIDTH, INPUT_HEIGHT),
                      Scalar(),
                      true,
                      false);

        net.setInput(blob);

        //  vector that will store the network’s output tensors.
        vector<Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        Mat output = outputs[0];

        //------------------------------------------
        // Convert shape if needed
        //------------------------------------------
        if (output.dims == 3)
        {
            int rows = output.size[1];
            int cols = output.size[2];

            output = output.reshape(1, rows);

            // (batch, channels, detections) shape of the output tensor
            // If exported as (1,84,8400), transpose
            if (rows < cols)
                transpose(output, output);
        }

        // prepare the info for the detected object box
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        float xFactor = (float)img_w / INPUT_WIDTH;
        float yFactor = (float)img_h / INPUT_HEIGHT;

        for (int i = 0; i < output.rows; i++)
        {
            float* data = output.ptr<float>(i);

            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            Mat scores(1,
                       output.cols - 4,
                       CV_32FC1,
                       data + 4);

            Point classIdPoint;
            double score;

            minMaxLoc(scores, 0, &score, 0, &classIdPoint);

            // if the detection pass the treshold
            if (score < SCORE_THRESHOLD)
                continue;

            float left = (cx - 0.5f * w) * xFactor;
            float top = (cy - 0.5f * h) * yFactor;
            float width = w * xFactor;
            float height = h * yFactor;

            boxes.emplace_back(
                Rect((int)left,
                     (int)top,
                     (int)width,
                     (int)height));

            classIds.push_back(classIdPoint.x);
            confidences.push_back((float)score);
        }

        // retriving ghe confidence index
        vector<int> indices;

        NMSBoxes(boxes,
                 confidences,
                 CONFIDENCE_THRESHOLD, // only for detected objects passing the thresholds
                 NMS_THRESHOLD,
                 indices);

        // For all detected objects in the frame
        for (int idx : indices)
        {
            Rect box = boxes[idx];
            
            double timeSec = frameNumber / videoFPS; // Calculating when was detected the object

            // Where is the object in the frame using the frame dimenisons to calcyulate the center
            float centerX = box.x + box.width * 0.5f;
            float centerY = box.y + box.height * 0.5f;

            string className; // The class id is matchin g the class position in the classes.name file

            if (classIds[idx] >= 0 &&
                static_cast<size_t>(classIds[idx]) < classNames.size())
            {
                className = classNames[classIds[idx]];
            }
            else
            {
                className = to_string(classIds[idx]);
            }

            // This is the record in the log for the detected object
            // This is keep in memory and at the end it writes the log file
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

            // Prepare the box and the label
            rectangle(frame,
                      box,
                      Scalar(0,255,0),
                      2);

            string label;

            if (classIds[idx] >= 0 &&
                static_cast<size_t>(classIds[idx]) < classNames.size())
            {
                label = classNames[classIds[idx]];
            }
            else
            {
                label = std::to_string(classIds[idx]);
            }

            char conf[32]; // insert a space between the class adn the confidence index
            snprintf(conf, sizeof(conf), " %.2f", confidences[idx]);
            label += conf;

            int baseline = 0;

            // Format the box label
            Size tsize =
                getTextSize(label,
                            FONT_HERSHEY_SIMPLEX,
                            0.5,
                            1,
                            &baseline);

            int y = max(box.y, tsize.height);

            // Draw the frame
            rectangle(frame,
                      Point(box.x, y - tsize.height - 4),
                      Point(box.x + tsize.width,
                            y + baseline),
                      Scalar(0,255,0),
                      FILLED);

            // Write the box label
            putText(frame,
                    label,
                    Point(box.x, y - 2),
                    FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(0,0,0),
                    1);
        }

        // Calculate the frames per second the program is able to process.
        // Time of processing per frame
        int64 currentTick = cv::getTickCount();
        double elapsed = (currentTick - prevTick) / cv::getTickFrequency();
        prevTick = currentTick;

        // Fps is the processing time needed per frame.
        // This can be translated as number of frames than can be processed by second
        fps = 1.0 / elapsed;

        // Format the FPS metric
        string fpsText = cv::format("FPS: %.2f", fps);

        int baseline = 0;
        Size fpsSize = getTextSize(fpsText,
                                   FONT_HERSHEY_SIMPLEX,
                                   0.7,
                                   2,
                                   &baseline);

        Point fpsOrigin(frame.cols - fpsSize.width - 10, 30);

        // All annotations are placed in the frame before it is displayed
        rectangle(frame,
                  Point(fpsOrigin.x - 5, fpsOrigin.y - fpsSize.height - 5),
                  Point(fpsOrigin.x + fpsSize.width + 5, fpsOrigin.y + baseline + 5),
                  Scalar(0, 0, 0),
                  FILLED);

        putText(frame,
                fpsText,
                fpsOrigin,
                FONT_HERSHEY_SIMPLEX,
                0.7,
                Scalar(255, 255, 255),
                2);
        imshow("YOLO11 Detection", frame);
        
        // Resize the frame to save disk space
        Mat outFrame;
        resize(frame, outFrame, Size(640, 480));
        writer.write(outFrame);
        
        // Produce the video
        writer.write(frame);

        // Wait for Esc or q to end the program
        int key = waitKey(1);

        if (key == 27 || key == 'q')
            break;
    }
    
    logFile.close();  // Write the log
    writer.release(); // Release resources before terminate the program
    cap.release();
    destroyAllWindows();

    // Terminate the program
    return 0;
}
