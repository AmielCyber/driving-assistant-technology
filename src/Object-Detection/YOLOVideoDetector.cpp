// YOLOVideoDetector.cpp
#include "YOLOVideoDetector.h"

#include <iostream>
#include <iomanip>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using namespace cv;
using namespace std;

// Constructor using parameters
YOLOVideoDetector::YOLOVideoDetector(
        const string& modelPath,
        const string& classFilePath,
        const bool log_data) // Control logging
    :
      env(ORT_LOGGING_LEVEL_WARNING, "YOLO"),
      memoryInfo(
          Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator,
              OrtMemTypeDefault))
{
    loadClasses(classFilePath);
    loadModel(modelPath);

    // Do log just if it is required by the user
    if (log_data) {
        // Open the log
        logFile.open("detections.csv");
        // This is the structure
        if (logFile.is_open())
        {
            logFile << "frame,class,confidence,"
                    << "center_x,center_y,"
                    << "left,top,width,height,"
                    << "frame_width,frame_height\n";
        }
        else // In case the log cannot be created
        {
            cerr << "Cannot open detections.csv for writing.\n";
        }
    }
}

// Destructor to close the log at the end of the program
YOLOVideoDetector::~YOLOVideoDetector()
{
    if (logFile.is_open())
    {
        logFile.close();
    }
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
/*
  Previous loadmodel for Mac and Linuxvoid YOLOVideoDetector::loadModel(const string& modelPath)
{
    // This provides flexibility to try new trained models
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);


    // Enable Apple Neural Engine / GPU through CoreML
#ifdef __APPLE__   
    Ort::ThrowOnError(
	OrtSessionOptionsAppendExecutionProvider_CoreML(
            sessionOptions,
            0));
#endif    
    session =
        std::make_unique<Ort::Session>(
            env,
            modelPath.c_str(),
            sessionOptions);
    
    // To understand how what is doing the process
    size_t providerCount = 0;

        auto providers =
            Ort::GetAvailableProviders();

        for (const auto& provider : providers)
        {
            std::cout << "Available provider: "
                      << provider
                      << std::endl;
        }
    
    Ort::AllocatorWithDefaultOptions allocator;

    // Input
    auto inputName =
        session->GetInputNameAllocated(0, allocator);

    inputNameStorage.push_back(inputName.get());
    inputNames.push_back(inputNameStorage.back().c_str());

    // Output
    auto outputName =
        session->GetOutputNameAllocated(0, allocator);

    outputNameStorage.push_back(outputName.get());
    outputNames.push_back(outputNameStorage.back().c_str());
}*/
// New LoaderModel including CUDA for Jetson
void YOLOVideoDetector::loadModel(const std::string& modelPath)
{
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
    // Apple Neural Engine / GPU
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CoreML(
            sessionOptions, 0));

#elif defined(__linux__) && defined(USE_CUDA)
    // NVIDIA GPU (Jetson / Linux with CUDA)
    OrtCUDAProviderOptions cuda_options{};
    cuda_options.device_id = 0;

    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_CUDA(
            sessionOptions, &cuda_options));

#else
    // CPU only (Ubuntu VM, generic Linux, etc.)
    std::cout << "Execution Provider: CPU" << std::endl;
#endif

    session = std::make_unique<Ort::Session>(
        env,
        modelPath.c_str(),
        sessionOptions);

    Ort::AllocatorWithDefaultOptions allocator;

    auto inputName = session->GetInputNameAllocated(0, allocator);
    inputNameStorage.push_back(inputName.get());
    inputNames.push_back(inputNameStorage.back().c_str());

    auto outputName = session->GetOutputNameAllocated(0, allocator);
    outputNameStorage.push_back(outputName.get());
    outputNames.push_back(outputNameStorage.back().c_str());
}

// Main method to detect objects, receive a frame and provide a processed frame
// This is our pipe line design
cv::Mat YOLOVideoDetector::process(cv::Mat &frame)
{
    // Variables for the detected objects and highlight them in the frame
    std::vector<int> classIds; // Vector to store the detected classes
    std::vector<float> confidences; // Vector to store the detected confidence rates
    std::vector<cv::Rect> boxes; // Vector to store the boxes for each detected object

    // Temp Timer
    //int64 frameStart = cv::getTickCount();
    
    // Launch the detection method and send the frame and recieve all the detected object info.
    detectObjects(frame, classIds, confidences, boxes);

    // sequence of the detected objects from the neural network
    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        boxes,
        confidences,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        indices
    );

    // for all detected object, create a box with level and confidence score in the output frame
    for (int idx : indices)
    {
        cv::Rect box = boxes[idx];
        std::string className = getClassName(classIds[idx]);
        
        // Write a record for each of the detected objects in the log
        logDetection(
            frameNumber, // Num of frame
            className, // Class
            confidences[idx], // Confidence of the detection
            box, // Box of the detected object
            frame.size() // Size of the frame
        );

        drawDetection(frame, box, className, confidences[idx]);
    }

    frameNumber++;   // next frame
    
    // Timer
    /*double frameTime =
        (cv::getTickCount() - frameStart) /
        cv::getTickFrequency();

    std::cout
        << "Frame: "
        << frameTime * 1000
        << " ms   FPS: "
        << 1.0/frameTime
        << "\n";*/
    
    return frame;
}

// Control for the main cpp to identify what feature is performing
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
    // Variables needed for the calculation frame size
    int imgWidth = frame.cols;
    int imgHeight = frame.rows;

    // Temp timers
    // int64 start = cv::getTickCount();
    
    // Convert the frame into a tensor which is the format the neural networks expect
    // Resize input frame to YOLO input size
    cv::Mat resized;

    cv::resize(
        frame,
        resized,
        cv::Size(INPUT_WIDTH, INPUT_HEIGHT)
    );

    // Convert BGR to RGB
    cv::cvtColor(
        resized,
        resized,
        cv::COLOR_BGR2RGB
    );

    // Prepare tensor data in CHW format
    std::vector<float> inputTensorValues(
        3 * INPUT_WIDTH * INPUT_HEIGHT
    );

    int channelSize =
        INPUT_WIDTH * INPUT_HEIGHT;


    for (int y = 0; y < INPUT_HEIGHT; y++)
    {
        for (int x = 0; x < INPUT_WIDTH; x++)
        {
            cv::Vec3b pixel =
                resized.at<cv::Vec3b>(y, x);

            // Red channel
            inputTensorValues[
                0 * channelSize +
                y * INPUT_WIDTH +
                x
            ] = pixel[0] / 255.0f;

            // Green channel
            inputTensorValues[
                1 * channelSize +
                y * INPUT_WIDTH +
                x
            ] = pixel[1] / 255.0f;

            // Blue channel
            inputTensorValues[
                2 * channelSize +
                y * INPUT_WIDTH +
                x
            ] = pixel[2] / 255.0f;
        }
    }

    std::array<int64_t,4> inputShape =
    {
        1,
        3,
        INPUT_HEIGHT,
        INPUT_WIDTH
    };
    
    Ort::Value inputTensor =
        Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            inputShape.data(),
            inputShape.size());

    // Timer
    //int64 inferenceStart = cv::getTickCount();

   // Pass the tnesor to the neural network
    auto outputTensors =
        session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            &inputTensor,
            1,
            outputNames.data(),
            1);

    // Timer
    /*double inferenceTime =
        (cv::getTickCount() - inferenceStart) /
        cv::getTickFrequency();

    std::cout
        << "Inference: "
        << inferenceTime * 1000
        << " ms\n";*/
    
    float* outputData =
        outputTensors[0].GetTensorMutableData<float>();

    // Convert the output to a cv::Mat
    auto info =
        outputTensors[0].GetTensorTypeAndShapeInfo();

    std::vector<int64_t> shape =
        info.GetShape();
    
    // Wrap the ONNX buffer as an OpenCV matrix.
    int dims[] =
    {
        (int)shape[0],
        (int)shape[1],
        (int)shape[2]
    };

    Mat output(
        3,
        dims,
        CV_32F,
        outputData);
    
    if (output.dims == 3)
    {
        int rows = output.size[1];
        int cols = output.size[2];

        output = output.reshape(1, rows);

        if (rows < cols)
            transpose(output, output);
    }

    // Preparing how the boxes will be located in the frame
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

        // Simpler loop thank use minMaxLot
        int bestClass = -1;
        float bestScore = 0.0f;

        for (int c = 4; c < output.cols; ++c)
        {
            if (data[c] > bestScore)
            {
                bestScore = data[c];
                bestClass = c - 4;
            }
        }

        // Pass only detected objects that meet the threshold
        // We can apply a stronger filter by increasing the threshold
        if (bestScore < SCORE_THRESHOLD)
            continue;

        float left = (cx - 0.5f * w) * xFactor;
        float top = (cy - 0.5f * h) * yFactor;
        float width = w * xFactor;
        float height = h * yFactor;

        // Compute bounding box
        boxes.emplace_back(
            Rect(
                static_cast<int>(left),
                static_cast<int>(top),
                static_cast<int>(width),
                static_cast<int>(height)
            )
        );

        // Pass the class id and the confidence score for each of the detected objects
        classIds.push_back(bestClass);
        confidences.push_back(bestScore);
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

// Method to write the detection log
void YOLOVideoDetector::logDetection(int frameNumber,
                                     const std::string& className,
                                     float confidence,
                                     const cv::Rect& box,
                                     const cv::Size& frameSize)
{
    if (!logFile.is_open())
        return;

    // Calculate the box center
    float centerX = box.x + box.width * 0.5f;
    float centerY = box.y + box.height * 0.5f;

    // format the info for the record
    logFile << frameNumber << ","
            << className << ","
            << std::fixed << std::setprecision(3) << confidence << ","
            << centerX << ","
            << centerY << ","
            << box.x << ","
            << box.y << ","
            << box.width << ","
            << box.height << ","
            << frameSize.width << ","
            << frameSize.height << "\n";
}
