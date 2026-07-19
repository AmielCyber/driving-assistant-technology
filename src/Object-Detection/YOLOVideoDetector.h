// YOLOVideoDetector.h
#ifndef YOLO_VIDEO_DETECTOR_H
#define YOLO_VIDEO_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <string>
#include <vector>
#include "../ADASFeature.h"

class YOLOVideoDetector : public ADASFeature
{
public:
    YOLOVideoDetector(const std::string& modelPath,
                      const std::string& classFilePath);

    cv::Mat process(cv::Mat &frame) override;
    std::string get_feature_name() override;
    
    ~YOLOVideoDetector();
    
private:
    static constexpr float SCORE_THRESHOLD = 0.35f;//0.25f;
    static constexpr float CONFIDENCE_THRESHOLD = 0.50f;//0.40f;
    static constexpr float NMS_THRESHOLD = 0.45f;//0.45f;

    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;

    // ONNX Runtime
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memoryInfo;

    // Cached model input/output names
    std::vector<std::string> inputNameStorage;
    std::vector<std::string> outputNameStorage;

    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::string> classNames;
    
    std::ofstream logFile;
    int frameNumber = 0;

    void loadClasses(const std::string& filename);
    void loadModel(const std::string& modelPath);

    void detectObjects(const cv::Mat& frame,
                       std::vector<int>& classIds,
                       std::vector<float>& confidences,
                       std::vector<cv::Rect>& boxes);

    void drawDetection(cv::Mat& frame,
                       const cv::Rect& box,
                       const std::string& label,
                       float confidence);

    void drawFPS(cv::Mat& frame, int64& previousTick);

    std::string getClassName(int classId) const;
    
    void logDetection(int frameNumber,
                      const std::string& className,
                      float confidence,
                      const cv::Rect& box,
                      const cv::Size& frameSize);
};

#endif
