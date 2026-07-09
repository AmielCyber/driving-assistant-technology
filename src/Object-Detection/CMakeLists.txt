cmake_minimum_required(VERSION 3.16)

project(onnx_yolo11_demo_threaded)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_DIR /opt/onnxruntime)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

add_executable(
    onnx_yolo11_demo_threaded
    onnx_yolo11_video_demo_threaded.cpp
)

target_link_libraries(
    onnx_yolo11_demo_threaded
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)

target_compile_options(
    onnx_yolo11_demo_threaded
    PRIVATE
    -O3
)
