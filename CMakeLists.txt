cmake_minimum_required(VERSION 3.16)
project(AppliedComputerVisionSelfDrivingVehicle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(OpenCV REQUIRED)

# -------------------------
# ONNX Runtime
# ----------------------------------
set(ONNXRUNTIME_DIR /opt/onnxruntime)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# ----------------------------------
# DriveMe
# ----------------------------------
add_executable(LaneDeparture
    src/main.cpp
    src/ADASFeature.h
    src/lane-departure/LaneDeparture.cpp
    src/lane-departure/LaneDeparture.h
    src/lane-departure/LaneState.h
)

target_link_libraries(LaneDeparture PRIVATE ${OpenCV_LIBS})

# ----------------------------------
# Detect Stop Sign
# ----------------------------------
add_executable(DetectStopSign
    src/DetectStopSign.cpp
)

target_link_libraries(DetectStopSign PRIVATE ${OpenCV_LIBS})
target_compile_options(DetectStopSign PRIVATE
        -O2
)

# ----------------------------------
# Train HOG Stop Sign
# ----------------------------------
add_executable(TrainHOGStopSign
    src/TrainHOGStopSign.cpp
)

target_link_libraries(TrainHOGStopSign PRIVATE ${OpenCV_LIBS})

# ----------------------------------
# ONNX YOLO11 Demo
# ----------------------------------
add_executable(onnx_yolo11_demo_threaded
    src/onnx_yolo11_video_demo_threaded.cpp
)

target_include_directories(onnx_yolo11_demo_threaded PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
)

target_link_libraries(onnx_yolo11_demo_threaded PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
    pthread
)

target_compile_options(onnx_yolo11_demo_threaded PRIVATE
    -O3
)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
