cmake_minimum_required(VERSION 3.16)

project(AppliedComputerVisionSelfDrivingVehicle LANGUAGES CXX)

# OpenCV's exported CUDA configuration still calls the legacy FindCUDA module.
if (POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif ()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4)
else ()
    # Warnings, Extra warnings, warnings if not using C++ standards, level 3 optimization
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif ()


# -------------------------------------------------------
# OpenCV
# -------------------------------------------------------

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version      : ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs : ${OpenCV_INCLUDE_DIRS}")


# -------------------------------------------------------
# ONNX Runtime
# -------------------------------------------------------

set(ONNXRUNTIME_DIR "" CACHE PATH
        "Path to ONNX Runtime installation")


# User provided path
if (ONNXRUNTIME_DIR)

    message(STATUS "Using ONNX Runtime path:")
    message(STATUS "${ONNXRUNTIME_DIR}")

    find_path(
            ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATHS
            ${ONNXRUNTIME_DIR}/include
            ${ONNXRUNTIME_DIR}/include/onnxruntime
    )

    find_library(
            ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS
            ${ONNXRUNTIME_DIR}/lib
    )


    # Automatic detection
else ()

    find_path(
            ONNXRUNTIME_INCLUDE_DIR
            NAMES onnxruntime_cxx_api.h
            PATHS
            /usr/include
            /usr/local/include
            /usr/local/include/onnxruntime
            /opt/onnxruntime/include
            /opt/onnxruntime/include/onnxruntime
            /opt/homebrew/include
            /opt/homebrew/opt/onnxruntime/include
            /opt/homebrew/opt/onnxruntime/include/onnxruntime
    )


    find_library(
            ONNXRUNTIME_LIBRARY
            NAMES onnxruntime
            PATHS
            /usr/lib
            /usr/local/lib
            /opt/onnxruntime/lib
            /opt/homebrew/lib
            /opt/homebrew/opt/onnxruntime/lib
    )

endif ()


if (NOT ONNXRUNTIME_INCLUDE_DIR)

    message(FATAL_ERROR
            "Cannot find ONNX Runtime headers.\n"
            "Install ONNX Runtime or specify:\n"
            "-DONNXRUNTIME_DIR=/path/to/onnxruntime")

endif ()


if (NOT ONNXRUNTIME_LIBRARY)

    message(FATAL_ERROR
            "Cannot find ONNX Runtime library.\n"
            "Install ONNX Runtime or specify:\n"
            "-DONNXRUNTIME_DIR=/path/to/onnxruntime")

endif ()


message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")


add_library(onnxruntime SHARED IMPORTED)

set_target_properties(
        onnxruntime PROPERTIES

        IMPORTED_LOCATION
        ${ONNXRUNTIME_LIBRARY}

        INTERFACE_INCLUDE_DIRECTORIES
        ${ONNXRUNTIME_INCLUDE_DIR}
)


# -------------------------------------------------------
# DriveMe Application
# -------------------------------------------------------

add_executable(DriveMe

        src/main.cpp

        src/ADASFeature.h
        src/ADASFeatureThreadPool.cpp
        src/ADASFeatureThreadPool.h
        src/AsyncLogger.cpp
        src/AsyncLogger.h

        src/lane-departure/LaneDeparture.cpp
        src/lane-departure/LaneDeparture.h
        src/lane-departure/LaneState.h
        src/lane-departure/LaneStateLogger.cpp
        src/lane-departure/LaneStateLogger.h

        src/code-for-stop-sign-training-and-detection/DetectStopSign.cpp
        src/code-for-stop-sign-training-and-detection/DetectStopSign.h

        src/Object-Detection/YOLOVideoDetector.cpp
        src/Object-Detection/YOLOVideoDetector.h
)


target_include_directories(
        DriveMe
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)


target_link_libraries(
        DriveMe
        PRIVATE
        ${OpenCV_LIBS}
        onnxruntime
)


# -------------------------------------------------------
# Train HOG Stop Sign
# -------------------------------------------------------

add_executable(
        TrainHOGStopSign

        src/TrainHOGStopSign.cpp
)


target_include_directories(
        TrainHOGStopSign
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)


target_link_libraries(
        TrainHOGStopSign
        PRIVATE
        ${OpenCV_LIBS}
)
