cmake_minimum_required(VERSION 3.16)
project(AppliedComputerVisionSelfDrivingVehicle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  # Warnings, Extra warnings, warnings if not using C++ standards, level 3 optimization
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(OpenCV REQUIRED)

# -------------------------
# ONNX Runtime
# -------------------------
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root folder")

if(NOT ONNXRUNTIME_DIR)
  if(APPLE)
    # Installed with HomeBrew
    set(ONNXRUNTIME_DIR "/opt/homebrew/opt/onnxruntime")
  elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ONNXRUNTIME_DIR "/opt/onnxruntime")
  endif()
endif()

message(STATUS "ONNX Runtime dir: ${ONNXRUNTIME_DIR}")

add_library(onnxruntime SHARED IMPORTED)

if(APPLE)
  set_target_properties(onnxruntime PROPERTIES
          IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib"
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
  )
else()
  set_target_properties(onnxruntime PROPERTIES
          IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
  )
endif()

# -------------------------
# DriveMe
# -------------------------
add_executable(DriveMe
        src/main.cpp
        src/ADASFeature.h
        src/lane-departure/LaneDeparture.cpp
        src/lane-departure/LaneDeparture.h
        src/lane-departure/LaneState.h
        src/code-for-stop-sign-training-and-detection/DetectStopSign.h
        src/code-for-stop-sign-training-and-detection/DetectStopSign.cpp
        src/Object-Detection/YOLOVideoDetector.h
        src/Object-Detection/YOLOVideoDetector.cpp
        src/lane-departure/LaneStateLogger.cpp
        src/lane-departure/LaneStateLogger.h
        src/ADASFeatureThreadPool.cpp
        src/ADASFeatureThreadPool.h
)

target_link_libraries(DriveMe PRIVATE ${OpenCV_LIBS})

# -------------------------
# Train HOG Stop Sign
# -------------------------
add_executable(TrainHOGStopSign
        src/TrainHOGStopSign.cpp
)

target_link_libraries(TrainHOGStopSign PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
