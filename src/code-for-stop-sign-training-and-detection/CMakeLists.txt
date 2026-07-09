cmake_minimum_required(VERSION 3.10)

project(StopSignProject)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler warnings
add_compile_options(-Wall -O2)

# Find OpenCV
find_package(OpenCV REQUIRED)

# -------------------------
# Detect Stop Sign
# -------------------------
add_executable(DetectStopSign
    DetectStopSign.cpp
)

target_link_libraries(DetectStopSign
    ${OpenCV_LIBS}
)

# -------------------------
# Train HOG Stop Sign
# -------------------------
add_executable(TrainHOGStopSign
    TrainHOGStopSign.cpp
)

target_link_libraries(TrainHOGStopSign
    ${OpenCV_LIBS}
)