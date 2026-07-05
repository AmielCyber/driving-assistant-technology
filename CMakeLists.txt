cmake_minimum_required(VERSION 3.16)
project(AppliedComputerVisionSelfDrivingVehicle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)            # Use C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)   # Enforce C++17, Don't use anything below
set(CMAKE_CXX_EXTENSIONS OFF)         # Force strict, portable ISO C++17


if(MSVC)
  # If using Microsoft's Compiler
  add_compile_options(/W4)
else()
  # Everything else such as CLang and GCC
  add_compile_options(-Wall -Wextra -pedantic)
endif()

find_package(OpenCV REQUIRED)

add_executable(DriveMe src/main.cpp
        src/ADASFeature.h
        src/LaneDeparture.cpp
        src/LaneDeparture.h)
target_link_libraries(DriveMe PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
