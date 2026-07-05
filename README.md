# Exercise 5 Extended Lab Proposal and Exploratory Code

AV/ADAS with C++ OpenCV, Nvidia Jetson. With lane departure warning system(Amiel), stop detection with HOG (Julia), and on-the-road object detection with machine learning using YOLO(Carlos).

## Requirements

[CMake](https://cmake.org/download/) is required
to build the generated Makefile for your system.

## Build Process

Generate Makefile for YOUR system in the build directory

```bash
cmake -B build
```

Build generated Makefile in build directory

```bash
make -C build
```

## Run

```bash
./build/DriveMe
```

## Project Directory Structure

```bash
Exercise5CSCI612/
├── CMakeLists.txt
├── build/
│   │── DriveMe
│   └── Makefile
├── data/
├── docs/
├── README.md
├── src/
└── test-data/
```

- build: where all build files are located including the Makefile and program executable
- data: All generated output from the exercise
- docs: Resources and documentation for this bash
- src: All source files produced for this exercise including
C++ headers and implementations along with Python scripts
- test-data: Used for development only where we store
our sample video files to test
