# Exercise 5 Extended Lab Proposal and Exploratory Code

AV/ADAS with C++ OpenCV, Nvidia Jetson. With lane departure warning system(Amiel), stop detection with HOG (Julia), and
on-the-road object detection with machine learning using YOLO(Carlos).

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

Usage: (--help)

```bash
Usage: DriveMe [params] 

        -h, --help (value:true)
                Print this message.
        -o, --store
                Store the results back in a video file name.
        -s, --show
                Features to show separated by commas. Example: --show=stops,lanes,objects
        -v, --video
                Video file name

```

Show all features

```bash
./build/DriveMe --video=./test-data/22400001.AVI --show=stops,lanes,objects --store=my-video.mp4
```

Show one feature

```bash
./build/DriveMe --video=./test-data/22400001.AVI --show=stops --store=my-video.mp4
```

## Project Directory Structure

```bash
Exercise5CSCI612/
├── CMakeLists.txt
├── build/
│   │── DriveMe
│   └── Makefile
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

## File Structure

```text
Exercise5CSCI612/
├── src/
│   └── code-for-stop-sign-training-and-detection/
│       ├── DetectStopSign.cpp
│       ├── stop_sign_hog_detector.yml
│       └── training/
│           ├── TrainHOGStopSign.cpp
│           ├── CMakeLists.txt
│           └── stop_sign_hog_detector.yml
```

## Building and Running Spefically for the Stop Sign Detection Feature

To build the project using the provided `CMakeLists.txt`, run the following commands from the root directory (`Exercise5CSCI612`):

```bash
cmake -B build
make -C build
```

Run the stop sign detector using either of the following commands:

```bash
./build/DetectStopSign --video=<challenge_set_video_file_name>
```

or

```bash
./build/DetectStopSign -v=GOPR0639.MP4 --store=annotated_GOPR0639_07_05_2026.MP4
```

## Training

> **Note:** The training dataset is **not** included in this repository. However, the HOG detector can be retrained by following the training steps described below.
