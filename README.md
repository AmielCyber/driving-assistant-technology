# Assisted Driving Technology with Stop Sign Detection, Road Object Detection, and Lane Departure Warning System

AV/ADAS with C++ OpenCV, Nvidia Jetson. With lane departure warning system(Amiel), stop sign detection with HOG (Julia), and
on-the-road object detection with YOLO(Carlos).

## Requirements

### CMake

[CMake](https://cmake.org/download/) is required
to build the generated Makefile for your system.

### ONNX Runtime

#### ONNX Runtime for Jetson Build

Download the release for Linux, in our case aarch64 for the Jetson and extract the archive

<https://github.com/microsoft/onnxruntime/releases>

Create the target directory

```bash
sudo mkdir -p /opt/onnxruntime
```

Copy the contents to the target directory

```bash
sudo cp -r onnxruntime-linux-aarch64-* /opt/onnxruntime/
```

Update Path

```bash
echo "/opt/onnxruntime/lib" | sudo tee /etc/ld.so.conf.d/onnxruntime.conf
sudo ldconfig
```

#### ONNX Runtime for macOS Installation

<https://formulae.brew.sh/formula/onnxruntime>

```bash
brew install onnxruntime
```

#### ONNX Runtime Cross-Platform Instructions

<https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx#install-onnx-runtime>

## Build Process

To perform a clean build remove the build directory

```bash
rm -rf build
```

Generate Makefile for YOUR system in the build directory

```bash
cmake -B build
```

**OR** generate Makefile with your custom onnxruntime path

```bash
cmake -B build -Donnxruntime_dir=/path/to/onnxruntime
```

Build generated Makefile in build directory

```bash
make -C build
```

## Run

Help Usage:

```bash
./build/DriveMe --help
```

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
./build/DriveMe --video=./test-data/22400001.AVI --show=stops,lanes,objects
```

Show Lane Departure System Feature

```bash
./build/LaneDeparture --video=./test-data/22400001.AVI --show=lanes 
```

Show Stop Sign Detection

```bash
./build/LaneDeparture --video=./test-data/22400001.AVI --show=stops 
```

Show Road Objects

```bash
./build/LaneDeparture --video=./test-data/22400001.AVI --show=objects 
```

## Training Source

> **Note:** The training dataset is **not** included in this repository. However, the HOG detector can be retrained by following the training steps described below.

### Training the HOG Stop Sign Detector

#### 1. Download the Dataset

Download the Road Sign Detection dataset from Kaggle:

<https://www.kaggle.com/datasets/andrewmvd/road-sign-detection>

---

#### 2. Organize the Dataset

The HOG detector requires two categories of images:

- **Positive images** — Images that contain the object of interest (stop signs).
- **Negative images** — Images that do **not** contain stop signs.

For this project:

- **Positive dataset:** Images from the **stop** class.
- **Negative dataset:** Images from the remaining classes, including:
  - Crosswalk
  - Speed Limit
  - Traffic Light

---

#### 3. Verify the Dataset

The following command was used to determine the number of images in each category:

```bash
grep -R "<name>" annotations | sed 's/.*<name>//;s/<\/name>.*//' | sort | uniq -c
```

The dataset contained the following categories:

| Category | Number of Images |
|-----------|-----------------:|
| Crosswalk | 200 |
| Speed Limit | 783 |
| Stop Sign (`stop`) | 91 |
| Traffic Light | 170 |

---

#### 4. Split the Dataset

The dataset was divided into training and testing sets.

- **80%** of the images were used for training.
- **20%** of the images were reserved for testing.

The resulting directory structure is:

```text
archive/
├── positives/
├── negatives/
└── test/
```

---

### Training the Detector

The initial HOG detector was trained using the following command:

```bash
./training \
-pd="archive/positives/*.png" \
-nd="archive/negatives/*.png" \
-td="archive/test/*.png" \
-fn=stop_detector.yml
```

---

### Improving the Detector

To improve detection performance, the detector was retrained using data augmentation.

The following augmentations were enabled:

- Horizontal image flipping
- Rotations

Training was performed using:

```bash
./TrainHOGStopSign \
-pd="archive/positives/*.png" \
-nd="archive/negatives/*.png" \
-td="archive/test/*.png" \
-dw=64 \
-dh=64 \
-f=true \
-d=true \
-fn=stop_sign_hog_detector.yml
```

The resulting detector is stored as:

```text
stop_sign_hog_detector.yml
```

---

### Training Environment

> **Note:** Training was performed on a macOS laptop rather than a Linux machine for improved training performance. The generated `stop_sign_hog_detector.yml` can be copied to the Linux environment and used without retraining.

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

- build: where all build files are located including the Makefile and program executable such as
  - DriveMe
  - TODO: ADD TRAINING EXECUTABLES
- data: All generated output from this exercise
- docs: Resources and documentation for this exercise
- src: All source files produced for this exercise including
  C++ headers and implementations along with Python scripts and training implementations
- test-data: Used for development only, where we store
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
