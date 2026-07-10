# Object Detection
This is the code to train a prototype YOLOv11n and test the model after training.
Training code is using C++ to request Python commands to do the training, validate it and in case of failure during training resume training based on the last check point (save_period).
This is a prototype and many of the variables were hard coded.

A demo prototype was created to test the accuracy of the taining by asking the user to provide a test video.
The demo shows the frames from the video with annotations showing the detected objects, the class and confidence index, and the frame process ratio of the detection.
A log is kept for each object detected in each frame, and a location in the image.
A samll video is produced during the demo as part of documented the detection outcome. this is a hard coded video name.
The user can stop the analysis by pressing ESC, "Q", or "q".

A makefile is also included so the source can be compile and then test it.

The training code was built for a MacBook Air M5, to use multi-GPUs as the training required massive computing power.

The traned model was ported to a Ubuntu24 virtual machine. The best.pt was converted to best.onnx.

## To compile the source code onnx_yolo11_video_demo_threaded.cpp in Linux follow this quick reference:
To select what ONNX library that  you need to install use:
uname -m

Quick reference
Linux hardware          	                            uname -m	  ONNX Runtime package
Intel/AMD PC	                                        x86_64	    onnxruntime-linux-x64-*.tgz
ARM64 Linux (Apple M VM, Raspberry Pi, RK3588, etc.)  aarch64     onnxruntime-linux-aarch64-*.tgz
NVIDIA Jetson	                                        aarch64	    Jetson-specific ONNX Runtime build (or generic ARM64 for CPU-only inference)

$$ To compile the source code, use the following command to compile:
g++ onnx_yolo11_video_demo_threaded.cpp -o onnx_yolo11_demo_threaded     -std=c++17 -O3 -pthread     $(pkg-config --cflags --libs opencv4)     -I/opt/onnxruntime/include     -L/opt/onnxruntime/lib     -lonnxruntime

Or use the provided CMakeLists.txt to generate a Makefile.

## To run the program use ./onnx_yolo11_video_demo_threaded video_file.mp4
The program processes the video and generate a log file called detections.csv and a new video called object_detection_demo.mp4. This version does not show the original video but eep some process indicators during execution, at the end a summary is provided.
Play the video to see the analysis amd all the objects that were detected.
