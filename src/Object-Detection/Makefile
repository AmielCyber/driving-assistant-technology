# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra $(shell pkg-config --cflags opencv4)

# Libraries
LDFLAGS = $(shell pkg-config --libs opencv4)

# Programs
PROGRAMS = YOLO-Training yolo11_video_demo yolo11_video_demo_threaded

all: $(PROGRAMS)

YOLO-Training: YOLO-Training.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

yolo11_demo: yolo11_video_demo.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
	
yolo11_demo_threaded: yolo11_video_demo_threaded.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

run-training: YOLO-Training
	./YOLO-Training

run-demo: yolo11_demo
	./yolo11_demo

run-demo-threaded: yolo11_demo_threaded
	./yolo11_demo_threaded

clean:
	rm -f $(PROGRAMS)

.PHONY: all clean run-training run-demo run-demo-threaded
