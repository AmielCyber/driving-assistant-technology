# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Source and target
SRC = YOLO-Training.cpp
TARGET = YOLO-Training

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Remove generated files
clean:
	rm -f $(TARGET)

.PHONY: all run clean
