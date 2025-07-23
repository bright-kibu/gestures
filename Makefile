camera-debug: CXXFLAGS_CAMERA = $(DEBUGFLAGS_CAMERA)
camera-debug: clean-camera camera
test_mock_detection: test_mock_detection.o $(CAMERA_LIBRARY)
test_simple_detection: test_simple_detection.o librobot_camera.so
test_simple_detection.o: test_simple_detection.cpp $(CAMERA_HEADERS)
test_headless_detection: test_headless_detection.o librobot_camera.so
test_headless_detection.o: test_headless_detection.cpp $(CAMERA_HEADERS)
test_pi5_live_detection: test_pi5_live_detection.cpp $(STATIC_LIBS)
test_main.cpp:
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)
test-functional: $(TARGET_FUNCTIONAL)
test-functional-hailo: CXXFLAGS += -DHAILO_SDK_AVAILABLE
test-functional-hailo: INCLUDES += $(HAILO_INCLUDES)
test-functional-hailo: LDFLAGS += $(HAILO_LIBS)
test-functional-hailo: $(TARGET_FUNCTIONAL)

# Minimal Makefile for main build and clean only

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2 -g
INCLUDES = -I.
LDFLAGS =

# Source files
SOURCES = Config.cpp Base.cpp Detector.cpp Landmark.cpp Gesture.cpp visualization.cpp main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = Config.hpp Base.hpp Detector.hpp Landmark.hpp Gesture.hpp visualization.hpp

# Main target
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
	@echo "Clean completed"

.PHONY: all clean
