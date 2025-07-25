# Build static library from core sources
LIBRARY = libgestures.a
LIB_SOURCES = Config.cpp Base.cpp Detector.cpp Landmark.cpp Gesture.cpp visualization.cpp
LIB_OBJECTS = $(LIB_SOURCES:.cpp=.o)

lib: $(LIBRARY)

$(LIBRARY): $(LIB_OBJECTS)
	ar rcs $@ $^

# Enhanced Makefile for DetectLiveCamera and existing C++ projects
# Supports both C++17 (for camera) and C++20 (for existing code)

CXX = g++
CXXFLAGS_LEGACY = -std=c++20 -Wall -Wextra -O2 -g -Wno-deprecated-enum-enum-conversion -Wno-deprecated-enum-float-conversion
CXXFLAGS_CAMERA = -std=c++17 -Wall -Wextra -O3 -DNDEBUG
DEBUGFLAGS_CAMERA = -std=c++17 -Wall -Wextra -g -O0 -DDEBUG

# OpenCV configuration - automatically detect or use provided path
ifdef OPENCV_PATH
	OPENCV_INCLUDES = -I$(OPENCV_PATH)/include/opencv4 -I$(OPENCV_PATH)/include
	OPENCV_LIBS = -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
else
	# Try to detect OpenCV using pkg-config first
	OPENCV_PKG_CONFIG := $(shell pkg-config --exists opencv4 2>/dev/null && echo "opencv4" || echo "")
	ifeq ($(OPENCV_PKG_CONFIG),opencv4)
		OPENCV_INCLUDES := $(shell pkg-config --cflags opencv4)
		OPENCV_LIBS := $(shell pkg-config --libs opencv4)
	else
		# Fallback to common installation paths
		UNAME_S := $(shell uname -s)
		ifeq ($(UNAME_S),Darwin)
			# macOS with Homebrew
			OPENCV_PATH = /opt/homebrew/Cellar/opencv/4.11.0_1
			OPENCV_INCLUDES = -I$(OPENCV_PATH)/include/opencv4
			OPENCV_LIBS = -L$(OPENCV_PATH)/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
		else
			# Linux (including Raspberry Pi)
			OPENCV_INCLUDES = -I/usr/include/opencv4 -I/usr/local/include/opencv4
			OPENCV_LIBS = -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
		endif
	endif
endif

# Hailo SDK configuration
# Default Hailo SDK paths (adjust as needed for your system)
# Also include project root to pick up local `hailo/` SDK copy
HAILO_INCLUDES = -I/usr/include/hailo -I/opt/hailo/include -I..
HAILO_LIBS = -lhailort

# libcamera configuration for modern camera support
# Set DISABLE_LIBCAMERA=1 to force OpenCV fallback
LIBCAMERA_AVAILABLE = $(shell pkg-config --exists libcamera && echo "yes" || echo "no")
ifndef DISABLE_LIBCAMERA
ifeq ($(LIBCAMERA_AVAILABLE),yes)
	LIBCAMERA_CFLAGS = $(shell pkg-config --cflags libcamera) -DLIBCAMERA2_AVAILABLE
	LIBCAMERA_LIBS = $(shell pkg-config --libs libcamera)
else
	LIBCAMERA_CFLAGS =
	LIBCAMERA_LIBS =
endif
else
	LIBCAMERA_CFLAGS =
	LIBCAMERA_LIBS =
endif

# Compilation flags
INCLUDES = -I. $(OPENCV_INCLUDES) $(HAILO_INCLUDES)
LDFLAGS = $(OPENCV_LIBS) $(HAILO_LIBS)

# Source files
SOURCES = Config.cpp Base.cpp Detector.cpp Landmark.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = Config.hpp Base.hpp Detector.hpp Landmark.hpp

TARGET = main

# Library target
LIBRARY = lib.a

# ================================================================================
# NEW CAMERA DETECTION TARGETS
# ================================================================================

# Camera-specific sources and targets
CAMERA_SOURCES = DetectLiveCamera.cpp visualization.cpp Config.cpp Base.cpp Detector.cpp Landmark.cpp Gesture.cpp
CAMERA_OBJECTS = DetectLiveCamera.o visualization.o Config_camera.o Base_camera.o Detector_camera.o Landmark_camera.o Gesture_camera.o
# CAMERA_LIBRARY = librobot_camera.so
# CAMERA_TEST = test_camera_detection

# Camera-specific flags
CAMERA_ALL_CFLAGS = $(CXXFLAGS_CAMERA) $(OPENCV_INCLUDES) $(LIBCAMERA_CFLAGS) $(HAILO_INCLUDES)
CAMERA_ALL_LIBS = $(OPENCV_LIBS) $(LIBCAMERA_LIBS) $(HAILO_LIBS) -lpthread


# Camera object files (use C++17 standard)
DetectLiveCamera.o: DetectLiveCamera.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

visualization.o: visualization.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

Config_camera.o: Config.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

Base_camera.o: Base.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

Detector_camera.o: Detector.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

Landmark_camera.o: Landmark.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<

Gesture_camera.o: Gesture.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -fPIC -c -o $@ $<


# Main application for Pi 5 CSI camera detection
main: main.o $(CAMERA_LIBRARY)
	$(CXX) -o $@ $< -L. -lrobot_camera -Wl,-rpath,. $(CAMERA_ALL_LIBS)
	@echo "Main detection application created successfully"

main.o: main.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -c -o $@ $<

# Default target
all: lib main

# Build library
lib: $(OBJECTS)
	ar rcs $@ $^
	@echo "Library $(LIBRARY) created successfully"

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f *.o $(LIBRARY) $(TARGET)
	@echo "Clean completed"

# Install library (optional)
install: $(LIBRARY)
	@echo "Installing library to /usr/local/lib and headers to /usr/local/include"
	sudo cp $(LIBRARY) /usr/local/lib/
	sudo cp $(HEADERS) /usr/local/include/
	@echo "Installation completed"

# Rebuild everything
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all       - Build library and test executable (default)"
	@echo "  library   - Build only the library"
	@echo "  test      - Build only the test executable"
	@echo "  camera    - Build modern camera detection library and test"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install library and headers to system"
	@echo "  rebuild   - Clean and rebuild everything"
	@echo "  help      - Show this help message"
	@echo "  help-camera - Show camera-specific help"
	@echo ""
	@echo "OpenCV Configuration:"
	@echo "  The Makefile automatically detects OpenCV using pkg-config or system paths"
	@echo "  To use a custom OpenCV installation:"
	@echo "    make OPENCV_PATH=/path/to/opencv"
	@echo ""
	@echo "Camera Dependencies Status:"
	@echo "  OpenCV:     $(shell pkg-config --exists opencv4 && echo "Found" || echo "NOT FOUND")"
	@echo "  libcamera:  $(LIBCAMERA_AVAILABLE)"
	@echo "  Hailo SDK:  Found"
	@echo ""
	@echo "Hailo SDK Support:"
	@echo "  Hailo SDK is always enabled in this build"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Auto-detect OpenCV, use Hailo SDK"
	@echo "  make OPENCV_PATH=/usr/local             # Custom OpenCV path, use Hailo SDK"
	@echo "  make HAILO_SDK_PATH=/path/to/hailo      # Custom Hailo path"

# Individual targets
library: $(LIBRARY)

# Configuration check target
config-check:
	@echo "=== Build Configuration ==="
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "OpenCV includes: $(OPENCV_INCLUDES)"
	@echo "OpenCV libs: $(OPENCV_LIBS)"
	@echo "Hailo includes: $(HAILO_INCLUDES)"
	@echo "Hailo libs: $(HAILO_LIBS)"
	@echo "Hailo SDK: ENABLED"
	@echo "==========================="


# Debug target with additional flags
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Release target with optimization
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)

# # Functional test targets
# test-functional: $(TARGET_FUNCTIONAL)
# 	@echo "Running functional palm detection test..."
# 	./$(TARGET_FUNCTIONAL)

# test-functional-hailo: INCLUDES += $(HAILO_INCLUDES)
# test-functional-hailo: LDFLAGS += $(HAILO_LIBS)
# test-functional-hailo: $(TARGET_FUNCTIONAL)
# 	@echo "Running functional palm detection test with Hailo SDK..."
# 	./$(TARGET_FUNCTIONAL)

.PHONY: all clean install rebuild help library test debug release hailo mock config-check test-functional test-functional-hailo camera camera-debug install-camera uninstall-camera clean-camera check-camera-deps help-camera