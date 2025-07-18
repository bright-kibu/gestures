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
# To build with Hailo SDK support, set HAILO_SDK_PATH or use: make HAILO_SDK_AVAILABLE=1 HAILO_SDK_PATH=/path/to/hailo
ifdef HAILO_SDK_PATH
	HAILO_SDK_AVAILABLE = 1
	HAILO_INCLUDES = -I$(HAILO_SDK_PATH)/include
	HAILO_LIBS = -L$(HAILO_SDK_PATH)/lib -lhailort
else ifdef HAILO_SDK_AVAILABLE
	# Default Hailo SDK paths (adjust as needed for your system)
	# Also include project root to pick up local `hailo/` SDK copy
	HAILO_INCLUDES = -I/usr/include/hailo -I/opt/hailo/include -I..
	HAILO_LIBS = -lhailort
else
	HAILO_INCLUDES = 
	HAILO_LIBS = 
endif

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

# Conditional compilation flags
ifdef HAILO_SDK_AVAILABLE
	CXXFLAGS += -DHAILO_SDK_AVAILABLE
	CXXFLAGS_CAMERA += -DHAILO_SDK_AVAILABLE
	INCLUDES = -I. $(OPENCV_INCLUDES) $(HAILO_INCLUDES)
	LDFLAGS = $(OPENCV_LIBS) $(HAILO_LIBS)
else
	INCLUDES = -I. $(OPENCV_INCLUDES)
	LDFLAGS = $(OPENCV_LIBS)
endif

# Source files
SOURCES = Config.cpp Base.cpp Detector.cpp Landmark.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = Config.hpp Base.hpp Detector.hpp Landmark.hpp

# Target executable (for testing)
TARGET = test_robot_config
TARGET_FULL = test_robot_full
TARGET_DETECTOR = test_robot_detector
TARGET_LANDMARK = test_robot_landmark
TARGET_FUNCTIONAL = test_palm_detection_functional
TARGET_LANDMARK_FUNCTIONAL = test_landmark_functional
TARGET_CPP_ONLY = test_cpp_only
EXAMPLE = example_usage

# Library target
LIBRARY = lib.a

# ================================================================================
# NEW CAMERA DETECTION TARGETS
# ================================================================================

# Camera-specific sources and targets
CAMERA_SOURCES = DetectLiveCamera.cpp visualization.cpp Config.cpp Base.cpp Detector.cpp Landmark.cpp Gesture.cpp
CAMERA_OBJECTS = DetectLiveCamera.o visualization.o Config_camera.o Base_camera.o Detector_camera.o Landmark_camera.o Gesture_camera.o
CAMERA_LIBRARY = librobot_camera.so
CAMERA_TEST = test_camera_detection

# Camera-specific flags
CAMERA_ALL_CFLAGS = $(CXXFLAGS_CAMERA) $(OPENCV_INCLUDES) $(LIBCAMERA_CFLAGS) $(HAILO_INCLUDES)
CAMERA_ALL_LIBS = $(OPENCV_LIBS) $(LIBCAMERA_LIBS) $(HAILO_LIBS) -lpthread

# Camera targets
camera: $(CAMERA_LIBRARY) $(CAMERA_TEST)

camera-debug: CXXFLAGS_CAMERA = $(DEBUGFLAGS_CAMERA)
camera-debug: clean-camera camera

# Camera library
$(CAMERA_LIBRARY): $(CAMERA_OBJECTS)
	$(CXX) -shared -o $@ $^ $(CAMERA_ALL_LIBS)
	@echo "Camera library $(CAMERA_LIBRARY) created successfully"

# Camera test executable
$(CAMERA_TEST): test_camera_detection.o $(CAMERA_LIBRARY)
	$(CXX) -o $@ $< -L. -lrobot_camera -Wl,-rpath,. $(CAMERA_ALL_LIBS)
	@echo "Camera test executable $(CAMERA_TEST) created successfully"

# Mock test executable (no camera hardware required)
test_mock_detection: test_mock_detection.o $(CAMERA_LIBRARY)
	$(CXX) -o $@ $< -L. -lrobot_camera -Wl,-rpath,. $(CAMERA_ALL_LIBS)
	@echo "Mock test executable $@ created successfully"

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

test_camera_detection.o: test_camera_detection.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -c -o $@ $<

test_mock_detection.o: test_mock_detection.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -c -o $@ $<

# Simple test target (single frame, limited detections)
test_simple_detection: test_simple_detection.o librobot_camera.so
	$(CXX) -o test_simple_detection test_simple_detection.o -L. -lrobot_camera -Wl,-rpath,. $(OPENCV_LIBS) $(LDFLAGS)
	@echo "Simple test executable test_simple_detection created successfully"

test_simple_detection.o: test_simple_detection.cpp $(CAMERA_HEADERS)
	$(CXX) $(CXXFLAGS_CAMERA) $(OPENCV_INCLUDES) -c -o test_simple_detection.o test_simple_detection.cpp

# Headless test target (no GUI, fast exit)
test_headless_detection: test_headless_detection.o librobot_camera.so
	$(CXX) -o test_headless_detection test_headless_detection.o -L. -lrobot_camera -Wl,-rpath,. $(OPENCV_LIBS) $(LDFLAGS)
	@echo "Headless test executable test_headless_detection created successfully"

test_headless_detection.o: test_headless_detection.cpp $(CAMERA_HEADERS)
	$(CXX) $(CXXFLAGS_CAMERA) $(OPENCV_INCLUDES) -c -o test_headless_detection.o test_headless_detection.cpp

# Pi 5 Live Detection Test (for real camera on Raspberry Pi 5)
test_pi5_live_detection: test_pi5_live_detection.cpp $(STATIC_LIBS)
	$(CXX) $(CXXFLAGS_CAMERA) $(OPENCV_INCLUDES) $< $(STATIC_LIBS) $(LIBS) -o $@

# Main application for Pi 5 CSI camera detection
main: main_simple.o $(CAMERA_LIBRARY)
	$(CXX) -o $@ $< -L. -lrobot_camera -Wl,-rpath,. $(CAMERA_ALL_LIBS)
	@echo "Main detection application created successfully"

main_simple.o: main_simple.cpp
	$(CXX) $(CAMERA_ALL_CFLAGS) -c -o $@ $<

# Camera installation
install-camera: $(CAMERA_LIBRARY) $(CAMERA_TEST)
	sudo mkdir -p /usr/local/lib
	sudo mkdir -p /usr/local/bin
	sudo mkdir -p /usr/local/include/robot
	sudo cp $(CAMERA_LIBRARY) /usr/local/lib/
	sudo cp $(CAMERA_TEST) /usr/local/bin/
	sudo cp DetectLiveCamera.hpp /usr/local/include/robot/
	sudo ldconfig
	@echo "Camera library and test installed successfully"

uninstall-camera:
	sudo rm -f /usr/local/lib/$(CAMERA_LIBRARY)
	sudo rm -f /usr/local/bin/$(CAMERA_TEST)
	sudo rm -f /usr/local/include/robot/DetectLiveCamera.hpp
	sudo ldconfig
	@echo "Camera library and test uninstalled successfully"

# Clean camera-specific files
clean-camera:
	rm -f DetectLiveCamera.o visualization.o test_camera_detection.o Config_camera.o Base_camera.o Detector_camera.o Landmark_camera.o $(CAMERA_LIBRARY) $(CAMERA_TEST)

# Camera dependency check
check-camera-deps:
	@echo "Checking camera dependencies..."
	@pkg-config --exists opencv4 || (echo "ERROR: OpenCV not found" && exit 1)
	@echo "OpenCV: OK"
	@if [ "$(LIBCAMERA_AVAILABLE)" = "yes" ]; then echo "libcamera: OK"; else echo "libcamera: Not found (OpenCV fallback will be used)"; fi
	@if [ "$(HAILO_SDK_AVAILABLE)" = "1" ]; then echo "Hailo SDK: OK"; else echo "Hailo SDK: Not found (mock implementation will be used)"; fi
	@echo "All required camera dependencies found!"

# Camera help
help-camera:
	@echo "Camera-specific targets:"
	@echo "  camera         - Build camera library and test executable"
	@echo "  camera-debug   - Build camera components in debug mode"
	@echo "  install-camera - Install camera components to system"
	@echo "  uninstall-camera - Remove camera components from system"
	@echo "  clean-camera   - Clean camera build artifacts"
	@echo "  check-camera-deps - Check camera dependencies"
	@echo "  help-camera    - Show camera-specific help"
	@echo ""
	@echo "Camera usage examples:"
	@echo "  ./$(CAMERA_TEST) --help"
	@echo "  ./$(CAMERA_TEST) -w 1280 -h 720 --show-fps"
	@echo "  ./$(CAMERA_TEST) --debug --profile"

# Default target
all: $(LIBRARY) $(TARGET) $(TARGET_FULL) $(TARGET_DETECTOR) $(TARGET_LANDMARK) $(TARGET_LANDMARK_FUNCTIONAL) $(EXAMPLE)

# Build library
$(LIBRARY): $(OBJECTS)
	ar rcs $@ $^
	@echo "Library $(LIBRARY) created successfully"

# Build test executable (original config test)
$(TARGET): $(OBJECTS) test_main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Test executable $(TARGET) created successfully"

# Build full test executable (config + base test)
$(TARGET_FULL): $(OBJECTS) test_robot_full.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Full test executable $(TARGET_FULL) created successfully"

# Build detector test executable (config + base + detector test)
$(TARGET_DETECTOR): $(OBJECTS) test_robot_detector.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Detector test executable $(TARGET_DETECTOR) created successfully"

# Build landmark test executable (config + base + landmark test)
$(TARGET_LANDMARK): $(OBJECTS) test_robot_landmark.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Landmark test executable $(TARGET_LANDMARK) created successfully"

# Build functional test executable (end-to-end palm detection test)
$(TARGET_FUNCTIONAL): $(OBJECTS) test_palm_detection_functional.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Functional test executable $(TARGET_FUNCTIONAL) created successfully"

# Build landmark functional test executable (end-to-end landmark test)  
$(TARGET_LANDMARK_FUNCTIONAL): $(OBJECTS) test_landmark_functional.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Landmark functional test executable $(TARGET_LANDMARK_FUNCTIONAL) created successfully"

# Build example usage executable
$(EXAMPLE): $(OBJECTS) example_usage.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "Example executable $(EXAMPLE) created successfully"

# Build C++ only test (simple validation)
$(TARGET_CPP_ONLY): $(OBJECTS) test_cpp_only.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)
	@echo "C++ only test executable $(TARGET_CPP_ONLY) created successfully"

# Compile source files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create a simple test file if it doesn't exist
test_main.cpp:
	@echo "Creating test_main.cpp..."
	@echo '#include "Config.hpp"' > test_main.cpp
	@echo '#include <iostream>' >> test_main.cpp
	@echo '' >> test_main.cpp
	@echo 'int main() {' >> test_main.cpp
	@echo '    robot::Config config;' >> test_main.cpp
	@echo '    std::cout << "Config initialized successfully!" << std::endl;' >> test_main.cpp
	@echo '    ' >> test_main.cpp
	@echo '    // Test get_model_config' >> test_main.cpp
	@echo '    auto model_config = config.get_model_config("robotpalm", 256, 256, 2944);' >> test_main.cpp
	@echo '    std::cout << "Palm model config - num_anchors: " << model_config.num_anchors << std::endl;' >> test_main.cpp
	@echo '    ' >> test_main.cpp
	@echo '    // Test get_anchor_options' >> test_main.cpp
	@echo '    auto anchor_options = config.get_anchor_options("robotpalm", 256, 256, 2944);' >> test_main.cpp
	@echo '    std::cout << "Palm anchor options - num_layers: " << anchor_options.num_layers << std::endl;' >> test_main.cpp
	@echo '    ' >> test_main.cpp
	@echo '    // Test generate_anchors' >> test_main.cpp
	@echo '    auto anchors = config.generate_anchors(anchor_options);' >> test_main.cpp
	@echo '    std::cout << "Generated " << anchors.size() << " anchors" << std::endl;' >> test_main.cpp
	@echo '    ' >> test_main.cpp
	@echo '    return 0;' >> test_main.cpp
	@echo '}' >> test_main.cpp

# Clean build artifacts
clean: clean-camera
	rm -f *.o $(LIBRARY) $(TARGET) $(TARGET_FULL) $(TARGET_DETECTOR) $(TARGET_LANDMARK) $(TARGET_FUNCTIONAL) $(TARGET_LANDMARK_FUNCTIONAL) $(TARGET_CPP_ONLY) $(EXAMPLE) test_main.cpp
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
	@echo "  Hailo SDK:  $(if $(HAILO_SDK_AVAILABLE),Found,Not found)"
	@echo ""
	@echo "Hailo SDK Support:"
	@echo "  To build with Hailo SDK support, use one of these options:"
	@echo "    make HAILO_SDK_AVAILABLE=1"
	@echo "    make HAILO_SDK_PATH=/path/to/hailo/sdk"
	@echo "  Without these flags, the code will build with mock Hailo interface"
	@echo ""
	@echo "Examples:"
	@echo "  make                                    # Auto-detect OpenCV, mock Hailo"
	@echo "  make HAILO_SDK_AVAILABLE=1              # Auto-detect OpenCV, default Hailo paths"
	@echo "  make OPENCV_PATH=/usr/local HAILO_SDK_AVAILABLE=1  # Custom paths"
	@echo "  make OPENCV_PATH=/opt/opencv            # Custom OpenCV path, mock Hailo"

# Individual targets
library: $(LIBRARY)
test: $(TARGET)

# Configuration check target
config-check:
	@echo "=== Build Configuration ==="
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "OpenCV includes: $(OPENCV_INCLUDES)"
	@echo "OpenCV libs: $(OPENCV_LIBS)"
ifdef HAILO_SDK_AVAILABLE
	@echo "Hailo includes: $(HAILO_INCLUDES)"
	@echo "Hailo libs: $(HAILO_LIBS)"
	@echo "Hailo SDK: ENABLED"
else
	@echo "Hailo SDK: DISABLED (mock interface)"
endif
	@echo "==========================="

# Hailo SDK specific targets
hailo:
	@echo "Building with Hailo SDK support..."
	$(MAKE) clean
	$(MAKE) HAILO_SDK_AVAILABLE=1 all
	@echo "Built with Hailo SDK support"

mock: 
	@echo "Building with mock Hailo interface..."
	$(MAKE) clean 
	$(MAKE) all
	@echo "Built with mock Hailo interface (no SDK required)"

# Debug target with additional flags
debug: CXXFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Release target with optimization
release: CXXFLAGS += -O3 -DNDEBUG
release: $(TARGET)

# Functional test targets
test-functional: $(TARGET_FUNCTIONAL)
	@echo "Running functional palm detection test..."
	./$(TARGET_FUNCTIONAL)

test-functional-hailo: CXXFLAGS += -DHAILO_SDK_AVAILABLE
test-functional-hailo: INCLUDES += $(HAILO_INCLUDES)
test-functional-hailo: LDFLAGS += $(HAILO_LIBS)
test-functional-hailo: $(TARGET_FUNCTIONAL)
	@echo "Running functional palm detection test with Hailo SDK..."
	./$(TARGET_FUNCTIONAL)

.PHONY: all clean install rebuild help library test debug release hailo mock config-check test-functional test-functional-hailo camera camera-debug install-camera uninstall-camera clean-camera check-camera-deps help-camera
