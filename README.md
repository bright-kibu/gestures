Gestures
=
Library for hand, landmark and gesture detection using the Hailo embedded AI processors.

## Building the Project

### Prerequisites

- C++ compiler with C++17/C++20 support (g++)
- OpenCV 4.x
- Hailo SDK
- libcamera, libcamera-dev

### Model Download

Before building, download the hef versions of the palm detection and landmark models and the supplied gesture model:

```bash
# Download the gesture model
wget https://github.com/bright-kibu/gestures/releases/download/0.0.1/gesture_model.hef

# Place it in the models directory
mkdir -p models
mv gesture_model.hef models/
```

The model file should be placed in the `models/` directory in your project root.

### Build Instructions

#### Build the Library

To build the static library (`libgestures.a`):

```bash
make lib
```

#### Build the Main Application

To build the main detection application:

```bash
make main
```

#### Build Everything

To build both the library and main application:

```bash
make all
```

### Build Targets

- `lib` - Build the static library only
- `main` - Build the main detection application
- `all` - Build both library and main application (default)
- `clean` - Remove build artifacts
- `help` - Show detailed build options

### Configuration Options

The Makefile automatically detects OpenCV and other dependencies. For custom installations:

```bash
# Use custom OpenCV path
make OPENCV_PATH=/path/to/opencv

```

### Dependencies Status Check

Check your build configuration:

```bash
make config-check
```

References
-

The gestures library is based on the following work:

- Google MediaPipe models : [google/mediapipe](https://github.com/google/mediapipe/blob/master/docs/solutions/models.md)
- Blaze Tutorial : [AlbertaBeef/blaze_app_python](https://github.com/AlbertaBeef/blaze_app_python)
- Hailo : [hailo-ai](https://github.com/hailo-ai), [hailo.ai](https://hailo.ai)
