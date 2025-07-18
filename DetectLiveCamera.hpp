#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

#include <libcamera/libcamera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>
#include <libcamera/formats.h>
#include <sys/mman.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

#include "Base.hpp"
#include "Gesture.hpp"

namespace robot {

class DetectLiveCamera {
private:
    std::unique_ptr<libcamera::CameraManager> camera_manager_;
    std::shared_ptr<libcamera::Camera> camera_;
    std::unique_ptr<libcamera::CameraConfiguration> config_;
    std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<libcamera::Request>> requests_;
    std::map<int, std::pair<void*, unsigned int>> mapped_buffers_;
    std::queue<libcamera::Request*> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable frame_available_;
    libcamera::Stream* stream_ = nullptr;
    bool camera_started_ = false;
    
    std::string window_name_ = "DetectLiveCamera";
    bool window_created_ = false;

    // For processing thread
    std::thread processing_thread_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable startup_cv_;
    bool new_frame_to_process_ = false;
    bool stop_thread_ = false;
    bool thread_ready_ = false;
    bool thread_init_success_ = false;
    cv::Mat frame_to_process_;
    std::vector<Detection> latest_detections_;
    std::vector<std::vector<cv::Point2f>> latest_landmarks_;
    std::string latest_gesture_;
    
    // Gesture recognizer for visualization (non-owning pointer)
    class Gesture* gesture_recognizer_ = nullptr;
    
public:
    DetectLiveCamera() = default;
    ~DetectLiveCamera();
    
    // Simple single function - config, read, display loop
    int run_camera_display_loop(int width = 640, int height = 480, volatile bool* external_running = nullptr);
    
private:
    void processing_loop();
    int init_camera();
    int configure_camera(int width, int height);
    int start_capture();
    bool read_frame(cv::Mat& frame);
    void stop_camera();
    void on_request_complete(libcamera::Request* request);
};

} // namespace robot
