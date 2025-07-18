#pragma once

#include "Detector.hpp"
#include "Landmark.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <thread>

namespace robot {

/**
 * Modern C++ class for live palm and hand landmark detection using libcamera on Raspberry Pi 5
 * Integrates Detector (palm detection) and Landmark (hand landmarks)
 */
class DetectLive {
public:
    struct CameraConfig {
        int width = 2304;
        int height = 1296; 
        int fps = 30;
        int rotation = 0;  // 0, 90, 180, 270 degrees
        std::string format = "RGB888";
    };
    
    struct DetectionConfig {
        float min_score_thresh = 0.5f;
        float roi_scale = 1.0f;  // Scale factor for ROI size
        bool debug = false;
        bool show_fps = false;
        bool show_scores = false;
        bool profile = false;
    };
    
    struct ProfileData {
        double resize_time = 0.0;
        double detector_pre_time = 0.0;
        double detector_model_time = 0.0;
        double detector_post_time = 0.0;
        double extract_roi_time = 0.0;
        double landmark_pre_time = 0.0;
        double landmark_model_time = 0.0;
        double landmark_post_time = 0.0;
        double annotate_time = 0.0;
        double total_time = 0.0;
        double fps = 0.0;
    };

private:
    // Camera components
    CameraConfig camera_config_;
    DetectionConfig detection_config_;
    
    //  components
    std::shared_ptr<HailoInference> hailo_infer_;
    std::unique_ptr<Detector> palm_detector_;
    std::unique_ptr<Landmark> hand_landmark_;
    
    // Camera state
    bool camera_initialized_ = false;
    bool is_running_ = false;
    std::atomic<bool> stop_requested_{false};
    
    // Libcamera process handle
    FILE* camera_pipe_ = nullptr;
    std::thread camera_thread_;
    
    // OpenCV fallback
    cv::VideoCapture opencv_cap_;
    bool use_opencv_fallback_ = false;
    
    // Performance tracking
    ProfileData current_profile_;
    int frame_count_ = 0;
    
    // FPS calculation
    std::chrono::steady_clock::time_point fps_start_time_;
    int fps_frame_count_ = 0;
    double current_fps_ = 0.0;
    
    // Output directory for captured images
    std::string output_dir_ = "./captured-images";
    
public:
    /**
     * Constructor
     */
    DetectLive(std::shared_ptr<HailoInference> hailo_infer);
    
    /**
     * Destructor
     */
    ~DetectLive();
    
    /**
     * Initialize camera system
     */
    bool initialize_camera(const CameraConfig& config = CameraConfig{});
    
    /**
     * Load detection and landmark models
     */
    bool load_models(const std::string& palm_model_path, const std::string& hand_model_path);
    
    /**
     * Configure detection parameters
     */
    void configure_detection(const DetectionConfig& config);
    
    /**
     * Start live detection
     */
    bool start_detection();
    
    /**
     * Stop live detection
     */
    void stop_detection();
    
    /**
     * Process a single frame (main detection pipeline)
     */
    cv::Mat process_frame(const cv::Mat& input_frame);
    
    /**
     * Capture and save current frame
     */
    bool capture_frame(const std::string& filename_prefix = "");
    
    /**
     * Get current performance profile
     */
    const ProfileData& get_profile() const { return current_profile_; }
    
    /**
     * Get current FPS
     */
    double get_fps() const { return current_fps_; }
    
    /**
     * Check if detection is running
     */
    bool is_running() const { return is_running_; }
    
    /**
     * Update detection threshold
     */
    void set_min_score_threshold(float threshold);
    
    /**
     * Set ROI scale factor
     */
    void set_roi_scale(float scale) { detection_config_.roi_scale = scale; }
    
    /**
     * Enable/disable debug mode
     */
    void set_debug(bool enable);
    
    /**
     * Enable/disable profiling
     */
    void set_profiling(bool enable);

private:
    /**
     * Initialize libcamera for Raspberry Pi 5
     */
    bool init_libcamera();
    
    /**
     * Initialize OpenCV fallback camera
     */
    bool init_opencv_camera();
    
    /**
     * Capture frame from libcamera
     */
    cv::Mat capture_libcamera_frame();
    
    /**
     * Capture frame from OpenCV
     */
    cv::Mat capture_opencv_frame();
    
    /**
     * Rotate image based on configuration
     */
    cv::Mat rotate_image(const cv::Mat& image, int angle);
    
    /**
     * Draw landmarks on image
     */
    void draw_hand_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    
    /**
     * Draw detection boxes on image
     */
    void draw_detections(cv::Mat& image, const std::vector<Detection>& detections);
    
    /**
     * Draw ROI boxes on image
     */
    void draw_roi_boxes(cv::Mat& image, const std::vector<cv::Rect2f>& roi_boxes);
    
    /**
     * Update FPS counter
     */
    void update_fps();
    
    /**
     * Create output directory if needed
     */
    void ensure_output_directory();
    
    /**
     * Camera capture thread function
     */
    void camera_thread_func();
};

} // namespace robot
