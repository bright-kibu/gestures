#include "DetectLive.hpp"
#include "visualization.hpp"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace robot {

// Hand landmark connections for visualization
static const std::vector<std::pair<int, int>> HAND_CONNECTIONS = {
    // Thumb
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    // Index finger
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    // Middle finger
    {0, 9}, {9, 10}, {10, 11}, {11, 12},
    // Ring finger
    {0, 13}, {13, 14}, {14, 15}, {15, 16},
    // Pinky finger
    {0, 17}, {17, 18}, {18, 19}, {19, 20}
};

DetectLive::DetectLive(std::shared_ptr<HailoInference> hailo_infer)
    : hailo_infer_(hailo_infer) {
    
    // Initialize  components
    palm_detector_ = std::make_unique<Detector>("robotpalm", hailo_infer_);
    hand_landmark_ = std::make_unique<Landmark>("hand_landmark", hailo_infer_);
    
    // Initialize FPS tracking
    fps_start_time_ = std::chrono::steady_clock::now();
    
    std::cout << "[DetectLive] Initialized with Hailo inference" << std::endl;
}

DetectLive::~DetectLive() {
    stop_detection();
    
    if (camera_pipe_) {
        pclose(camera_pipe_);
        camera_pipe_ = nullptr;
    }
    
    if (opencv_cap_.isOpened()) {
        opencv_cap_.release();
    }
    
    std::cout << "[DetectLive] Destroyed" << std::endl;
}

bool DetectLive::initialize_camera(const CameraConfig& config) {
    camera_config_ = config;
    
    std::cout << "[DetectLive] Initializing camera..." << std::endl;
    std::cout << "  Resolution: " << config.width << "x" << config.height << std::endl;
    std::cout << "  FPS: " << config.fps << std::endl;
    std::cout << "  Rotation: " << config.rotation << "°" << std::endl;
    
    // Try libcamera first (modern Pi camera interface)
    if (init_libcamera()) {
        std::cout << "[DetectLive] Using libcamera" << std::endl;
        camera_initialized_ = true;
        use_opencv_fallback_ = false;
        return true;
    }
    
    // Fallback to OpenCV
    std::cout << "[DetectLive] libcamera failed, trying OpenCV fallback..." << std::endl;
    if (init_opencv_camera()) {
        std::cout << "[DetectLive] Using OpenCV camera" << std::endl;
        camera_initialized_ = true;
        use_opencv_fallback_ = true;
        return true;
    }
    
    std::cerr << "[DetectLive] ERROR: Failed to initialize any camera interface!" << std::endl;
    return false;
}

// bool DetectLive::init_libcamera() {
//     try {
//         // Build libcamera-still command for continuous capture
//         std::stringstream cmd;
//         cmd << "libcamera-vid";
//         cmd << " --width " << camera_config_.width;
//         cmd << " --height " << camera_config_.height;
//         cmd << " --framerate " << camera_config_.fps;
//         cmd << " --timeout 0";  // Continuous
//         cmd << " --codec yuv420";
//         cmd << " --output -";   // Output to stdout
//         cmd << " --nopreview";  // No preview window
//         cmd << " --immediate";  // Start immediately
        
//         // Add rotation if specified
//         if (camera_config_.rotation != 0) {
//             // libcamera uses transform parameter
//             if (camera_config_.rotation == 90) {
//                 cmd << " --transform rot90";
//             } else if (camera_config_.rotation == 180) {
//                 cmd << " --transform rot180"; 
//             } else if (camera_config_.rotation == 270) {
//                 cmd << " --transform rot270";
//             }
//         }
        
//         std::string command = cmd.str();
//         std::cout << "[DetectLive] libcamera command: " << command << std::endl;
        
//         // Test if libcamera-vid is available
//         int result = system("which libcamera-vid > /dev/null 2>&1");
//         if (result != 0) {
//             std::cout << "[DetectLive] libcamera-vid not found" << std::endl;
//             return false;
//         }
        
//         // Test camera availability
//         result = system("libcamera-hello --list-cameras > /dev/null 2>&1");
//         if (result != 0) {
//             std::cout << "[DetectLive] No cameras detected by libcamera" << std::endl;
//             return false;
//         }
        
//         return true;  // libcamera is available, will start pipe when needed
        
//     } catch (const std::exception& e) {
//         std::cerr << "[DetectLive] libcamera initialization error: " << e.what() << std::endl;
//         return false;
//     }
// }

bool DetectLive::init_opencv_camera() {
    try {
        // Try different camera indices
        for (int camera_id = 0; camera_id < 4; ++camera_id) {
            opencv_cap_.open(camera_id);
            if (opencv_cap_.isOpened()) {
                std::cout << "[DetectLive] Found camera at index " << camera_id << std::endl;
                
                // Set camera properties
                opencv_cap_.set(cv::CAP_PROP_FRAME_WIDTH, camera_config_.width);
                opencv_cap_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_config_.height);
                opencv_cap_.set(cv::CAP_PROP_FPS, camera_config_.fps);
                
                // Verify settings
                int actual_width = static_cast<int>(opencv_cap_.get(cv::CAP_PROP_FRAME_WIDTH));
                int actual_height = static_cast<int>(opencv_cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
                double actual_fps = opencv_cap_.get(cv::CAP_PROP_FPS);
                
                std::cout << "[DetectLive] Camera configured: " 
                          << actual_width << "x" << actual_height 
                          << " @ " << actual_fps << " FPS" << std::endl;
                
                return true;
            }
        }
        
        std::cerr << "[DetectLive] No OpenCV cameras found" << std::endl;
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "[DetectLive] OpenCV camera error: " << e.what() << std::endl;
        return false;
    }
}

bool DetectLive::load_models(const std::string& palm_model_path, const std::string& hand_model_path) {
    std::cout << "[DetectLive] Loading models..." << std::endl;
    std::cout << "  Palm detection: " << palm_model_path << std::endl;
    std::cout << "  Hand landmark: " << hand_model_path << std::endl;
    
    // Load palm detection model
    if (!palm_detector_->load_model(palm_model_path)) {
        std::cerr << "[DetectLive] ERROR: Failed to load palm detection model" << std::endl;
        return false;
    }
    
    // Load hand landmark model
    if (!hand_landmark_->load_model(hand_model_path)) {
        std::cerr << "[DetectLive] ERROR: Failed to load hand landmark model" << std::endl;
        return false;
    }
    
    std::cout << "[DetectLive] Models loaded successfully" << std::endl;
    return true;
}

void DetectLive::configure_detection(const DetectionConfig& config) {
    detection_config_ = config;
    
    // Configure palm detector
    palm_detector_->set_min_score_threshold(config.min_score_thresh);
    palm_detector_->set_debug(config.debug);
    
    // Configure hand landmark detector
    hand_landmark_->set_debug(config.debug);
    
    std::cout << "[DetectLive] Detection configured:" << std::endl;
    std::cout << "  Min score threshold: " << config.min_score_thresh << std::endl;
    std::cout << "  ROI scale: " << config.roi_scale << std::endl;
    std::cout << "  Debug: " << (config.debug ? "ON" : "OFF") << std::endl;
}

cv::Mat DetectLive::capture_libcamera_frame() {
    // For libcamera, we would need to implement a proper streaming interface
    // This is a simplified version - in practice, you'd want to use libcamera C++ API directly
    // For now, return empty frame to indicate libcamera capture not fully implemented
    return cv::Mat();
}

cv::Mat DetectLive::capture_opencv_frame() {
    cv::Mat frame;
    if (opencv_cap_.isOpened()) {
        opencv_cap_ >> frame;
        
        // Apply rotation if specified
        if (!frame.empty() && camera_config_.rotation != 0) {
            frame = rotate_image(frame, camera_config_.rotation);
        }
    }
    return frame;
}

cv::Mat DetectLive::rotate_image(const cv::Mat& image, int angle) {
    if (angle == 0) {
        return image;
    } else if (angle == 90) {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
        return rotated;
    } else if (angle == 180) {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_180);
        return rotated;
    } else if (angle == 270) {
        cv::Mat rotated;
        cv::rotate(image, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
        return rotated;
    } else {
        std::cout << "[DetectLive] WARNING: Unsupported rotation " << angle << "°" << std::endl;
        return image;
    }
}

bool DetectLive::start_detection() {
    if (!camera_initialized_) {
        std::cerr << "[DetectLive] ERROR: Camera not initialized" << std::endl;
        return false;
    }
    
    if (is_running_) {
        std::cout << "[DetectLive] Detection already running" << std::endl;
        return true;
    }
    
    ensure_output_directory();
    
    std::cout << "[DetectLive] Starting live detection..." << std::endl;
    
    is_running_ = true;
    stop_requested_ = false;
    
    // Start camera capture thread if using libcamera
    if (!use_opencv_fallback_) {
        // Note: Full libcamera implementation would start streaming thread here
        std::cout << "[DetectLive] Note: libcamera streaming not fully implemented in this demo" << std::endl;
    }
    
    return true;
}

void DetectLive::stop_detection() {
    if (!is_running_) {
        return;
    }
    
    std::cout << "[DetectLive] Stopping detection..." << std::endl;
    
    stop_requested_ = true;
    is_running_ = false;
    
    // Stop camera thread
    if (camera_thread_.joinable()) {
        camera_thread_.join();
    }
    
    std::cout << "[DetectLive] Detection stopped" << std::endl;
}

cv::Mat DetectLive::process_frame(const cv::Mat& input_frame) {
    if (input_frame.empty()) {
        return cv::Mat();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Prepare output image
    cv::Mat output = input_frame.clone();
    cv::Mat rgb_frame;
    
    // Convert BGR to RGB for processing
    auto resize_start = std::chrono::high_resolution_clock::now();
    cv::cvtColor(input_frame, rgb_frame, cv::COLOR_BGR2RGB);
    
    // Resize and pad for palm detection
    auto [resized_img, scale, pad] = palm_detector_->resize_pad(rgb_frame);
    auto resize_end = std::chrono::high_resolution_clock::now();
    current_profile_.resize_time = std::chrono::duration<double>(resize_end - resize_start).count();
    
    // Run palm detection
    auto detections = palm_detector_->predict_on_image(resized_img);
    current_profile_.detector_pre_time = palm_detector_->get_profile_pre();
    current_profile_.detector_model_time = palm_detector_->get_profile_model();
    current_profile_.detector_post_time = palm_detector_->get_profile_post();
    
    if (!detections.empty()) {
        auto extract_start = std::chrono::high_resolution_clock::now();
        
        // Denormalize detections
        auto denorm_detections = palm_detector_->denormalize_detections(detections, scale, pad);
        
        // Convert detections to ROI parameters
        std::vector<float> xc, yc, roi_scale, theta;
        for (const auto& detection : denorm_detections) {
            // Extract center, scale, and rotation from detection
            float cx = detection.bbox[0] + detection.bbox[2] / 2.0f;
            float cy = detection.bbox[1] + detection.bbox[3] / 2.0f;
            float w = detection.bbox[2];
            float h = detection.bbox[3];
            float s = std::max(w, h) * detection_config_.roi_scale;
            
            xc.push_back(cx);
            yc.push_back(cy);
            roi_scale.push_back(s);
            theta.push_back(0.0f);  // Assuming no rotation for now
        }
        
        // Extract ROIs for hand landmark detection
        auto [roi_images, roi_affine, roi_boxes] = hand_landmark_->extract_roi(rgb_frame, xc, yc, theta, roi_scale);
        auto extract_end = std::chrono::high_resolution_clock::now();
        current_profile_.extract_roi_time = std::chrono::duration<double>(extract_end - extract_start).count();
        
        // Run hand landmark detection
        auto [flags, normalized_landmarks] = hand_landmark_->predict(roi_images);
        current_profile_.landmark_pre_time = hand_landmark_->get_profile_pre();
        current_profile_.landmark_model_time = hand_landmark_->get_profile_model();
        current_profile_.landmark_post_time = hand_landmark_->get_profile_post();
        
        // Draw results
        auto annotate_start = std::chrono::high_resolution_clock::now();
        
        // Denormalize landmarks and draw them
        auto landmarks = hand_landmark_->denormalize_landmarks(normalized_landmarks, roi_affine);
        
        for (size_t i = 0; i < landmarks.size(); ++i) {
            std::cout << "  Landmark detection " << i << " confidence: " << flags[i] << std::endl;
            
            if (flags[i] > 0.5f) {  // Confidence threshold
                std::cout << "  Processing landmarks for detection " << i << " (confidence passed)" << std::endl;
                
                // Convert to cv::Point2f for drawing
                std::vector<cv::Point2f> landmark_points;
                for (const auto& landmark : landmarks[i]) {
                    landmark_points.emplace_back(landmark.x, landmark.y);
                }
                
                // Debug: Print first few landmark coordinates
                if (landmark_points.size() >= 3) {
                    std::cout << "  Landmark coordinates for detection " << i << ":" << std::endl;
                    std::cout << "    [0]: (" << landmark_points[0].x << ", " << landmark_points[0].y << ")" << std::endl;
                    std::cout << "    [1]: (" << landmark_points[1].x << ", " << landmark_points[1].y << ")" << std::endl;
                    std::cout << "    [2]: (" << landmark_points[2].x << ", " << landmark_points[2].y << ")" << std::endl;
                }
                
                draw_hand_landmarks(output, landmark_points);
            } else {
                std::cout << "  Skipping landmarks for detection " << i << " (confidence too low: " << flags[i] << ")" << std::endl;
            }
        }
        
        // Draw detections and ROIs
        draw_detections(output, denorm_detections);
        draw_roi_boxes(output, roi_boxes);
        
        auto annotate_end = std::chrono::high_resolution_clock::now();
        current_profile_.annotate_time = std::chrono::duration<double>(annotate_end - annotate_start).count();
    }
    
    // Calculate total time and FPS
    auto end_time = std::chrono::high_resolution_clock::now();
    current_profile_.total_time = std::chrono::duration<double>(end_time - start_time).count();
    current_profile_.fps = 1.0 / current_profile_.total_time;
    
    // Update frame counter and FPS
    frame_count_++;
    update_fps();
    
    // Draw FPS if enabled
    if (detection_config_.show_fps) {
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(current_fps_));
        cv::putText(output, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
    
    return output;
}

void DetectLive::draw_hand_landmarks(cv::Mat& image, const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() != 21) {
        return;  // Expected 21 hand landmarks
    }
    
    // Draw landmark points
    for (const auto& point : landmarks) {
        cv::circle(image, point, 3, cv::Scalar(0, 255, 0), -1);
    }
    
    // Draw connections
    for (const auto& connection : HAND_CONNECTIONS) {
        if (connection.first < landmarks.size() && connection.second < landmarks.size()) {
            cv::line(image, landmarks[connection.first], landmarks[connection.second], 
                    cv::Scalar(255, 0, 0), 2);
        }
    }
}

void DetectLive::draw_detections(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& detection : detections) {
        cv::Rect rect(detection.bbox[0], detection.bbox[1], detection.bbox[2], detection.bbox[3]);
        cv::rectangle(image, rect, cv::Scalar(255, 255, 0), 2);
        
        // Draw score
        if (detection_config_.show_scores) {
            std::string score_text = std::to_string(detection.score).substr(0, 4);
            cv::putText(image, score_text, cv::Point(rect.x, rect.y - 5), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
        }
    }
}

void DetectLive::draw_roi_boxes(cv::Mat& image, const std::vector<cv::Rect2f>& roi_boxes) {
    for (const auto& roi : roi_boxes) {
        cv::rectangle(image, roi, cv::Scalar(0, 255, 255), 1);
    }
}

void DetectLive::update_fps() {
    fps_frame_count_++;
    
    if (fps_frame_count_ >= 10) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - fps_start_time_).count();
        
        current_fps_ = fps_frame_count_ / elapsed;
        
        fps_frame_count_ = 0;
        fps_start_time_ = current_time;
    }
}

void DetectLive::ensure_output_directory() {
    try {
        if (!std::filesystem::exists(output_dir_)) {
            std::filesystem::create_directories(output_dir_);
            std::cout << "[DetectLive] Created output directory: " << output_dir_ << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[DetectLive] Warning: Could not create output directory: " << e.what() << std::endl;
    }
}

bool DetectLive::capture_frame(const std::string& filename_prefix) {
    try {
        // Capture current frame
        cv::Mat frame;
        if (use_opencv_fallback_) {
            frame = capture_opencv_frame();
        } else {
            frame = capture_libcamera_frame();
        }
        
        if (frame.empty()) {
            std::cerr << "[DetectLive] No frame to capture" << std::endl;
            return false;
        }
        
        // Generate filename
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::stringstream ss;
        ss << output_dir_ << "/";
        if (!filename_prefix.empty()) {
            ss << filename_prefix << "_";
        }
        ss << "frame_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") << ".jpg";
        
        std::string filename = ss.str();
        
        // Save frame
        if (cv::imwrite(filename, frame)) {
            std::cout << "[DetectLive] Captured frame: " << filename << std::endl;
            return true;
        } else {
            std::cerr << "[DetectLive] Failed to save frame: " << filename << std::endl;
            return false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[DetectLive] Capture error: " << e.what() << std::endl;
        return false;
    }
}

void DetectLive::set_min_score_threshold(float threshold) {
    detection_config_.min_score_thresh = threshold;
    palm_detector_->set_min_score_threshold(threshold);
}

void DetectLive::set_debug(bool enable) {
    detection_config_.debug = enable;
    palm_detector_->set_debug(enable);
    hand_landmark_->set_debug(enable);
}

void DetectLive::set_profiling(bool enable) {
    detection_config_.profile = enable;
    // Note: Profiling is always collected, this just controls if it's displayed
}

} // namespace robot
