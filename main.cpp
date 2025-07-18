#include "DetectLiveCamera.hpp"
#include <signal.h>
#include <thread>
#include <chrono>

using namespace robot;

volatile bool running = true;
std::shared_ptr<HailoInference> global_hailo_inference;

void signal_handler(int) {
    std::cout << "\nShutting down..." << std::endl;
    running = false;
    if (global_hailo_inference) {
        global_hailo_inference->request_shutdown();
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "Raspberry Pi 5 Hand Detection - CSI Camera" << std::endl;
    std::cout << "Press 'q' or Ctrl+C to exit" << std::endl;
    std::cout << "============================================" << std::endl;
    
    try {
        // Initialize Hailo inference with real SDK
        global_hailo_inference = std::make_shared<HailoInference>();
        auto hailo_inference = global_hailo_inference;
        
        std::cout << "Using real Hailo SDK for inference" << std::endl;
        
        // Create camera detector
        DetectLiveCamera detector(hailo_inference);
        
        // Configure for Pi 5 performance
        DetectLiveCamera::DetectionConfig config;
        config.min_score_thresh = 0.8f;  // Much higher threshold to reduce false detections
        config.roi_scale = 1.2f;
        config.debug = false;
        config.show_fps = true;
        config.show_scores = true;
        config.profile = true;
        config.max_detections = 2;
        config.capture_enabled = true;  // Enable display
        detector.configure_detection(config);
        
        // Load models
        std::cout << "Loading models..." << std::endl;
        if (!detector.load_models("models/palm_detection_lite.hef", "models/hand_landmark_lite.hef")) {
            std::cerr << "Failed to load models. Check that HEF files exist in models/" << std::endl;
            return 1;
        }
        
        // Set threshold AFTER loading models (to prevent config_model() from overwriting it)
        std::cout << "Setting detection threshold to 0.8..." << std::endl;
        detector.set_min_score_threshold(0.8f);
        
        // Initialize CSI camera
        std::cout << "Initializing Pi 5 CSI camera..." << std::endl;
        DetectLiveCamera::CameraConfig camera_config;
        // Use camera's native resolution for best quality
        camera_config.width = 1536;   // Native IMX708 width
        camera_config.height = 864;   // Native IMX708 height
        camera_config.fps = 30;
        camera_config.rotation = 0;
        
        if (!detector.initialize_camera(camera_config)) {
            std::cerr << "Failed to initialize camera. Is CSI camera connected?" << std::endl;
            return 1;
        }
        
        std::cout << "Starting detection. Show your hands!" << std::endl;
        
        // Start the detection pipeline (runs in background threads)
        if (!detector.start_detection()) {
            std::cerr << "Failed to start detection pipeline" << std::endl;
            return 1;
        }
        
        // Main loop - just wait for exit signal
        std::cout << "Detection running. Press 'q' or Ctrl+C to exit..." << std::endl;
        
        while (running) {
            // Check if detection is still running (in case of camera failure)
            if (!detector.is_running()) {
                std::cout << "Detection stopped (camera may have failed). Exiting..." << std::endl;
                running = false;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Stop detection
        std::cout << "Stopping detection..." << std::endl;
        detector.stop_detection();
        
        // Clean up global reference
        global_hailo_inference.reset();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        global_hailo_inference.reset();
        return 1;
    }
    
    std::cout << "Done." << std::endl;
    return 0;
}
