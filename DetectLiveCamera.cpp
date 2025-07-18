#include "DetectLiveCamera.hpp"
#include "Detector.hpp"
#include "Landmark.hpp"
#include "Gesture.hpp"
#include "visualization.hpp"
#include <iostream>

namespace robot {

DetectLiveCamera::~DetectLiveCamera() {
    stop_thread_ = true;
    if (processing_thread_.joinable()) {
        cv_.notify_one();
        processing_thread_.join();
    }

    stop_camera();
    if (window_created_) {
        cv::destroyAllWindows();
    }
}

std::string detect_gesture(const std::vector<cv::Point3f>& landmarks_3d, Gesture& gesture_recognizer) {
    if (landmarks_3d.size() < 21) {
        std::cout << "Not enough landmarks for gesture recognition: " << landmarks_3d.size() << std::endl;
        return "";
    }

    try {
        // Use the gesture recognizer to predict the gesture
        std::string prediction = gesture_recognizer.predict(landmarks_3d);
        if (!prediction.empty()) {
            std::cout << "Gesture prediction: " << prediction << std::endl;
        } else {
            std::cout << "Gesture prediction returned empty" << std::endl;
        }
        return prediction;
    } catch (const std::exception& e) {
        std::cerr << "Error in gesture recognition: " << e.what() << std::endl;
        return "";
    }
}

void DetectLiveCamera::processing_loop() {
    auto inference_engine = std::make_shared<HailoInference>();
    Detector robot_detector("robotpalm", inference_engine);
    Landmark robot_landmark("robothandlandmark", inference_engine);
    Gesture robot_gesture("robotgesture", inference_engine);
    thread_init_success_ = false;
    try {
        robot_detector.load_model("models/palm_detection_lite.hef");
        robot_landmark.load_model("models/hand_landmark_lite.hef");
        robot_gesture.load_model("models/gesture_recognition.hef");
        robot_detector.set_min_score_threshold(0.3f);
        
        // Store the gesture recognizer for visualization
        gesture_recognizer_ = &robot_gesture;
        
        thread_init_success_ = true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load models in processing thread: " << e.what() << std::endl;
    }

    {
        std::lock_guard<std::mutex> lock(mtx_);
        thread_ready_ = true;
    }
    startup_cv_.notify_one();

    if (!thread_init_success_) {
        return;
    }

    while (!stop_thread_) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return new_frame_to_process_ || stop_thread_; });

            if (stop_thread_) {
                break;
            }

            frame = frame_to_process_.clone();
            new_frame_to_process_ = false;
        }

        if (frame.empty()) {
            continue;
        }

        cv::Mat image;
        cv::cvtColor(frame, image, cv::COLOR_BGR2RGB);

        // This part of the loop seems to be missing the declaration of robot_detector and robot_landmark
        // Let's assume they are available from the initialization part of the function.
        
        // Step 1: Resize and pad image for detector
        cv::Mat img1;
        double scale1;
        cv::Point2f pad1;
        std::tie(img1, scale1, pad1) = robot_detector.resize_pad(image);

        if (img1.empty()) {
            continue;
        }

        // Step 2: Palm detection
        auto batch_results = robot_detector.predict_on_batch(std::vector<cv::Mat>{img1});
        std::vector<Detection> normalized_detections = batch_results.empty() ? std::vector<Detection>() : batch_results[0];

        std::vector<Detection> current_detections;
        std::vector<std::vector<cv::Point2f>> current_landmarks;
        std::string current_gesture;

        if (!normalized_detections.empty()) {
            // Step 3: Denormalize detections
            std::vector<Detection> detections = robot_detector.denormalize_detections(normalized_detections, scale1, pad1);
            current_detections = detections;

            // Step 4: Convert detections to ROIs
            std::vector<ROI> rois = robot_detector.detection2roi(detections);

            std::vector<double> xc, yc, theta, scale;
            for (const auto& roi : rois) {
                xc.push_back(roi.xc);
                yc.push_back(roi.yc);
                theta.push_back(roi.theta);
                scale.push_back(roi.scale);
            }

            if (!xc.empty()) {
                // Step 5: Extract ROI images
                std::vector<cv::Mat> roi_imgs;
                std::vector<cv::Mat> roi_affine;
                std::vector<std::vector<cv::Point2f>> roi_boxes;
                std::tie(roi_imgs, roi_affine, roi_boxes) = robot_landmark.extract_roi(image, xc, yc, theta, scale);

                if (!roi_imgs.empty()) {
                    // Step 6: Hand landmark detection
                    std::vector<std::vector<double>> flags;
                    std::vector<std::vector<std::vector<double>>> normalized_landmarks_vec;
                    std::tie(flags, normalized_landmarks_vec) = robot_landmark.predict(roi_imgs);

                    // Step 7: Convert to Point3d
                    std::vector<std::vector<cv::Point3d>> landmarks_3d;
                    for (const auto& batch : normalized_landmarks_vec) {
                        std::vector<cv::Point3d> landmark_batch;
                        for (const auto& landmark : batch) {
                            landmark_batch.emplace_back(landmark.size() > 0 ? landmark[0] : 0, landmark.size() > 1 ? landmark[1] : 0, landmark.size() > 2 ? landmark[2] : 0);
                        }
                        landmarks_3d.push_back(landmark_batch);
                    }

                    // Step 8: Denormalize landmarks
                    std::vector<std::vector<cv::Point3d>> landmarks = robot_landmark.denormalize_landmarks(landmarks_3d, roi_affine);

                    // Step 9: Process landmarks
                    for (size_t i = 0; i < flags.size() && i < landmarks.size(); ++i) {
                        double confidence = flags[i].empty() ? 0.0 : flags[i][0];
                        if (confidence > 0.3f) {
                            std::vector<cv::Point2f> landmark_points;
                            std::vector<cv::Point3f> landmark_points_3d;
                            
                            for (const auto& landmark : landmarks[i]) {
                                if (landmark.x >= 0 && landmark.x < image.cols && landmark.y >= 0 && landmark.y < image.rows) {
                                    landmark_points.emplace_back(landmark.x, landmark.y);
                                    landmark_points_3d.emplace_back(static_cast<float>(landmark.x), 
                                                                   static_cast<float>(landmark.y), 
                                                                   static_cast<float>(landmark.z));
                                }
                            }
                            if (!landmark_points.empty() && !landmark_points_3d.empty()) {
                                current_landmarks.push_back(landmark_points);
                                // Use the proper gesture recognition with 3D landmarks
                                current_gesture = detect_gesture(landmark_points_3d, robot_gesture);
                            }
                        }
                    }
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            latest_detections_ = current_detections;
            latest_landmarks_ = current_landmarks;
            latest_gesture_ = current_gesture;
        }
    }
}

int DetectLiveCamera::run_camera_display_loop(int width, int height, volatile bool* external_running) {
    std::cout << "Starting camera display loop..." << std::endl;
    
    // Initialize camera
    if (init_camera() != 0) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return -1;
    }
    
    // Configure camera
    if (configure_camera(width, height) != 0) {
        std::cerr << "Failed to configure camera" << std::endl;
        return -1;
    }
    
    // Start capture
    if (start_capture() != 0) {
        std::cerr << "Failed to start capture" << std::endl;
        return -1;
    }
    
    // Start the processing thread
    stop_thread_ = false;
    thread_ready_ = false;
    processing_thread_ = std::thread(&DetectLiveCamera::processing_loop, this);

    // Wait for the processing thread to be ready
    {
        std::unique_lock<std::mutex> lock(mtx_);
        startup_cv_.wait(lock, [this] { return thread_ready_; });
    }

    if (!thread_init_success_) {
        std::cerr << "Processing thread failed to initialize. Exiting." << std::endl;
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        return -1;
    }


    // Create display window
    cv::namedWindow(window_name_, cv::WINDOW_AUTOSIZE);
    window_created_ = true;
    
    std::cout << "Camera started. Press 'q' to quit." << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    // Simple display loop - exactly like test_libcamera_simple.cpp
    while (true) {
        // Check external shutdown signal
        if (external_running && !*external_running) {
            std::cout << "External shutdown requested" << std::endl;
            break;
        }
        
        // Read frame
        if (read_frame(frame)) {
            frame_count++;
            
            // Validate frame before processing
            if (frame.empty()) {
                std::cerr << "Frame is empty, skipping processing" << std::endl;
                continue;
            }
            
            if (frame.cols <= 0 || frame.rows <= 0) {
                std::cerr << "Frame has invalid dimensions: " << frame.cols << "x" << frame.rows << std::endl;
                continue;
            }
            
            {
                std::lock_guard<std::mutex> lock(mtx_);
                frame_to_process_ = frame.clone();
                new_frame_to_process_ = true;
            }
            cv_.notify_one();

            cv::Mat output = frame.clone();
            
            {
                std::lock_guard<std::mutex> lock(mtx_);
                if (!latest_detections_.empty()) {
                    draw_detections(output, latest_detections_);
                }
                if (!latest_landmarks_.empty()) {
                    for(const auto& l : latest_landmarks_) {
                        draw_landmarks(output, l, HAND_CONNECTIONS);
                    }
                }
                if (!latest_gesture_.empty()) {
                    // Draw gesture name with larger font
                    cv::putText(output, "Gesture: " + latest_gesture_, cv::Point(10, 70), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
                    
                    // Draw finger status visualization in the top-right corner
                    if (gesture_recognizer_) {
                        gesture_recognizer_->draw_finger_status(output, cv::Point(output.cols - 250, 30), 1);
                    }
                }
            }
            
            // Add simple frame counter overlay
            cv::putText(output, "Frame " + std::to_string(frame_count), cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            
            // Display frame
            try {
                cv::imshow(window_name_, output);
            } catch (const cv::Exception& e) {
                std::cerr << "OpenCV display error: " << e.what() << std::endl;
                break;
            }
            
            // Calculate and display FPS every 30 frames
            if (frame_count % 30 == 0) {
                auto current_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                double fps = 30000.0 / elapsed.count();
                std::cout << "Frame " << frame_count << ", FPS: " << fps << std::endl;
                start_time = current_time;
            }
            
            // Process keyboard input
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                std::cout << "Quit requested" << std::endl;
                break;
            }
        } else {
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }
    }
    
    // Cleanup
    stop_thread_ = true;
    cv_.notify_one();
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }

    cv::destroyAllWindows();
    window_created_ = false;
    stop_camera();
    
    std::cout << "Total frames captured: " << frame_count << std::endl;
    return 0;
}

int DetectLiveCamera::init_camera() {
    // Initialize camera manager
    camera_manager_ = std::make_unique<libcamera::CameraManager>();
    int ret = camera_manager_->start();
    if (ret) {
        std::cerr << "Failed to start camera manager: " << ret << std::endl;
        return ret;
    }
    
    // Get first available camera
    auto cameras = camera_manager_->cameras();
    if (cameras.empty()) {
        std::cerr << "No cameras found" << std::endl;
        return -1;
    }
    
    camera_ = cameras[0];
    std::cout << "Using camera: " << camera_->id() << std::endl;
    
    // Acquire camera
    ret = camera_->acquire();
    if (ret) {
        std::cerr << "Failed to acquire camera: " << ret << std::endl;
        return ret;
    }
    
    return 0;
}

int DetectLiveCamera::configure_camera(int width, int height) {
    // Generate configuration
    config_ = camera_->generateConfiguration({libcamera::StreamRole::VideoRecording});
    if (!config_) {
        std::cerr << "Failed to generate camera configuration" << std::endl;
        return -1;
    }
    
    // Configure stream
    auto& stream_config = config_->at(0);
    stream_config.size = libcamera::Size(width, height);
    stream_config.pixelFormat = libcamera::formats::RGB888;
    
    // Validate configuration
    libcamera::CameraConfiguration::Status validation = config_->validate();
    if (validation == libcamera::CameraConfiguration::Invalid) {
        std::cerr << "Camera configuration invalid" << std::endl;
        return -1;
    }
    
    if (validation == libcamera::CameraConfiguration::Adjusted) {
        std::cout << "Camera configuration adjusted" << std::endl;
        std::cout << "  Adjusted size: " << stream_config.size.width << "x" << stream_config.size.height << std::endl;
    }
    
    // Apply configuration
    int ret = camera_->configure(config_.get());
    if (ret) {
        std::cerr << "Failed to configure camera: " << ret << std::endl;
        return ret;
    }
    
    std::cout << "Camera configured: " << stream_config.size.width << "x" << stream_config.size.height << std::endl;
    return 0;
}

int DetectLiveCamera::start_capture() {
    // Connect request completion signal
    camera_->requestCompleted.connect(this, &DetectLiveCamera::on_request_complete);
    
    // Create frame buffer allocator
    allocator_ = std::make_unique<libcamera::FrameBufferAllocator>(camera_);
    
    // Allocate buffers for each stream
    for (libcamera::StreamConfiguration& cfg : *config_) {
        int ret = allocator_->allocate(cfg.stream());
        if (ret < 0) {
            std::cerr << "Can't allocate buffers: " << ret << std::endl;
            return ret;
        }
        
        unsigned int allocated = allocator_->buffers(cfg.stream()).size();
        std::cout << "Allocated " << allocated << " buffers for stream" << std::endl;
    }
    
    // Create requests
    for (unsigned int i = 0; i < allocator_->buffers(config_->at(0).stream()).size(); ++i) {
        std::unique_ptr<libcamera::Request> request = camera_->createRequest();
        if (!request) {
            std::cerr << "Can't create request" << std::endl;
            return -1;
        }
        
        libcamera::Stream* stream = config_->at(0).stream();
        const std::vector<std::unique_ptr<libcamera::FrameBuffer>>& buffers = allocator_->buffers(stream);
        const std::unique_ptr<libcamera::FrameBuffer>& buffer = buffers[i];
        
        int ret = request->addBuffer(stream, buffer.get());
        if (ret < 0) {
            std::cerr << "Can't set buffer for request: " << ret << std::endl;
            return ret;
        }
        
        // Map buffer memory
        for (const libcamera::FrameBuffer::Plane& plane : buffer->planes()) {
            void* memory = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
            if (memory == MAP_FAILED) {
                std::cerr << "Failed to mmap buffer" << std::endl;
                return -1;
            }
            mapped_buffers_[plane.fd.get()] = std::make_pair(memory, plane.length);
        }
        
        requests_.push_back(std::move(request));
    }
    
    // Start camera
    int ret = camera_->start();
    if (ret) {
        std::cerr << "Failed to start capture: " << ret << std::endl;
        return ret;
    }
    camera_started_ = true;
    
    // Queue all requests
    for (std::unique_ptr<libcamera::Request>& request : requests_) {
        ret = camera_->queueRequest(request.get());
        if (ret < 0) {
            std::cerr << "Can't queue request: " << ret << std::endl;
            camera_->stop();
            return ret;
        }
    }
    
    stream_ = config_->at(0).stream();
    std::cout << "Camera capture started" << std::endl;
    return 0;
}

bool DetectLiveCamera::read_frame(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    
    // Wait for frame with timeout
    bool frame_ready = frame_available_.wait_for(lock, std::chrono::seconds(2), [this] {
        return !frame_queue_.empty();
    });
    
    if (!frame_ready) {
        std::cerr << "Frame timeout" << std::endl;
        return false;
    }
    
    libcamera::Request* request = frame_queue_.front();
    frame_queue_.pop();
    lock.unlock();
    
    try {
        // Convert to OpenCV Mat
        const libcamera::Request::BufferMap& buffers = request->buffers();
        for (auto it = buffers.begin(); it != buffers.end(); ++it) {
            libcamera::FrameBuffer* buffer = it->second;
            
            // Validate buffer
            if (!buffer || buffer->planes().empty()) {
                std::cerr << "Invalid buffer or no planes" << std::endl;
                continue;
            }
            
            // Get the first plane (assuming RGB888 single plane)
            const libcamera::FrameBuffer::Plane& plane = buffer->planes()[0];
            const libcamera::FrameMetadata::Plane& meta = buffer->metadata().planes()[0];
            
            // Validate plane data
            auto it_mapped = mapped_buffers_.find(plane.fd.get());
            if (it_mapped == mapped_buffers_.end()) {
                std::cerr << "Buffer not found in mapped buffers" << std::endl;
                continue;
            }
            
            void* data = it_mapped->second.first;
            unsigned int length = std::min(meta.bytesused, plane.length);
            
            if (!data || length == 0) {
                std::cerr << "Invalid buffer data or zero length" << std::endl;
                continue;
            }
            
            // Calculate dimensions from stream config
            const libcamera::StreamConfiguration& cfg = stream_->configuration();
            int width = cfg.size.width;
            int height = cfg.size.height;
            int stride = cfg.stride;
            
            // Validate dimensions
            if (width <= 0 || height <= 0 || stride <= 0) {
                std::cerr << "Invalid frame dimensions" << std::endl;
                continue;
            }
            
            // Check if we have enough data
            size_t expected_size = height * stride;
            if (length < expected_size) {
                // Try with actual buffer length
                stride = width * 3; // Assume RGB888 without padding
                expected_size = height * stride;
                if (length < expected_size) {
                    std::cerr << "Buffer still too small even without stride padding" << std::endl;
                    continue;
                }
            }
            
            // Create Mat from buffer data (RGB888 format)
            cv::Mat temp_frame(height, width, CV_8UC3, data, stride);

            // Validate the Mat
            if (temp_frame.empty()) {
                std::cerr << "Failed to create valid Mat from buffer" << std::endl;
                continue;
            }

            // Clone the frame and rotate it 180 degrees
            cv::Mat cloned_frame = temp_frame.clone();
            cv::rotate(cloned_frame, frame, cv::ROTATE_180);
            if (frame.empty()) {
                std::cerr << "Frame is empty after clone" << std::endl;
                continue;
            }
            break; // Successfully processed first buffer
            
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in read_frame: " << e.what() << std::endl;
        return false;
    }
    
    // Reuse the request
    try {
        request->reuse(libcamera::Request::ReuseBuffers);
        int ret = camera_->queueRequest(request);
        if (ret < 0) {
            std::cerr << "Failed to requeue request: " << ret << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during request requeue: " << e.what() << std::endl;
        return false;
    }
    
    return !frame.empty();
}

void DetectLiveCamera::stop_camera() {
    if (camera_started_) {
        camera_->stop();
        camera_started_ = false;
    }
    
    if (camera_) {
        camera_->requestCompleted.disconnect(this);
    }
    
    // Unmap buffers
    for (auto& iter : mapped_buffers_) {
        std::pair<void*, unsigned int> pair = iter.second;
        munmap(pair.first, pair.second);
    }
    mapped_buffers_.clear();
    
    requests_.clear();
    allocator_.reset();
    
    if (camera_) {
        camera_->release();
        camera_.reset();
    }
    
    camera_manager_.reset();
}

void DetectLiveCamera::on_request_complete(libcamera::Request* request) {
    if (request->status() == libcamera::Request::RequestCancelled) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    frame_queue_.push(request);
    frame_available_.notify_one();
}

} // namespace robot
