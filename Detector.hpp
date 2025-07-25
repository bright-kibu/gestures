#pragma once

#include "Base.hpp"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>

// Hailo SDK includes
#include "hailo/hailort.hpp"
// Using hailort.hpp which includes vstream, network_group, and defaults
using namespace hailort;

namespace robot {

/**
 * VStream Info wrapper
 * Abstracts Hailo VStream information
 */
struct VStreamInfo {
    std::string name;
    std::vector<size_t> shape;
    
    VStreamInfo() = default;
    VStreamInfo(const std::string& stream_name, const std::vector<size_t>& stream_shape)
        : name(stream_name), shape(stream_shape) {}
        
    VStreamInfo(const hailo_vstream_info_t& hailo_info) 
        : name(hailo_info.name) {
        // Convert hailo_3d_image_shape_t to std::vector<size_t>
        shape = {hailo_info.shape.height, hailo_info.shape.width, hailo_info.shape.features};
    }
};

/**
 * Hailo Inference class
 * Handles Hailo device management and model inference
 */
class HailoInference {
public:
    HailoInference();
    ~HailoInference() = default;
    
    // Model management
    int load_model(const std::string& hef_path);
    
    // Inference
    std::map<std::string, cv::Mat> infer(const std::map<std::string, cv::Mat>& input_data, int hef_id);
    
    // Get model information
    std::vector<VStreamInfo> get_input_vstream_infos(int hef_id) const;
    std::vector<VStreamInfo> get_output_vstream_infos(int hef_id) const;
    
    // Shutdown control
    void request_shutdown() { shutdown_requested_ = true; }
    bool is_shutdown_requested() const { return shutdown_requested_; }
    
private:
    struct ModelInfo {
        std::string hef_path;
        std::vector<VStreamInfo> input_infos;
        std::vector<VStreamInfo> output_infos;
        
        std::shared_ptr<Hef> hef;
        std::shared_ptr<ConfiguredNetworkGroup> network_group;
        std::vector<InputVStream> input_vstreams;
        std::vector<OutputVStream> output_vstreams;
    };
    
    std::map<int, ModelInfo> models_;
    int next_model_id_;
    
    // Shutdown control
    std::atomic<bool> shutdown_requested_;
    
    std::unique_ptr<VDevice> device_;
};

/**
 * Detector class - C++ port of robot_hailo/robotdetector.py
 * 
 * This class provides the same functionality as the Python version but uses
 * a mock Hailo interface since no Hailo SDK is available on this system.
 * On target systems, the HailoInference class would be replaced with actual
 * Hailo SDK calls.
 */
class Detector : public DetectorBase {
public:
    /**
     * Constructor
     * @param robot_app The application type (e.g., "robotpalm", "robotface")
     * @param hailo_infer Pointer to HailoInference instance
     */
    Detector(const std::string& robot_app, std::shared_ptr<HailoInference> hailo_infer);
    
    /**
     * Destructor
     */
    virtual ~Detector() = default;
    
    /**
     * Load a Hailo model from file
     * @param model_path Path to the HEF model file
     */
    void load_model(const std::string& model_path);
    
    /**
     * Preprocess image data for inference
     * @param input Input image as cv::Mat
     * @return Preprocessed data as cv::Mat ready for inference
     */
    cv::Mat preprocess(const ImageType& input);
    
    /**
     * Make prediction on a single image
     * @param img Input image of shape (H, W, 3)
     * @return Detection results
     */
    std::vector<Detection> predict_on_image(const ImageType& img);
    
    /**
     * Make prediction on a batch of images
     * @param x Input batch of images of shape (b, H, W, 3)
     * @return Vector of detection results for each image
     */
    std::vector<std::vector<Detection>> predict_on_batch(const std::vector<ImageType>& x);
    
    // Profiling accessors
    double get_profile_pre() const { return profile_pre; }
    double get_profile_model() const { return profile_model; }
    double get_profile_post() const { return profile_post; }
    
    // Configuration methods
    void set_min_score_threshold(float threshold);

private:
    /**
     * Process raw model outputs into standardized tensor format
     * @param infer_results Raw model output from Hailo inference
     * @return Pair of processed tensors (scores, boxes)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_model_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection v0.07 outputs (6 outputs)
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_v07_outputs(const std::map<std::string, cv::Mat>& infer_results);
    
    /**
     * Process palm detection lite outputs (4 outputs)  
     */
    std::pair<std::vector<std::vector<std::vector<float>>>, 
              std::vector<std::vector<std::vector<float>>>>
    process_palm_lite_outputs(const std::map<std::string, cv::Mat>& infer_results);

private:
    std::string robot_app_;
    std::shared_ptr<HailoInference> hailo_infer_;
    int hef_id_;
    
    // Model information
    std::vector<VStreamInfo> input_vstream_infos_;
    std::vector<VStreamInfo> output_vstream_infos_;
    int num_inputs_;
    int num_outputs_;
    std::vector<size_t> input_shape_;
    std::vector<size_t> output_shape1_;
    std::vector<size_t> output_shape2_;
    
    // Profiling
    double profile_pre;
    double profile_model;
    double profile_post;
};

} // namespace robot
