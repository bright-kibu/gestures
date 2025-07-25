#include "Detector.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cassert>

using namespace hailort;

namespace robot {

// ============================================================================
// HailoInference Implementation
// ============================================================================

HailoInference::HailoInference() : next_model_id_(0), shutdown_requested_(false) {
    auto expected_device = VDevice::create();
    if (!expected_device) {
        throw std::runtime_error("Failed to create Hailo VDevice");
    }
    device_ = std::move(expected_device.value());
    std::cout << "[HailoInference] Hailo VDevice created successfully" << std::endl;
}

int HailoInference::load_model(const std::string& hef_path) {
    int model_id = next_model_id_++;
    
    std::cout << "[HailoInference.load_model] Loading: " << hef_path << std::endl;
    std::cout << "[HailoInference.load_model] Assigned Model ID: " << model_id << std::endl;
    
    ModelInfo model_info;
    model_info.hef_path = hef_path;
    
    try {
        // Load HEF file
        auto expected_hef = Hef::create(hef_path);
        if (!expected_hef) {
            throw std::runtime_error("Failed to load HEF file: " + hef_path);
        }
        model_info.hef = std::make_shared<Hef>(std::move(expected_hef.value()));
        
        // Configure network group
        auto expected_network_groups = device_->configure(*model_info.hef);
        if (!expected_network_groups) {
            throw std::runtime_error("Failed to configure network group");
        }
        auto network_groups = expected_network_groups.value();
        if (network_groups.empty()) {
            throw std::runtime_error("No network groups configured");
        }
        model_info.network_group = network_groups[0];
        
        // Get input/output stream info
        auto expected_input_vstream_infos = model_info.hef->get_input_vstream_infos();
        if (!expected_input_vstream_infos) {
            throw std::runtime_error("Failed to get input vstream infos");
        }
        auto input_vstream_infos = expected_input_vstream_infos.value();
        
        auto expected_output_vstream_infos = model_info.hef->get_output_vstream_infos();
        if (!expected_output_vstream_infos) {
            throw std::runtime_error("Failed to get output vstream infos");
        }
        auto output_vstream_infos = expected_output_vstream_infos.value();
        
        for (const auto& info : input_vstream_infos) {
            // Convert hailo_vstream_info_t to VStreamInfo
            std::vector<size_t> shape_vec = {info.shape.height, info.shape.width, info.shape.features};
            model_info.input_infos.emplace_back(info.name, shape_vec);
        }
        
        for (const auto& info : output_vstream_infos) {
            // Convert hailo_vstream_info_t to VStreamInfo
            std::vector<size_t> shape_vec = {info.shape.height, info.shape.width, info.shape.features};
            model_info.output_infos.emplace_back(info.name, shape_vec);
        }
        
        // Create VStreams once during model loading (but with optimized approach)
        std::cout << "[HailoInference.load_model] Creating VStreams..." << std::endl;
        
        // Create VStreams using Hailo SDK: obtain vstream parameters and request FLOAT32 outputs
        auto expected_input_params = model_info.network_group->make_input_vstream_params(
            false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!expected_input_params) {
            throw std::runtime_error("Failed to get input vstream params");
        }
        auto input_params = std::move(expected_input_params.value());
        auto expected_input_vstreams = model_info.network_group->create_input_vstreams(input_params);
        if (!expected_input_vstreams) {
            throw std::runtime_error("Failed to create input VStreams");
        }
        model_info.input_vstreams = std::move(expected_input_vstreams.value());
        std::cout << "[HailoInference.load_model] Created " << model_info.input_vstreams.size() << " input VStreams" << std::endl;

        auto expected_output_params = model_info.network_group->make_output_vstream_params(
            false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!expected_output_params) {
            throw std::runtime_error("Failed to get output vstream params");
        }
        auto output_params = std::move(expected_output_params.value());
        std::cout << "[HailoInference.load_model] Requesting FLOAT32 outputs" << std::endl;
        auto expected_output_vstreams = model_info.network_group->create_output_vstreams(output_params);
        if (!expected_output_vstreams) {
            throw std::runtime_error("Failed to create output VStreams");
        }
        model_info.output_vstreams = std::move(expected_output_vstreams.value());
        std::cout << "[HailoInference.load_model] Created " << model_info.output_vstreams.size() << " output VStreams" << std::endl;
        
        std::cout << "[HailoInference.load_model] Successfully loaded HEF with " 
                  << model_info.input_infos.size() << " inputs and " 
                  << model_info.output_infos.size() << " outputs" << std::endl;
                  
    } catch (const std::exception& e) {
        std::cerr << "[HailoInference.load_model] Error: " << e.what() << std::endl;
        throw;
    }
    
    models_[model_id] = std::move(model_info);
    return model_id;
}

std::map<std::string, cv::Mat> 
HailoInference::infer(const std::map<std::string, cv::Mat>& input_data, int hef_id) {
    std::map<std::string, cv::Mat> results;
    
    // Check for shutdown request immediately
    if (shutdown_requested_.load()) {
        return results; // Return empty results
    }
    
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    
    const auto& model_info = models_[hef_id];
    
    try {
        // Use pre-created VStreams with optimized approach (no sleep delays)
        auto& input_vstreams = const_cast<std::vector<InputVStream>&>(model_info.input_vstreams);
        auto& output_vstreams = const_cast<std::vector<OutputVStream>&>(model_info.output_vstreams);
        
        // Write input data to all streams first
        for (auto& input_vstream : input_vstreams) {
            // Check for shutdown before processing each stream
            if (shutdown_requested_.load()) {
                return results; // Return empty results
            }
            
            const std::string& stream_name = input_vstream.name();
            if (input_data.find(stream_name) != input_data.end()) {
                const cv::Mat& input_mat = input_data.at(stream_name);
                
                // Convert cv::Mat to uint8 buffer
                cv::Mat input_uint8;
                input_mat.convertTo(input_uint8, CV_8UC3);
                
                size_t data_size = input_uint8.total() * input_uint8.elemSize();
                
                // Create MemoryView for input data
                MemoryView input_buffer(input_uint8.data, data_size);
                auto status = input_vstream.write(input_buffer);
                if (HAILO_SUCCESS != status) {
                    std::cerr << "[HailoInference.infer] Failed to write to input stream " << stream_name 
                              << " with status: " << status << std::endl;
                    throw std::runtime_error("Failed to write to input stream: " + stream_name);
                }
            }
        }
        
        // Read float32 output data immediately (no sleep delay)
        for (auto& output_vstream : output_vstreams) {
            if (shutdown_requested_.load()) {
                return results;
            }
            const std::string& stream_name = output_vstream.name();
            auto info = output_vstream.get_info();
            // Calculate output size
            size_t output_size = info.shape.height * info.shape.width * info.shape.features;
            size_t buffer_bytes = output_size * sizeof(float);
            cv::Mat output_float32(1, static_cast<int>(output_size), CV_32F);
            MemoryView view(output_float32.data, buffer_bytes);
            auto status = output_vstream.read(view);
            if (HAILO_SUCCESS != status) {
                std::cerr << "[HailoInference.infer] Failed to read float32 from " << stream_name
                          << " status: " << status << std::endl;
                continue;
            }
            int h = static_cast<int>(info.shape.height);
            int w = static_cast<int>(info.shape.width);
            int c = static_cast<int>(info.shape.features);
            cv::Mat shaped(h, w, CV_32FC(c), output_float32.data);
            results[stream_name] = shaped.clone();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[HailoInference.infer] Error: " << e.what() << std::endl;
        throw;
    }
    
    return results;
}

std::vector<VStreamInfo> HailoInference::get_input_vstream_infos(int hef_id) const {
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    return models_.at(hef_id).input_infos;
}

std::vector<VStreamInfo> HailoInference::get_output_vstream_infos(int hef_id) const {
    if (models_.find(hef_id) == models_.end()) {
        throw std::runtime_error("Invalid model ID: " + std::to_string(hef_id));
    }
    return models_.at(hef_id).output_infos;
}

// ============================================================================
// Detector Implementation
// ============================================================================

Detector::Detector(const std::string& robot_app, std::shared_ptr<HailoInference> hailo_infer)
    : DetectorBase()
    , robot_app_(robot_app)
    , hailo_infer_(hailo_infer)
    , hef_id_(-1)
    , num_inputs_(0)
    , num_outputs_(0)
    , profile_pre(0.0)
    , profile_model(0.0)
    , profile_post(0.0) {
}

void Detector::load_model(const std::string& model_path) {
    if (DEBUG) {
        std::cout << "[Detector.load_model] Model File: " << model_path << std::endl;
    }
    
    hef_id_ = hailo_infer_->load_model(model_path);
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] HEF Id: " << hef_id_ << std::endl;
    }
    
    // Get model information
    input_vstream_infos_ = hailo_infer_->get_input_vstream_infos(hef_id_);
    output_vstream_infos_ = hailo_infer_->get_output_vstream_infos(hef_id_);
    
    num_inputs_ = static_cast<int>(input_vstream_infos_.size());
    num_outputs_ = static_cast<int>(output_vstream_infos_.size());
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] Number of Inputs: " << num_inputs_ << std::endl;
        for (int i = 0; i < num_inputs_; ++i) {
            std::cout << "[Detector.load_model] Input[" << i << "] Shape: (";
            for (size_t j = 0; j < input_vstream_infos_[i].shape.size(); ++j) {
                std::cout << input_vstream_infos_[i].shape[j];
                if (j < input_vstream_infos_[i].shape.size() - 1) std::cout << ", ";
            }
            std::cout << ") Name: " << input_vstream_infos_[i].name << std::endl;
        }
        
        std::cout << "[Detector.load_model] Number of Outputs: " << num_outputs_ << std::endl;
        for (int i = 0; i < num_outputs_; ++i) {
            std::cout << "[Detector.load_model] Output[" << i << "] Shape: (";
            for (size_t j = 0; j < output_vstream_infos_[i].shape.size(); ++j) {
                std::cout << output_vstream_infos_[i].shape[j];
                if (j < output_vstream_infos_[i].shape.size() - 1) std::cout << ", ";
            }
            std::cout << ") Name: " << output_vstream_infos_[i].name << std::endl;
        }
    }
    
    input_shape_ = input_vstream_infos_[0].shape;
    
    // Configure output shapes based on model type
    if (robot_app_ == "robotpalm" && num_outputs_ == 6) {
        output_shape1_ = {1, 2944, 1};
        output_shape2_ = {1, 2944, 18};
    } else if (robot_app_ == "robotpalm" && num_outputs_ == 4) {
        output_shape1_ = {1, 2016, 1};
        output_shape2_ = {1, 2016, 18};
    } else {
        // Default configuration
        output_shape1_ = {1, 2944, 1};
        output_shape2_ = {1, 2944, 18};
    }
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] Input Shape: (";
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            std::cout << input_shape_[i];
            if (i < input_shape_.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        
        std::cout << "[Detector.load_model] Output1 Shape: (";
        for (size_t i = 0; i < output_shape1_.size(); ++i) {
            std::cout << output_shape1_[i];
            if (i < output_shape1_.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        
        std::cout << "[Detector.load_model] Output2 Shape: (";
        for (size_t i = 0; i < output_shape2_.size(); ++i) {
            std::cout << output_shape2_[i];
            if (i < output_shape2_.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }
    
    // Set scales and anchors
    x_scale = static_cast<double>(input_shape_[1]);
    y_scale = static_cast<double>(input_shape_[0]);
    h_scale = static_cast<double>(input_shape_[0]);
    w_scale = static_cast<double>(input_shape_[1]);
    
    num_anchors = static_cast<int>(output_shape2_[1]);
    
    if (DEBUG) {
        std::cout << "[Detector.load_model] Num Anchors: " << num_anchors << std::endl;
    }
    
    config_model(robot_app_);
}

cv::Mat Detector::preprocess(const ImageType& input) {
    // Convert to uint8 format for Hailo input
    cv::Mat processed;
    input.convertTo(processed, CV_8UC3);
    
    return processed;
}

std::vector<Detection> Detector::predict_on_image(const ImageType& img) {
    // Use resize_pad to handle arbitrary input image sizes
    auto [resized_img, scale, pad] = resize_pad(img);
    
    // Convert single image to batch format
    std::vector<ImageType> batch = {resized_img};
    
    // Call predict_on_batch with properly sized image
    auto detections = predict_on_batch(batch);
    
    // Return first element from batch results
    if (!detections.empty()) {
        return detections[0];
    } else {
        return {};
    }
}

std::vector<std::vector<Detection>> Detector::predict_on_batch(const std::vector<ImageType>& x) {
    profile_pre = 0.0;
    profile_model = 0.0;
    profile_post = 0.0;
    
    // Validate input dimensions
    assert(x[0].channels() == 3);
    assert(x[0].rows == static_cast<int>(y_scale));
    assert(x[0].cols == static_cast<int>(x_scale));
    
    // 1. Preprocess the images
    auto start = std::chrono::high_resolution_clock::now();
    
    std::map<std::string, cv::Mat> input_data;
    auto preprocessed = preprocess(x[0]);
    input_data[input_vstream_infos_[0].name] = preprocessed;
    
    auto end = std::chrono::high_resolution_clock::now();
    profile_pre = std::chrono::duration<double>(end - start).count();
    
    // 2. Run inference
    start = std::chrono::high_resolution_clock::now();
    
    auto infer_results = hailo_infer_->infer(input_data, hef_id_);
    
    end = std::chrono::high_resolution_clock::now();
    profile_model = std::chrono::duration<double>(end - start).count();
    
    // 3. Post-process results
    start = std::chrono::high_resolution_clock::now();
    
    auto processed_outputs = process_model_outputs(infer_results);
    auto out1 = processed_outputs.first;   // scores
    auto out2 = processed_outputs.second;  // boxes
    
    // Validate output shapes
    assert(out1.size() == 1); // batch
    assert(out1[0].size() == static_cast<size_t>(num_anchors));
    assert(out1[0][0].size() == 1);
    
    assert(out2.size() == 1); // batch
    assert(out2[0].size() == static_cast<size_t>(num_anchors));
    assert(out2[0][0].size() == static_cast<size_t>(num_coords));
    
    // Convert to detection format (convert float to double)
    std::vector<std::vector<std::vector<double>>> out1_double(out1.size());
    std::vector<std::vector<std::vector<double>>> out2_double(out2.size());
    
    for (size_t i = 0; i < out1.size(); ++i) {
        out1_double[i].resize(out1[i].size());
        for (size_t j = 0; j < out1[i].size(); ++j) {
            out1_double[i][j].resize(out1[i][j].size());
            for (size_t k = 0; k < out1[i][j].size(); ++k) {
                out1_double[i][j][k] = static_cast<double>(out1[i][j][k]);
            }
        }
    }
    
    for (size_t i = 0; i < out2.size(); ++i) {
        out2_double[i].resize(out2[i].size());
        for (size_t j = 0; j < out2[i].size(); ++j) {
            out2_double[i][j].resize(out2[i][j].size());
            for (size_t k = 0; k < out2[i][j].size(); ++k) {
                out2_double[i][j][k] = static_cast<double>(out2[i][j][k]);
            }
        }
    }
    
    auto detections = tensors_to_detections(out2_double, out1_double, anchors_);
    
    // Apply non-maximum suppression
    std::cout << "[Detector.predict_on_batch] NMS threshold: " << min_score_thresh << std::endl;
    std::cout << "[Detector.predict_on_batch] output_vstream_infos_:" << std::endl;
    for (const auto &info : output_vstream_infos_) {
        std::cout << "  - " << info.name << " shape: (";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j] << (j + 1 < info.shape.size() ? ", " : "");
        }
        std::cout << ")" << std::endl;
    }
    std::vector<std::vector<Detection>> filtered_detections;
    for (size_t i = 0; i < detections.size(); ++i) {
        std::cout << "[Detector.predict_on_batch] Before NMS: " << detections[i].size() << " detections" << std::endl;
        auto wnms_detections = weighted_non_max_suppression(detections[i]);
        std::cout << "[Detector.predict_on_batch] After NMS: " << wnms_detections.size() << " detections" << std::endl;
        if (!wnms_detections.empty()) {
            filtered_detections.push_back(wnms_detections);
        }
    }
    
    end = std::chrono::high_resolution_clock::now();
    profile_post = std::chrono::duration<double>(end - start).count();
    
    return filtered_detections;
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>>
Detector::process_model_outputs(const std::map<std::string, cv::Mat>& infer_results) {
    
    if (robot_app_ == "robotpalm" && num_outputs_ == 6) {
        return process_palm_v07_outputs(infer_results);
    } else if (robot_app_ == "robotpalm" && num_outputs_ == 4) {
        return process_palm_lite_outputs(infer_results);
    } else {
        // Default to lite processing
        return process_palm_lite_outputs(infer_results);
    }
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>>
Detector::process_palm_v07_outputs(const std::map<std::string, cv::Mat>& infer_results) {
    
    // Process palm_detection_v0_07 outputs (6 outputs)
    // Score outputs: conv47 (8x8x6), conv44 (16x16x2), conv47_2 (32x32x2)
    // Box outputs: conv42 (8x8x108), conv45 (16x16,36), conv48 (32x32x36)
    
    // Get score tensors
    const cv::Mat& conv47_8x8x6 = infer_results.at(output_vstream_infos_[0].name);  // 8x8x6
    const cv::Mat& conv44_16x16x2 = infer_results.at(output_vstream_infos_[1].name); // 16x16x2
    const cv::Mat& conv47_32x32x2 = infer_results.at(output_vstream_infos_[2].name); // 32x32x2
    
    // Reshape and concatenate score tensors
    std::vector<std::vector<std::vector<float>>> out1(1);
    out1[0].resize(2944);
    
    // Flatten and concatenate: 32x32x2 (2048) + 16x16x2 (512) + 8x8x6 (384) = 2944
    int idx = 0;
    
    // Convert and flatten 32x32x2 -> 2048x1
    cv::Mat conv47_32_flat = conv47_32x32x2.reshape(1, 32*32*2);
    conv47_32_flat.convertTo(conv47_32_flat, CV_32F);
    for (int i = 0; i < conv47_32_flat.rows; ++i) {
        out1[0][idx++] = {conv47_32_flat.at<float>(i, 0)};
    }
    
    // Convert and flatten 16x16x2 -> 512x1
    cv::Mat conv44_flat = conv44_16x16x2.reshape(1, 16*16*2);
    conv44_flat.convertTo(conv44_flat, CV_32F);
    for (int i = 0; i < conv44_flat.rows; ++i) {
        out1[0][idx++] = {conv44_flat.at<float>(i, 0)};
    }
    
    // Convert and flatten 8x8x6 -> 384x1
    cv::Mat conv47_8_flat = conv47_8x8x6.reshape(1, 8*8*6);
    conv47_8_flat.convertTo(conv47_8_flat, CV_32F);
    for (int i = 0; i < conv47_8_flat.rows; ++i) {
        out1[0][idx++] = {conv47_8_flat.at<float>(i, 0)};
    }
    
    // Get box tensors
    const cv::Mat& conv42_8x8x108 = infer_results.at(output_vstream_infos_[3].name);  // 8x8x108
    const cv::Mat& conv45_16x16x36 = infer_results.at(output_vstream_infos_[4].name); // 16x16,36  
    const cv::Mat& conv48_32x32x36 = infer_results.at(output_vstream_infos_[5].name); // 32x32x36
    
    // Reshape and concatenate box tensors
    std::vector<std::vector<std::vector<float>>> out2(1);
    out2[0].resize(2944);
    
    idx = 0;
    
    // Process 32x32x36 -> 2048x18
    cv::Mat conv48_flat = conv48_32x32x36.reshape(1, 32*32*36);
    conv48_flat.convertTo(conv48_flat, CV_32F);
    
    for (int h = 0; h < 32; ++h) {
        for (int w = 0; w < 32; ++w) {
            for (int c = 0; c < 2; ++c) { // Only first 2 anchor boxes per cell
                std::vector<float> box_coords(18);
                for (int coord = 0; coord < 18; ++coord) {
                    int flat_idx = (h * 32 + w) * 36 + c * 18 + coord;
                    box_coords[coord] = conv48_flat.at<float>(flat_idx, 0);
                }
                out2[0][idx++] = box_coords;
            }
        }
    }
    
    // Process 16x16x36 -> 512x18
    cv::Mat conv45_flat = conv45_16x16x36.reshape(1, 16*16*36);
    conv45_flat.convertTo(conv45_flat, CV_32F);
    
    for (int h = 0; h < 16; ++h) {
        for (int w = 0; w < 16; ++w) {
            for (int c = 0; c < 2; ++c) {
                std::vector<float> box_coords(18);
                for (int coord = 0; coord < 18; ++coord) {
                    int flat_idx = (h * 16 + w) * 36 + c * 18 + coord;
                    box_coords[coord] = conv45_flat.at<float>(flat_idx, 0);
                }
                out2[0][idx++] = box_coords;
            }
        }
    }
    
    // Process 8x8x108 -> 384x18
    cv::Mat conv42_flat = conv42_8x8x108.reshape(1, 8*8*108);
    conv42_flat.convertTo(conv42_flat, CV_32F);
    
    for (int h = 0; h < 8; ++h) {
        for (int w = 0; w < 8; ++w) {
            for (int c = 0; c < 6; ++c) {
                std::vector<float> box_coords(18);
                for (int coord = 0; coord < 18; ++coord) {
                    int flat_idx = (h * 8 + w) * 108 + c * 18 + coord;
                    box_coords[coord] = conv42_flat.at<float>(flat_idx, 0);
                }
                out2[0][idx++] = box_coords;
            }
        }
    }
    
    return std::make_pair(out1, out2);
}

std::pair<std::vector<std::vector<std::vector<float>>>, 
          std::vector<std::vector<std::vector<float>>>>
Detector::process_palm_lite_outputs(const std::map<std::string, cv::Mat>& infer_results) {
    
    // Process palm_detection_lite outputs (4 outputs)
    // Score outputs: conv29 (24x24x2), conv24 (12x12x6)
    // Box outputs: conv30 (24x24x36), conv25 (12x12x108)
    
    // Debug: Print available output keys
    std::cout << "[Detector.process_palm_lite_outputs] Available outputs:" << std::endl;
    for (const auto& [key, value] : infer_results) {
        std::cout << "  - " << key << " shape: " << value.rows << "x" << value.cols << std::endl;
    }
    
    // Check if we have the expected outputs
    if (output_vstream_infos_.size() < 4) {
        throw std::runtime_error("Expected 4 outputs but got " + std::to_string(output_vstream_infos_.size()));
    }
    
    // Safely get score tensors with error checking
    auto it1 = infer_results.find(output_vstream_infos_[1].name);
    auto it0 = infer_results.find(output_vstream_infos_[0].name);
    
    if (it1 == infer_results.end() || it0 == infer_results.end()) {
        std::cerr << "[Detector.process_palm_lite_outputs] Missing score outputs!" << std::endl;
        std::cerr << "  Expected: " << output_vstream_infos_[1].name << ", " << output_vstream_infos_[0].name << std::endl;
        throw std::runtime_error("Missing required score outputs");
    }
    
    const cv::Mat& conv29_24x24x2 = it1->second;  // 24x24x2
    const cv::Mat& conv24_12x12x6 = it0->second;  // 12x12x6
    
    // Reshape and concatenate score tensors
    std::vector<std::vector<std::vector<float>>> out1(1);
    out1[0].resize(2016);
    
    // Flatten and concatenate: 24x24x2 (1152) + 12x12x6 (864) = 2016
    int idx = 0;
    
    // Convert cv::Mat to float and flatten 24x24x2 -> 1152x1
    cv::Mat conv29_flat = conv29_24x24x2.reshape(1, 24*24*2);
    conv29_flat.convertTo(conv29_flat, CV_32F);
    for (int i = 0; i < conv29_flat.rows; ++i) {
        out1[0][idx++] = {conv29_flat.at<float>(i, 0)};
    }
    
    // Convert cv::Mat to float and flatten 12x12x6 -> 864x1
    cv::Mat conv24_flat = conv24_12x12x6.reshape(1, 12*12*6);
    conv24_flat.convertTo(conv24_flat, CV_32F);
    for (int i = 0; i < conv24_flat.rows; ++i) {
        out1[0][idx++] = {conv24_flat.at<float>(i, 0)};
    }
    
    // Safely get box tensors with error checking
    auto it3 = infer_results.find(output_vstream_infos_[3].name);
    auto it2 = infer_results.find(output_vstream_infos_[2].name);
    
    if (it3 == infer_results.end() || it2 == infer_results.end()) {
        std::cerr << "[Detector.process_palm_lite_outputs] Missing box outputs!" << std::endl;
        std::cerr << "  Expected: " << output_vstream_infos_[3].name << ", " << output_vstream_infos_[2].name << std::endl;
        throw std::runtime_error("Missing required box outputs");
    }
    
    const cv::Mat& conv30_24x24x36 = it3->second; // 24x24x36
    const cv::Mat& conv25_12x12x108 = it2->second; // 12x12x108
    
    // Reshape and concatenate box tensors
    std::vector<std::vector<std::vector<float>>> out2(1);
    out2[0].resize(2016);
    
    idx = 0;
    
    // Convert cv::Mat to float and process 24x24x36 -> 1152x18
    cv::Mat conv30_flat = conv30_24x24x36.reshape(1, 24*24*36);
    conv30_flat.convertTo(conv30_flat, CV_32F);
    
    for (int h = 0; h < 24; ++h) {
        for (int w = 0; w < 24; ++w) {
            for (int c = 0; c < 2; ++c) { // 2 anchor boxes per cell
                std::vector<float> box_coords(18);
                for (int coord = 0; coord < 18; ++coord) {
                    int flat_idx = (h * 24 + w) * 36 + c * 18 + coord;
                    box_coords[coord] = conv30_flat.at<float>(flat_idx, 0);
                }
                out2[0][idx++] = box_coords;
            }
        }
    }
    
    // Convert cv::Mat to float and process 12x12x108 -> 864x18  
    cv::Mat conv25_flat = conv25_12x12x108.reshape(1, 12*12*108);
    conv25_flat.convertTo(conv25_flat, CV_32F);
    
    for (int h = 0; h < 12; ++h) {
        for (int w = 0; w < 12; ++w) {
            for (int c = 0; c < 6; ++c) { // 6 anchor boxes per cell
                std::vector<float> box_coords(18);
                for (int coord = 0; coord < 18; ++coord) {
                    int flat_idx = (h * 12 + w) * 108 + c * 18 + coord;
                    box_coords[coord] = conv25_flat.at<float>(flat_idx, 0);
                }
                out2[0][idx++] = box_coords;
            }
        }
    }
    
    return std::make_pair(out1, out2);
}

void Detector::set_min_score_threshold(float threshold) {
    min_score_thresh = static_cast<double>(threshold);
    std::cout << "[Detector.set_min_score_threshold] Set threshold to: " << min_score_thresh << std::endl;
}

} // namespace robot
